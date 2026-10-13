"""Exact repeated-update regression against an independently supplied 3999f37 archive.

Run with the same backend/runtime for both paths. Does not alter production loss
or random draws. Source hashes establish that the oracle is the original code.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from agents import PathBridgerAgent
from utils.core_dataset import CoreDataset
from utils.training import own_training_state, update_training_state
from utils.flax_utils import save_agent, restore_agent, load_agent
from utils.critic_value_params import extract_critic_value_params
from legacy_extract import load_extract_critic_value_params

ROOT = Path(__file__).resolve().parents[1]


def oracle(root, temp, *, reference_scene_layout=False):
    provenance = json.loads((ROOT / 'pathbridger_core/PROVENANCE.json').read_text())
    target = Path(temp) / 'reference3999'
    for rel, entry in provenance['files'].items():
        source = (root / rel).read_text()
        assert hashlib.sha256(source.encode()).hexdigest() == entry['original_sha256'], rel
        production = (ROOT / 'pathbridger_core' / rel).read_text()
        expected = source
        for patch in provenance.get('compatibility_patches', []):
            if patch['file'] == rel:
                assert expected.count(patch['replace_before']) == 1, rel
                expected = expected.replace(patch['replace_before'], patch['replace_after'], 1)
        assert production.replace('from pathbridger_core.', 'from ') == expected, rel
        assert hashlib.sha256(production.encode()).hexdigest() == entry['vendored_sha256'], rel
        out = target / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(source.replace('from utils.', 'from reference3999.utils.').replace('from agents.', 'from reference3999.agents.'))
    for rel in ['__init__.py', 'utils/__init__.py', 'agents/__init__.py']:
        (target / rel).touch()
    sys.path.insert(0, temp)
    from reference3999.agents.dynamics import DynamicsAgent
    from reference3999.agents.critic import CriticAgent
    from reference3999.utils.datasets import Dataset, PathHGCDataset
    from reference3999.utils.critic_sequence_dataset import CriticSequenceDataset
    if reference_scene_layout:
        # The untouched original cannot initialize scene. Use the independently
        # verified OGBench layout as an explicit reference-only correction;
        # never describe this reference as unmodified 3999f37 equivalence.
        import reference3999.utils.goal_representation as goals
        original_parse = goals._parse_scene_compact_layout
        goals._parse_scene_compact_layout = lambda dim: (1, 2, 2) if int(dim) == 40 else original_parse(dim)
    return DynamicsAgent, CriticAgent, Dataset, PathHGCDataset, CriticSequenceDataset


def exact(a, b, label):
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b), label + '/structure'
    aa, bb = jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b)
    assert len(aa) == len(bb), (label, len(aa), len(bb))
    for i, (x, y) in enumerate(zip(aa, bb)):
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape and x.dtype == y.dtype, (label, i, x.shape, y.shape, x.dtype, y.dtype)
        if np.issubdtype(x.dtype, np.number):
            assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)), (label, i, 'non-finite')
        assert x.tobytes() == y.tobytes(), (label, i, float(np.max(np.abs(x.astype(float)-y.astype(float)))))
    return {'exact': True, 'leaves': len(aa), 'max_abs': 0.0}


def states(a, d, c, label):
    return {name: exact(flax.serialization.to_state_dict(new), flax.serialization.to_state_dict(old), label+'/'+name)
            for name, new, old in [('dynamics', a.dynamics, d), ('critic', a.critic, c)]}


def rng_equal(a, b):
    assert a[0] == b[0] and a[2:] == b[2:]
    exact(a[1], b[1], 'numpy RNG')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--reference-run', type=Path)
    p.add_argument('--env', default='puzzle_3x3')
    p.add_argument('--distribution', default='pbf')
    p.add_argument('--dataset', required=True)
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference-scene-layout', action='store_true',
                   help='Compare scene to original code plus the explicit (1,2,2) layout correction.')
    p.add_argument('--reuse-train-buffers', action='store_true')
    args = p.parse_args()
    def update(agent, batch):
        return update_training_state(agent, batch) if args.reuse_train_buffers else agent.update(batch)
    cfg = importlib.import_module(f'configs.{args.distribution}.{args.env}').get_config()
    result = {'backend': jax.default_backend(), 'batch_size': 1024, 'steps': args.steps,
              'reuse_train_buffers': args.reuse_train_buffers, 'env': cfg.env_name, 'reference': '3999f37 + scene layout (1,2,2)' if args.reference_scene_layout else 'unmodified 3999f37', 'xla_flags': os.environ.get('XLA_FLAGS'), 'initialization': {}, 'trace': []}
    with tempfile.TemporaryDirectory() as temp:
        D, C, Dataset, DS, CS = oracle(args.reference_root, temp, reference_scene_layout=args.reference_scene_layout)
        raw = ogbench.load_dataset(args.dataset, compact_dataset=True)
        new_sampler = CoreDataset(raw, cfg)
        dc, cc = new_sampler.dynamics.config, new_sampler.critic.config
        old_d, old_c = DS(Dataset.create(**raw), dc), CS(Dataset.create(**raw), cc)
        valid = np.intersect1d(old_d.path_valid_idxs, old_c.valid_starts, assume_unique=False).astype(np.int64)
        exact(valid, new_sampler.valid_starts, 'valid starts')
        if args.reference_run:
            flags = json.loads((args.reference_run / 'flags.json').read_text())
            for name, computed in [('dynamics', dc), ('critic_agent', cc)]:
                assert json.loads(json.dumps(computed)) == flags[name], name
            result['original_config_match'] = True
        def reference_sample():
            idx = valid[np.random.randint(len(valid), size=1024)]
            return old_d.sample(1024, idxs=idx), old_c.sample(1024, idxs=idx)
        np.random.seed(1)
        db, cb = new_sampler.sample(1024)
        after = np.random.get_state()
        np.random.seed(1)
        odb, ocb = reference_sample()
        exact((db, cb), (odb, ocb), 'initial batch')
        rng_equal(after, np.random.get_state())
        a = PathBridgerAgent.create(1, db['observations'], db['actions'], cfg, critic_batch=cb)
        d = D.create(1, odb['observations'], dc, ex_actions=odb['actions'])
        c = C.create(1, ocb['observations'], None, ocb['action_chunk_actions'], cc, ex_goals=ocb.get('value_goals'))
        result['initialization'] = states(a, d, c, 'initialization')
        # Prefer AST extraction from the pinned PathBridger_legacy main.py.
        legacy_extract, helper_source = load_extract_critic_value_params()
        result['legacy_value_helper'] = helper_source
        batches = []
        for i in range(args.steps):
            before = np.random.get_state()
            batch = new_sampler.sample(1024)
            after = np.random.get_state()
            np.random.set_state(before)
            ref = reference_sample()
            exact(batch, ref, f'batch{i}')
            rng_equal(after, np.random.get_state())
            batches.append(jax.tree_util.tree_map(jnp.asarray, batch))
        result['sampler'] = {'exact': True, 'batches': 1+args.steps, 'numpy_rng_exact': True,
                             'valid_starts': len(valid), 'keys': [sorted(db), sorted(cb)]}
        print('INITIALIZATION_AND_SAMPLER_PASS', flush=True)
        if args.reuse_train_buffers:
            a = own_training_state(a)
        # Three updates from fresh initialization, including every original Q head.
        for i, batch in enumerate(batches[:3]):
            a, _ = update(a, batch)
            d, _ = d.update(batch[0], critic_value_params=legacy_extract(c))
            c, _ = c.update(batch[1])
            states(a, d, c, f'fresh {i+1}')
        result['fresh_updates_exact'] = min(3, len(batches))
        if args.reference_run:
            for name in ['dynamics', 'critic']:
                payload = pickle.load((args.reference_run / f'checkpoints/{name}/params_100000.pkl').open('rb'))['agent']
                a = a.replace(**{name: flax.serialization.from_state_dict(getattr(a, name), payload)})
                if name == 'dynamics':
                    d = flax.serialization.from_state_dict(d, payload)
                else:
                    c = flax.serialization.from_state_dict(c, payload)
            states(a, d, c, 'restored original 100k')
            if args.reuse_train_buffers:
                a = own_training_state(a)
        result['starting_optimizer_count'] = int(d.network.opt_state[0].count)
        # Native loss paths consume identical internal Flow noise/time keys.
        def gradients(d, c, batch, key, value_params):
            ld, gd = jax.value_and_grad(lambda params: d.total_loss(batch[0], params, key,
                value_params), has_aux=True)(d.network.params)
            lc, gc = jax.value_and_grad(lambda params: c.total_loss(batch[1], params), has_aux=True)(c.network.params)
            return ld, gd, lc, gc
        gf = jax.jit(gradients)
        new_grad = gf(a.dynamics, a.critic, batches[0], jax.random.PRNGKey(9013),
                      extract_critic_value_params(a.critic))
        old_grad = gf(d, c, batches[0], jax.random.PRNGKey(9013), legacy_extract(c))
        result['loss_metrics_and_gradients'] = exact(new_grad, old_grad, 'gradients')
        for i, batch in enumerate(batches):
            _, loss_key = jax.random.split(d.rng)
            dep_value = extract_critic_value_params(a.critic)
            leg_value = legacy_extract(c)
            step_gradients = exact(gf(a.dynamics, a.critic, batch, loss_key, dep_value),
                                   gf(d, c, batch, loss_key, leg_value), f'gradient {i+1}')
            a, info = update(a, batch)
            d, di = d.update(batch[0], critic_value_params=leg_value)
            c, ci = c.update(batch[1])
            metrics = {**{'dynamics/'+k: v for k, v in di.items()}, **{'critic/'+k: v for k, v in ci.items()}}
            row = {'step': i+1, 'gradients': step_gradients, 'states': states(a, d, c, f'update {i+1}'),
                   'metrics': exact(info, metrics, 'metrics')}
            result['trace'].append(row)
            print('UPDATE_EXACT', i+1, flush=True)
        # Checkpoint must retain original Adam, EMA, model RNG and host sampling.
        n = int(a.dynamics.network.step)-1
        checkpoint = save_agent(a, Path(temp)/'checkpoints', n)
        expected_batch = new_sampler.sample(1024)
        expected_rng = np.random.get_state()
        restored = restore_agent(a, checkpoint)
        exact(expected_batch, new_sampler.sample(1024), 'resumed batch')
        rng_equal(expected_rng, np.random.get_state())
        updated, _ = update(own_training_state(a), batches[0])
        resumed, _ = update(own_training_state(restored), batches[0])
        result['resume'] = states(resumed, updated.dynamics, updated.critic, 'resume update')
        loaded = load_agent(checkpoint)
        result['self_contained_checkpoint'] = states(loaded, a.dynamics, a.critic, 'load without data')
        # Reject config drift explicitly.
        altered = a.replace(config=a.config.copy(add_or_replace={'discount': .5}))
        try:
            restore_agent(altered, checkpoint)
        except ValueError:
            result['reject_config_drift'] = True
        else:
            raise AssertionError('Config drift accepted')
        result['passed'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('ALL_EXACT', args.output, flush=True)


if __name__ == '__main__':
    main()
