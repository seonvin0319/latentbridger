"""Compare fixed deployment entrypoint paths to legacy 3999f37 for 256 updates.

Legacy always receives critic_value_params from the AST-extracted original
``_extract_critic_value_params``. Deployment is exercised once through
``PathBridgerAgent.update`` and once through ``update_training_state``
(main's default buffer-reuse path). Prefetch is off so host RNG is recorded
after the batch for that step is obtained.

This is a manually constructed training-loop check, not a full CLI invocation.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.runtime import configure_training_runtime

configure_training_runtime()

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench

from agents import PathBridgerAgent
from utils.core_dataset import CoreDataset
from utils.core_config import legacy_configs
from utils.training import own_training_state, update_training_state
from utils.critic_value_params import extract_critic_value_params
from legacy_extract import extract_helper_source, load_extract_critic_value_params

LEGACY_ROOT = Path('/home/choi/pb_reimpl_workspace/PathBridger_legacy')
DATASET = Path('/home/choi/.ogbench/data/puzzle-3x3-play-v0-val.npz')
OUT_DIR = ROOT / 'validation'
STEPS = 256
SEED = 1
BATCH = 1024


def sha_array(arr) -> str:
    arr = np.asarray(arr)
    return hashlib.sha256(arr.tobytes() + str(arr.shape).encode() + str(arr.dtype).encode()).hexdigest()


def sha_tree(obj) -> str:
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(obj):
        arr = np.asarray(leaf)
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def exact(a, b, label: str):
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b), label + '/structure'
    for i, (x, y) in enumerate(zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))):
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape and x.dtype == y.dtype, (label, i, x.shape, y.shape, x.dtype, y.dtype)
        if np.issubdtype(x.dtype, np.number):
            assert np.all(np.isfinite(x)) and np.all(np.isfinite(y)), (label, i, 'non-finite')
        if x.tobytes() != y.tobytes():
            max_abs = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if np.issubdtype(x.dtype, np.number) else None
            raise AssertionError(f'{label}/leaf{i} max_abs={max_abs}')


def rng_equal(a, b, label='numpy RNG'):
    assert a[0] == b[0] and a[2:] == b[2:], label
    exact(a[1], b[1], label)


def load_legacy_modules(temp: str):
    provenance = json.loads((ROOT / 'pathbridger_core/PROVENANCE.json').read_text())
    target = Path(temp) / 'legacy3999'
    for rel in provenance['files']:
        source = (LEGACY_ROOT / rel).read_text()
        expected = provenance['files'][rel]['original_sha256']
        assert hashlib.sha256(source.encode()).hexdigest() == expected, rel
        out = target / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        text = (
            source.replace('from utils.', 'from legacy3999.utils.')
            .replace('from agents.', 'from legacy3999.agents.')
        )
        out.write_text(text)
    for rel in ['__init__.py', 'utils/__init__.py', 'agents/__init__.py']:
        (target / rel).touch()
    sys.path.insert(0, temp)
    from legacy3999.agents.dynamics import DynamicsAgent
    from legacy3999.agents.critic import CriticAgent
    from legacy3999.utils.datasets import Dataset, PathHGCDataset
    from legacy3999.utils.critic_sequence_dataset import CriticSequenceDataset
    return DynamicsAgent, CriticAgent, Dataset, PathHGCDataset, CriticSequenceDataset


def assert_helper_matches_deploy(legacy_source: str):
    """Deployed helper body must match the AST-extracted original body."""
    # Strip def lines and compare bodies.
    leg_body = '\n'.join(legacy_source.splitlines()[1:])
    dep_source = Path(ROOT / 'utils/critic_value_params.py').read_text()
    tree = __import__('ast').parse(dep_source)
    for node in tree.body:
        if isinstance(node, __import__('ast').FunctionDef) and node.name == 'extract_critic_value_params':
            dep_fn = __import__('ast').get_source_segment(dep_source, node)
            dep_body = '\n'.join(dep_fn.splitlines()[1:])
            assert dep_body == leg_body, (dep_body, leg_body)
            return
    raise AssertionError('extract_critic_value_params missing')


def run_path(mode: str, update_fn, *, reuse_buffers: bool) -> dict:
    extract_fn, helper_source = load_extract_critic_value_params()
    assert_helper_matches_deploy(helper_source)
    # Deployed selector must agree with the AST-extracted original on objects.
    class _Probe:
        def __init__(self, params, has_trl=False):
            self.network = type('N', (), {'params': params})()
            if has_trl:
                self._is_trl = lambda: True

    for params, has_trl in [
        ({'modules_value': 'online', 'modules_target_value': 'target'}, False),
        ({'modules_value': 'online', 'modules_target_value': 'target'}, True),
        (None, False),
    ]:
        if params is None:
            assert extract_fn(None) is None and extract_critic_value_params(None) is None
        else:
            probe = _Probe(params, has_trl=has_trl)
            assert extract_fn(probe) is extract_critic_value_params(probe)

    cfg = importlib.import_module('configs.pbf.puzzle_3x3').get_config()
    raw = ogbench.load_dataset(str(DATASET), compact_dataset=True)
    result = {
        'mode': mode,
        'verification_kind': 'manual_training_loop',
        'cli_verification': False,
        'reuse_train_buffers': reuse_buffers,
        'legacy_helper': 'AST-extracted _extract_critic_value_params',
        'helper_source': helper_source,
        'legacy_reference_commit': '3999f3759fe6bfce1e62966bb38f0ce277add843',
        'steps': STEPS,
        'seed': SEED,
        'batch_size': BATCH,
        'prefetch': False,
        'dataset_file': str(DATASET),
        'dataset_file_sha256': hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        'runtime': {
            'jax': jax.__version__,
            'jaxlib': importlib.import_module('jaxlib').__version__,
            'flax': importlib.import_module('flax').__version__,
            'optax': importlib.import_module('optax').__version__,
            'numpy': np.__version__,
            'backend': jax.default_backend(),
            'devices': [str(d) for d in jax.devices()],
            'xla_flags': os.environ.get('XLA_FLAGS'),
        },
        'first_mismatch': None,
        'passed': False,
    }

    with tempfile.TemporaryDirectory() as temp:
        DynamicsAgent, CriticAgent, Dataset, PathHGCDataset, CriticSequenceDataset = load_legacy_modules(temp)
        dc, cc = legacy_configs(cfg, raw['observations'].shape[-1], raw['actions'].shape[-1])

        np.random.seed(SEED)
        dep_dataset = CoreDataset(raw, cfg)
        result['valid_starts_count'] = int(len(dep_dataset.valid_starts))
        result['valid_starts_sha256'] = sha_array(dep_dataset.valid_starts)
        np.random.seed(SEED)
        dep_ex_d, dep_ex_c = dep_dataset.sample(BATCH)
        dep_agent = PathBridgerAgent.create(
            SEED, dep_ex_d['observations'], dep_ex_d['actions'], cfg, critic_batch=dep_ex_c
        )
        if reuse_buffers:
            dep_agent = own_training_state(dep_agent)
        dep_host_after_init = np.random.get_state()

        np.random.seed(SEED)
        leg_data = Dataset.create(**dict(raw))
        leg_dyn_ds = PathHGCDataset(leg_data, dc)
        leg_crit_ds = CriticSequenceDataset(leg_data, cc)
        leg_valid = np.asarray(
            np.intersect1d(leg_dyn_ds.path_valid_idxs, leg_crit_ds.valid_starts, assume_unique=False),
            dtype=np.int64,
        )
        exact(dep_dataset.valid_starts, leg_valid, 'valid_starts')
        np.random.seed(SEED)
        picked = np.random.randint(len(leg_valid), size=BATCH)
        idxs = leg_valid[picked]
        leg_ex_d = leg_dyn_ds.sample(BATCH, idxs=idxs)
        leg_ex_c = leg_crit_ds.sample(BATCH, idxs=idxs)
        exact((dep_ex_d, dep_ex_c), (leg_ex_d, leg_ex_c), 'example_batch')
        leg_d = DynamicsAgent.create(SEED, leg_ex_d['observations'], dc, ex_actions=leg_ex_d['actions'])
        leg_c = CriticAgent.create(
            SEED, leg_ex_c['observations'], None, leg_ex_c['action_chunk_actions'], cc,
            ex_goals=leg_ex_c.get('value_goals'),
        )
        exact(flax.serialization.to_state_dict(dep_agent.dynamics), flax.serialization.to_state_dict(leg_d), 'init dynamics')
        exact(flax.serialization.to_state_dict(dep_agent.critic), flax.serialization.to_state_dict(leg_c), 'init critic')
        rng_equal(dep_host_after_init, np.random.get_state(), 'host RNG after init')
        assert not hasattr(leg_c, '_is_trl')
        assert not hasattr(dep_agent.critic, '_is_trl')
        result['initialization_exact'] = True
        print(f'{mode}: INIT_PASS', flush=True)

        # Gradient equality probe uses the same passed value params.
        def gradients(d, c, batch, key, value_params):
            ld, gd = jax.value_and_grad(
                lambda params: d.total_loss(batch[0], params, key, critic_value_params=value_params),
                has_aux=True,
            )(d.network.params)
            lc, gc = jax.value_and_grad(lambda params: c.total_loss(batch[1], params), has_aux=True)(c.network.params)
            return ld, gd, lc, gc

        gf = jax.jit(gradients)

        for step in range(1, STEPS + 1):
            before = np.random.get_state()
            dep_batch = dep_dataset.sample(BATCH)
            after_dep = np.random.get_state()
            np.random.set_state(before)
            picked = np.random.randint(len(leg_valid), size=BATCH)
            idxs = leg_valid[picked]
            leg_batch = (leg_dyn_ds.sample(BATCH, idxs=idxs), leg_crit_ds.sample(BATCH, idxs=idxs))
            after_leg = np.random.get_state()
            try:
                exact(dep_batch, leg_batch, f'batch@{step}')
                rng_equal(after_dep, after_leg, f'host_rng_after_sample@{step}')
            except AssertionError as exc:
                result['first_mismatch'] = {'step': step, 'stage': 'sample', 'detail': str(exc)}
                raise

            leg_value = extract_fn(leg_c)
            dep_value = extract_critic_value_params(dep_agent.critic)
            assert sha_tree(leg_value) == sha_tree(dep_value), f'value params diverge before update@{step}'

            # Compare loss/gradients at the pre-update state with identical keys.
            _, loss_key = jax.random.split(leg_d.rng)
            try:
                exact(
                    gf(dep_agent.dynamics, dep_agent.critic, dep_batch, loss_key, dep_value),
                    gf(leg_d, leg_c, leg_batch, loss_key, leg_value),
                    f'gradients@{step}',
                )
            except AssertionError as exc:
                result['first_mismatch'] = {'step': step, 'stage': 'gradients', 'detail': str(exc)}
                raise

            dep_agent, dep_info = update_fn(dep_agent, dep_batch)
            leg_d, leg_di = leg_d.update(leg_batch[0], critic_value_params=leg_value)
            leg_c, leg_ci = leg_c.update(leg_batch[1])
            leg_info = {
                **{'dynamics/' + k: v for k, v in leg_di.items()},
                **{'critic/' + k: v for k, v in leg_ci.items()},
            }
            try:
                exact(dep_info, leg_info, f'metrics@{step}')
                exact(
                    flax.serialization.to_state_dict(dep_agent.dynamics),
                    flax.serialization.to_state_dict(leg_d),
                    f'dynamics_state@{step}',
                )
                exact(
                    flax.serialization.to_state_dict(dep_agent.critic),
                    flax.serialization.to_state_dict(leg_c),
                    f'critic_state@{step}',
                )
            except AssertionError as exc:
                result['first_mismatch'] = {
                    'step': step,
                    'stage': 'update',
                    'detail': str(exc),
                    'legacy_passed_module': 'modules_value',
                    'legacy_passed_sha256': sha_tree(leg_value),
                }
                raise

            if step == 1 or step % 32 == 0 or step == STEPS:
                print(f'{mode}: UPDATE_EXACT {step}/{STEPS}', flush=True)

        result['updates_exact'] = STEPS
        result['passed'] = True
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # 1) Public non-destructive update path.
    plain = run_path(
        'plain_update',
        lambda agent, batch: agent.update(batch),
        reuse_buffers=False,
    )
    (OUT_DIR / 'entrypoint_256_plain.json').write_text(json.dumps(plain, indent=2) + '\n')
    print('PLAIN_PASS', flush=True)

    # 2) Main default: reuse_train_buffers via update_training_state.
    reuse = run_path(
        'reuse_train_buffers',
        update_training_state,
        reuse_buffers=True,
    )
    (OUT_DIR / 'entrypoint_256_reuse.json').write_text(json.dumps(reuse, indent=2) + '\n')
    print('REUSE_PASS', flush=True)

    summary = {
        'plain_update': {'passed': plain['passed'], 'steps': plain['steps'], 'verification_kind': plain['verification_kind']},
        'reuse_train_buffers': {'passed': reuse['passed'], 'steps': reuse['steps'], 'verification_kind': reuse['verification_kind']},
        'cli_verification': False,
        'passed': plain['passed'] and reuse['passed'],
        'corrected_from_invalid_commit': '1a5a3324a9c359a7d14f0f0987e9e44beb2c9057',
        'note': (
            '1a5a332 incorrectly passed modules_target_value to both sides. '
            'These results use the original helper on the legacy side and the '
            'matching deployment selector.'
        ),
    }
    (OUT_DIR / 'entrypoint_256.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('ALL_PATHS_EXACT', flush=True)


if __name__ == '__main__':
    main()
