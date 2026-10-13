"""Compare 127f540-style deployment (hardcoded target value) to legacy entrypoint.

Legacy dynamics.update receives critic_value_params from the AST-extracted
original ``_extract_critic_value_params``. Deployment uses the broken
modules_target_value hardcode that shipped in 127f540 / 1a5a332.
Records the first mismatch; does not force agreement.
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
from legacy_extract import extract_helper_source, load_extract_critic_value_params

LEGACY_ROOT = Path('/home/choi/pb_reimpl_workspace/PathBridger_legacy')
DATASET = Path('/home/choi/.ogbench/data/puzzle-3x3-play-v0-val.npz')
OUT = ROOT / 'validation' / 'entrypoint_pre_fix_127f540.json'
STEPS = 256
SEED = 1
BATCH = 1024

FM_KEYS = (
    'dynamics/phase1/subgoal_flow_fm_raw',
    'dynamics/phase1/subgoal_flow_weight_mean',
    'dynamics/phase1/subgoal_flow_weighted_fm',
    'dynamics/phase1/subgoal_weighted_mse',
    'dynamics/phase1/subgoal_mse_weight_mean',
)


def sha_tree(obj) -> str:
    leaves = jax.tree_util.tree_leaves(obj)
    h = hashlib.sha256()
    for leaf in leaves:
        arr = np.asarray(leaf)
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def exact(a, b, label: str):
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b), label + '/structure'
    for i, (x, y) in enumerate(zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))):
        x, y = np.asarray(x), np.asarray(y)
        assert x.shape == y.shape and x.dtype == y.dtype, (label, i, x.shape, y.shape)
        if x.tobytes() != y.tobytes():
            max_abs = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if np.issubdtype(x.dtype, np.number) else None
            raise AssertionError(f'{label}/leaf{i} max_abs={max_abs}')


def metric_snapshot(info) -> dict:
    host = jax.device_get(info)
    out = {}
    for key in FM_KEYS:
        if key in host:
            out[key] = float(np.asarray(host[key]).reshape(()))
    # Always keep a few dynamics keys if present under alternate names.
    for key, value in host.items():
        if 'subgoal_flow' in key or 'subgoal_weighted' in key or 'subgoal_mse_weight' in key:
            arr = np.asarray(value)
            if arr.size == 1:
                out[str(key)] = float(arr.reshape(()))
    return out


def describe_module(params, chosen) -> str:
    value = params.get('modules_value')
    target = params.get('modules_target_value')
    if chosen is value:
        return 'modules_value'
    if chosen is target:
        return 'modules_target_value'
    # Identity may fail across tree copies; fall back to bytes.
    if value is not None and sha_tree(chosen) == sha_tree(value):
        same_target = target is not None and sha_tree(value) == sha_tree(target)
        return 'modules_value' + (' (== modules_target_value by content)' if same_target else '')
    if target is not None and sha_tree(chosen) == sha_tree(target):
        return 'modules_target_value'
    return 'unknown'


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


def broken_deploy_update(agent, batch):
    """127f540 / 1a5a332 PathBridgerAgent.update: hardcoded target value."""
    db, cb = batch
    d, di = agent.dynamics.update(
        db, critic_value_params=agent.critic.network.params['modules_target_value']
    )
    c, ci = agent.critic.update(cb)
    info = {**{'dynamics/' + k: v for k, v in di.items()},
            **{'critic/' + k: v for k, v in ci.items()}}
    return agent.replace(dynamics=d, critic=c), info


def main():
    extract_fn, helper_source = load_extract_critic_value_params()
    assert extract_helper_source() == helper_source

    cfg = importlib.import_module('configs.pbf.puzzle_3x3').get_config()
    raw = ogbench.load_dataset(str(DATASET), compact_dataset=True)
    result = {
        'comparison_baseline_commit': '127f54039aea221b11cae70f7b433f5be0597d8f',
        'invalid_prior_commit': '1a5a3324a9c359a7d14f0f0987e9e44beb2c9057',
        'legacy_reference_commit': '3999f3759fe6bfce1e62966bb38f0ce277add843',
        'helper_source': helper_source,
        'deployment_update': 'hardcoded modules_target_value (127f540 behavior)',
        'legacy_update': 'AST-extracted _extract_critic_value_params',
        'steps_requested': STEPS,
        'seed': SEED,
        'batch_size': BATCH,
        'first_mismatch': None,
        'passed': False,
        'note': (
            'Prior 1a5a332 check_entrypoint_256.py passed modules_target_value to BOTH '
            'sides, so its exact result is not evidence of original entrypoint equivalence.'
        ),
    }

    with tempfile.TemporaryDirectory() as temp:
        DynamicsAgent, CriticAgent, Dataset, PathHGCDataset, CriticSequenceDataset = load_legacy_modules(temp)
        from utils.core_config import legacy_configs

        dc, cc = legacy_configs(cfg, raw['observations'].shape[-1], raw['actions'].shape[-1])

        np.random.seed(SEED)
        dep_dataset = CoreDataset(raw, cfg)
        np.random.seed(SEED)
        dep_ex_d, dep_ex_c = dep_dataset.sample(BATCH)
        dep_agent = PathBridgerAgent.create(
            SEED, dep_ex_d['observations'], dep_ex_d['actions'], cfg, critic_batch=dep_ex_c
        )

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
        assert not hasattr(leg_c, '_is_trl'), 'legacy CriticAgent unexpectedly defines _is_trl'
        assert not hasattr(dep_agent.critic, '_is_trl'), 'deploy CriticAgent unexpectedly defines _is_trl'

        for step in range(1, STEPS + 1):
            before = np.random.get_state()
            dep_batch = dep_dataset.sample(BATCH)
            after_dep = np.random.get_state()
            np.random.set_state(before)
            picked = np.random.randint(len(leg_valid), size=BATCH)
            idxs = leg_valid[picked]
            leg_batch = (leg_dyn_ds.sample(BATCH, idxs=idxs), leg_crit_ds.sample(BATCH, idxs=idxs))
            after_leg = np.random.get_state()
            exact(dep_batch, leg_batch, f'batch@{step}')
            assert after_dep[0] == after_leg[0] and after_dep[2:] == after_leg[2:]
            exact(after_dep[1], after_leg[1], f'host_rng@{step}')

            # Capture which modules each side will pass BEFORE the update.
            dep_passed = dep_agent.critic.network.params['modules_target_value']
            leg_passed = extract_fn(leg_c)
            dep_module = describe_module(dep_agent.critic.network.params, dep_passed)
            leg_module = describe_module(leg_c.network.params, leg_passed)
            dep_hash = sha_tree(dep_passed)
            leg_hash = sha_tree(leg_passed)
            value_hash = sha_tree(leg_c.network.params['modules_value'])
            target_hash = sha_tree(leg_c.network.params['modules_target_value'])

            dep_agent, dep_info = broken_deploy_update(dep_agent, dep_batch)
            leg_d, leg_di = leg_d.update(leg_batch[0], critic_value_params=leg_passed)
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
                    f'dynamics@{step}',
                )
                exact(
                    flax.serialization.to_state_dict(dep_agent.critic),
                    flax.serialization.to_state_dict(leg_c),
                    f'critic@{step}',
                )
            except AssertionError as exc:
                result['first_mismatch'] = {
                    'step': step,
                    'detail': str(exc),
                    'deployment_value_module': dep_module,
                    'legacy_value_module': leg_module,
                    'deployment_passed_params_sha256': dep_hash,
                    'legacy_passed_params_sha256': leg_hash,
                    'modules_value_sha256': value_hash,
                    'modules_target_value_sha256': target_hash,
                    'passed_params_identical': dep_hash == leg_hash,
                    'deployment_fm_metrics': metric_snapshot(dep_info),
                    'legacy_fm_metrics': metric_snapshot(leg_info),
                    'legacy_critic_has__is_trl': hasattr(leg_c, '_is_trl'),
                }
                OUT.parent.mkdir(parents=True, exist_ok=True)
                OUT.write_text(json.dumps(result, indent=2) + '\n')
                print('FIRST_MISMATCH', json.dumps(result['first_mismatch'], indent=2), flush=True)
                print('WROTE', OUT, flush=True)
                return

            if step == 1 or step % 32 == 0:
                print(
                    f'STILL_EXACT {step} dep={dep_module} leg={leg_module} '
                    f'params_same={dep_hash == leg_hash}',
                    flush=True,
                )

        result['passed'] = True
        result['updates_compared'] = STEPS
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2) + '\n')
        print('UNEXPECTED_FULL_MATCH', OUT, flush=True)


if __name__ == '__main__':
    main()
