"""Compare legacy 3999f37 vs deployment entrypoint paths for 256 updates.

Each side initializes and samples through its own code path. Prefetch is off so
host RNG is recorded at the same logical point (immediately after the batch used
for that update is obtained, before the update consumes model RNG).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

# Match training runtime before importing JAX.
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

LEGACY_ROOT = Path('/home/choi/pb_reimpl_workspace/PathBridger_legacy')
DATASET = Path('/home/choi/.ogbench/data/puzzle-3x3-play-v0-val.npz')
OUT = ROOT / 'validation' / 'entrypoint_256.json'
STEPS = 256
SEED = 1
BATCH = 1024


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha_array(arr) -> str:
    arr = np.asarray(arr)
    return hashlib.sha256(arr.tobytes() + str(arr.shape).encode() + str(arr.dtype).encode()).hexdigest()


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


def rng_state():
    return np.random.get_state()


def rng_equal(a, b, label='numpy RNG'):
    assert a[0] == b[0] and a[2:] == b[2:], label
    exact(a[1], b[1], label)


def load_legacy_modules(temp: str):
    """Import unmodified legacy agents/datasets under an isolated package name."""
    provenance = json.loads((ROOT / 'pathbridger_core/PROVENANCE.json').read_text())
    target = Path(temp) / 'legacy3999'
    for rel in provenance['files']:
        source = (LEGACY_ROOT / rel).read_text()
        expected = provenance['files'][rel]['original_sha256']
        assert hashlib.sha256(source.encode()).hexdigest() == expected, rel
        out = target / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        text = source.replace('from utils.', 'from legacy3999.utils.').replace(
            'from agents.', 'from legacy3999.agents.'
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


def state_dict(agent_or_module):
    return flax.serialization.to_state_dict(agent_or_module)


def compare_states(dep_agent, leg_d, leg_c, label: str):
    exact(state_dict(dep_agent.dynamics), state_dict(leg_d), label + '/dynamics')
    exact(state_dict(dep_agent.critic), state_dict(leg_c), label + '/critic')


def main():
    assert LEGACY_ROOT.exists(), LEGACY_ROOT
    assert (LEGACY_ROOT / '.git').exists() or True
    legacy_head = os.popen(f'git -C {LEGACY_ROOT} rev-parse HEAD').read().strip()
    assert legacy_head == '3999f3759fe6bfce1e62966bb38f0ce277add843', legacy_head

    cfg = importlib.import_module('configs.pbf.puzzle_3x3').get_config()
    raw = ogbench.load_dataset(str(DATASET), compact_dataset=True)
    obs_dim = int(raw['observations'].shape[-1])
    act_dim = int(raw['actions'].shape[-1])
    dc, cc = legacy_configs(cfg, obs_dim, act_dim)

    result = {
        'legacy_reference_commit': legacy_head,
        'main_comparison_start_commit': os.popen(
            f'git -C {ROOT} rev-parse HEAD'
        ).read().strip(),
        'steps': STEPS,
        'seed': SEED,
        'batch_size': BATCH,
        'prefetch': False,
        'dataset_file': str(DATASET),
        'dataset_file_sha256': sha_file(DATASET),
        'array_hashes': {k: sha_array(v) for k, v in raw.items()},
        'runtime': {
            'jax': jax.__version__,
            'jaxlib': importlib.import_module('jaxlib').__version__,
            'flax': importlib.import_module('flax').__version__,
            'optax': importlib.import_module('optax').__version__,
            'numpy': np.__version__,
            'python': sys.version.split()[0],
            'backend': jax.default_backend(),
            'devices': [str(d) for d in jax.devices()],
            'xla_flags': os.environ.get('XLA_FLAGS'),
            'xla_python_client_preallocate': os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'),
        },
        'resolved_config': {
            'public': dict(cfg),
            'dynamics': dc,
            'critic': cc,
        },
        'paper_run_settings': {
            'status': 'partially_confirmed',
            'confirmed_from_supplementary_best_table': {
                'env': 'puzzle-3x3-play-v0',
                'endpoint_distribution': 'flow',
                'horizon_K': 25,
                'discount_gamma': 0.99,
                'endpoint_value_scale_gap': 10.0,
                'value_distance_weight_power_lambda': 0.5,
                'eval_num_candidates_N': 32,
                'eval_temperature_T': 1.0,
                'batch_size': 1024,
            },
            'note': (
                'Public configs/pbf/puzzle_3x3.py matches the paper Supplementary best '
                'table. Local research runs under Pathbridger_flow often used other '
                'K/gap values (e.g. K=40 gap=0.5); those are not treated as the paper run.'
            ),
            'estimated_or_unconfirmed': [
                'exact paper-run JAX/CUDA/driver versions on the original host',
                'exact paper-run git working tree dirty state',
            ],
        },
        'first_mismatch': None,
        'passed': False,
    }

    with tempfile.TemporaryDirectory() as temp:
        DynamicsAgent, CriticAgent, Dataset, PathHGCDataset, CriticSequenceDataset = load_legacy_modules(temp)

        # --- Deployment entrypoint path (official main.py semantics, no prefetch) ---
        np.random.seed(SEED)
        dep_dataset = CoreDataset(raw, cfg)
        result['valid_starts_count'] = int(len(dep_dataset.valid_starts))
        result['valid_starts_sha256'] = sha_array(dep_dataset.valid_starts)
        np.random.seed(SEED)
        dep_ex_d, dep_ex_c = dep_dataset.sample(BATCH)
        dep_agent = PathBridgerAgent.create(
            SEED, dep_ex_d['observations'], dep_ex_d['actions'], cfg, critic_batch=dep_ex_c
        )
        dep_host_rng_after_init = rng_state()

        # --- Legacy entrypoint path (main.py shared sampling, no SPI, no prefetch) ---
        np.random.seed(SEED)
        leg_data = Dataset.create(**dict(raw))
        leg_dyn_ds = PathHGCDataset(leg_data, dc)
        leg_crit_ds = CriticSequenceDataset(leg_data, cc)
        leg_valid = np.asarray(
            np.intersect1d(leg_dyn_ds.path_valid_idxs, leg_crit_ds.valid_starts, assume_unique=False),
            dtype=np.int64,
        )
        exact(dep_dataset.valid_starts, leg_valid, 'valid_starts')
        # Official reseeds before the example batch; legacy creates agents after an
        # example sample drawn with the same shared-index helper.
        np.random.seed(SEED)
        picked = np.random.randint(len(leg_valid), size=BATCH)
        idxs = leg_valid[picked]
        leg_ex_d = leg_dyn_ds.sample(BATCH, idxs=idxs)
        leg_ex_c = leg_crit_ds.sample(BATCH, idxs=idxs)
        exact((dep_ex_d, dep_ex_c), (leg_ex_d, leg_ex_c), 'example_batch')
        leg_d = DynamicsAgent.create(SEED, leg_ex_d['observations'], dc, ex_actions=leg_ex_d['actions'])
        leg_c = CriticAgent.create(
            SEED,
            leg_ex_c['observations'],
            None,
            leg_ex_c['action_chunk_actions'],
            cc,
            ex_goals=leg_ex_c.get('value_goals'),
        )
        compare_states(dep_agent, leg_d, leg_c, 'initialization')
        rng_equal(dep_host_rng_after_init, rng_state(), 'host RNG after init')

        result['initialization_exact'] = True
        print('INIT_PASS', flush=True)

        for step in range(1, STEPS + 1):
            # Record host RNG before sampling so prefetch cannot shift the probe point.
            before = rng_state()
            dep_batch = dep_dataset.sample(BATCH)
            after_dep = rng_state()

            np.random.set_state(before)
            picked = np.random.randint(len(leg_valid), size=BATCH)
            idxs = leg_valid[picked]
            leg_batch = (leg_dyn_ds.sample(BATCH, idxs=idxs), leg_crit_ds.sample(BATCH, idxs=idxs))
            after_leg = rng_state()
            try:
                exact(dep_batch, leg_batch, f'batch@{step}')
                rng_equal(after_dep, after_leg, f'host_rng_after_sample@{step}')
            except AssertionError as exc:
                result['first_mismatch'] = {
                    'step': step,
                    'stage': 'sample',
                    'detail': str(exc),
                    'batch_indices_dep_sha256': sha_array(idxs),
                }
                raise

            # Flow / model RNG live inside agent state; compare full state after update.
            dep_agent, dep_info = dep_agent.update(dep_batch)
            leg_d, leg_di = leg_d.update(
                leg_batch[0], critic_value_params=leg_c.network.params['modules_target_value']
            )
            leg_c, leg_ci = leg_c.update(leg_batch[1])
            leg_info = {
                **{'dynamics/' + k: v for k, v in leg_di.items()},
                **{'critic/' + k: v for k, v in leg_ci.items()},
            }
            try:
                exact(dep_info, leg_info, f'metrics@{step}')
                compare_states(dep_agent, leg_d, leg_c, f'state@{step}')
            except AssertionError as exc:
                result['first_mismatch'] = {
                    'step': step,
                    'stage': 'update',
                    'detail': str(exc),
                }
                raise

            if step == 1 or step % 32 == 0 or step == STEPS:
                print(f'UPDATE_EXACT {step}/{STEPS}', flush=True)

        result['updates_exact'] = STEPS
        result['passed'] = True

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Drop bulky nested configs from the on-disk summary; keep hashes / verdict.
    slim = dict(result)
    slim['resolved_config'] = {
        'public': result['resolved_config']['public'],
        'dynamics_key_count': len(result['resolved_config']['dynamics']),
        'critic_key_count': len(result['resolved_config']['critic']),
        'dynamics_sha256': hashlib.sha256(
            json.dumps(result['resolved_config']['dynamics'], sort_keys=True, default=str).encode()
        ).hexdigest(),
        'critic_sha256': hashlib.sha256(
            json.dumps(result['resolved_config']['critic'], sort_keys=True, default=str).encode()
        ).hexdigest(),
    }
    # Full resolved configs for audit.
    (ROOT / 'validation' / 'entrypoint_resolved_config.json').write_text(
        json.dumps(result['resolved_config'], indent=2, default=str) + '\n'
    )
    OUT.write_text(json.dumps(slim, indent=2, default=str) + '\n')
    print('ALL_EXACT', OUT, flush=True)


if __name__ == '__main__':
    main()
