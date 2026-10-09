"""Fast synthetic checks for the learned-goalspace contracts."""

import ast
import json
import random
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents.pathbridger import BridgeResidual, InverseDynamics
from configs.learned_goalspace.puzzle_3x3 import get_config
from learned_goalspace.checkpoints import (
    VARIANT_ARCHITECTURE,
    load_goal_encoder,
    restore_pretrainer,
    save_pretrain_checkpoint,
)
from learned_goalspace.dataset import (
    LONG_BAND_MIN,
    MEDIUM_BAND,
    FutureNCEDataset,
    MultiHorizonNCEDataset,
    SHORT_BAND,
)
from learned_goalspace.downstream import FrozenLearnedGoalspaceAgent
from learned_goalspace.fixed_representations import (
    FixedLinearEncoder,
    encoder_params,
    fit_pca16,
    load_fixed_representation,
    make_random16,
    save_fixed_representation,
)
from learned_goalspace.multihorizon import MultiHorizonNCEPretrainer
from learned_goalspace.pretrain import FutureNCEPretrainer
from learned_goalspace.probes import frozen_embeddings
from main_learned_goalspace import _evaluation_complete
from scripts.run_learned_goalspace_queue import (
    downstream_complete,
    downstream_dir,
    pretrain_complete,
    pretrain_dir,
    probes_complete,
)
from utils.datasets import Dataset
from utils.flax_utils import restore_agent, save_agent

OBS_DIM = 55


def compact_dataset():
    observations = np.arange(12 * OBS_DIM, dtype=np.float32).reshape(12, OBS_DIM)
    terminals = np.zeros(12, dtype=np.float32)
    terminals[[5, 11]] = 1
    return Dataset.create(observations=observations, terminals=terminals)


def pretrainer(seed=0):
    observations = np.linspace(-1, 1, 4 * OBS_DIM, dtype=np.float32).reshape(4, OBS_DIM)
    return FutureNCEPretrainer.create(
        seed,
        observations[:2],
        observations.std(axis=0),
        env_name='puzzle-3x3-play-v0',
    )


def downstream(agent=None):
    agent = pretrainer() if agent is None else agent
    observations = jnp.zeros((2, OBS_DIM), dtype=jnp.float32)
    actions = jnp.zeros((2, 2), dtype=jnp.float32)
    return FrozenLearnedGoalspaceAgent.create(
        0,
        observations,
        actions,
        get_config().to_dict(),
        agent.network.params['modules_goal_encoder'],
    )


def downstream_batch(seed=0, batch_size=4):
    rng = np.random.default_rng(seed)
    observations = rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32)
    offsets = np.arange(1, batch_size + 1, dtype=np.float32)
    return {
        'observations': observations,
        'next_observations': observations + 0.01,
        'actions': rng.normal(size=(batch_size, 2)).astype(np.float32),
        'bridge_targets': rng.normal(size=(batch_size, 5, OBS_DIM)).astype(np.float32),
        'endpoint_goals': rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32),
        'endpoint_targets': observations + 0.2,
        'value_goals': rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32),
        'value_offsets': offsets,
        'base_goals': rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32),
        'base_offsets': np.minimum(offsets, 5),
        'transitive_subgoals': rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32),
        'transitive_offsets': np.maximum(offsets - 1, 0),
        'transitive_valids': (offsets > 1).astype(np.float32),
    }


def test_contract_01_pretrain_uses_compact_full_observations():
    sampler = FutureNCEDataset(compact_dataset(), 0.99)
    batch = sampler.sample(3, anchors=np.array([0, 2, 7]))
    assert batch['queries'].shape == (3, OBS_DIM)
    np.testing.assert_array_equal(batch['queries'], compact_dataset()['observations'][[0, 2, 7]])


def test_contract_02_oracle_absent_from_pretrain_import_and_loss_boundary():
    root = Path(__file__).parents[1] / 'learned_goalspace'
    for name in (
        'pretrain.py',
        'dataset.py',
        'checkpoints.py',
        'fixed_representations.py',
        'multihorizon.py',
    ):
        tree = ast.parse((root / name).read_text())
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        assert not any('goal_representation' in ast.unparse(node) for node in imports)


def test_contract_03_terminals_are_never_crossed():
    sampler = FutureNCEDataset(compact_dataset(), 0.99)
    np.random.seed(7)
    anchors, futures, _ = sampler.sample_indices(4, np.array([0, 4, 6, 10]))
    assert np.all(futures <= sampler.final_for_index[anchors])
    assert np.all(futures > anchors)


def test_contract_04_geometric_positive_is_clipped_to_episode(monkeypatch):
    sampler = FutureNCEDataset(compact_dataset(), 0.99)
    monkeypatch.setattr(np.random, 'geometric', lambda p, size: np.array([1, 9], dtype=np.int64))
    anchors, futures, offsets = sampler.sample_indices(2, np.array([1, 3]))
    np.testing.assert_array_equal(futures, [2, 5])
    np.testing.assert_array_equal(offsets, [1, 2])


def test_contract_05_query_and_goal_encoder_are_distinct_subtrees():
    params = pretrainer().network.params
    assert 'modules_query_encoder' in params and 'modules_goal_encoder' in params
    query_paths = {str(path) for path, _ in jax.tree_util.tree_leaves_with_path(params['modules_query_encoder'])}
    goal_paths = {str(path) for path, _ in jax.tree_util.tree_leaves_with_path(params['modules_goal_encoder'])}
    assert any('projection' in path for path in query_paths)
    assert any('latent' in path for path in goal_paths)


def test_contract_06_exported_goal_encoder_is_16_dimensional():
    agent = pretrainer()
    encoded = agent.encode_goal(jnp.zeros((3, OBS_DIM), dtype=jnp.float32))
    assert encoded.shape == (3, 16)


def test_contract_07_probe_materializes_numpy_without_gradient_path():
    agent = pretrainer()
    values = frozen_embeddings(
        agent.network.params['modules_goal_encoder'],
        np.zeros((2, OBS_DIM), dtype=np.float32),
    )
    assert type(values) is np.ndarray
    assert values.shape == (2, 16)


def test_contract_08_downstream_encoder_has_zero_gradient_and_update():
    agent = downstream()
    params = agent.network.params
    gradients = jax.grad(lambda values: agent._encode(jnp.ones((2, OBS_DIM), dtype=jnp.float32), params=values).sum())(
        params
    )
    for leaf in jax.tree_util.tree_leaves(gradients['modules_goal_encoder']):
        np.testing.assert_array_equal(leaf, np.zeros_like(leaf))
    all_ones = jax.tree_util.tree_map(jnp.ones_like, params)
    updated = agent.network.apply_gradients(all_ones)
    for before, after in zip(
        jax.tree_util.tree_leaves(params['modules_goal_encoder']),
        jax.tree_util.tree_leaves(updated.params['modules_goal_encoder']),
    ):
        np.testing.assert_array_equal(before, after)


def test_contract_09_proposer_current_input_remains_full_state():
    agent = downstream()
    kernels = [
        np.asarray(value)
        for path, value in jax.tree_util.tree_leaves_with_path(agent.network.params['modules_endpoint'])
        if 'kernel' in str(path)
    ]
    # flow input = full s + phi(g=9) + full displacement + time
    assert any(kernel.ndim == 2 and kernel.shape[0] == 2 * OBS_DIM + 9 + 1 for kernel in kernels)


def test_contract_10_bridge_and_idm_are_original_definitions():
    modules = downstream().network.model_def.modules
    assert isinstance(modules['bridge'], BridgeResidual)
    assert isinstance(modules['idm'], InverseDynamics)


def test_contract_11_s_z_g_all_use_learned_encoder_in_high_level_paths():
    source = (Path(__file__).parents[1] / 'learned_goalspace' / 'downstream.py').read_text()
    assert 'for key in (' in source
    for key in ('observations', 'base_goals', 'value_goals', 'transitive_subgoals'):
        assert f"'{key}'" in source
    assert 'latent_s = self._encode(flat_s)' in source
    assert 'latent_z = self._encode(flat_z)' in source
    assert 'latent_g = self._encode(flat_g)' in source


def test_contract_12_checkpoint_resume_is_deterministic(tmp_path):
    agent = pretrainer(3)
    np.random.seed(19)
    random.seed(19)
    checkpoint = save_pretrain_checkpoint(agent, tmp_path, 7)
    batch = {
        'queries': jnp.linspace(-1, 1, 4 * OBS_DIM, dtype=jnp.float32).reshape(4, OBS_DIM),
        'goals': jnp.linspace(1, -1, 4 * OBS_DIM, dtype=jnp.float32).reshape(4, OBS_DIM),
    }
    expected_agent, expected_info = agent.update(batch)
    expected_numpy = np.random.random(4)
    expected_python = [random.random() for _ in range(4)]
    restored = restore_pretrainer(pretrainer(3), checkpoint, step=7)
    resumed_agent, resumed_info = restored.update(batch)
    np.testing.assert_array_equal(np.random.random(4), expected_numpy)
    np.testing.assert_array_equal([random.random() for _ in range(4)], expected_python)
    for left, right in zip(
        jax.tree_util.tree_leaves(expected_agent),
        jax.tree_util.tree_leaves(resumed_agent),
    ):
        np.testing.assert_array_equal(left, right)
    for key in expected_info:
        np.testing.assert_array_equal(expected_info[key], resumed_info[key])
    params, metadata = load_goal_encoder(
        checkpoint,
        env_name='puzzle-3x3-play-v0',
        obs_dim=OBS_DIM,
        step=7,
        allow_nonproduction_step=True,
    )
    assert metadata['step'] == 7
    assert params.keys() == agent.network.params['modules_goal_encoder'].keys()


def test_contract_12b_downstream_checkpoint_resume_is_deterministic(tmp_path):
    agent = downstream()
    batch = downstream_batch()
    checkpoint = save_agent(agent, tmp_path, 11)
    expected_agent, expected_info = agent.update(batch)
    restored = restore_agent(downstream(), checkpoint)
    resumed_agent, resumed_info = restored.update(batch)
    for left, right in zip(
        jax.tree_util.tree_leaves(expected_agent),
        jax.tree_util.tree_leaves(resumed_agent),
    ):
        np.testing.assert_array_equal(left, right)
    for key in expected_info:
        np.testing.assert_array_equal(expected_info[key], resumed_info[key])


def test_contract_13_existing_goalspace_suite_remains_separate():
    assert (Path(__file__).parent / 'test_goalspace_transitive_distance.py').is_file()


def test_queue_completion_requires_all_artifacts(tmp_path):
    pretrain = pretrain_dir(tmp_path, 'puzzle_3x3')
    (pretrain / 'checkpoints').mkdir(parents=True)
    (pretrain / 'complete.json').write_text('{"steps": 20}')
    (pretrain / 'checkpoints' / 'params_20.pkl').touch()
    assert pretrain_complete(tmp_path, 'puzzle_3x3', 20)

    run = downstream_dir(tmp_path, 'puzzle_3x3', 'LGS_TRL_W_FROZEN')
    (run / 'checkpoints').mkdir(parents=True)
    (run / 'complete.json').write_text('{"steps": 20}')
    (run / 'checkpoints' / 'params_20.pkl').touch()
    (run / 'downstream_results.csv').write_text('checkpoint\n20\n')
    assert not downstream_complete(
        tmp_path,
        'puzzle_3x3',
        'LGS_TRL_W_FROZEN',
        20,
        smoke=True,
    )
    smoke_result = {
        'checkpoint': 19,
        'h': 5,
        'env': 'puzzle-3x3-play-v0',
        'method': 'LGS_TRL_W_FROZEN',
        'variant': 'fullobs_future_nce',
        'seed': 0,
        'episodes_per_task': 1,
        'N': 32,
        'temperature': 1.0,
    }
    (run / 'smoke_evaluation.json').write_text(json.dumps(smoke_result))
    assert not _evaluation_complete(run, 20, smoke=True)
    assert not downstream_complete(
        tmp_path,
        'puzzle_3x3',
        'LGS_TRL_W_FROZEN',
        20,
        smoke=True,
    )
    smoke_result['checkpoint'] = 20
    (run / 'smoke_evaluation.json').write_text(json.dumps(smoke_result))
    assert _evaluation_complete(run, 20, smoke=True)
    assert downstream_complete(
        tmp_path,
        'puzzle_3x3',
        'LGS_TRL_W_FROZEN',
        20,
        smoke=True,
    )

    probes = tmp_path / 'probes' / 'puzzle_3x3' / 'fullobs_future_nce' / 'seed0'
    probes.mkdir(parents=True)
    header = 'env,variant,seed,checkpoint,representation,metric,value\n'
    row = 'puzzle-3x3-play-v0,fullobs_future_nce,0,20,learned_E,effective_rank,8.0\n'
    (probes / 'representation_metrics.csv').write_text(header + row)
    assert not probes_complete(probes, (20,), 'puzzle_3x3')
    (probes / 'probe_metrics.csv').write_text(header + row)
    assert probes_complete(probes, (20,), 'puzzle_3x3')


def test_16g_wrapper_checks_full_cgroup_contract():
    for script in (
        'run_learned_goalspace_16g.sh',
        'run_learned_goalspace_phase2_16g.sh',
    ):
        source = (Path(__file__).parents[1] / 'scripts' / script).read_text()
        for requirement in (
            'memory.max',
            'memory.high',
            'memory.swap.max',
            'memory.oom.group',
            'MemoryHigh=15G',
            'MemoryMax=16G',
            'MemorySwapMax=0',
            'OOMPolicy=kill',
            'KillMode=control-group',
        ):
            assert requirement in source


def test_pca16_fit_is_deterministic_and_sixteen_dimensional():
    observations = np.linspace(-1, 1, 40 * OBS_DIM, dtype=np.float32).reshape(40, OBS_DIM)
    left = fit_pca16(observations)
    right = fit_pca16(observations)
    np.testing.assert_allclose(left['mean'], right['mean'])
    np.testing.assert_allclose(left['kernel'], right['kernel'])
    assert left['kernel'].shape == (OBS_DIM, 16)
    assert left['mean'].shape == (OBS_DIM,)
    encoded = (observations - left['mean']) @ left['kernel']
    assert encoded.shape == (40, 16)


def test_random16_projection_is_seed_deterministic():
    left = make_random16(OBS_DIM, seed=11)
    right = make_random16(OBS_DIM, seed=11)
    other = make_random16(OBS_DIM, seed=12)
    np.testing.assert_array_equal(left['kernel'], right['kernel'])
    assert not np.array_equal(left['kernel'], other['kernel'])
    np.testing.assert_array_equal(left['mean'], np.zeros(OBS_DIM, dtype=np.float32))
    scale = np.sqrt(np.mean(np.square(left['kernel'])) * 16)
    assert 0.5 < float(scale) < 2.0


def test_fixed_representation_roundtrip_and_encoder(tmp_path):
    payload = fit_pca16(np.random.default_rng(0).normal(size=(20, OBS_DIM)).astype(np.float32))
    path = save_fixed_representation(tmp_path / 'representation.pkl', payload)
    loaded = load_fixed_representation(path)
    np.testing.assert_array_equal(loaded['kernel'], payload['kernel'])
    params = encoder_params(loaded)
    module = FixedLinearEncoder()
    observations = jnp.ones((3, OBS_DIM), dtype=jnp.float32)
    encoded = module.apply({'params': params}, observations, normalize=False)
    expected = (np.ones((3, OBS_DIM), dtype=np.float32) - payload['mean']) @ payload['kernel']
    np.testing.assert_allclose(np.asarray(encoded), expected, rtol=1e-5, atol=1e-5)
    normalized = module.apply({'params': params}, observations, normalize=True)
    norms = np.linalg.norm(np.asarray(normalized), axis=-1)
    np.testing.assert_allclose(norms, np.ones_like(norms), rtol=1e-5, atol=1e-5)


def test_fixed_encoder_frozen_zero_update():
    payload = make_random16(OBS_DIM, seed=3)
    observations = jnp.zeros((2, OBS_DIM), dtype=jnp.float32)
    actions = jnp.zeros((2, 2), dtype=jnp.float32)
    config = get_config('PCA16_GS_TRL_W').to_dict()
    agent = FrozenLearnedGoalspaceAgent.create(
        0,
        observations,
        actions,
        config,
        encoder_params(payload),
        encoder_module=FixedLinearEncoder(),
    )
    params = agent.network.params
    gradients = jax.grad(lambda values: agent._encode(jnp.ones((2, OBS_DIM), dtype=jnp.float32), params=values).sum())(
        params
    )
    for leaf in jax.tree_util.tree_leaves(gradients['modules_goal_encoder']):
        np.testing.assert_array_equal(leaf, np.zeros_like(leaf))
    all_ones = jax.tree_util.tree_map(jnp.ones_like, params)
    updated = agent.network.apply_gradients(all_ones)
    for before, after in zip(
        jax.tree_util.tree_leaves(params['modules_goal_encoder']),
        jax.tree_util.tree_leaves(updated.params['modules_goal_encoder']),
    ):
        np.testing.assert_array_equal(before, after)


def test_multihorizon_bands_never_cross_terminal_and_mask_impossible():
    sampler = MultiHorizonNCEDataset(compact_dataset(), 0.99)
    # Anchor 4 can only reach offset 1 (final=5); medium/long impossible.
    # Anchor 0 can reach short; medium/long impossible within first episode.
    # Anchor 6 has room for short/medium; long needs >=21 so impossible on this short episode.
    np.random.seed(0)
    batch = sampler.sample(3, anchors=np.array([4, 0, 6]))
    finals = sampler.final_for_index[np.array([4, 0, 6])]
    for band, low, high in (
        ('short', SHORT_BAND[0], SHORT_BAND[1]),
        ('medium', MEDIUM_BAND[0], MEDIUM_BAND[1]),
        ('long', LONG_BAND_MIN, 10**9),
    ):
        mask = batch[f'{band}_mask']
        offsets = batch[f'{band}_offsets'].astype(np.int64)
        assert np.all((mask < 0.5) | ((offsets >= low) & (offsets <= high)))
        assert np.all((mask < 0.5) | ((batch['anchor_indices'] + offsets) <= finals))
        assert np.all((mask < 0.5) | (offsets >= 1))
    assert batch['medium_mask'][0] == 0.0
    assert batch['long_mask'][0] == 0.0
    assert batch['long_mask'][2] == 0.0


def test_multihorizon_has_three_query_heads_and_shared_goal_encoder():
    observations = np.linspace(-1, 1, 4 * OBS_DIM, dtype=np.float32).reshape(4, OBS_DIM)
    agent = MultiHorizonNCEPretrainer.create(
        0,
        observations[:2],
        observations.std(axis=0),
        env_name='puzzle-3x3-play-v0',
    )
    params = agent.network.params
    assert 'modules_goal_encoder' in params
    assert 'modules_query_encoder' in params
    query_paths = {str(path) for path, _ in jax.tree_util.tree_leaves_with_path(params['modules_query_encoder'])}
    assert any('q_short' in path for path in query_paths)
    assert any('q_medium' in path for path in query_paths)
    assert any('q_long' in path for path in query_paths)
    short = params['modules_query_encoder']['q_short']['kernel']
    medium = params['modules_query_encoder']['q_medium']['kernel']
    long = params['modules_query_encoder']['q_long']['kernel']
    assert not np.array_equal(np.asarray(short), np.asarray(medium))
    assert not np.array_equal(np.asarray(medium), np.asarray(long))
    encoded = agent.encode_goal(jnp.zeros((2, OBS_DIM), dtype=jnp.float32))
    assert encoded.shape == (2, 16)
    assert agent.config['variant'] == 'fullobs_multihorizon_nce'
    assert VARIANT_ARCHITECTURE[agent.config['variant']] == agent.config['architecture']


def test_multihorizon_update_and_checkpoint_roundtrip(tmp_path):
    observations = np.linspace(-1, 1, 8 * OBS_DIM, dtype=np.float32).reshape(8, OBS_DIM)
    agent = MultiHorizonNCEPretrainer.create(
        1,
        observations[:2],
        np.ones(OBS_DIM, dtype=np.float32),
        env_name='puzzle-3x3-play-v0',
    )
    batch = {
        'queries': jnp.asarray(observations[:4]),
        'goals_short': jnp.asarray(observations[1:5]),
        'goals_medium': jnp.asarray(observations[2:6]),
        'goals_long': jnp.asarray(observations[3:7]),
        'short_mask': jnp.ones((4,), dtype=jnp.float32),
        'medium_mask': jnp.array([1, 1, 0, 0], dtype=jnp.float32),
        'long_mask': jnp.zeros((4,), dtype=jnp.float32),
    }
    updated, info = agent.update(batch)
    assert 'loss_short' in info and 'recall_at_1_medium' in info
    assert float(info['loss_long']) == 0.0
    checkpoint = save_pretrain_checkpoint(updated, tmp_path, 500_000)
    restored = restore_pretrainer(
        MultiHorizonNCEPretrainer.create(
            1,
            observations[:2],
            np.ones(OBS_DIM, dtype=jnp.float32),
            env_name='puzzle-3x3-play-v0',
        ),
        checkpoint,
        step=500_000,
    )
    params, metadata = load_goal_encoder(
        checkpoint,
        env_name='puzzle-3x3-play-v0',
        obs_dim=OBS_DIM,
        step=500_000,
        expected_variant='fullobs_multihorizon_nce',
    )
    assert metadata['variant'] == 'fullobs_multihorizon_nce'
    assert params.keys() == updated.network.params['modules_goal_encoder'].keys()
    for left, right in zip(
        jax.tree_util.tree_leaves(updated),
        jax.tree_util.tree_leaves(restored),
    ):
        np.testing.assert_array_equal(left, right)


def test_phase2_queue_has_no_oracle_run_gate_and_correct_order():
    source = (Path(__file__).parents[1] / 'scripts' / 'run_learned_goalspace_phase2_queue.py').read_text()
    assert 'MH_EXACT_ACC_THRESHOLD' not in source
    assert '--force-mh-downstream' not in source
    assert 'ALLOW_MH_DOWNSTREAM' not in source
    assert 'PREDECLARED_PRETRAIN_STEP = 500_000' in source
    assert 'unconditionally' in source
    assert "('fixed', 'pca16')" in source
    assert "('downstream', 'PCA16_GS_TRL_W')" in source
    # PCA fit must precede PCA downstream; Random fit precedes Random downstream.
    assert source.index("('fixed', 'pca16')") < source.index("('downstream', 'PCA16_GS_TRL_W')")
    assert source.index("('downstream', 'PCA16_GS_TRL_W')") < source.index("('fixed', 'random16')")
    assert source.index("('fixed', 'random16')") < source.index("('downstream', 'RANDOM16_GS_TRL_W')")
    assert source.index("('probes_mh', '')") < source.index("('downstream_mh', 'MH_LGS_TRL_W_FROZEN')")
    wrapper = (Path(__file__).parents[1] / 'scripts' / 'run_learned_goalspace_phase2_16g.sh').read_text()
    assert 'run_learned_goalspace_phase2_queue.py' in wrapper
    assert 'run_learned_goalspace_queue.py' not in wrapper
