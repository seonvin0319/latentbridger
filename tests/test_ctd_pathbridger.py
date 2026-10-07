"""CPU tests for the temporal quasimetric PathBridger."""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from agents.contrastive_transitive_distance_pathbridger import (
    ContrastiveTransitiveDistanceAgent,
    logits_from_distance,
    path_candidate_positive_mask,
    quasimetric_distance,
    safe_l2_norm,
    value_from_distance,
)
from agents.pathbridger import PathBridgerAgent, _ENDPOINT_WEIGHT_CAP, _TARGET_TAU, _replace_module_params
from configs.ctd.antmaze_medium import get_config as antmaze_config
from configs.ctd.cube_double import get_config as cube_double_config
from configs.ctd.cube_single import get_config as cube_single_config
from configs.ctd.puzzle_3x3 import get_config as puzzle_config

OBS_DIM = 8
ACTION_DIM = 2
BATCH = 8


def _config(variant):
    config = antmaze_config(variant)
    config.eval_num_candidates = 4
    return config


def _agent(variant='ctd_weighted', config=None, seed=0):
    observations = jnp.zeros((2, OBS_DIM), dtype=jnp.float32)
    actions = jnp.zeros((2, ACTION_DIM), dtype=jnp.float32)
    return ContrastiveTransitiveDistanceAgent.create(
        seed,
        observations,
        actions,
        (_config(variant) if config is None else config).to_dict(),
    )


def _batch(agent=None, seed=0):
    rng = np.random.default_rng(seed)
    observations = rng.normal(size=(BATCH, OBS_DIM)).astype(np.float32)
    actions = rng.normal(size=(BATCH, ACTION_DIM)).astype(np.float32)
    offsets = np.array([1, 2, 3, 4, 8, 12, 20, 30], dtype=np.float32)
    base_offsets = np.array([1, 2, 3, 4, 5, 1, 2, 5], dtype=np.float32)
    transitive_offsets = np.array([0, 0, 0, 0, 3, 4, 6, 10], dtype=np.float32)
    start = np.arange(BATCH, dtype=np.int32) * 100
    path_indices = start[:, None] + np.array([2, 4, 6, 12], dtype=np.int32)[None, :]
    episode = np.arange(BATCH, dtype=np.int32)
    return {
        'observations': observations,
        'next_observations': observations + 0.01,
        'actions': actions,
        'bridge_targets': rng.normal(size=(BATCH, 5, OBS_DIM)).astype(np.float32),
        'endpoint_goals': rng.normal(size=(BATCH, OBS_DIM)).astype(np.float32),
        'endpoint_targets': observations + 0.2,
        'value_goals': rng.normal(size=(BATCH, OBS_DIM)).astype(np.float32),
        'value_offsets': offsets,
        'base_goals': rng.normal(size=(BATCH, OBS_DIM)).astype(np.float32),
        'base_offsets': base_offsets,
        'transitive_subgoals': rng.normal(size=(BATCH, OBS_DIM)).astype(np.float32),
        'transitive_offsets': transitive_offsets,
        'transitive_valids': (offsets > 5).astype(np.float32),
        'path_positive_states': rng.normal(size=(BATCH, 4, OBS_DIM)).astype(np.float32),
        'path_positive_mask': np.tile(np.array([1, 1, 1, 0], np.float32), (BATCH, 1)),
        'path_start_indices': start,
        'path_goal_indices': start + 10,
        'path_positive_indices': path_indices,
        'path_anchor_episode': episode,
        'path_positive_episode': np.repeat(episode[:, None], 4, axis=1),
        'endpoint_offsets': np.full(BATCH, 5, np.float32),
        'future_actions': rng.normal(size=(BATCH, 5, ACTION_DIM)).astype(np.float32),
    }


def test_distance_is_nonnegative_and_zero_on_diagonal():
    h = jax.random.normal(jax.random.PRNGKey(0), (6, 64))
    p = jax.random.normal(jax.random.PRNGKey(1), (6, 64))
    distance = np.asarray(quasimetric_distance(h, p, h, p))
    assert np.all(distance >= -1e-6)
    np.testing.assert_allclose(distance, 0.0, atol=1e-6)
    zero_gradient = jax.grad(lambda x: safe_l2_norm(x))(jnp.zeros(64))
    assert np.isfinite(np.asarray(zero_gradient)).all()
    np.testing.assert_array_equal(zero_gradient, jnp.zeros(64))


def test_asymmetry_is_possible():
    h = jnp.zeros((64,))
    left = jnp.zeros((64,))
    right = jnp.ones((64,))
    forward = float(quasimetric_distance(h, left, h, right))
    backward = float(quasimetric_distance(h, right, h, left))
    assert forward == 1.0
    assert backward == 0.0


def test_triangle_inequality_on_random_embeddings():
    h = jax.random.normal(jax.random.PRNGKey(2), (5, 64))
    p = jax.random.normal(jax.random.PRNGKey(3), (5, 64))
    for i in range(5):
        for j in range(5):
            for k in range(5):
                direct = quasimetric_distance(h[i], p[i], h[j], p[j])
                composed = quasimetric_distance(h[i], p[i], h[k], p[k]) + quasimetric_distance(h[k], p[k], h[j], p[j])
                assert float(direct) <= float(composed) + 1e-4


def test_network_distance_obeys_the_same_identities():
    agent = _agent()
    states = jax.random.normal(jax.random.PRNGKey(4), (5, OBS_DIM))
    same = np.asarray(agent._metric_distance(states, states, name='value'))
    np.testing.assert_allclose(same, 0.0, atol=1e-5)
    assert np.all(same >= -1e-5)
    h, p = agent._metric_encode(states, name='value')
    for i in range(5):
        for j in range(5):
            for k in range(5):
                direct = quasimetric_distance(h[i], p[i], h[j], p[j])
                composed = quasimetric_distance(h[i], p[i], h[k], p[k]) + quasimetric_distance(h[k], p[k], h[j], p[j])
                assert float(direct) <= float(composed) + 1e-4


def test_state_and_goal_share_one_encoder():
    agent = _agent()
    keys = ['.'.join(str(part) for part in path) for path, _ in jax.tree_util.tree_leaves_with_path(agent.network.params['modules_value'])]
    assert any('trunk' in key for key in keys)
    assert sum('trunk' in key and 'kernel' in key for key in keys) == 3
    assert not any('goal' in key or 'psi' in key or 'phi' in key for key in keys)
    flat = ' '.join(jax.tree_util.tree_flatten(agent.network.params)[1] and [
        '.'.join(str(part) for part in path)
        for path, _ in jax.tree_util.tree_leaves_with_path(agent.network.params)
    ])
    assert 'modules_phi' not in flat and 'modules_psi' not in flat


def test_value_is_gamma_to_the_distance():
    discount = 0.99
    distance = jnp.array([0.5, 2.0, 10.0])
    expected = np.clip(discount ** np.asarray(distance), 1e-6, 1 - 1e-6)
    np.testing.assert_allclose(value_from_distance(distance, discount), expected, rtol=1e-5)
    logits = logits_from_distance(distance, discount)
    np.testing.assert_allclose(jax.nn.sigmoid(logits), expected, rtol=1e-5)


def test_short_target_matches_distance_k_and_trl_terms_exist():
    agent = _agent('dtrl_uniform')
    batch = _batch(agent)
    _, info = agent.value_loss(batch, agent.network.params)
    expected = np.mean(0.99 ** batch['base_offsets'])
    np.testing.assert_allclose(float(info['value/base_target_mean']), expected, rtol=1e-5)
    for key in ('value/self_loss', 'value/base_loss', 'value/transitive_loss'):
        assert np.isfinite(float(info[key]))


def test_transitive_product_is_distance_addition():
    discount = 0.99
    left = value_from_distance(jnp.array(3.0), discount)
    right = value_from_distance(jnp.array(4.0), discount)
    composed = value_from_distance(jnp.array(7.0), discount)
    np.testing.assert_allclose(left * right, composed, rtol=1e-5)


def test_target_metric_ema_uses_the_pathbridger_rate():
    agent = _agent('dtrl_weighted')
    batch = _batch(agent)
    before = jax.tree_util.tree_leaves(agent.network.params['modules_target_value'])
    updated, _ = agent.update({key: jnp.asarray(value) for key, value in batch.items()})
    online = jax.tree_util.tree_leaves(updated.network.params['modules_value'])
    target = jax.tree_util.tree_leaves(updated.network.params['modules_target_value'])
    for old, new_online, new_target in zip(before, online, target):
        expected = _TARGET_TAU * new_online + (1.0 - _TARGET_TAU) * old
        np.testing.assert_allclose(new_target, expected, rtol=1e-5, atol=1e-5)


def test_nce_uses_the_metric_and_diagonal_positives():
    agent = _agent('ctd_weighted')
    batch = _batch(agent)
    normalized, info = agent.nce_loss(batch, agent.network.params)
    h_state, p_state = agent._metric_encode(batch['observations'], name='value')
    h_goal, p_goal = agent._metric_encode(batch['value_goals'], name='value')
    from agents.contrastive_transitive_distance_pathbridger import pairwise_distance
    distance = pairwise_distance(h_state, p_state, h_goal, p_goal)
    bias = agent._metric_bias(batch['value_goals'])
    expected = bias[None, :] - distance / float(agent.config['contrastive_temperature'])
    positive = jnp.diag(expected)
    expected_loss = (jax.nn.logsumexp(expected, axis=1) - positive).mean() / jnp.log(BATCH)
    temperature = float(agent.config['contrastive_temperature'])
    assert temperature == float(agent.config['horizon'])
    np.testing.assert_allclose(normalized, expected_loss, rtol=1e-5)
    assert float(info['nce/positive_rank']) >= 1.0
    assert distance.shape == (BATCH, BATCH)
    assert np.isfinite(float(info['nce/bias_mean']))


def test_nuisance_bias_receives_only_future_nce_gradient():
    agent = _agent('ctd_weighted')
    batch = {key: jnp.asarray(value) for key, value in _batch(agent).items()}

    def bias_grad(grads):
        return jax.tree_util.tree_leaves(grads['modules_value']['nce_bias'])

    value_grads = jax.grad(lambda params: agent.value_loss(batch, params)[0])(agent.network.params)
    assert all(float(jnp.max(jnp.abs(leaf))) == 0.0 for leaf in bias_grad(value_grads))
    nce_grads = jax.grad(lambda params: agent.nce_loss(batch, params)[0])(agent.network.params)
    assert any(float(jnp.max(jnp.abs(leaf))) > 0.0 for leaf in bias_grad(nce_grads))
    source = Path('agents/contrastive_transitive_distance_pathbridger.py').read_text()
    inference = source[source.index('def sample_action_chunks'):source.index('@classmethod', source.index('def sample_action_chunks'))]
    weighting = source[source.index('def _endpoint_weights'):source.index('@partial', source.index('def _endpoint_weights'))]
    assert '_metric_bias' not in inference
    assert '_metric_bias' not in weighting


def test_lambda_zero_removes_nce_exactly():
    agent = _agent('dtrl_uniform')
    batch = {key: jnp.asarray(value) for key, value in _batch(agent).items()}
    rng = jax.random.PRNGKey(7)
    loss, info = agent.total_loss(batch, agent.network.params, rng)
    parent, _ = PathBridgerAgent.total_loss(agent, batch, agent.network.params, rng)
    np.testing.assert_allclose(loss, parent, rtol=0, atol=0)
    assert float(info['nce/lambda']) == 0.0
    assert float(agent.config['lambda_nce']) == 0.0


def test_uniform_weights_are_one_and_weighted_matches_the_cap_formula():
    uniform = _agent('dtrl_uniform')
    weighted = _agent('dtrl_weighted')
    batch = _batch(uniform)
    ones, gap = uniform._endpoint_weights(
        batch['observations'], batch['endpoint_goals'], batch['endpoint_targets'],
    )
    np.testing.assert_array_equal(np.asarray(ones), np.ones(BATCH))
    np.testing.assert_array_equal(np.asarray(gap), np.zeros(BATCH))
    weights, delta = weighted._endpoint_weights(
        batch['observations'], batch['endpoint_goals'], batch['endpoint_targets'],
    )
    scale = float(weighted.config['endpoint_value_scale'])
    expected = np.minimum(_ENDPOINT_WEIGHT_CAP, np.exp(scale * np.asarray(delta)))
    np.testing.assert_allclose(weights, expected, rtol=1e-5, atol=1e-5)
    assert _ENDPOINT_WEIGHT_CAP == 5.0
    assert float(np.asarray(weights).max()) <= 5.0 + 1e-5


def test_proposer_has_no_freeze_and_receives_an_update():
    source = Path('agents/contrastive_transitive_distance_pathbridger.py').read_text()
    assert 'freeze_proposer' not in source
    assert 'stop_endpoint' not in source
    agent = _agent('ctd_weighted')
    batch = {key: jnp.asarray(value) for key, value in _batch(agent).items()}
    before = jax.tree_util.tree_leaves(agent.network.params['modules_endpoint'])
    updated, info = agent.update(batch)
    after = jax.tree_util.tree_leaves(updated.network.params['modules_endpoint'])
    assert any(not np.allclose(left, right) for left, right in zip(before, after))
    assert np.isfinite(float(info['endpoint/loss']))
    assert int(updated.network.step) > int(agent.network.step)


def test_distance_ranking_matches_value_product_ranking():
    discount = 0.99
    distance_sz = np.array([1.0, 4.0, 0.5, 2.0])
    distance_zg = np.array([3.0, 0.2, 2.5, 2.0])
    distance_score = distance_sz + distance_zg
    product = (discount ** distance_sz) * (discount ** distance_zg)
    assert int(distance_score.argmin()) == int(product.argmax())


def test_bridge_and_idm_match_pathbridger_for_a_single_candidate():
    observations = jnp.zeros((2, OBS_DIM))
    actions = jnp.zeros((2, ACTION_DIM))
    pb_config = _config('dtrl_uniform').to_dict()
    pb_config['eval_num_candidates'] = 1
    pb = PathBridgerAgent.create(0, observations, actions, pb_config)
    ctd = ContrastiveTransitiveDistanceAgent.create(1, observations, actions, pb_config)
    params = ctd.network.params
    for name in ('endpoint', 'bridge', 'idm'):
        params = _replace_module_params(params, name, pb.network.params[f'modules_{name}'])
    ctd = ctd.replace(network=ctd.network.replace(params=params))
    obs = jax.random.normal(jax.random.PRNGKey(8), (3, OBS_DIM))
    goals = jax.random.normal(jax.random.PRNGKey(9), (3, OBS_DIM))
    endpoints = obs + 0.3
    np.testing.assert_allclose(
        pb._construct_bridge_prefix(obs, endpoints),
        ctd._construct_bridge_prefix(obs, endpoints),
        atol=1e-6,
    )
    pb_actions = pb.sample_action_chunks(obs, goals, jax.random.PRNGKey(10), num_candidates=1, temperature=0.0)
    ctd_actions = ctd.sample_action_chunks(obs, goals, jax.random.PRNGKey(10), num_candidates=1, temperature=0.0)
    np.testing.assert_allclose(pb_actions, ctd_actions, atol=1e-5)


def test_environment_settings_match_pbf():
    expected = {
        cube_single_config: dict(horizon=40, discount=0.99, eval_num_candidates=1, eval_temperature=0.0, endpoint_value_scale=5.0, value_distance_weight_power=0.7),
        cube_double_config: dict(horizon=40, discount=0.99, eval_num_candidates=8, eval_temperature=0.25, endpoint_value_scale=10.0, value_distance_weight_power=1.0),
        puzzle_config: dict(horizon=25, discount=0.99, eval_num_candidates=32, eval_temperature=1.0, endpoint_value_scale=10.0, value_distance_weight_power=0.5),
        antmaze_config: dict(horizon=25, discount=0.99, eval_num_candidates=8, eval_temperature=0.25, endpoint_value_scale=10.0, value_distance_weight_power=0.0),
    }
    for getter, values in expected.items():
        for variant, lambda_nce, weighting in (
            ('dtrl_uniform', 0.0, 'uniform'),
            ('dtrl_weighted', 0.0, 'transitive'),
            ('ctd_uniform', 1.0, 'uniform'),
            ('ctd_weighted', 1.0, 'transitive'),
        ):
            config = getter(variant)
            for key, value in values.items():
                assert config[key] == value
            assert config.lambda_nce == lambda_nce
            assert config.proposer_weighting == weighting
            assert config.contrastive_temperature == float(config.horizon)
            assert config.tau_path == 5.0
            assert config.lambda_pathnce == 0.0
            assert config.lambda_bridge_geo == 0.0
            assert tuple(config.critic_p) == (0.0, 1.0, 0.0, 0.0)


def _path_batch(batch, start, goal, indices, mask, episode):
    row_count = len(start)
    out = {
        key: (value[:row_count].copy() if hasattr(value, 'shape') and value.shape[0] == BATCH else value)
        for key, value in batch.items()
    }
    states = np.broadcast_to(batch['observations'][:1], (len(indices), batch['observations'].shape[-1])).copy()
    states += indices[:, None].astype(np.float32)
    out.update({
        'path_positive_states': states[None].repeat(row_count, axis=0).astype(np.float32),
        'path_positive_mask': np.asarray(mask, dtype=np.float32),
        'path_start_indices': np.asarray(start, dtype=np.int32),
        'path_goal_indices': np.asarray(goal, dtype=np.int32),
        'path_positive_indices': np.asarray(indices, dtype=np.int32)[None].repeat(len(start), axis=0),
        'path_anchor_episode': np.asarray(episode, dtype=np.int32),
        'path_positive_episode': np.full((len(start), len(indices)), episode[0], dtype=np.int32),
        'endpoint_offsets': np.full(len(start), 5, dtype=np.int32),
    })
    return out


def test_path_positives_are_ordered_and_do_not_cross_terminals():
    from utils.datasets import Dataset, PathBridgerDataset

    length = 30
    observations = np.arange(length, dtype=np.float32)[:, None]
    terminals = np.zeros(length, np.float32)
    terminals[14] = 1.0
    terminals[-1] = 1.0
    dataset = PathBridgerDataset(
        Dataset.create(observations=observations, actions=np.zeros((length, 1), np.float32), terminals=terminals),
        SimpleNamespace(horizon=5, discount=0.99, dynamics_p=(0.0, 0.0, 1.0, 0.0), critic_p=(0.0, 1.0, 0.0, 0.0)),
    )
    fields = dataset._path_positive_fields(np.array([0, 16]), np.array([10, 20]), observations)
    assert fields['path_positive_mask'][0].sum() == 4
    assert np.all((fields['path_positive_indices'][0] > 0) & (fields['path_positive_indices'][0] < 10))
    assert fields['path_positive_mask'][1].sum() == 3
    assert np.all(fields['path_positive_indices'][1][fields['path_positive_mask'][1] > 0] <= 19)
    assert np.all(fields['path_positive_indices'][1][fields['path_positive_mask'][1] > 0] >= 17)
    offsets = dataset.sample(4, idxs=np.array([0, 5, 16, 20]))['endpoint_offsets']
    assert np.all((offsets >= 1) & (offsets <= 5))


def test_in_interval_candidate_is_positive_not_false_negative():
    mask = path_candidate_positive_mask(
        start=np.array([0, 0]),
        goal=np.array([10, 10]),
        anchor_episode=np.array([0, 1]),
        candidate_index=np.array([3, 12, 4]),
        candidate_episode=np.array([0, 0, 1]),
        candidate_valid=np.array([True, True, True]),
    )
    assert bool(mask[0, 0])
    assert not bool(mask[0, 1])
    assert not bool(mask[0, 2])
    assert bool(mask[1, 2])


def test_pathnce_residual_gradient_stays_on_metric_and_uniform_weight_is_one():
    agent = _agent('ctd_pathnce_uniform', _config('ctd_pathnce_uniform'))
    batch = _path_batch(
        _batch(),
        start=np.array([0, 0]),
        goal=np.array([8, 8]),
        indices=np.array([2, 4, 6, 9]),
        mask=np.array([[1, 1, 1, 0], [1, 1, 0, 0]], np.float32),
        episode=np.array([0, 1]),
    )
    batch['path_positive_episode'] = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], np.int32)
    batch['path_positive_indices'] = np.array([[2, 4, 6, 9], [2, 4, 6, 9]], np.int32)

    def residual_sum(params):
        distance, _ = agent.path_nce_loss(batch, params)
        return distance

    value = jax.grad(residual_sum)(agent.network.params)
    assert any(
        float(jnp.max(jnp.abs(leaf))) > 0
        for leaf in jax.tree_util.tree_leaves(value['modules_value'])
    )
    for module in ('endpoint', 'bridge', 'idm'):
        leaves = jax.tree_util.tree_leaves(value[f'modules_{module}'])
        assert all(float(jnp.max(jnp.abs(leaf))) == 0.0 for leaf in leaves)

    weights, _ = agent._endpoint_weights(
        jnp.asarray(batch['observations']),
        jnp.asarray(batch['value_goals']),
        jnp.asarray(batch['endpoint_targets']),
    )
    assert jnp.allclose(weights, jnp.ones_like(weights))

    weighted = _agent('ctd_pathnce_weighted', _config('ctd_pathnce_weighted'))
    parent = PathBridgerAgent._endpoint_weights(
        weighted,
        jnp.asarray(batch['observations']),
        jnp.asarray(batch['value_goals']),
        jnp.asarray(batch['endpoint_targets']),
    )[0]
    child, _ = weighted._endpoint_weights(
        jnp.asarray(batch['observations']),
        jnp.asarray(batch['value_goals']),
        jnp.asarray(batch['endpoint_targets']),
    )
    assert jnp.allclose(child, parent)

    plain = _agent('ctd_uniform', _config('ctd_uniform'))
    rng = jax.random.PRNGKey(17)
    left, _ = plain.total_loss(batch, plain.network.params, rng)
    base, _ = PathBridgerAgent.total_loss(plain, batch, plain.network.params, rng)
    nce, _ = plain.nce_loss(batch, plain.network.params)
    assert jnp.allclose(left, base + nce)

    source = Path('agents/contrastive_transitive_distance_pathbridger.py').read_text()
    path_source = source[source.index('def path_nce_loss'):source.index('def _frozen_metric_params')]
    assert 'scores = -path_cost / temperature' in path_source
    assert 'scores = -residual' not in path_source


def test_bridge_geometry_freezes_metric_and_trains_bridge():
    agent = _agent('ctd_pathnce_uniform_bridgegeo', _config('ctd_pathnce_uniform_bridgegeo'))
    batch = _path_batch(
        _batch(),
        start=np.array([0, 100]),
        goal=np.array([8, 108]),
        indices=np.array([2, 4, 6, 12]),
        mask=np.array([[1, 1, 1, 0], [1, 1, 1, 0]], np.float32),
        episode=np.array([0, 1]),
    )
    batch['path_positive_episode'] = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], np.int32)
    batch['path_positive_indices'] = np.array([[2, 4, 6, 12], [102, 104, 106, 112]], np.int32)
    batch['endpoint_offsets'] = np.array([5, 3], np.float32)
    elapsed = np.minimum(np.arange(1, 6)[None, :], batch['endpoint_offsets'][:, None])
    remaining = np.maximum(batch['endpoint_offsets'][:, None] - np.arange(1, 6)[None, :], 0)
    assert elapsed.tolist() == [[1, 2, 3, 4, 5], [1, 2, 3, 3, 3]]
    assert remaining.tolist() == [[4, 3, 2, 1, 0], [2, 1, 0, 0, 0]]

    def geo(params):
        loss, _ = agent.bridge_geometry_loss(batch, params)
        return loss

    grads = jax.grad(geo)(agent.network.params)
    metric = jax.tree_util.tree_leaves(grads['modules_value'])
    bridge = jax.tree_util.tree_leaves(grads['modules_bridge'])
    assert all(float(jnp.max(jnp.abs(leaf))) == 0.0 for leaf in metric)
    assert any(float(jnp.max(jnp.abs(leaf))) > 0.0 for leaf in bridge)

    zero = _agent('ctd_pathnce_uniform', _config('ctd_pathnce_uniform'))
    rng = jax.random.PRNGKey(18)
    full, _ = zero.total_loss(batch, zero.network.params, rng)
    base, _ = PathBridgerAgent.total_loss(zero, batch, zero.network.params, rng)
    nce, _ = zero.nce_loss(batch, zero.network.params)
    path, _ = zero.path_nce_loss(batch, zero.network.params)
    assert jnp.allclose(full, base + nce + path)
    bridge = np.asarray(agent._construct_bridge(
        jnp.asarray(batch['observations']),
        jnp.asarray(batch['endpoint_targets']),
    ))
    assert np.allclose(bridge[:, 0], batch['observations'])
    assert np.allclose(bridge[:, -1], batch['endpoint_targets'])

