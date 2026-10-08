"""CPU invariants for the goal-space temporal-quasimetric pilot."""

import jax
import jax.numpy as jnp
import numpy as np

from agents.contrastive_transitive_distance_pathbridger import ContrastiveTransitiveDistanceAgent
from configs.gsctd.puzzle_3x3 import get_config
from scripts.run_goalspace_suite import QUEUE
from utils.goal_representation import infer_phi_goal_obs_indices

OBS_DIM = 55
ACTION_DIM = 2
BATCH = 8


def make_agent(variant='gsdtrl_weighted', seed=0):
    return ContrastiveTransitiveDistanceAgent.create(
        seed,
        jnp.zeros((2, OBS_DIM), dtype=jnp.float32),
        jnp.zeros((2, ACTION_DIM), dtype=jnp.float32),
        get_config(variant).to_dict(),
    )


def nce_batch(seed=0):
    rng = np.random.default_rng(seed)
    return {
        'observations': rng.normal(size=(BATCH, OBS_DIM)).astype(np.float32),
        'value_goals': rng.normal(size=(BATCH, OBS_DIM)).astype(np.float32),
    }


def test_goalspace_configs_are_locked_to_phi_and_original_weighting():
    for variant, mode, lambda_nce in (
        ('gsdtrl_weighted', 'fixed', 0.0),
        ('gsctd_learned_temp', 'learned', 1.0),
        ('gsctd_fixed', 'fixed', 1.0),
    ):
        config = get_config(variant)
        assert config.metric_representation == 'phi'
        assert config.proposer_weighting == 'transitive'
        assert config.nce_temperature_mode == mode
        assert config.lambda_nce == lambda_nce
        assert config.lambda_pathnce == 0.0
        assert config.lambda_bridge_geo == 0.0


def test_pilot_queue_is_exact_and_stops_before_other_environments():
    assert QUEUE == (
        ('puzzle_3x3', 'gsdtrl_weighted'),
        ('cube_double', 'gsdtrl_weighted'),
        ('puzzle_3x3', 'gsctd_learned_temp'),
        ('cube_double', 'gsctd_learned_temp'),
        ('puzzle_3x3', 'gsctd_fixed'),
    )


def test_metric_is_invariant_to_puzzle_nuisance_coordinates():
    agent = make_agent()
    indices = infer_phi_goal_obs_indices('puzzle-3x3-play-v0', OBS_DIM)
    assert len(indices) == 9
    left = np.zeros((3, OBS_DIM), dtype=np.float32)
    right = np.zeros_like(left)
    nuisance = [index for index in range(OBS_DIM) if index not in indices]
    right[:, nuisance] = np.arange(1, len(nuisance) + 1, dtype=np.float32)
    np.testing.assert_allclose(
        agent._metric_distance(jnp.asarray(left), jnp.asarray(right), name='value'),
        0.0,
        atol=1e-6,
    )


def test_metric_trunk_sees_phi_but_proposer_keeps_full_state_input():
    agent = make_agent()
    phi_dim = len(infer_phi_goal_obs_indices('puzzle-3x3-play-v0', OBS_DIM))
    value_kernels = [
        np.asarray(leaf)
        for path, leaf in jax.tree_util.tree_leaves_with_path(agent.network.params['modules_value'])
        if 'kernel' in '.'.join(str(part) for part in path)
    ]
    endpoint_kernels = [
        np.asarray(leaf)
        for path, leaf in jax.tree_util.tree_leaves_with_path(agent.network.params['modules_endpoint'])
        if 'kernel' in '.'.join(str(part) for part in path)
    ]
    assert any(kernel.ndim == 2 and kernel.shape[0] == phi_dim for kernel in value_kernels)
    # Flow proposer input = full current state + phi(goal) + full noisy
    # displacement + scalar flow time.
    expected_flow_input = 2 * OBS_DIM + phi_dim + 1
    assert any(kernel.ndim == 2 and kernel.shape[0] == expected_flow_input for kernel in endpoint_kernels)


def test_learned_temperature_starts_at_inverse_horizon_and_gets_nce_gradient():
    agent = make_agent('gsctd_learned_temp')
    expected = 1.0 / float(agent.config['horizon'])
    np.testing.assert_allclose(float(agent._metric_alpha()), expected, rtol=1e-5)
    batch = {key: jnp.asarray(value) for key, value in nce_batch().items()}
    value_grads = jax.grad(lambda params: agent.network.select('value')(
        batch['observations'], batch['value_goals'], params=params,
    ).sum())(agent.network.params)
    np.testing.assert_array_equal(
        np.asarray(value_grads['modules_value']['raw_alpha']),
        0.0,
    )
    grads = jax.grad(lambda params: agent.nce_loss(batch, params)[0])(agent.network.params)
    raw = np.asarray(grads['modules_value']['raw_alpha'])
    assert np.isfinite(raw).all()
    assert np.any(np.abs(raw) > 0)
    _, info = agent.nce_loss(batch, agent.network.params)
    np.testing.assert_allclose(float(info['nce/effective_temperature']), 1.0 / expected, rtol=1e-5)


def test_fixed_temperature_has_no_trainable_alpha_and_matches_horizon():
    agent = make_agent('gsctd_fixed')
    assert 'raw_alpha' not in agent.network.params['modules_value']
    _, info = agent.nce_loss(
        {key: jnp.asarray(value) for key, value in nce_batch().items()},
        agent.network.params,
    )
    np.testing.assert_allclose(float(info['nce/alpha']), 1.0 / agent.config['horizon'])
    np.testing.assert_allclose(float(info['nce/effective_temperature']), agent.config['horizon'])
