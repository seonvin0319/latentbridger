"""Non-oracle PCA-BTRL16 contract tests."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np

from learned_goalspace.fixed_representations import fit_pca16
from learned_goalspace.pca_btrl import (
    METHOD,
    PCABTRLAgent,
    PCAResidualGoalEncoder,
    representation_diagnostics,
)


def _agent():
    obs = np.random.randn(256, 55).astype(np.float32)
    pca = fit_pca16(obs)
    acts = np.random.randn(4, 5).astype(np.float32)
    config = dict(
        env_name='puzzle-3x3-play-v0',
        horizon=25,
        discount=0.99,
        endpoint_distribution='flow',
        endpoint_value_scale=10.0,
        value_distance_weight_power=0.5,
        eval_num_candidates=4,
        eval_temperature=1.0,
        proposer_weighting='transitive',
        pca_beta=0.1,
    )
    return PCABTRLAgent.create(0, obs[:4], acts, config, pca_payload=pca), pca, obs


def test_encoder_source_has_no_oracle_phi():
    source = inspect.getsource(PCAResidualGoalEncoder) + inspect.getsource(PCABTRLAgent)
    assert 'goal_representation' not in source
    assert 'infer_phi_goal_obs_indices' not in source


def test_init_matches_pca_when_residual_zero():
    agent, pca, obs = _agent()
    batch = jnp.asarray(obs[:32])
    encoded = agent._encode(batch)
    expected = (batch - jnp.asarray(pca['mean'])) @ jnp.asarray(pca['kernel'])
    np.testing.assert_allclose(np.asarray(encoded), np.asarray(expected), atol=1e-5)
    diag = representation_diagnostics(agent, np.asarray(obs[:128]))
    assert diag['diag/residual_norm_mean'] == 0.0
    assert diag['diag/drift_from_pca_mean'] == 0.0
    assert diag['diag/beta'] == 0.1


def test_flags_and_pca_frozen_under_update():
    agent, pca, _obs = _agent()
    assert agent.config['method'] == METHOD
    assert agent.config['high_level_oracle_phi'] is False
    assert agent.config['proposer_oracle_phi'] is False
    batch = {
        'observations': np.random.randn(8, 55).astype(np.float32),
        'next_observations': np.random.randn(8, 55).astype(np.float32),
        'actions': np.random.randn(8, 5).astype(np.float32),
        'bridge_targets': np.random.randn(8, 5, 55).astype(np.float32),
        'endpoint_goals': np.random.randn(8, 55).astype(np.float32),
        'endpoint_targets': np.random.randn(8, 55).astype(np.float32),
        'value_goals': np.random.randn(8, 55).astype(np.float32),
        'value_offsets': np.full((8,), 10.0, np.float32),
        'base_goals': np.random.randn(8, 55).astype(np.float32),
        'base_offsets': np.ones((8,), np.float32),
        'transitive_subgoals': np.random.randn(8, 55).astype(np.float32),
        'transitive_offsets': np.ones((8,), np.float32),
        'transitive_valids': np.ones((8,), np.float32),
    }
    mean0 = np.asarray(agent.network.params['modules_goal_encoder']['mean']).copy()
    kernel0 = np.asarray(agent.network.params['modules_goal_encoder']['kernel']).copy()
    updated, _info = agent.update(batch)
    np.testing.assert_array_equal(np.asarray(updated.network.params['modules_goal_encoder']['mean']), mean0)
    np.testing.assert_array_equal(np.asarray(updated.network.params['modules_goal_encoder']['kernel']), kernel0)
    # Residual should receive TRL gradients / move.
    before = jax.tree_util.tree_leaves(agent.network.params['modules_goal_encoder']['residual'])
    after = jax.tree_util.tree_leaves(updated.network.params['modules_goal_encoder']['residual'])
    assert any(not np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(before, after))


def test_endpoint_loss_does_not_train_residual():
    agent, _pca, _obs = _agent()
    obs = jnp.asarray(np.random.randn(8, 55), dtype=jnp.float32)
    goals = jnp.asarray(np.random.randn(8, 55), dtype=jnp.float32)
    targets = jnp.asarray(np.random.randn(8, 55), dtype=jnp.float32)
    batch = {'observations': obs, 'endpoint_goals': goals, 'endpoint_targets': targets}

    def endpoint_only(params):
        return agent.endpoint_loss(batch, params, jax.random.PRNGKey(0))

    grads, _ = jax.grad(endpoint_only, has_aux=True)(agent.network.params)
    residual_grads = grads['modules_goal_encoder']['residual']
    assert all(np.allclose(np.asarray(g), 0.0) for g in jax.tree_util.tree_leaves(residual_grads))
    # PCA params may get None/zero grads; residual is the trainable abstraction path.
    assert all(
        np.allclose(np.asarray(g), 0.0) for g in jax.tree_util.tree_leaves(grads['modules_goal_encoder']['mean'])
    )
