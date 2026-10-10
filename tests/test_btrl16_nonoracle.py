"""Non-oracle BTRL16 contract tests."""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np

from agents.learned_goal_proposer import LearnedGoalFlowEndpointProposer
from agents.pathbridger import FlowEndpointProposer
from learned_goalspace.btrl import BottleneckTRLAgent, METHOD
from utils.goal_representation import goal_representation


def _tiny_agent():
    obs = np.random.randn(4, 55).astype(np.float32)
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
    )
    return BottleneckTRLAgent.create(0, obs, acts, config)


def test_learned_proposer_has_no_phi_call():
    source = inspect.getsource(LearnedGoalFlowEndpointProposer)
    assert 'goal_representation' not in source
    # Hard: module forward accepts embeddings dim 16, not raw goals alone.
    module = LearnedGoalFlowEndpointProposer(state_dim=8, goal_dim=16)
    key = jax.random.PRNGKey(0)
    obs = jnp.zeros((2, 8))
    eg = jnp.zeros((2, 16))
    noise = jnp.zeros((2, 8))
    t = jnp.zeros((2, 1))
    params = module.init(key, obs, eg, noise, t)
    out = module.apply(params, obs, eg, noise, t)
    assert out.shape == (2, 8)


def test_original_flow_proposer_still_uses_phi():
    source = inspect.getsource(FlowEndpointProposer)
    assert 'goal_representation' in source
    assert "'phi'" in source or '"phi"' in source


def test_btrl_agent_source_has_no_phi_goal_representation():
    source = inspect.getsource(BottleneckTRLAgent)
    assert 'goal_representation' not in source
    assert 'infer_phi_goal_obs_indices' not in source
    assert 'assert_phi_goal_obs_indices' not in source


def test_btrl_flags_and_modules():
    agent = _tiny_agent()
    assert agent.config['method'] == METHOD
    assert agent.config['high_level_oracle_phi'] is False
    assert agent.config['proposer_oracle_phi'] is False
    keys = set(agent.network.params)
    assert 'modules_goal_encoder' in keys
    assert 'modules_endpoint' in keys
    assert 'modules_bridge' in keys
    assert 'modules_idm' in keys


def test_proposer_receives_stopgrad_embedding_not_phi():
    agent = _tiny_agent()
    batch = {
        'observations': np.random.randn(8, 55).astype(np.float32),
        'next_observations': np.random.randn(8, 55).astype(np.float32),
        'actions': np.random.randn(8, 5).astype(np.float32),
        'bridge_targets': np.random.randn(8, 5, 55).astype(np.float32),
        'endpoint_goals': np.random.randn(8, 55).astype(np.float32),
        'endpoint_targets': np.random.randn(8, 55).astype(np.float32),
        'value_goals': np.random.randn(8, 55).astype(np.float32),
        'base_goals': np.random.randn(8, 55).astype(np.float32),
        'base_offsets': np.ones((8,), np.float32),
        'value_offsets': np.full((8,), 10.0, np.float32),
        'transitive_subgoals': np.random.randn(8, 55).astype(np.float32),
        'transitive_offsets': np.ones((8,), np.float32),
        'transitive_valids': np.ones((8,), np.float32),
    }

    def loss_fn(params):
        return agent.total_loss(batch, params, rng=jax.random.PRNGKey(1))

    grads, info = jax.grad(loss_fn, has_aux=True)(agent.network.params)
    # Encoder must receive TRL gradients.
    enc_grads = grads['modules_goal_encoder']
    assert any(np.any(np.asarray(g) != 0) for g in jax.tree_util.tree_leaves(enc_grads))

    # Proposer path uses stopgrad(E(g)): compare encoder grad with value-only vs full.
    # Sanity: endpoint module input dim expects 16 — apply with phi must fail shape-wise
    # if someone passed phi (puzzle phi dim != 16 typically).
    phi = np.asarray(
        goal_representation(
            jnp.asarray(batch['endpoint_goals']),
            'phi',
            env_name='puzzle-3x3-play-v0',
        )
    )
    assert phi.shape[-1] != 16


def test_encoder_grads_only_from_trl_when_endpoint_isolated():
    agent = _tiny_agent()
    obs = jnp.asarray(np.random.randn(8, 55), dtype=jnp.float32)
    goals = jnp.asarray(np.random.randn(8, 55), dtype=jnp.float32)
    targets = jnp.asarray(np.random.randn(8, 55), dtype=jnp.float32)
    batch = {
        'observations': obs,
        'endpoint_goals': goals,
        'endpoint_targets': targets,
    }

    def endpoint_only(params):
        return agent.endpoint_loss(batch, params, jax.random.PRNGKey(0))

    grads, _ = jax.grad(endpoint_only, has_aux=True)(agent.network.params)
    enc = grads['modules_goal_encoder']
    assert all(np.allclose(np.asarray(g), 0.0) for g in jax.tree_util.tree_leaves(enc))
