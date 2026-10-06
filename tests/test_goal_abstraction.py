"""Contracts for the goal-abstraction actors.

The raw SGCRL agent and the waypoint bridges are covered by
``test_online_sgcrl.py``.  These tests pin the new actors: what they
condition on, which parameters move, and what is kept out of the loss.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.goal_abstraction import (  # noqa: E402
    GoalAbstractionAgent,
    export_agent,
    get_config,
    import_agent,
)
from agents.online_sgcrl import (  # noqa: E402
    OnlineSGCRLAgent,
    VARIANT_SETTINGS as RAW_SETTINGS,
    _sample_tanh_gaussian,
    get_config as raw_config,
)
from utils.online_evaluation import online_episode_manifest  # noqa: E402

OBS_DIM = 6
ACTION_DIM = 3


def make_agent(variant='sgcrl_psi_goal', seed=0):
    config = get_config(variant)
    config.batch_size = 8
    config.hidden_dims = (16, 16)
    config.abstraction_hidden_dims = (16, 16)
    config.repr_dim = 4
    return GoalAbstractionAgent.create(
        seed,
        np.zeros((2, OBS_DIM), dtype=np.float32),
        np.zeros((2, ACTION_DIM), dtype=np.float32),
        config,
    )


def make_batch(batch_size=8, seed=0, **extra):
    rng = np.random.default_rng(seed)
    batch = {
        'observations': rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32),
        'actions': np.clip(rng.normal(size=(batch_size, ACTION_DIM)), -1, 1).astype(
            np.float32
        ),
        'goals': rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32),
    }
    batch.update(extra)
    return batch


def _leaves_unchanged(before, after) -> bool:
    return all(
        np.allclose(np.asarray(left), np.asarray(right))
        for left, right in zip(
            jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(after)
        )
    )


def _any_nonzero(tree) -> bool:
    return any(np.any(np.abs(np.asarray(leaf)) > 0) for leaf in jax.tree_util.tree_leaves(tree))


def test_raw_sgcrl_actor_still_conditions_on_the_full_goal():
    """The existing baseline's first layer still sees concat(s, g_full)."""

    config = raw_config()
    config.variant = 'online_sgcrl'
    for key, value in RAW_SETTINGS['online_sgcrl'].items():
        config[key] = value
    config.hidden_dims = (16, 16)
    config.bridge_hidden_dims = (16, 16)
    config.repr_dim = 4
    agent = OnlineSGCRLAgent.create(
        0,
        np.zeros((2, OBS_DIM), dtype=np.float32),
        np.zeros((2, ACTION_DIM), dtype=np.float32),
        config,
    )
    kernel = agent.actor.params['modules_actor']['Dense_0']['kernel']
    assert kernel.shape[0] == OBS_DIM + OBS_DIM
    assert 'reward' not in inspect.getsource(OnlineSGCRLAgent.actor_loss)
    assert 'oracle' not in inspect.getsource(OnlineSGCRLAgent.actor_loss)


def test_psi_goal_actor_input_has_repr_dim_and_is_stopped():
    agent = make_agent('sgcrl_psi_goal')
    batch = make_batch()
    features = agent.goal_features(batch['observations'], batch['goals'])
    assert features['goal_input'].shape == (8, 4)
    psi = agent.critic.select('psi')(batch['goals'])
    np.testing.assert_allclose(
        np.asarray(features['goal_input']), np.asarray(jax.lax.stop_gradient(psi))
    )

    def embedding(critic_params):
        alt = agent.replace(critic=agent.critic.replace(params=critic_params))
        return alt.goal_features(batch['observations'], batch['goals'])['goal_input'].sum()

    grads = jax.grad(embedding)(agent.critic.params)
    assert not _any_nonzero(grads)


def test_actor_update_does_not_change_psi_or_the_critic():
    agent = make_agent('sgcrl_psi_goal')
    batch = make_batch()
    before = jax.tree.map(lambda leaf: np.array(jax.device_get(leaf)), agent.critic.params)
    rng = jax.random.PRNGKey(3)
    agent.policy.apply_loss_fn(loss_fn=lambda params: agent.actor_loss(batch, params, rng))
    after = jax.tree.map(lambda leaf: np.array(jax.device_get(leaf)), agent.critic.params)
    assert _leaves_unchanged(before, after)


@pytest.mark.parametrize('variant', ['sgcrl_psi_goal', 'sgcrl_state_goal', 'sgcrl_state_mask'])
def test_a_joint_update_moves_the_critic_and_stays_finite(variant):
    agent = make_agent(variant)
    updated, info = agent.update(make_batch())
    assert not _leaves_unchanged(agent.critic.params, updated.critic.params)
    values = [float(np.asarray(value)) for value in jax.tree_util.tree_leaves(info)]
    assert all(np.isfinite(value) for value in values)


def test_state_goal_modules_receive_gradients_and_match_repr_dim():
    agent = make_agent('sgcrl_state_goal')
    batch = make_batch(seed=1)
    features = agent.goal_features(batch['observations'], batch['goals'])
    assert features['goal_input'].shape == (8, agent.config['repr_dim'])
    rng = jax.random.PRNGKey(5)
    grads = jax.grad(
        lambda params: agent.actor_loss(batch, params, rng)[0]
    )(agent.policy.params)
    assert _any_nonzero(grads['modules_state_encoder'])
    assert _any_nonzero(grads['modules_abstraction'])
    assert _any_nonzero(grads['modules_residual'])
    # The stopped psi encoder is not a policy parameter, and the critic is frozen.
    before = jax.tree.map(lambda leaf: np.array(jax.device_get(leaf)), agent.critic.params)
    agent.policy.apply_loss_fn(loss_fn=lambda params: agent.actor_loss(batch, params, rng))
    after = jax.tree.map(lambda leaf: np.array(jax.device_get(leaf)), agent.critic.params)
    assert _leaves_unchanged(before, after)


def test_state_mask_is_a_probability_and_builds_the_interpolated_goal():
    agent = make_agent('sgcrl_state_mask')
    batch = make_batch(seed=2)
    states, goals = batch['observations'], batch['goals']
    features = agent.goal_features(states, goals)
    mask = np.asarray(features['mask'])
    goal_input = np.asarray(features['goal_input'])
    assert mask.shape == states.shape
    assert np.all(mask >= 0.0) and np.all(mask <= 1.0)
    expected = states + mask * (goals - states)
    np.testing.assert_allclose(goal_input, expected, atol=1e-6)


def test_the_actor_consumes_g_hat_while_the_critic_scores_g_full():
    agent = make_agent('sgcrl_state_mask')
    batch = make_batch(seed=4)
    rng = jax.random.PRNGKey(7)
    states, goals = agent._actor_batch(batch)
    features = agent.goal_features(states, goals, params=agent.policy.params)
    loc, scale = agent.actor_distribution(states, features['goal_input'])
    actions, _ = _sample_tanh_gaussian(loc, scale, rng)
    full_scores = agent.critic_scores(states, actions, goals)
    hat_scores = agent.critic_scores(states, actions, features['goal_input'])
    assert not np.allclose(np.asarray(full_scores), np.asarray(hat_scores))
    loss, info = agent.actor_loss(batch, agent.policy.params, rng)
    np.testing.assert_allclose(
        float(info['actor/critic_score']), float(jnp.mean(full_scores)), atol=1e-5
    )
    expected = jnp.mean(-full_scores) + float(agent.config['mask_coef']) * jnp.mean(
        features['mask']
    )
    np.testing.assert_allclose(float(loss), float(expected), atol=1e-5)


@pytest.mark.parametrize('variant', ['sgcrl_psi_goal', 'sgcrl_state_goal', 'sgcrl_state_mask'])
def test_reward_return_and_oracle_fields_do_not_change_the_loss(variant):
    agent = make_agent(variant)
    batch = make_batch(seed=3)
    rng = jax.random.PRNGKey(1)
    loss, _ = agent.actor_loss(batch, agent.policy.params, rng)
    critic_loss, _ = agent.critic_loss(batch, agent.critic.params)
    rewarded = dict(batch, reward=np.ones((8,), np.float32), returns=np.ones((8,), np.float32))
    probed = dict(batch, oracle_xyz=np.zeros((8, 3), np.float32))
    loss_reward, _ = agent.actor_loss(rewarded, agent.policy.params, rng)
    loss_oracle, _ = agent.actor_loss(probed, agent.policy.params, rng)
    critic_reward, _ = agent.critic_loss(rewarded, agent.critic.params)
    critic_oracle, _ = agent.critic_loss(probed, agent.critic.params)
    np.testing.assert_allclose(float(loss), float(loss_reward))
    np.testing.assert_allclose(float(loss), float(loss_oracle))
    np.testing.assert_allclose(float(critic_loss), float(critic_reward))
    np.testing.assert_allclose(float(critic_loss), float(critic_oracle))
    source = inspect.getsource(GoalAbstractionAgent.actor_loss) + inspect.getsource(
        GoalAbstractionAgent.critic_loss
    ) + inspect.getsource(GoalAbstractionAgent.update)
    assert 'reward' not in source
    assert 'returns' not in source
    assert 'oracle' not in source


def test_checkpoint_round_trip_restores_weights_and_rng():
    agent = make_agent('sgcrl_state_goal', seed=9)
    agent, _ = agent.update(make_batch(seed=6))
    restored = import_agent(make_agent('sgcrl_state_goal', seed=9), export_agent(agent))
    assert _leaves_unchanged(agent.critic.params, restored.critic.params)
    assert _leaves_unchanged(agent.policy.params, restored.policy.params)
    np.testing.assert_array_equal(np.asarray(agent.rng), np.asarray(restored.rng))
    again_a, info_a = agent.update(make_batch(seed=8))
    again_b, info_b = restored.update(make_batch(seed=8))
    assert _leaves_unchanged(again_a.policy.params, again_b.policy.params)
    np.testing.assert_allclose(float(info_a['actor/loss']), float(info_b['actor/loss']))


def test_deterministic_actions_do_not_depend_on_the_rng():
    agent = make_agent('sgcrl_psi_goal')
    states = np.zeros((1, OBS_DIM), dtype=np.float32)
    goals = np.ones((1, OBS_DIM), dtype=np.float32)
    first = agent.act(states, goals, jax.random.PRNGKey(0), deterministic=True)
    second = agent.act(states, goals, jax.random.PRNGKey(1), deterministic=True)
    np.testing.assert_allclose(np.asarray(first), np.asarray(second))


def test_evaluation_manifest_is_reproducible_and_matches_the_recorded_scheme():
    first = online_episode_manifest(1, 100, 0)
    second = online_episode_manifest(1, 100, 0)
    assert first == second
    assert first[0]['env_seed'] == 10000
    assert first[0]['action_space_seed'] == 10005
    assert first[3]['env_seed'] == 10003
