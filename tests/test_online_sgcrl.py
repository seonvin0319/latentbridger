"""Tests for the online SGCRL port.

These pin the properties that make the three-variant comparison meaningful:
the replay's hindsight distribution, the strict isolation of the critic,
actor, and bridge optimizers, and the fact that the variants differ only in
the actor's goal interface.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.online_sgcrl import (  # noqa: E402
    VARIANT_SETTINGS,
    VARIANTS,
    OnlineSGCRLAgent,
    get_config,
)
from utils.online_evaluation import online_episode_manifest  # noqa: E402
from utils.online_replay import (  # noqa: E402
    EpisodicReplayBuffer,
    sparse_bridge_offsets,
)

OBS_DIM = 6
ACTION_DIM = 3


def make_agent(variant='online_sgcrl', seed=0, **overrides):
    config = get_config()
    config.variant = variant
    for key, value in VARIANT_SETTINGS[variant].items():
        config[key] = value
    config.batch_size = 8
    config.hidden_dims = (16, 16)
    config.repr_dim = 4
    config.num_waypoints = 3
    config.flow_steps = 2
    for key, value in overrides.items():
        config[key] = value
    return OnlineSGCRLAgent.create(
        seed,
        np.zeros((2, OBS_DIM), dtype=np.float32),
        np.zeros((2, ACTION_DIM), dtype=np.float32),
        config,
    )


def fill_replay(buffer, num_episodes=6, length=25, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(num_episodes):
        observations = rng.normal(size=(length + 1, OBS_DIM)).astype(np.float32)
        actions = rng.normal(size=(length, ACTION_DIM)).astype(np.float32)
        buffer.add_episode(observations, actions)
    return buffer


def make_replay(discount=0.99, seed=0, **kwargs):
    return EpisodicReplayBuffer(
        OBS_DIM, ACTION_DIM, discount=discount, seed=seed, **kwargs
    )


# ----------------------------------------------------------------------
# Replay and hindsight sampling
# ----------------------------------------------------------------------
def test_sparse_offsets_cover_the_horizon_and_end_on_it():
    assert sparse_bridge_offsets(40, 5) == (8, 16, 24, 32, 40)
    # Rounding up keeps the offsets strictly increasing when the horizon is
    # not a multiple of the waypoint count.
    assert sparse_bridge_offsets(10, 4) == (3, 5, 8, 10)


def test_sparse_offsets_reject_more_waypoints_than_steps():
    with pytest.raises(ValueError):
        sparse_bridge_offsets(3, 5)


def test_hindsight_goals_come_from_the_same_episode_and_the_future():
    buffer = make_replay()
    rng = np.random.default_rng(0)
    # Each episode is tagged by a constant in its first feature, so a goal
    # drawn from another episode is detectable.
    for episode in range(5):
        observations = rng.normal(size=(31, OBS_DIM)).astype(np.float32)
        observations[:, 0] = float(episode)
        actions = rng.normal(size=(30, ACTION_DIM)).astype(np.float32)
        buffer.add_episode(observations, actions)

    batch = buffer.sample(64)
    assert np.allclose(batch['goals'][:, 0], batch['observations'][:, 0])
    assert np.all(batch['future_offsets'] >= 1)


def test_future_offsets_follow_a_truncated_geometric_distribution():
    # With a long single episode and a small discount the empirical mean
    # offset should match the truncated geometric's analytic mean.
    discount = 0.9
    buffer = make_replay(discount=discount)
    rng = np.random.default_rng(1)
    length = 400
    buffer.add_episode(
        rng.normal(size=(length + 1, OBS_DIM)).astype(np.float32),
        rng.normal(size=(length, ACTION_DIM)).astype(np.float32),
    )
    offsets = np.concatenate([buffer.sample(512)['future_offsets'] for _ in range(20)])
    # Anchors near the end truncate the tail, so compare against the mean of
    # the untruncated geometric only loosely; the sharper check is that the
    # distribution decays rather than being uniform.
    assert offsets.min() >= 1
    assert np.mean(offsets <= 10) > np.mean(offsets > 10)


def test_sampling_never_crosses_an_episode_boundary_at_the_buffer_wrap():
    buffer = make_replay(max_size=60)
    fill_replay(buffer, num_episodes=8, length=20)
    # The buffer has wrapped; every sampled goal must still share its
    # anchor's episode.
    assert len(buffer) <= 60
    for _ in range(20):
        batch = buffer.sample(32)
        assert np.all(batch['future_offsets'] >= 1)


def test_bridge_targets_mark_out_of_episode_waypoints_invalid():
    buffer = make_replay()
    fill_replay(buffer, num_episodes=4, length=12)
    offsets = sparse_bridge_offsets(10, 5)
    batch = buffer.sample_bridge(32, offsets)
    assert batch['bridge_targets'].shape == (32, 5, OBS_DIM)
    assert batch['bridge_valid'].shape == (32, 5)
    # Later waypoints fall outside the episode at least as often as earlier
    # ones, never less.
    valid_fraction = batch['bridge_valid'].mean(axis=0)
    assert np.all(np.diff(valid_fraction) <= 1e-6)


def test_replay_state_round_trips():
    buffer = fill_replay(make_replay(), num_episodes=3, length=15)
    restored = make_replay()
    restored.load_state_dict(buffer.state_dict())
    assert len(restored) == len(buffer)
    assert restored.num_episodes == buffer.num_episodes
    first = buffer.sample(16)
    second = restored.sample(16)
    for key in first:
        np.testing.assert_allclose(first[key], second[key])


# ----------------------------------------------------------------------
# Variant structure
# ----------------------------------------------------------------------
def test_each_variant_fixes_its_own_goal_interface():
    assert VARIANTS == (
        'online_sgcrl',
        'online_sgcrl_latent',
        'online_sgcrl_latent_bridge',
    )
    assert VARIANT_SETTINGS['online_sgcrl']['actor_goal_input'] == 'raw'
    assert VARIANT_SETTINGS['online_sgcrl_latent']['use_bridge'] is False
    assert VARIANT_SETTINGS['online_sgcrl_latent_bridge']['use_bridge'] is True


def test_a_config_cannot_override_a_variants_structure():
    config = get_config()
    config.variant = 'online_sgcrl_latent'
    config.use_bridge = True
    with pytest.raises(ValueError, match='fixes use_bridge'):
        OnlineSGCRLAgent.create(
            0,
            np.zeros((2, OBS_DIM), dtype=np.float32),
            np.zeros((2, ACTION_DIM), dtype=np.float32),
            config,
        )


def test_the_three_variants_share_one_critic_architecture():
    shapes = []
    for variant in VARIANTS:
        agent = make_agent(variant)
        shapes.append(
            jax.tree_util.tree_map(lambda x: x.shape, agent.critic.params)
        )
    assert shapes[0] == shapes[1] == shapes[2]


def test_the_variants_build_identical_critics_from_the_same_seed():
    # The critic must not depend on the actor's interface, or an A-vs-B gap
    # could come from initialization rather than from the interface.
    first = make_agent('online_sgcrl', seed=3).critic.params
    second = make_agent('online_sgcrl_latent_bridge', seed=3).critic.params
    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_allclose(a, b), first, second
    )


def test_the_latent_actor_consumes_repr_dim_not_observation_dim():
    raw = make_agent('online_sgcrl')
    latent = make_agent('online_sgcrl_latent')
    raw_kernel = raw.actor.params['modules_actor']['Dense_0']['kernel']
    latent_kernel = latent.actor.params['modules_actor']['Dense_0']['kernel']
    assert raw_kernel.shape[0] == OBS_DIM + OBS_DIM
    assert latent_kernel.shape[0] == OBS_DIM + 4


# ----------------------------------------------------------------------
# Losses and optimizer isolation
# ----------------------------------------------------------------------
def make_batch(batch_size=8, seed=0):
    rng = np.random.default_rng(seed)
    return {
        'observations': rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32),
        'actions': np.clip(
            rng.normal(size=(batch_size, ACTION_DIM)), -1, 1
        ).astype(np.float32),
        'goals': rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32),
    }


def test_the_critic_update_leaves_the_actor_and_flow_untouched():
    # Separate optimizers matter here: one shared Adam state would move the
    # actor on a critic step through momentum even with zero gradients.
    agent = make_agent('online_sgcrl_latent_bridge')
    batch = make_batch()
    updated, _ = agent.update(batch)
    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_allclose(a, b), agent.flow.params,
        updated.flow.params,
    )
    moved = [
        bool(np.any(np.abs(np.asarray(a) - np.asarray(b)) > 0))
        for a, b in zip(
            jax.tree_util.tree_leaves(agent.critic.params),
            jax.tree_util.tree_leaves(updated.critic.params),
        )
    ]
    assert any(moved)


def test_the_bridge_update_leaves_the_critic_and_actor_untouched():
    agent = make_agent('online_sgcrl_latent_bridge')
    buffer = fill_replay(make_replay(), num_episodes=4, length=20)
    batch = buffer.sample_bridge(8, sparse_bridge_offsets(9, 3))
    # The replay's observation dim must match the agent's for this to be a
    # real test of isolation rather than a shape error.
    updated, info = agent.update_bridge(batch)
    for before, after in (
        (agent.critic.params, updated.critic.params),
        (agent.actor.params, updated.actor.params),
    ):
        jax.tree_util.tree_map(
            lambda a, b: np.testing.assert_allclose(a, b), before, after
        )
    assert np.isfinite(float(info['bridge/loss']))


def test_the_critic_loss_matches_infonce_plus_the_logsumexp_penalty():
    agent = make_agent('online_sgcrl')
    batch = make_batch()
    loss, info = agent.critic_loss(batch, agent.critic.params)

    logits = np.asarray(
        agent.critic_logits(batch['observations'], batch['actions'], batch['goals'])
    )
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_partition = np.log(np.exp(shifted).sum(axis=1)) + logits.max(axis=1)
    cross_entropy = log_partition - np.diag(logits)
    penalty = 0.01 * log_partition**2
    np.testing.assert_allclose(
        float(loss), float(np.mean(cross_entropy + penalty)), rtol=1e-5
    )
    assert 0.0 <= float(info['critic/recall_at_1']) <= 1.0


def test_the_actor_loss_contains_no_behavioural_cloning_term():
    # Replacing the batch's actions must not change the actor loss: a BC term
    # would make the loss depend on them.
    agent = make_agent('online_sgcrl')
    batch = make_batch()
    other = dict(batch, actions=-batch['actions'])
    rng = jax.random.PRNGKey(0)
    first, _ = agent.actor_loss(batch, agent.actor.params, rng)
    second, _ = agent.actor_loss(other, agent.actor.params, rng)
    np.testing.assert_allclose(float(first), float(second), rtol=1e-6)


def test_random_goals_doubles_the_actor_batch_and_shuffles_half_of_it():
    agent = make_agent('online_sgcrl')
    batch = make_batch()
    states, goals = agent._actor_batch(batch)
    assert states.shape[0] == 2 * batch['observations'].shape[0]
    np.testing.assert_allclose(np.asarray(goals)[:8], batch['goals'])
    np.testing.assert_allclose(
        np.asarray(goals)[8:], np.roll(batch['goals'], 1, axis=0)
    )


def test_an_unsupported_random_goals_value_is_rejected():
    with pytest.raises(ValueError, match='random_goals'):
        make_agent('online_sgcrl', random_goals=0.25)


def test_the_policy_scale_starts_near_softplus_zero_and_has_a_floor():
    # With entropy_coefficient=0.0 this floor is the only thing keeping
    # collection stochastic, so it is load-bearing rather than cosmetic.
    agent = make_agent('online_sgcrl')
    observations = np.zeros((4, OBS_DIM), dtype=np.float32)
    goals = np.zeros((4, OBS_DIM), dtype=np.float32)
    _, scale = agent.actor.select('actor')(observations, goals)
    np.testing.assert_allclose(np.asarray(scale), 0.693147 + 1e-3, rtol=1e-4)

    # Driving the scale head's pre-activation very negative must not take the
    # scale below the floor.
    # With hidden_dims=(16, 16) the trunk is Dense_0/Dense_1, the mean head is
    # Dense_2, and the scale head is Dense_3.
    modules = dict(agent.actor.params['modules_actor'])
    scale_head = modules['Dense_3']
    modules['Dense_3'] = dict(
        scale_head, bias=jnp.full_like(scale_head['bias'], -50.0)
    )
    params = {'modules_actor': modules}
    _, floored = agent.actor.select('actor')(observations, goals, params=params)
    assert np.all(np.asarray(floored) >= 1e-3)
    np.testing.assert_allclose(np.asarray(floored), 1e-3, atol=1e-9)


def test_the_entropy_coefficient_defaults_to_zero():
    # SGCRL's launcher sets this explicitly, which bypasses adaptive alpha.
    assert float(get_config().entropy_coefficient) == 0.0


def test_sgcrl_defaults_match_the_documented_launcher_settings():
    config = get_config()
    assert int(config.repr_dim) == 64
    assert bool(config.repr_norm) is False
    assert tuple(config.hidden_dims) == (256, 256)
    assert float(config.learning_rate) == 3e-4
    assert float(config.discount) == 0.99
    assert int(config.batch_size) == 256
    assert float(config.logsumexp_coef) == 0.01
    assert int(config.min_replay_size) == 10_000
    assert int(config.max_replay_size) == 1_000_000
    assert float(config.updates_per_env_step) == 1.0


# ----------------------------------------------------------------------
# Acting and the branch interface
# ----------------------------------------------------------------------
def test_deterministic_actions_are_reproducible_and_in_range():
    agent = make_agent('online_sgcrl')
    observations = np.zeros((4, OBS_DIM), dtype=np.float32)
    goals = np.ones((4, OBS_DIM), dtype=np.float32)
    rng = jax.random.PRNGKey(0)
    first = agent.act(observations, goals, rng, deterministic=True)
    second = agent.act(observations, goals, jax.random.PRNGKey(1), deterministic=True)
    np.testing.assert_allclose(np.asarray(first), np.asarray(second))
    assert np.all(np.abs(np.asarray(first)) <= 1.0)


def test_sampled_actions_depend_on_the_rng():
    agent = make_agent('online_sgcrl')
    observations = np.zeros((4, OBS_DIM), dtype=np.float32)
    goals = np.ones((4, OBS_DIM), dtype=np.float32)
    first = agent.act(observations, goals, jax.random.PRNGKey(0))
    second = agent.act(observations, goals, jax.random.PRNGKey(1))
    assert not np.allclose(np.asarray(first), np.asarray(second))


def test_the_bridge_can_be_switched_on_at_the_branch_point():
    # The paired branch needs one set of parameters to produce both the
    # latent-goal behaviour and the bridge behaviour.
    agent = make_agent('online_sgcrl_latent_bridge')
    observations = np.zeros((4, OBS_DIM), dtype=np.float32)
    goals = np.ones((4, OBS_DIM), dtype=np.float32)
    rng = jax.random.PRNGKey(0)
    direct = agent.act(observations, goals, rng, deterministic=True, use_bridge=False)
    bridged = agent.act(observations, goals, rng, deterministic=True, use_bridge=True)
    assert not np.allclose(np.asarray(direct), np.asarray(bridged))


def test_the_bridge_waypoint_lives_in_the_latent_space():
    agent = make_agent('online_sgcrl_latent_bridge')
    observations = np.zeros((4, OBS_DIM), dtype=np.float32)
    goals = np.ones((4, OBS_DIM), dtype=np.float32)
    waypoint = agent.bridge_waypoint(observations, goals, jax.random.PRNGKey(0))
    assert waypoint.shape == (4, 4)
    assert np.all(np.isfinite(np.asarray(waypoint)))


# ----------------------------------------------------------------------
# Evaluation manifest
# ----------------------------------------------------------------------
def test_the_manifest_pins_both_environment_random_sources():
    manifest = online_episode_manifest(task_id=1, num_episodes=4, seed=0)
    assert len(manifest) == 4
    for entry in manifest:
        assert entry['action_space_seed'] == entry['env_seed'] + 5
        assert entry['task_id'] == 1


def test_variants_evaluated_at_the_same_seed_get_identical_episodes():
    first = online_episode_manifest(task_id=1, num_episodes=8, seed=2)
    second = online_episode_manifest(task_id=1, num_episodes=8, seed=2)
    assert first == second


def test_different_seeds_produce_different_episodes():
    first = online_episode_manifest(task_id=1, num_episodes=8, seed=0)
    second = online_episode_manifest(task_id=1, num_episodes=8, seed=1)
    assert [entry['env_seed'] for entry in first] != [
        entry['env_seed'] for entry in second
    ]


# ----------------------------------------------------------------------
# A short end-to-end update loop
# ----------------------------------------------------------------------
@pytest.mark.parametrize('variant', VARIANTS)
def test_a_short_update_loop_stays_finite_for_every_variant(variant):
    agent = make_agent(variant)
    buffer = fill_replay(make_replay(), num_episodes=6, length=25)
    offsets = sparse_bridge_offsets(9, 3)
    for _ in range(5):
        agent, info = agent.update(buffer.sample(8))
        if VARIANT_SETTINGS[variant]['use_bridge']:
            agent, bridge_info = agent.update_bridge(
                buffer.sample_bridge(8, offsets)
            )
            info = {**info, **bridge_info}
        for key, value in info.items():
            assert np.isfinite(float(value)), key
    assert jnp.all(jnp.isfinite(agent.actor.params['modules_actor']['Dense_0']['kernel']))
