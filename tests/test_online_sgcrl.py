"""Tests for the online SGCRL raw-state-bridge comparison.

These pin the properties the three-way comparison depends on: the replay's
hindsight and waypoint distributions, the strict isolation of the critic,
actor, and bridge optimizers, the fact that the actor interface stays raw,
and the exactness of the shared-warmup branch point.
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
    BRIDGE_MODES,
    VARIANT_SETTINGS,
    VARIANTS,
    OnlineSGCRLAgent,
    get_config,
)
from utils.online_evaluation import online_episode_manifest  # noqa: E402
from utils.online_replay import EpisodicReplayBuffer, waypoint_index  # noqa: E402

OBS_DIM = 6
ACTION_DIM = 3


def make_agent(variant='online_sgcrl', seed=0, **overrides):
    config = get_config()
    config.variant = variant
    for key, value in VARIANT_SETTINGS[variant].items():
        config[key] = value
    config.batch_size = 8
    config.hidden_dims = (16, 16)
    config.bridge_hidden_dims = (16, 16)
    config.repr_dim = 4
    config.flow_steps = 2
    config.diagnostic_flow_samples = 4
    for key, value in overrides.items():
        config[key] = value
    return OnlineSGCRLAgent.create(
        seed,
        np.zeros((2, OBS_DIM), dtype=np.float32),
        np.zeros((2, ACTION_DIM), dtype=np.float32),
        config,
    )


def make_replay(discount=0.99, seed=0, **kwargs):
    return EpisodicReplayBuffer(
        OBS_DIM, ACTION_DIM, discount=discount, seed=seed, **kwargs
    )


def fill_replay(buffer, num_episodes=40, length=25, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(num_episodes):
        observations = rng.normal(size=(length + 1, OBS_DIM)).astype(np.float32)
        actions = rng.normal(size=(length, ACTION_DIM)).astype(np.float32)
        buffer.add_episode(observations, actions)
    return buffer


def make_batch(batch_size=8, seed=0):
    rng = np.random.default_rng(seed)
    return {
        'observations': rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32),
        'actions': np.clip(rng.normal(size=(batch_size, ACTION_DIM)), -1, 1).astype(
            np.float32
        ),
        'goals': rng.normal(size=(batch_size, OBS_DIM)).astype(np.float32),
    }


# ----------------------------------------------------------------------
# 1. Future goal sampling never crosses a terminal
# ----------------------------------------------------------------------
def test_hindsight_goals_never_cross_an_episode_boundary():
    buffer = make_replay()
    rng = np.random.default_rng(0)
    # Each episode is tagged by a constant in its first feature, so a goal
    # drawn from a neighbouring episode would be detectable.
    for episode in range(8):
        observations = rng.normal(size=(31, OBS_DIM)).astype(np.float32)
        observations[:, 0] = float(episode)
        buffer.add_episode(
            observations, rng.normal(size=(30, ACTION_DIM)).astype(np.float32)
        )
    for _ in range(10):
        batch = buffer.sample(64)
        np.testing.assert_allclose(batch['goals'][:, 0], batch['observations'][:, 0])
        assert np.all(batch['future_offsets'] >= 1)


def test_hindsight_goals_stay_inside_the_episode_after_the_buffer_wraps():
    buffer = make_replay(max_size=60)
    rng = np.random.default_rng(1)
    for episode in range(12):
        observations = rng.normal(size=(21, OBS_DIM)).astype(np.float32)
        observations[:, 0] = float(episode)
        buffer.add_episode(
            observations, rng.normal(size=(20, ACTION_DIM)).astype(np.float32)
        )
    assert len(buffer) <= 60
    batch = buffer.sample(64)
    np.testing.assert_allclose(batch['goals'][:, 0], batch['observations'][:, 0])


def test_future_offsets_decay_rather_than_being_uniform():
    buffer = make_replay(discount=0.9)
    rng = np.random.default_rng(2)
    buffer.add_episode(
        rng.normal(size=(401, OBS_DIM)).astype(np.float32),
        rng.normal(size=(400, ACTION_DIM)).astype(np.float32),
    )
    offsets = np.concatenate([buffer.sample(512)['future_offsets'] for _ in range(20)])
    assert offsets.min() >= 1
    assert np.mean(offsets <= 10) > np.mean(offsets > 10)


# ----------------------------------------------------------------------
# 2 & 3. Waypoint target strictly between start and endpoint; alpha index
# ----------------------------------------------------------------------
def test_alpha_half_picks_the_segment_midpoint():
    np.testing.assert_array_equal(
        waypoint_index(np.array([2, 4, 6, 7, 10]), 0.5), np.array([1, 2, 3, 3, 5])
    )


def test_the_waypoint_index_stays_strictly_inside_the_segment():
    segments = np.arange(2, 200)
    for alpha in (0.05, 0.25, 0.5, 0.75, 0.95):
        offsets = waypoint_index(segments, alpha)
        assert np.all(offsets >= 1)
        assert np.all(offsets <= segments - 1)


def test_an_alpha_outside_the_open_unit_interval_is_rejected():
    for alpha in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match='alpha'):
            waypoint_index(np.array([4]), alpha)


def test_bridge_segments_are_at_least_two_steps_long():
    buffer = fill_replay(make_replay())
    for _ in range(20):
        batch = buffer.sample_bridge(64, alpha=0.5)
        assert np.all(batch['segment_lengths'] >= 2)
        assert np.all(batch['waypoint_offsets'] >= 1)
        assert np.all(batch['waypoint_offsets'] <= batch['segment_lengths'] - 1)


def test_the_waypoint_is_an_observed_state_not_an_interpolation():
    # Build episodes whose states are distinguishable integers so a blended
    # state would be detectable: an interpolation of two integers is not one.
    buffer = make_replay()
    for episode in range(8):
        observations = np.arange(31, dtype=np.float32)[:, None] * np.ones(
            (1, OBS_DIM), dtype=np.float32
        )
        observations[:, 0] = float(episode)
        buffer.add_episode(
            observations, np.zeros((30, ACTION_DIM), dtype=np.float32)
        )
    batch = buffer.sample_bridge(128, alpha=0.5)
    waypoint_steps = batch['waypoints'][:, 1]
    np.testing.assert_allclose(waypoint_steps, np.round(waypoint_steps))
    # And it sits strictly between the anchor's step and the goal's step.
    assert np.all(batch['observations'][:, 1] < waypoint_steps)
    assert np.all(waypoint_steps < batch['goals'][:, 1])
    # Same episode throughout.
    np.testing.assert_allclose(batch['goals'][:, 0], batch['observations'][:, 0])
    np.testing.assert_allclose(batch['waypoints'][:, 0], batch['observations'][:, 0])


def test_bridge_holdout_episodes_are_disjoint_from_training_episodes():
    buffer = make_replay(holdout_every=4)
    for episode in range(40):
        observations = np.zeros((21, OBS_DIM), dtype=np.float32)
        observations[:, 0] = float(episode)
        buffer.add_episode(
            observations, np.zeros((20, ACTION_DIM), dtype=np.float32)
        )
    train_ids = set(np.unique(buffer.sample_bridge(512, holdout=False)['goals'][:, 0]))
    holdout_ids = set(np.unique(buffer.sample_bridge(512, holdout=True)['goals'][:, 0]))
    assert train_ids and holdout_ids
    assert not (train_ids & holdout_ids)
    # The critic's sampler still sees everything, so its data stays SGCRL's.
    assert set(np.unique(buffer.sample(512)['goals'][:, 0])) >= holdout_ids


# ----------------------------------------------------------------------
# 4, 6, 7. Bridge shapes
# ----------------------------------------------------------------------
def test_the_deterministic_bridge_outputs_a_raw_state():
    agent = make_agent('online_sgcrl_det_bridge')
    batch = make_batch(5)
    predicted = agent.det_waypoint(batch['observations'], batch['goals'])
    assert predicted.shape == (5, OBS_DIM)
    assert np.all(np.isfinite(np.asarray(predicted)))


def test_the_flow_velocity_has_the_shape_of_a_state_displacement():
    agent = make_agent('online_sgcrl_rf_bridge')
    batch = make_batch(5)
    displacement = jnp.zeros((5, OBS_DIM))
    times = jnp.full((5, 1), 0.3)
    velocity = agent.rf_bridge.select('bridge')(
        batch['observations'], batch['goals'], displacement, times
    )
    assert velocity.shape == (5, OBS_DIM)


def test_flow_sampling_produces_a_raw_state_shaped_waypoint():
    agent = make_agent('online_sgcrl_rf_bridge')
    batch = make_batch(5)
    waypoint = agent.rf_waypoint(
        batch['observations'], batch['goals'], jax.random.PRNGKey(0)
    )
    assert waypoint.shape == (5, OBS_DIM)
    assert np.all(np.isfinite(np.asarray(waypoint)))


def test_the_flow_bridge_is_stochastic_and_the_deterministic_one_is_not():
    agent = make_agent('online_sgcrl_rf_bridge')
    batch = make_batch(5)
    first = agent.rf_waypoint(
        batch['observations'], batch['goals'], jax.random.PRNGKey(0)
    )
    second = agent.rf_waypoint(
        batch['observations'], batch['goals'], jax.random.PRNGKey(1)
    )
    assert not np.allclose(np.asarray(first), np.asarray(second))

    det_first = agent.det_waypoint(batch['observations'], batch['goals'])
    det_second = agent.det_waypoint(batch['observations'], batch['goals'])
    np.testing.assert_allclose(np.asarray(det_first), np.asarray(det_second))


# ----------------------------------------------------------------------
# 5. Bridge gradients only touch bridge parameters
# ----------------------------------------------------------------------
def test_the_deterministic_bridge_update_touches_only_its_own_parameters():
    agent = make_agent('online_sgcrl_det_bridge')
    buffer = fill_replay(make_replay())
    updated, info = agent.update_det_bridge(buffer.sample_bridge(8, alpha=0.5))
    for before, after in (
        (agent.critic.params, updated.critic.params),
        (agent.actor.params, updated.actor.params),
        (agent.rf_bridge.params, updated.rf_bridge.params),
    ):
        jax.tree_util.tree_map(
            lambda a, b: np.testing.assert_allclose(a, b), before, after
        )
    assert np.isfinite(float(info['det_bridge/loss']))
    assert any(
        bool(np.any(np.asarray(a) != np.asarray(b)))
        for a, b in zip(
            jax.tree_util.tree_leaves(agent.det_bridge.params),
            jax.tree_util.tree_leaves(updated.det_bridge.params),
        )
    )


def test_the_flow_bridge_update_touches_only_its_own_parameters():
    agent = make_agent('online_sgcrl_rf_bridge')
    buffer = fill_replay(make_replay())
    updated, info = agent.update_rf_bridge(buffer.sample_bridge(8, alpha=0.5))
    for before, after in (
        (agent.critic.params, updated.critic.params),
        (agent.actor.params, updated.actor.params),
        (agent.det_bridge.params, updated.det_bridge.params),
    ):
        jax.tree_util.tree_map(
            lambda a, b: np.testing.assert_allclose(a, b), before, after
        )
    assert np.isfinite(float(info['rf_bridge/loss']))


def test_training_a_bridge_does_not_disturb_the_reinforcement_learning_rng():
    # The three variants must consume identical randomness in their critic
    # and actor updates given identical data, so the bridges draw from their
    # own stream.
    agent = make_agent('online_sgcrl_rf_bridge')
    buffer = fill_replay(make_replay())
    batch = make_batch()

    without, _ = agent.update(batch)
    bridged, _ = agent.update_rf_bridge(buffer.sample_bridge(8, alpha=0.5))
    bridged, _ = bridged.update_det_bridge(buffer.sample_bridge(8, alpha=0.5))
    after, _ = bridged.update(batch)

    np.testing.assert_array_equal(np.asarray(without.rng), np.asarray(after.rng))
    jax.tree_util.tree_map(
        lambda a, b: np.testing.assert_allclose(a, b),
        without.actor.params,
        after.actor.params,
    )


def test_the_critic_update_leaves_the_bridges_untouched():
    # Separate optimizers matter: one shared Adam state would move the other
    # modules on a critic step through momentum even with zero gradients.
    agent = make_agent('online_sgcrl_rf_bridge')
    updated, _ = agent.update(make_batch())
    for before, after in (
        (agent.det_bridge.params, updated.det_bridge.params),
        (agent.rf_bridge.params, updated.rf_bridge.params),
    ):
        jax.tree_util.tree_map(
            lambda a, b: np.testing.assert_allclose(a, b), before, after
        )


# ----------------------------------------------------------------------
# 8, 9, 10, 11. Actor interface and objective
# ----------------------------------------------------------------------
def test_the_actor_interface_is_raw_state_and_raw_goal_in_every_variant():
    for variant in VARIANTS:
        agent = make_agent(variant)
        kernel = agent.actor.params['modules_actor']['Dense_0']['kernel']
        # Raw goal, not a repr_dim=4 latent.
        assert kernel.shape[0] == OBS_DIM + OBS_DIM


def test_the_three_variants_share_identical_critic_and_actor_parameters():
    # Any A-vs-B gap must come from behaviour, not from initialization.
    reference = make_agent('online_sgcrl', seed=3)
    for variant in VARIANTS[1:]:
        other = make_agent(variant, seed=3)
        for first, second in (
            (reference.critic.params, other.critic.params),
            (reference.actor.params, other.actor.params),
        ):
            jax.tree_util.tree_map(
                lambda a, b: np.testing.assert_allclose(a, b), first, second
            )


def test_the_actor_loss_contains_no_behavioural_cloning_term():
    # Negating the batch's actions must not change the actor loss; a BC term
    # would make the loss depend on them.
    agent = make_agent('online_sgcrl')
    batch = make_batch()
    rng = jax.random.PRNGKey(0)
    first, _ = agent.actor_loss(batch, agent.actor.params, rng)
    second, _ = agent.actor_loss(
        dict(batch, actions=-batch['actions']), agent.actor.params, rng
    )
    np.testing.assert_allclose(float(first), float(second), rtol=1e-6)


def test_the_actor_loss_is_identical_across_the_three_variants():
    # "Same actor objective" is a fairness requirement, so the bridge must
    # not appear in the actor loss at all.
    batch = make_batch()
    rng = jax.random.PRNGKey(0)
    losses = []
    for variant in VARIANTS:
        agent = make_agent(variant, seed=5)
        loss, _ = agent.actor_loss(batch, agent.actor.params, rng)
        losses.append(float(loss))
    assert losses[0] == losses[1] == losses[2]


def test_no_action_nce_term_exists_in_the_critic_loss():
    # Action-NCE would add a second softmax over shuffled actions, making the
    # loss depend on a permutation of the action column beyond its own row.
    agent = make_agent('online_sgcrl')
    batch = make_batch()
    loss, _ = agent.critic_loss(batch, agent.critic.params)
    logits = np.asarray(
        agent.critic_logits(batch['observations'], batch['actions'], batch['goals'])
    )
    log_partition = jax.scipy.special.logsumexp(logits, axis=1)
    expected = np.mean(
        (log_partition - np.diag(logits)) + 0.01 * np.asarray(log_partition) ** 2
    )
    np.testing.assert_allclose(float(loss), float(expected), rtol=1e-5)


def test_the_baseline_never_consults_a_bridge():
    agent = make_agent('online_sgcrl')
    observations = np.zeros((4, OBS_DIM), dtype=np.float32)
    goals = np.ones((4, OBS_DIM), dtype=np.float32)
    # In 'none' mode the actor's goal slot is the final goal, untouched.
    pointed_at = agent.waypoint(
        observations, goals, jax.random.PRNGKey(0), bridge_mode='none'
    )
    np.testing.assert_allclose(np.asarray(pointed_at), goals)


def test_each_variant_fixes_its_own_bridge_mode():
    assert VARIANTS == (
        'online_sgcrl',
        'online_sgcrl_det_bridge',
        'online_sgcrl_rf_bridge',
    )
    assert [VARIANT_SETTINGS[v]['bridge_mode'] for v in VARIANTS] == list(BRIDGE_MODES)


def test_a_config_cannot_override_a_variants_bridge_mode():
    config = get_config()
    config.variant = 'online_sgcrl'
    config.bridge_mode = 'rectified_flow'
    with pytest.raises(ValueError, match='fixes bridge_mode'):
        OnlineSGCRLAgent.create(
            0,
            np.zeros((2, OBS_DIM), dtype=np.float32),
            np.zeros((2, ACTION_DIM), dtype=np.float32),
            config,
        )


def test_random_goals_doubles_the_actor_batch_and_shuffles_half_of_it():
    agent = make_agent('online_sgcrl')
    batch = make_batch()
    states, goals = agent._actor_batch(batch)
    assert states.shape[0] == 2 * batch['observations'].shape[0]
    np.testing.assert_allclose(np.asarray(goals)[:8], batch['goals'])
    np.testing.assert_allclose(
        np.asarray(goals)[8:], np.roll(batch['goals'], 1, axis=0)
    )


def test_sgcrl_defaults_match_the_documented_launcher_settings():
    config = get_config()
    assert int(config.repr_dim) == 64
    assert bool(config.repr_norm) is False
    assert tuple(config.hidden_dims) == (256, 256)
    assert float(config.learning_rate) == 3e-4
    assert float(config.actor_learning_rate) == 3e-4
    assert float(config.discount) == 0.99
    assert int(config.batch_size) == 256
    assert float(config.logsumexp_coef) == 0.01
    assert float(config.entropy_coefficient) == 0.0
    assert int(config.min_replay_size) == 10_000
    assert int(config.max_replay_size) == 1_000_000
    assert float(config.updates_per_env_step) == 1.0
    assert float(config.bridge_alpha) == 0.5


def test_the_policy_scale_matches_acmes_softplus_parameterization():
    # softplus(0) + actor_min_std, with SGCRL's actor_min_std = 1e-6.
    agent = make_agent('online_sgcrl')
    observations = np.zeros((4, OBS_DIM), dtype=np.float32)
    _, scale = agent.actor_distribution(observations, observations)
    np.testing.assert_allclose(np.asarray(scale), 0.693147 + 1e-6, rtol=1e-5)


def test_the_policy_mean_is_squashed_before_the_outer_tanh():
    # distributional.py maps loc -> 10 * tanh(loc / 10), so |loc| < 10 always.
    agent = make_agent('online_sgcrl')
    modules = dict(agent.actor.params['modules_actor'])
    mean_head = modules['Dense_2']
    modules['Dense_2'] = dict(mean_head, bias=jnp.full_like(mean_head['bias'], 1e4))
    observations = np.zeros((4, OBS_DIM), dtype=np.float32)
    loc, _ = agent.actor_distribution(
        observations, observations, params={'modules_actor': modules}
    )
    assert np.all(np.abs(np.asarray(loc)) <= 10.0)
    np.testing.assert_allclose(np.asarray(loc), 10.0, rtol=1e-5)


# ----------------------------------------------------------------------
# 12. Branch snapshot equality
# ----------------------------------------------------------------------
def test_three_variants_built_from_one_snapshot_start_bit_identical():
    import main_online

    warm = make_agent('online_sgcrl', seed=7)
    buffer = fill_replay(make_replay(), num_episodes=10, length=20)
    # Move the parameters off their initialization so equality is not trivial.
    warm, _ = warm.update(make_batch())
    warm, _ = warm.update_det_bridge(buffer.sample_bridge(8, alpha=0.5))
    warm, _ = warm.update_rf_bridge(buffer.sample_bridge(8, alpha=0.5))
    reference = main_online._snapshot_digests(warm, buffer)

    for variant in VARIANTS:
        fresh = make_agent(variant, seed=99)
        restored = fresh.replace(
            critic=fresh.critic.replace(params=warm.critic.params),
            actor=fresh.actor.replace(params=warm.actor.params),
            det_bridge=fresh.det_bridge.replace(params=warm.det_bridge.params),
            rf_bridge=fresh.rf_bridge.replace(params=warm.rf_bridge.params),
        )
        assert main_online._snapshot_digests(restored, buffer) == reference


def test_the_snapshot_digest_notices_a_single_changed_weight():
    import main_online

    agent = make_agent('online_sgcrl', seed=7)
    buffer = fill_replay(make_replay(), num_episodes=10, length=20)
    before = main_online._snapshot_digests(agent, buffer)

    modules = dict(agent.actor.params['modules_actor'])
    dense = modules['Dense_0']
    kernel = jnp.asarray(dense['kernel']).at[0, 0].add(1e-6)
    modules['Dense_0'] = dict(dense, kernel=kernel)
    perturbed = agent.replace(
        actor=agent.actor.replace(params={'modules_actor': modules})
    )
    assert main_online._snapshot_digests(perturbed, buffer)['actor'] != before['actor']


def test_the_replay_digest_covers_the_holdout_assignment():
    import main_online

    agent = make_agent('online_sgcrl', seed=7)
    first = fill_replay(make_replay(), num_episodes=10, length=20)
    second = fill_replay(make_replay(), num_episodes=10, length=20)
    assert (
        main_online._snapshot_digests(agent, first)['replay']
        == main_online._snapshot_digests(agent, second)['replay']
    )
    second.add_episode(
        np.zeros((21, OBS_DIM), dtype=np.float32),
        np.zeros((20, ACTION_DIM), dtype=np.float32),
    )
    assert (
        main_online._snapshot_digests(agent, first)['replay']
        != main_online._snapshot_digests(agent, second)['replay']
    )


def test_the_replay_round_trips_through_its_state_dict():
    buffer = fill_replay(make_replay(), num_episodes=10, length=15)
    restored = make_replay()
    restored.load_state_dict(buffer.state_dict())
    assert len(restored) == len(buffer)
    assert restored.num_episodes == buffer.num_episodes
    first = buffer.sample(16)
    second = restored.sample(16)
    for key in first:
        np.testing.assert_allclose(first[key], second[key])
    np.testing.assert_allclose(
        buffer.sample_bridge(16, alpha=0.5)['waypoints'],
        restored.sample_bridge(16, alpha=0.5)['waypoints'],
    )


# ----------------------------------------------------------------------
# 13 & 14. Paired evaluation reproducibility and RNG seeding
# ----------------------------------------------------------------------
def test_the_manifest_pins_both_environment_random_sources():
    manifest = online_episode_manifest(task_id=1, num_episodes=4, seed=0)
    assert len(manifest) == 4
    for entry in manifest:
        # reset(seed=...) covers env.np_random; OGBench builds the goal by
        # sampling the action space, which carries its own generator.
        assert entry['action_space_seed'] == entry['env_seed'] + 5
        assert entry['task_id'] == 1


def test_variants_evaluated_at_the_same_seed_get_identical_episodes():
    assert online_episode_manifest(1, 8, 2) == online_episode_manifest(1, 8, 2)


def test_different_seeds_produce_different_episodes():
    first = online_episode_manifest(1, 8, 0)
    second = online_episode_manifest(1, 8, 1)
    assert [e['env_seed'] for e in first] != [e['env_seed'] for e in second]


def test_deterministic_actions_ignore_the_rng_for_the_baseline():
    agent = make_agent('online_sgcrl')
    observations = np.zeros((4, OBS_DIM), dtype=np.float32)
    goals = np.ones((4, OBS_DIM), dtype=np.float32)
    first = agent.act(observations, goals, jax.random.PRNGKey(0), deterministic=True)
    second = agent.act(observations, goals, jax.random.PRNGKey(1), deterministic=True)
    np.testing.assert_allclose(np.asarray(first), np.asarray(second))
    assert np.all(np.abs(np.asarray(first)) <= 1.0)


def test_evaluation_actions_reproduce_for_a_fixed_rng_in_every_mode():
    observations = np.zeros((4, OBS_DIM), dtype=np.float32)
    goals = np.ones((4, OBS_DIM), dtype=np.float32)
    for variant, mode in zip(VARIANTS, BRIDGE_MODES):
        agent = make_agent(variant)
        rng = jax.random.PRNGKey(3)
        first = agent.act(observations, goals, rng, deterministic=True, bridge_mode=mode)
        second = agent.act(
            observations, goals, rng, deterministic=True, bridge_mode=mode
        )
        np.testing.assert_allclose(np.asarray(first), np.asarray(second))


def test_the_three_bridge_modes_point_the_actor_somewhere_different():
    # One parameter set must be able to drive all three behaviours, which is
    # what the shared-warmup branch relies on.
    agent = make_agent('online_sgcrl_rf_bridge')
    observations = np.zeros((4, OBS_DIM), dtype=np.float32)
    goals = np.ones((4, OBS_DIM), dtype=np.float32)
    rng = jax.random.PRNGKey(0)
    actions = {
        mode: np.asarray(
            agent.act(observations, goals, rng, deterministic=True, bridge_mode=mode)
        )
        for mode in BRIDGE_MODES
    }
    assert not np.allclose(actions['none'], actions['deterministic'])
    assert not np.allclose(actions['deterministic'], actions['rectified_flow'])


# ----------------------------------------------------------------------
# Diagnostics and a short end-to-end loop
# ----------------------------------------------------------------------
def test_bridge_diagnostics_report_every_documented_quantity():
    agent = make_agent('online_sgcrl_rf_bridge')
    buffer = fill_replay(make_replay())
    batch = buffer.sample_bridge(16, alpha=0.5, holdout=True)
    info = agent.bridge_diagnostics(batch, jax.random.PRNGKey(0))
    expected = {
        'diagnostics/final_goal_score_direct',
        'diagnostics/det_waypoint_mse',
        'diagnostics/det_displacement_norm',
        'diagnostics/det_final_goal_score',
        'diagnostics/det_waypoint_score',
        'diagnostics/rf_waypoint_mse',
        'diagnostics/rf_displacement_norm',
        'diagnostics/rf_final_goal_score',
        'diagnostics/rf_waypoint_score',
        'diagnostics/rf_pairwise_distance',
        'diagnostics/rf_sample_std',
        'diagnostics/rf_conditional_variance',
        'diagnostics/rf_action_diversity',
        'diagnostics/rf_score_std',
        'diagnostics/true_displacement_norm',
    }
    assert expected <= set(info)
    for key, value in info.items():
        assert np.isfinite(float(value)), key
    # The pairwise distance is a mean over sample pairs *and* over the batch,
    # so it has to sit on the same scale as the per-sample spread rather than
    # accumulating one factor of the batch size.
    spread = float(info['diagnostics/rf_sample_std']) * np.sqrt(OBS_DIM)
    assert 0.0 < float(info['diagnostics/rf_pairwise_distance']) < 10.0 * spread


@pytest.mark.parametrize('variant', VARIANTS)
def test_a_short_update_loop_stays_finite_for_every_variant(variant):
    agent = make_agent(variant)
    buffer = fill_replay(make_replay())
    mode = VARIANT_SETTINGS[variant]['bridge_mode']
    for _ in range(5):
        agent, info = agent.update(buffer.sample(8))
        if mode == 'deterministic':
            agent, extra = agent.update_det_bridge(buffer.sample_bridge(8, alpha=0.5))
            info = {**info, **extra}
        elif mode == 'rectified_flow':
            agent, extra = agent.update_rf_bridge(buffer.sample_bridge(8, alpha=0.5))
            info = {**info, **extra}
        for key, value in info.items():
            assert np.isfinite(float(value)), key
    assert np.all(
        np.isfinite(np.asarray(agent.actor.params['modules_actor']['Dense_0']['kernel']))
    )
