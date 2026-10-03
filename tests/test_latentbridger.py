"""Unit tests for the LatentBridger ablation branch.

Observations in the synthetic fixtures are self-describing: channel 0 holds the
episode index and channel 1 holds the global transition index.  Episode
boundaries and sampled offsets can therefore be checked exactly, instead of
being inferred from the sampler's own bookkeeping.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.latentbridger import (
    VARIANT_SETTINGS,
    LatentBridgerAgent,
    trainable_modules,
)
from configs.latent.cube_single import get_config
from utils.datasets import Dataset
from utils.latent_datasets import LatentBridgerDataset

_EPISODE_INDEX = 0
_GLOBAL_INDEX = 1
_OBS_DIM = 6
_ACTION_DIM = 3


def _make_dataset(episode_lengths=(50, 50, 50, 50), seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    size = int(sum(episode_lengths))
    observations = rng.normal(size=(size, _OBS_DIM)).astype(np.float32)
    terminals = np.zeros(size, dtype=np.float32)

    cursor = 0
    for episode, length in enumerate(episode_lengths):
        observations[cursor : cursor + length, _EPISODE_INDEX] = float(episode)
        observations[cursor : cursor + length, _GLOBAL_INDEX] = np.arange(
            cursor,
            cursor + length,
            dtype=np.float32,
        )
        cursor += length
        terminals[cursor - 1] = 1.0

    actions = rng.uniform(-1.0, 1.0, size=(size, _ACTION_DIM)).astype(np.float32)
    return Dataset.create(
        observations=observations,
        actions=actions,
        terminals=terminals,
    )


def _make_config(variant: str = 'sa_cl_bc', **overrides):
    config = get_config(variant)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _make_agent(
    config,
    batch,
    *,
    stage: str,
    action_low=-1.0,
    action_high=1.0,
    seed: int = 0,
) -> LatentBridgerAgent:
    return LatentBridgerAgent.create(
        seed,
        batch['observations'],
        batch['actions'],
        config,
        stage=stage,
        action_low=np.broadcast_to(np.asarray(action_low, dtype=np.float32), (_ACTION_DIM,)),
        action_high=np.broadcast_to(np.asarray(action_high, dtype=np.float32), (_ACTION_DIM,)),
    )


def _trees_equal(left, right) -> bool:
    return bool(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(lambda a, b: jnp.array_equal(a, b), left, right)
        )
    )


@pytest.fixture(autouse=True)
def _deterministic_host_rng():
    np.random.seed(12345)


# ----------------------------------------------------------------------
# 1-3. Dataset supervision and episode boundaries
# ----------------------------------------------------------------------
def test_future_positive_sampling_never_crosses_episode_boundaries():
    dataset = _make_dataset()
    for sampling in ('geometric', 'uniform', 'trajectory'):
        sampler = LatentBridgerDataset(
            dataset,
            _make_config(future_sampling=sampling, actor_goal_max_offset=5),
        )
        for _ in range(20):
            batch = sampler.sample(256)
            anchors = batch['observations']
            for goal_key, offset_key in (
                ('contrastive_goals', 'contrastive_offsets'),
                ('actor_goals', 'actor_offsets'),
                ('bridge_goals', 'bridge_goal_offsets'),
            ):
                goals = batch[goal_key]
                assert np.array_equal(
                    goals[:, _EPISODE_INDEX],
                    anchors[:, _EPISODE_INDEX],
                ), f'{goal_key} crossed an episode boundary with {sampling} sampling.'
                offsets = batch[offset_key]
                assert np.all(offsets >= 1.0)
                assert np.allclose(
                    goals[:, _GLOBAL_INDEX] - anchors[:, _GLOBAL_INDEX],
                    offsets,
                )


def test_bridge_targets_never_cross_episode_boundaries():
    dataset = _make_dataset(episode_lengths=(4, 7, 50))
    sampler = LatentBridgerDataset(dataset, _make_config())
    for _ in range(50):
        batch = sampler.sample(128)
        anchors = batch['observations']
        targets = batch['bridge_targets']
        assert targets.shape[1] == 5
        assert np.array_equal(
            targets[:, :, _EPISODE_INDEX],
            np.repeat(anchors[:, _EPISODE_INDEX : _EPISODE_INDEX + 1], 5, axis=1),
        )
        # Every prefix state is a real in-episode successor of the anchor.
        steps = targets[:, :, _GLOBAL_INDEX] - anchors[:, _GLOBAL_INDEX : _GLOBAL_INDEX + 1]
        assert np.all(steps >= 1.0)
        assert np.all(steps <= 5.0)
        assert np.all(np.diff(steps, axis=1) >= 0.0)


def test_close_goal_prefix_is_clipped_and_padded():
    # Episodes of four states force bridge goals that land before t+5.
    dataset = _make_dataset(episode_lengths=(4, 4, 4, 4))
    sampler = LatentBridgerDataset(dataset, _make_config())
    saw_padding = False
    for _ in range(50):
        batch = sampler.sample(128)
        anchors = batch['observations']
        goals = batch['bridge_goals']
        targets = batch['bridge_targets']
        goal_steps = goals[:, _GLOBAL_INDEX] - anchors[:, _GLOBAL_INDEX]
        prefix_steps = (
            targets[:, :, _GLOBAL_INDEX] - anchors[:, _GLOBAL_INDEX : _GLOBAL_INDEX + 1]
        )
        # The prefix is clipped at the goal and padded with it afterwards.
        expected = np.minimum(
            np.arange(1, 6, dtype=np.float32)[None, :],
            goal_steps[:, None],
        )
        assert np.allclose(prefix_steps, expected)
        padded = prefix_steps[:, -1] == goal_steps
        assert np.all(padded)
        saw_padding = saw_padding or bool(np.any(goal_steps < 5.0))
        np.testing.assert_allclose(
            targets[np.arange(len(goals)), -1],
            goals,
        )
    assert saw_padding, 'Short episodes should have produced close-goal padding.'


# ----------------------------------------------------------------------
# 4-6. Contrastive critic
# ----------------------------------------------------------------------
def test_infonce_uses_diagonal_positive_labels():
    dataset = _make_dataset()
    config = _make_config('sa_cl_bc')
    sampler = LatentBridgerDataset(dataset, config)
    batch = sampler.sample(32)
    agent = _make_agent(config, batch, stage='critic')

    loss, info = agent.contrastive_loss(batch, agent.network.params)
    logits = np.asarray(
        agent.critic_score_matrix(
            batch['observations'],
            batch['actions'],
            batch['contrastive_goals'],
        )
    )
    log_probabilities = logits - jax.scipy.special.logsumexp(
        logits,
        axis=-1,
        keepdims=True,
    )
    expected = -np.mean(np.diag(np.asarray(log_probabilities)))

    np.testing.assert_allclose(float(loss), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        float(info['contrastive/positive_logit']),
        float(np.mean(np.diag(logits))),
        rtol=1e-5,
        atol=1e-5,
    )
    # A permuted goal batch moves the positives off the diagonal, so the loss
    # must get worse; this is what pins the label convention down.
    permuted = dict(batch)
    permuted['contrastive_goals'] = np.roll(batch['contrastive_goals'], 1, axis=0)
    permuted_loss, _ = agent.contrastive_loss(permuted, agent.network.params)
    assert float(permuted_loss) != pytest.approx(float(loss))


def test_state_action_critic_output_shapes():
    dataset = _make_dataset()
    config = _make_config('sa_cl_bc')
    sampler = LatentBridgerDataset(dataset, config)
    batch = sampler.sample(16)
    agent = _make_agent(config, batch, stage='critic')
    repr_dim = int(config.repr_dim)

    anchors = agent.encode_state_action(batch['observations'], batch['actions'])
    goal_latents = agent.goal_latents(batch['contrastive_goals'])
    assert anchors.shape == (16, repr_dim)
    assert goal_latents.shape == (16, repr_dim)
    assert agent.critic_scores(
        batch['observations'],
        batch['actions'],
        batch['contrastive_goals'],
    ).shape == (16,)
    assert agent.critic_score_matrix(
        batch['observations'],
        batch['actions'],
        batch['contrastive_goals'],
    ).shape == (16, 16)
    # repr_norm=True must actually put the embeddings on the unit sphere.
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(anchors), axis=-1),
        np.ones(16),
        rtol=1e-5,
        atol=1e-5,
    )


def test_state_only_critic_output_shapes_and_action_invariance():
    dataset = _make_dataset()
    config = _make_config('state_cl')
    sampler = LatentBridgerDataset(dataset, config)
    batch = sampler.sample(16)
    agent = _make_agent(config, batch, stage='critic')
    repr_dim = int(config.repr_dim)

    anchors = agent.encode_state(batch['observations'])
    assert anchors.shape == (16, repr_dim)
    assert agent.critic_score_matrix(
        batch['observations'],
        batch['actions'],
        batch['contrastive_goals'],
    ).shape == (16, 16)

    other_actions = np.roll(batch['actions'], 1, axis=0)
    np.testing.assert_allclose(
        np.asarray(
            agent.critic_scores(
                batch['observations'],
                batch['actions'],
                batch['contrastive_goals'],
            )
        ),
        np.asarray(
            agent.critic_scores(
                batch['observations'],
                other_actions,
                batch['contrastive_goals'],
            )
        ),
    )


# ----------------------------------------------------------------------
# 7-8. Actor bounds and action-contrastive supervision
# ----------------------------------------------------------------------
def test_actor_actions_respect_configured_bounds():
    dataset = _make_dataset()
    config = _make_config('sa_cl_bc')
    sampler = LatentBridgerDataset(dataset, config)
    batch = sampler.sample(64)
    low = np.array([-0.5, -2.0, 0.25], dtype=np.float32)
    high = np.array([1.0, 0.25, 3.0], dtype=np.float32)
    agent = _make_agent(config, batch, stage='actor', action_low=low, action_high=high)

    actions = np.asarray(
        agent.sample_actions_from_goals(batch['observations'], batch['actor_goals'])
    )
    assert actions.shape == (64, _ACTION_DIM)
    assert np.all(actions >= low - 1e-5)
    assert np.all(actions <= high + 1e-5)

    # Bounds must survive training, not just initialization.
    for _ in range(5):
        agent, _ = agent.update(batch)
    trained_actions = np.asarray(
        agent.sample_actions_from_goals(batch['observations'], batch['actor_goals'])
    )
    assert np.all(trained_actions >= low - 1e-5)
    assert np.all(trained_actions <= high + 1e-5)


def test_action_nce_shapes_and_gradients():
    dataset = _make_dataset()
    config = _make_config('sa_cl_bc_actnce', num_action_negatives=8)
    sampler = LatentBridgerDataset(dataset, config)
    batch = sampler.sample(32)
    agent = _make_agent(config, batch, stage='critic')

    loss, info = agent.action_contrastive_loss(batch, agent.network.params)
    assert loss.shape == ()
    for key in (
        'action_nce/loss',
        'action_nce/positive_score',
        'action_nce/negative_score',
        'action_nce/gap',
    ):
        assert info[key].shape == ()
    assert float(info['action_nce/num_negatives']) == 8.0
    # A uniform (1 + K)-way softmax starts at log(1 + K).
    np.testing.assert_allclose(float(loss), np.log(9.0), rtol=0.5)

    grads = jax.grad(lambda params: agent.action_contrastive_loss(batch, params)[0])(
        agent.network.params
    )
    phi_sa_grads = jax.tree_util.tree_leaves(grads['modules_phi_sa'])
    assert phi_sa_grads
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in phi_sa_grads)
    assert any(np.any(np.asarray(leaf) != 0.0) for leaf in phi_sa_grads)


# ----------------------------------------------------------------------
# 9-10. Latent rectified flow
# ----------------------------------------------------------------------
def test_latent_flow_training_tensor_shapes():
    dataset = _make_dataset()
    config = _make_config('latent_rf')
    sampler = LatentBridgerDataset(dataset, config)
    batch = sampler.sample(16)
    agent = _make_agent(config, batch, stage='flow')
    repr_dim = int(config.repr_dim)
    horizon = int(config.action_horizon)

    assert batch['bridge_targets'].shape == (16, horizon, _OBS_DIM)
    prefix_latents = agent._encode_prefix(jnp.asarray(batch['bridge_targets']))
    assert prefix_latents.shape == (16, horizon, repr_dim)

    velocities = agent.network.select('flow')(
        jnp.zeros((16, horizon, repr_dim)),
        jnp.zeros((16, 1)),
        jnp.zeros((16, repr_dim)),
        jnp.zeros((16, repr_dim)),
    )
    assert velocities.shape == (16, horizon, repr_dim)

    loss, info = agent.flow_loss(batch, agent.network.params, jax.random.PRNGKey(0))
    assert loss.shape == ()
    assert np.isfinite(float(info['flow/loss']))


def test_euler_integration_produces_prefix_shape():
    dataset = _make_dataset()
    config = _make_config('latent_rf')
    sampler = LatentBridgerDataset(dataset, config)
    batch = sampler.sample(8)
    agent = _make_agent(config, batch, stage='flow')
    repr_dim = int(config.repr_dim)
    horizon = int(config.action_horizon)

    prefix = agent.sample_latent_prefix(
        batch['observations'],
        batch['bridge_goals'],
        jax.random.PRNGKey(0),
    )
    assert prefix.shape == (8, horizon, repr_dim)
    assert np.all(np.isfinite(np.asarray(prefix)))
    # flow_renormalize keeps generated latents on the actor's training manifold.
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(prefix), axis=-1),
        np.ones((8, horizon)),
        rtol=1e-5,
        atol=1e-5,
    )

    single = agent.sample_latent_prefix(
        batch['observations'][0],
        batch['bridge_goals'][0],
        jax.random.PRNGKey(0),
    )
    assert single.shape == (horizon, repr_dim)


# ----------------------------------------------------------------------
# 11-12. Staged freezing
# ----------------------------------------------------------------------
def test_frozen_critic_is_unchanged_by_actor_updates():
    dataset = _make_dataset()
    config = _make_config('sa_cl_bc')
    sampler = LatentBridgerDataset(dataset, config)
    batch = sampler.sample(32)
    agent = _make_agent(config, batch, stage='actor')

    before = jax.tree_util.tree_map(jnp.copy, agent.network.params)
    for _ in range(5):
        agent, _ = agent.update(sampler.sample(32))
    after = agent.network.params

    for module in ('phi_sa', 'phi_s', 'psi'):
        assert _trees_equal(
            before[f'modules_{module}'], after[f'modules_{module}']
        ), f'{module} changed during an actor-only stage.'
    assert not _trees_equal(before['modules_actor'], after['modules_actor'])
    assert trainable_modules('actor') == ('actor',)


def test_frozen_critic_and_actor_are_unchanged_by_flow_updates():
    dataset = _make_dataset()
    config = _make_config('latent_rf')
    sampler = LatentBridgerDataset(dataset, config)
    batch = sampler.sample(32)
    agent = _make_agent(config, batch, stage='flow')

    before = jax.tree_util.tree_map(jnp.copy, agent.network.params)
    for _ in range(5):
        agent, _ = agent.update(sampler.sample(32))
    after = agent.network.params

    for module in ('phi_sa', 'phi_s', 'psi', 'actor'):
        assert _trees_equal(
            before[f'modules_{module}'], after[f'modules_{module}']
        ), f'{module} changed during a flow-only stage.'
    assert not _trees_equal(before['modules_flow'], after['modules_flow'])


def test_variant_table_isolates_one_knob_per_comparison():
    """The ablation claims only hold if neighbouring variants differ by one field."""

    def difference(left: str, right: str) -> set[str]:
        return {
            key
            for key in VARIANT_SETTINGS[left]
            if VARIANT_SETTINGS[left][key] != VARIANT_SETTINGS[right][key]
        }

    assert difference('sa_cl', 'sa_cl_bc') == {'actor_bc_coef'}
    assert difference('sa_cl_bc', 'sa_cl_bc_actnce') == {'action_nce_coef'}
    assert difference('sa_cl_bc', 'latent_rf') == {'use_flow', 'eval_mode'}
    # latent_rf_actnce is Module B on the action-sensitive critic, and differs
    # from latent_rf by exactly the action-NCE coefficient.
    assert difference('sa_cl_bc_actnce', 'latent_rf_actnce') == {
        'use_flow',
        'eval_mode',
    }
    assert difference('latent_rf', 'latent_rf_actnce') == {'action_nce_coef'}
    # v1: the only thing separating local from multi-horizon is the offset set,
    # and the only thing separating multi-horizon from the sparse bridge is
    # Module B (plus the replanning rate that a sparse bridge forces).
    assert difference('actnce_local', 'actnce_multihorizon') == {
        'actor_goal_offsets'
    }
    assert difference('actnce_multihorizon', 'latent_rf_sparse') == {
        'use_flow',
        'eval_mode',
        'flow_target_mode',
        'replan_interval',
    }
    # actnce_local must be numerically the v0 action-NCE controller, so the two
    # pilots remain comparable.
    assert difference('sa_cl_bc_actnce', 'actnce_local') == set()


def test_suite_runner_shares_critic_and_actor_across_variants():
    """The flow variants must build on the controller they claim to extend."""

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        'run_latentbridger_suite',
        Path(__file__).resolve().parents[1] / 'scripts' / 'run_latentbridger_suite.py',
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    for base, extended in (
        ('sa_cl_bc', 'latent_rf'),
        ('sa_cl_bc_actnce', 'latent_rf_actnce'),
        ('actnce_multihorizon', 'latent_rf_sparse'),
    ):
        assert runner.critic_signature(base) == runner.critic_signature(extended)
        assert runner.actor_signature(base) == runner.actor_signature(extended)

    # Variants that should *not* share must stay separate.
    assert runner.critic_signature('sa_cl_bc') != runner.critic_signature(
        'sa_cl_bc_actnce'
    )
    assert runner.actor_signature('sa_cl') != runner.actor_signature('sa_cl_bc')
    assert runner.critic_signature('gcbc') is None

    assert runner.eval_jobs('sa_cl_bc', (1, 5)) == (('direct_goal', None),)
    assert runner.eval_jobs('latent_rf_actnce', (1, 5)) == (
        ('direct_goal', None),
        ('latent_flow', 1),
        ('latent_flow', 5),
    )
    # The multi-horizon actor must not be confused with the local one.
    assert runner.actor_signature('actnce_local') != runner.actor_signature(
        'actnce_multihorizon'
    )
    # A sparse bridge is only defined at replan_interval=1, whatever is asked.
    assert runner.eval_jobs('latent_rf_sparse', (1, 5)) == (
        ('direct_goal', None),
        ('latent_flow', 1),
    )


# ----------------------------------------------------------------------
# v1: latent temporal-scale mismatch
# ----------------------------------------------------------------------
def test_sparse_prefix_offsets_span_the_planning_horizon():
    from utils.latent_datasets import sparse_prefix_offsets

    # The spec's worked example: PathBridger H=40 with a five-waypoint prefix.
    assert sparse_prefix_offsets(40, 5) == (8, 16, 24, 32, 40)
    assert sparse_prefix_offsets(25, 5) == (5, 10, 15, 20, 25)
    # Rounds up so the last waypoint always lands exactly on the horizon.
    assert sparse_prefix_offsets(13, 5) == (3, 6, 8, 11, 13)
    for horizon in (5, 13, 25, 40):
        offsets = sparse_prefix_offsets(horizon, 5)
        assert offsets[-1] == horizon
        assert list(offsets) == sorted(set(offsets))
    with pytest.raises(ValueError, match='horizon >= action_horizon'):
        sparse_prefix_offsets(3, 5)


def test_multihorizon_actor_offsets_are_drawn_from_the_configured_set():
    dataset = _make_dataset()
    config = _make_config('actnce_multihorizon')
    sampler = LatentBridgerDataset(dataset, config)
    assert sampler.actor_goal_offsets == (1, 2, 4, 8, 16)

    batch = sampler.sample(4096)
    offsets = batch['actor_offsets'].astype(np.int64)
    assert set(np.unique(offsets)) <= {1, 2, 4, 8, 16}
    # Every horizon must actually appear, or the ablation is not testing one.
    assert set(np.unique(offsets)) == {1, 2, 4, 8, 16}

    # The goal really is s_{t+delta} of the same episode.
    anchors = batch['observations']
    goals = batch['actor_goals']
    assert np.array_equal(anchors[:, _EPISODE_INDEX], goals[:, _EPISODE_INDEX])
    assert np.array_equal(
        goals[:, _GLOBAL_INDEX] - anchors[:, _GLOBAL_INDEX],
        batch['actor_offsets'],
    )


def test_multihorizon_offsets_never_cross_the_episode_boundary():
    # Episodes far shorter than the largest offset: the draw must fall back to
    # the horizons that fit rather than clamping everything onto the terminal.
    dataset = _make_dataset(episode_lengths=(6, 6, 6, 6))
    sampler = LatentBridgerDataset(dataset, _make_config('actnce_multihorizon'))
    batch = sampler.sample(2048)
    assert set(np.unique(batch['actor_offsets'].astype(np.int64))) <= {1, 2, 4}
    assert np.array_equal(
        batch['observations'][:, _EPISODE_INDEX],
        batch['actor_goals'][:, _EPISODE_INDEX],
    )


def test_local_variant_sampling_is_unchanged_from_v0():
    """actnce_local must reproduce the released delta=1 actor supervision."""

    dataset = _make_dataset()
    np.random.seed(7)
    v0 = LatentBridgerDataset(dataset, _make_config('sa_cl_bc_actnce')).sample(256)
    np.random.seed(7)
    v1 = LatentBridgerDataset(dataset, _make_config('actnce_local')).sample(256)
    for key in v0:
        assert np.array_equal(v0[key], v1[key]), key
    assert np.all(v0['actor_offsets'] == 1.0)


def test_sparse_bridge_targets_are_long_horizon_waypoints():
    dataset = _make_dataset(episode_lengths=(200, 200))
    sampler = LatentBridgerDataset(dataset, _make_config('latent_rf_sparse'))
    assert sampler.prefix_offsets == (8, 16, 24, 32, 40)

    batch = sampler.sample(512)
    anchors = batch['observations'][:, _GLOBAL_INDEX]
    targets = batch['bridge_targets'][:, :, _GLOBAL_INDEX]
    realized = targets - anchors[:, None]
    assert np.array_equal(realized, batch['bridge_target_offsets'])

    goal_offsets = batch['bridge_goal_offsets']
    for index, offset in enumerate(sampler.prefix_offsets):
        # Each waypoint is its offset, or the goal when the goal comes first.
        expected = np.minimum(offset, goal_offsets)
        assert np.array_equal(realized[:, index], expected)
    # Targets stay inside the anchor's episode.
    assert np.all(
        batch['bridge_targets'][:, :, _EPISODE_INDEX]
        == batch['observations'][:, None, _EPISODE_INDEX]
    )


def test_sparse_bridge_requires_single_step_replanning():
    dataset = _make_dataset()
    batch = LatentBridgerDataset(dataset, _make_config('latent_rf_sparse')).sample(8)
    agent = _make_agent(_make_config('latent_rf_sparse'), batch, stage='flow')
    assert tuple(agent.config['flow_target_offsets']) == (8, 16, 24, 32, 40)
    assert int(agent.config['replan_interval']) == 1

    with pytest.raises(ValueError, match="requires replan_interval=1"):
        _make_agent(
            _make_config('latent_rf_sparse', replan_interval=5),
            batch,
            stage='flow',
        )


def test_lsnr_compares_flow_error_against_true_latent_motion():
    """LSNR_h must be D_h / E_h on the same rows, excluding clipped waypoints."""

    from utils.latent_diagnostics import flow_reconstruction

    dataset = _make_dataset(episode_lengths=(200, 200))
    config = _make_config('latent_rf_sparse')
    sampler = LatentBridgerDataset(dataset, config)
    agent = _make_agent(config, sampler.sample(8), stage='flow')

    metrics = flow_reconstruction(agent, sampler, batch_size=128, num_batches=2)
    for offset in (8, 16, 24, 32, 40):
        assert f'flow/D_h{offset}' in metrics
        assert f'flow/E_h{offset}' in metrics
        assert metrics[f'flow/lsnr_h{offset}'] == pytest.approx(
            metrics[f'flow/D_h{offset}'] / metrics[f'flow/E_h{offset}'],
            rel=1e-5,
        )
        assert 0.0 <= metrics[f'flow/unclipped_fraction_h{offset}'] <= 1.0
        assert 0.0 <= metrics[f'flow/retrieval_h{offset}'] <= 1.0
    assert metrics['flow/min_lsnr'] == min(
        metrics[f'flow/lsnr_h{offset}'] for offset in (8, 16, 24, 32, 40)
    )
    # Clipping at the conditioning goal can only bite harder for a farther
    # waypoint, so the surviving fraction must be non-increasing in h.
    kept = [metrics[f'flow/unclipped_fraction_h{offset}'] for offset in (8, 16, 24, 32, 40)]
    assert kept == sorted(kept, reverse=True)
    assert metrics['flow/target_offset_step_1'] == 8.0


def test_actor_goal_sensitivity_detects_a_goal_blind_actor():
    from utils.latent_diagnostics import actor_goal_sensitivity

    dataset = _make_dataset(episode_lengths=(200, 200))
    config = _make_config('actnce_multihorizon')
    sampler = LatentBridgerDataset(dataset, config)
    agent = _make_agent(config, sampler.sample(8), stage='actor')

    metrics = actor_goal_sensitivity(agent, sampler, batch_size=128)
    for horizon in (1, 2, 4, 8, 16):
        assert metrics[f'actor_goal_sensitivity/action_delta_{horizon}'] >= 0.0
        assert f'actor_goal_sensitivity/critic_gain_{horizon}' in metrics

    class _GoalBlindAgent:
        """An actor that ignores its conditioning vector entirely."""

        def __init__(self, inner):
            self._inner = inner
            self.config = inner.config

        def sample_actions_from_goals(self, observations, goals):
            del goals
            return np.zeros((len(observations), _ACTION_DIM), dtype=np.float32)

        def critic_scores(self, *args, **kwargs):
            return self._inner.critic_scores(*args, **kwargs)

    blind = actor_goal_sensitivity(_GoalBlindAgent(agent), sampler, batch_size=128)
    for horizon in (1, 2, 4, 8, 16):
        assert blind[f'actor_goal_sensitivity/action_delta_{horizon}'] == 0.0
        assert blind[f'actor_goal_sensitivity/critic_gain_{horizon}'] == 0.0


def test_hard_action_sensitivity_is_stricter_than_a_single_uniform_action():
    from utils.latent_diagnostics import action_sensitivity

    dataset = _make_dataset()
    config = _make_config('actnce_local')
    sampler = LatentBridgerDataset(dataset, config)
    agent = _make_agent(config, sampler.sample(8), stage='critic')

    metrics = action_sensitivity(
        agent,
        sampler,
        action_low=-np.ones(_ACTION_DIM, dtype=np.float32),
        action_high=np.ones(_ACTION_DIM, dtype=np.float32),
        batch_size=128,
        num_batches=2,
        num_hard_candidates=16,
    )
    # Beating the best of 16 is never easier than beating one draw.
    assert metrics['action_sensitivity/p_data_gt_hard'] <= (
        metrics['action_sensitivity/p_data_gt_uniform'] + 1e-6
    )
    assert metrics['action_sensitivity/margin_hard'] <= (
        metrics['action_sensitivity/margin_uniform'] + 1e-6
    )


# ----------------------------------------------------------------------
# Receding-horizon latent control
# ----------------------------------------------------------------------
class _CountingAgent:
    """Forward everything to the agent while counting flow replans."""

    def __init__(self, agent):
        self._agent = agent
        self.replans = 0

    def __getattr__(self, name):
        return getattr(self._agent, name)

    def sample_latent_prefix(self, *args, **kwargs):
        self.replans += 1
        return self._agent.sample_latent_prefix(*args, **kwargs)


class _StubEnv:
    """A minimal OGBench-shaped environment that never reports success."""

    class _Spec:
        max_episode_steps = 20

    class _ActionSpace:
        def __init__(self, action_dim):
            self.low = -np.ones(action_dim, dtype=np.float32)
            self.high = np.ones(action_dim, dtype=np.float32)

    def __init__(self, obs_dim, action_dim):
        self.spec = self._Spec()
        self.action_space = self._ActionSpace(action_dim)
        self._obs_dim = obs_dim
        self.steps = 0

    def _observation(self):
        return np.zeros(self._obs_dim, dtype=np.float32)

    def reset(self, options=None):
        self.steps = 0
        return self._observation(), {'goal': self._observation()}

    def step(self, action):
        assert np.all(action >= self.action_space.low - 1e-5)
        assert np.all(action <= self.action_space.high + 1e-5)
        self.steps += 1
        return self._observation(), 0.0, False, False, {'success': False}


@pytest.mark.parametrize(
    ('replan_interval', 'expected_replans'),
    [(5, 4), (1, 20)],
)
def test_latent_flow_replans_from_the_actual_state(replan_interval, expected_replans):
    from utils.latent_evaluation import evaluate_latent

    dataset = _make_dataset()
    config = _make_config('latent_rf')
    sampler = LatentBridgerDataset(dataset, config)
    batch = sampler.sample(8)
    agent = _CountingAgent(_make_agent(config, batch, stage='flow'))
    env = _StubEnv(_OBS_DIM, _ACTION_DIM)

    metrics = evaluate_latent(
        agent,
        env,
        mode='latent_flow',
        task_ids=(1,),
        episodes_per_task=1,
        seed=0,
        replan_interval=replan_interval,
    )
    # 20 environment steps consumed replan_interval latents at a time.
    assert env.steps == _StubEnv._Spec.max_episode_steps
    assert agent.replans == expected_replans
    assert metrics['replan_interval'] == replan_interval
    assert metrics['overall_success'] == 0.0


def test_direct_goal_mode_never_touches_the_flow():
    from utils.latent_evaluation import evaluate_latent

    dataset = _make_dataset()
    config = _make_config('latent_rf')
    sampler = LatentBridgerDataset(dataset, config)
    batch = sampler.sample(8)
    agent = _CountingAgent(_make_agent(config, batch, stage='flow'))
    env = _StubEnv(_OBS_DIM, _ACTION_DIM)

    evaluate_latent(agent, env, mode='direct_goal', task_ids=(1,), episodes_per_task=1)
    assert agent.replans == 0
    assert env.steps == _StubEnv._Spec.max_episode_steps


def test_replan_interval_is_validated_against_the_action_horizon():
    from utils.latent_evaluation import evaluate_latent

    dataset = _make_dataset()
    config = _make_config('latent_rf')
    sampler = LatentBridgerDataset(dataset, config)
    batch = sampler.sample(8)
    agent = _make_agent(config, batch, stage='flow')
    env = _StubEnv(_OBS_DIM, _ACTION_DIM)

    with pytest.raises(ValueError, match='replan_interval'):
        evaluate_latent(
            agent,
            env,
            mode='latent_flow',
            task_ids=(1,),
            episodes_per_task=1,
            replan_interval=6,
        )
    with pytest.raises(ValueError, match='replan_interval'):
        _make_agent(
            _make_config('latent_rf', replan_interval=0),
            batch,
            stage='flow',
        )


# ----------------------------------------------------------------------
# 13. The released PathBridger must keep working untouched
# ----------------------------------------------------------------------
def test_original_pathbridger_still_imports_and_updates():
    from agents import PathBridgerAgent
    from agents.pathbridger import get_config as get_pathbridger_config
    from utils.datasets import PathBridgerDataset

    dataset = _make_dataset(episode_lengths=(60, 60, 60))
    config = get_pathbridger_config()
    config.env_name = 'antmaze-medium-navigate-v0'
    config.horizon = 25

    sampler = PathBridgerDataset(dataset, config)
    batch = sampler.sample(16)
    agent = PathBridgerAgent.create(
        0,
        batch['observations'],
        batch['actions'],
        config,
    )
    updated, info = agent.update(batch)
    assert np.isfinite(float(info['loss/total']))
    assert not _trees_equal(agent.network.params, updated.network.params)

    chunks = updated.sample_action_chunks(
        observations=jnp.asarray(batch['observations'][:1]),
        goals=jnp.asarray(batch['value_goals'][:1]),
        seed=jax.random.PRNGKey(0),
        num_candidates=int(config.eval_num_candidates),
        temperature=float(config.eval_temperature),
    )
    assert chunks.shape == (1, 5, _ACTION_DIM)
