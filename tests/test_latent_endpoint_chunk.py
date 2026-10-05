"""Correctness tests for the factorized latent endpoint chunk planner."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import jax
import numpy as np

from agents.latent_endpoint_chunk import (
    LatentEndpointChunkAgent,
    get_config,
    restore_latent_endpoint_params,
)
from utils.chunk_relabeling import LatentEndpointChunkDataset
from utils.datasets import Dataset
from utils.flax_utils import save_agent
from utils.latent_endpoint_chunk_evaluation import (
    chunk_prefix,
    episode_manifest,
    evaluate_latent_endpoint_chunk,
)

OBS_DIM = 6
ACTION_DIM = 2
H = 5


def make_dataset(episode_length=14, episodes=3):
    observations, actions, terminals = [], [], []
    for episode in range(episodes):
        for step in range(episode_length + 1):
            observations.append([episode, step, episode + step, 2 * step, -step, 1.0])
            actions.append([episode * 0.1 + step * 0.01, -step * 0.02])
            terminals.append(step == episode_length)
    size = len(observations)
    return Dataset.create(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        terminals=np.asarray(terminals, dtype=np.float32),
        rewards=np.full(size, 12_345.0, dtype=np.float32),
        returns=np.full(size, -54_321.0, dtype=np.float32),
    )


def config(variant='latent_endpoint_awr', **overrides):
    value = get_config()
    value.variant = variant
    value.policy_type = 'awr' if variant.endswith('awr') else 'td3bc'
    value.env_name = 'antmaze-medium-navigate-v0'
    value.chunk_horizon = H
    value.execute_h = 2
    value.contrastive_hidden_dims = (16, 16)
    value.policy_hidden_dims = (16, 16)
    value.repr_dim = 8
    value.num_action_negatives = 3
    for key, item in overrides.items():
        value[key] = item
    return value


def make_sampler(**overrides):
    return LatentEndpointChunkDataset(make_dataset(), config(**overrides))


def make_agent(stage='critic', variant='latent_endpoint_awr', **overrides):
    cfg = config(variant, **overrides)
    sampler = LatentEndpointChunkDataset(make_dataset(), cfg)
    batch = sampler.sample(8)
    agent = LatentEndpointChunkAgent.create(
        0,
        batch['observations'],
        batch['action_chunks'],
        batch['endpoint_states'],
        batch['goals'],
        cfg,
        stage=stage,
        action_low=np.full(ACTION_DIM, -0.5),
        action_high=np.full(ACTION_DIM, 0.75),
    )
    return agent, sampler


def leaves_differ(left, right):
    return any(
        np.any(np.asarray(a) != np.asarray(b))
        for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right))
    )


def test_terminal_safe_sampling_delta_and_exact_endpoint():
    sampler = make_sampler()
    starts = np.asarray([0, 1, 2, 15, 16, 17, 30, 31], dtype=np.int64)
    batch = sampler.sample(len(starts), idxs=starts)
    finals = sampler.final_for_idx[starts]
    assert np.all(batch['endpoint_indices'] == starts + H)
    assert np.all(batch['endpoint_indices'] <= finals)
    assert np.all(batch['goal_indices'] <= finals)
    assert np.all(batch['future_offsets'] >= H)
    np.testing.assert_allclose(
        batch['endpoint_states'], np.asarray(sampler.dataset['observations'])[starts + H]
    )
    assert np.all(batch['observations'][:, 0] == batch['endpoint_states'][:, 0])


def test_chunks_are_exactly_h_consecutive_actions():
    sampler = make_sampler()
    starts = np.asarray([0, 2, 15, 17], dtype=np.int64)
    batch = sampler.sample(len(starts), idxs=starts)
    actions = np.asarray(sampler.dataset['actions'])
    expected = np.stack([actions[start : start + H].reshape(-1) for start in starts])
    np.testing.assert_allclose(batch['action_chunks'], expected)


def test_training_losses_ignore_reward_and_return_fields():
    agent, sampler = make_agent(stage='critic')
    batch = sampler.sample(8)
    rng = jax.random.PRNGKey(5)
    first, _ = agent.critic_loss(batch, agent.network.params, rng)
    poisoned = dict(batch, rewards=np.full(8, np.nan), returns=np.full(8, np.inf))
    second, _ = agent.critic_loss(poisoned, agent.network.params, rng)
    np.testing.assert_allclose(first, second)


def test_proposal_likelihood_updates_only_proposal():
    agent, sampler = make_agent(stage='proposal')
    batch = sampler.sample(8)
    loss, info = agent.proposal_loss(batch, agent.network.params)
    assert np.isfinite(float(loss))
    assert np.isfinite(float(info['proposal/nll']))
    before = copy.deepcopy(agent.network.params)
    updated, _ = agent.update(batch)
    assert leaves_differ(before['modules_proposal'], updated.network.params['modules_proposal'])
    for key in ('modules_endpoint', 'modules_state', 'modules_goal', 'modules_awr_policy'):
        jax.tree_util.tree_map(
            lambda a, b: np.testing.assert_allclose(a, b),
            before[key],
            updated.network.params[key],
        )


def test_critic_parameters_are_frozen_during_policy_stage():
    agent, sampler = make_agent(stage='policy_awr')
    batch = sampler.sample(8)
    before = copy.deepcopy(agent.network.params)
    updated, _ = agent.update(batch)
    for key in ('modules_endpoint', 'modules_state', 'modules_goal', 'modules_proposal'):
        jax.tree_util.tree_map(
            lambda a, b: np.testing.assert_allclose(a, b),
            before[key],
            updated.network.params[key],
        )
    assert leaves_differ(
        before['modules_awr_policy'], updated.network.params['modules_awr_policy']
    )


def test_awr_loss_has_no_critic_or_proposal_gradient():
    agent, sampler = make_agent(stage='policy_awr')
    batch = sampler.sample(8)
    gradients = jax.grad(lambda params: agent.awr_loss(batch, params)[0])(
        agent.network.params
    )
    for key in ('modules_endpoint', 'modules_state', 'modules_goal', 'modules_proposal'):
        for leaf in jax.tree_util.tree_leaves(gradients[key]):
            np.testing.assert_array_equal(np.asarray(leaf), 0.0)
    assert any(
        np.linalg.norm(np.asarray(leaf)) > 0
        for leaf in jax.tree_util.tree_leaves(gradients['modules_awr_policy'])
    )


def test_action_nce_negatives_come_from_q_beta():
    agent, sampler = make_agent(stage='critic')
    batch = sampler.sample(8)
    loss, info = agent.action_nce_loss(batch, agent.network.params, jax.random.PRNGKey(7))
    assert np.isfinite(float(loss))
    assert 0.0 <= float(info['action/p_positive_gt_q']) <= 1.0


def test_support_threshold_filters_candidates_and_never_leaves_empty_set():
    agent, sampler = make_agent(
        stage='policy_awr', support_logprob_threshold=1e9
    )
    batch = sampler.sample(4)
    selected, candidates, details, metrics = agent.plan_action_chunks(
        batch['observations'], batch['goals'], jax.random.PRNGKey(9), num_proposals=4
    )
    assert selected.shape == (4, H * ACTION_DIM)
    assert candidates.shape == (4, 5, H * ACTION_DIM)
    np.testing.assert_array_equal(np.sum(np.asarray(details['allowed']), axis=1), 1)
    assert np.all(np.asarray(metrics['filtered_fraction']) > 0.0)


def test_direct_and_planning_restore_from_the_same_checkpoint(tmp_path):
    agent, sampler = make_agent(stage='policy_awr')
    checkpoint = save_agent(agent, tmp_path, 11)
    fresh, _ = make_agent(stage='policy_awr')
    restored = restore_latent_endpoint_params(fresh, checkpoint)
    batch = sampler.sample(4)
    direct = restored.sample_action_chunks(batch['observations'], batch['goals'])
    planned, candidates, _, _ = restored.plan_action_chunks(
        batch['observations'], batch['goals'], jax.random.PRNGKey(1), num_proposals=2
    )
    np.testing.assert_allclose(np.asarray(candidates)[:, 0], np.asarray(direct))
    assert planned.shape == direct.shape


def test_chunk_prefix_executes_only_requested_actions():
    flat = np.arange(H * ACTION_DIM, dtype=np.float32)
    prefix = chunk_prefix(flat, horizon=H, action_dim=ACTION_DIM, execute_h=2)
    np.testing.assert_array_equal(prefix, flat.reshape(H, ACTION_DIM)[:2])


class _Space:
    low = np.full(ACTION_DIM, -1.0)
    high = np.full(ACTION_DIM, 1.0)

    def seed(self, _seed):
        return None


class _Env:
    action_space = _Space()
    spec = SimpleNamespace(max_episode_steps=5)

    def reset(self, seed, options):
        del seed, options
        self.steps = 0
        return np.zeros(OBS_DIM, dtype=np.float32), {
            'goal': np.ones(OBS_DIM, dtype=np.float32)
        }

    def step(self, action):
        assert np.asarray(action).shape == (ACTION_DIM,)
        self.steps += 1
        done = self.steps == 5
        return (
            np.full(OBS_DIM, self.steps, dtype=np.float32),
            99.0,
            done,
            False,
            {'success': done},
        )


class _CountingAgent:
    def __init__(self):
        self.calls = 0
        self.config = {
            'chunk_horizon': H,
            'execute_h': 2,
            'action_dim': ACTION_DIM,
            'goal_representation': 'phi',
            'env_name': 'antmaze-medium-navigate-v0',
        }

    def sample_action_chunks(self, observations, goals):
        self.calls += 1
        return np.zeros((len(observations), H * ACTION_DIM), dtype=np.float32)

    def composed_scores(self, observations, chunks, goals):
        return np.zeros(len(observations), dtype=np.float32)

    def proposal_log_prob(self, observations, goals, chunks):
        return np.zeros(len(observations), dtype=np.float32)


def test_execute_h_replans_after_each_prefix():
    agent = _CountingAgent()
    result = evaluate_latent_endpoint_chunk(
        agent,
        _Env(),
        manifest=[{'task_id': 1, 'env_seed': 3, 'action_space_seed': 4}],
        execute_h=2,
    )
    assert agent.calls == 3
    assert result['mean_replans'] == 3
    assert result['num_successes'] == 1


def test_default_manifest_is_new_250_episode_paired_protocol():
    manifest = episode_manifest(seed=17)
    assert len(manifest) == 250
    assert len({(row['env_seed'], row['action_space_seed']) for row in manifest}) == 250
