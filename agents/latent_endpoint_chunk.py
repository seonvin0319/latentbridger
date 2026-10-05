"""Latent endpoint contrastive critic with support-aware chunk extraction.

The critic is deliberately factorized:

``F(s, A) ~= phi_s(s[t + H])``
``V_C(s, g) = phi_s(s)^T psi(g)``
``S(s, A, g) = F(s, A)^T psi(g)``

The primary actor is weighted chunk behavior cloning (AWR).  TD3+BC is kept
only as an explicit ablation.  Planning never performs a pure critic argmax;
it combines a candidate-normalized critic score with behavior log likelihood.
"""

from __future__ import annotations

import pickle
import random
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, resolve_checkpoint
from utils.networks import MLP

VARIANTS = ('latent_endpoint_awr', 'latent_endpoint_td3bc')
VARIANT_SETTINGS: dict[str, dict[str, Any]] = {
    'latent_endpoint_awr': {'policy_type': 'awr'},
    'latent_endpoint_td3bc': {'policy_type': 'td3bc'},
}
STAGES = ('proposal', 'critic', 'policy_awr', 'policy_td3bc')
_MODULES = ('endpoint', 'state', 'goal', 'awr_policy', 'td3bc_policy', 'proposal')
_STAGE_TRAINABLE = {
    'proposal': ('proposal',),
    'critic': ('endpoint', 'state', 'goal'),
    'policy_awr': ('awr_policy',),
    'policy_td3bc': ('td3bc_policy',),
}
_LOG_2PI = float(np.log(2.0 * np.pi))
_LOC_LIMIT = 10.0


def _init():
    return nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform')


class ReluMLP(nn.Module):
    widths: Sequence[int]

    @nn.compact
    def __call__(self, value):
        for index, width in enumerate(self.widths):
            value = nn.Dense(width, kernel_init=_init())(value)
            if index + 1 < len(self.widths):
                value = nn.relu(value)
        return value


class EndpointModel(nn.Module):
    hidden_dims: tuple[int, ...]
    repr_dim: int

    @nn.compact
    def __call__(self, observations, action_chunks):
        value = jnp.concatenate([observations, action_chunks], axis=-1)
        return ReluMLP((*self.hidden_dims, self.repr_dim))(value)


class StateEncoder(nn.Module):
    hidden_dims: tuple[int, ...]
    repr_dim: int

    @nn.compact
    def __call__(self, states):
        return ReluMLP((*self.hidden_dims, self.repr_dim))(states)


class GoalEncoder(nn.Module):
    hidden_dims: tuple[int, ...]
    repr_dim: int

    @nn.compact
    def __call__(self, goals):
        return ReluMLP((*self.hidden_dims, self.repr_dim))(goals)


class GaussianChunkPolicy(nn.Module):
    chunk_dim: int
    hidden_dims: tuple[int, ...]
    layer_norm: bool

    @nn.compact
    def __call__(self, observations, goals):
        value = MLP(
            self.hidden_dims,
            activate_final=True,
            layer_norm=self.layer_norm,
        )(jnp.concatenate([observations, goals], axis=-1))
        mean = nn.Dense(self.chunk_dim, kernel_init=_init())(value)
        log_std = nn.Dense(self.chunk_dim, kernel_init=_init())(value)
        return mean, jnp.clip(log_std, -5.0, 2.0)


class DeterministicChunkPolicy(nn.Module):
    chunk_dim: int
    hidden_dims: tuple[int, ...]
    layer_norm: bool

    @nn.compact
    def __call__(self, observations, goals):
        return MLP(
            (*self.hidden_dims, self.chunk_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(jnp.concatenate([observations, goals], axis=-1))


class BehaviorChunkProposal(nn.Module):
    chunk_dim: int
    hidden_dims: tuple[int, ...]
    min_scale: float
    layer_norm: bool

    @nn.compact
    def __call__(self, observations, goals):
        value = MLP(
            self.hidden_dims,
            activate_final=True,
            layer_norm=self.layer_norm,
        )(jnp.concatenate([observations, goals], axis=-1))
        loc = nn.Dense(self.chunk_dim, kernel_init=_init())(value)
        loc = _LOC_LIMIT * jnp.tanh(loc / _LOC_LIMIT)
        raw_scale = nn.Dense(self.chunk_dim, kernel_init=_init())(value)
        return loc, jax.nn.softplus(raw_scale) + self.min_scale


def _freeze_config(config: dict[str, Any]):
    return flax.core.FrozenDict(
        {
            key: tuple(value) if isinstance(value, (list, tuple)) else value
            for key, value in config.items()
        }
    )


def _bounds(value: Any, size: int, name: str) -> tuple[float, ...]:
    values = np.broadcast_to(np.asarray(value, dtype=np.float32), (size,))
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{name} must be finite.')
    return tuple(float(item) for item in values)


class LatentEndpointChunkAgent(flax.struct.PyTreeNode):
    """Unified proposal, factorized critic, and chunk-policy checkpoint."""

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    def _maybe_normalize(self, value):
        if bool(self.config['repr_norm']):
            return value / jnp.maximum(
                jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-8
            )
        return value

    def endpoint_latents(self, observations, chunks, params=None):
        return self._maybe_normalize(
            self.network.select('endpoint')(observations, chunks, params=params)
        )

    def state_latents(self, states, params=None):
        return self._maybe_normalize(
            self.network.select('state')(states, params=params)
        )

    def goal_latents(self, goals, params=None):
        return self._maybe_normalize(
            self.network.select('goal')(goals, params=params)
        )

    def composed_scores(self, observations, chunks, goals, params=None):
        endpoint = self.endpoint_latents(observations, chunks, params=params)
        goal = self.goal_latents(goals, params=params)
        return jnp.sum(endpoint * goal, axis=-1)

    def state_goal_values(self, observations, goals, params=None):
        state = self.state_latents(observations, params=params)
        goal = self.goal_latents(goals, params=params)
        return jnp.sum(state * goal, axis=-1)

    def _infonce(self, left, right, prefix):
        temperature = float(self.config['contrastive_temperature'])
        logits = left @ right.T / temperature
        labels = jnp.arange(logits.shape[0])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        penalty = jnp.mean(jnp.square(jax.scipy.special.logsumexp(logits, axis=-1)))
        loss = loss + float(self.config['logsumexp_coef']) * penalty
        positive = jnp.diag(logits)
        ranks = jnp.sum(logits >= positive[:, None], axis=-1)
        return loss, {
            f'{prefix}/loss': loss,
            f'{prefix}/recall_at_1': jnp.mean(ranks <= 1),
            f'{prefix}/recall_at_5': jnp.mean(ranks <= 5),
            f'{prefix}/positive_rank': jnp.mean(ranks),
        }

    def endpoint_loss(self, batch, params):
        predicted = self.endpoint_latents(
            batch['observations'], batch['action_chunks'], params=params
        )
        target = self.state_latents(batch['endpoint_states'], params=params)
        return self._infonce(predicted, target, 'endpoint')

    def state_goal_loss(self, batch, params):
        states = self.state_latents(batch['observations'], params=params)
        goals = self.goal_latents(batch['goals'], params=params)
        return self._infonce(states, goals, 'state_goal')

    def proposal_distribution(self, observations, goals, params=None):
        return self.network.select('proposal')(observations, goals, params=params)

    def _normalize_chunks(self, chunks):
        return (chunks - jnp.asarray(self.config['action_center'])) / jnp.asarray(
            self.config['action_half_range']
        )

    def _bounded_chunks(self, normalized):
        return jnp.asarray(self.config['action_center']) + jnp.asarray(
            self.config['action_half_range']
        ) * normalized

    def proposal_log_prob(self, observations, goals, chunks, params=None):
        loc, scale = self.proposal_distribution(observations, goals, params=params)
        normalized = jnp.clip(self._normalize_chunks(chunks), -0.999, 0.999)
        pre_tanh = jnp.arctanh(normalized)
        base = jax.scipy.stats.norm.logpdf(pre_tanh, loc, scale)
        correction = jnp.log(1.0 - jnp.square(normalized) + 1e-6)
        bound_scale = jnp.log(jnp.asarray(self.config['action_half_range']))
        return jnp.sum(base - correction - bound_scale, axis=-1)

    def proposal_loss(self, batch, params):
        log_prob = self.proposal_log_prob(
            batch['observations'], batch['goals'], batch['action_chunks'], params=params
        )
        loss = -jnp.mean(log_prob)
        return loss, {'proposal/loss': loss, 'proposal/nll': loss}

    def _sample_proposal(self, observations, goals, rng, count, params=None):
        loc, scale = self.proposal_distribution(observations, goals, params=params)
        noise = jax.random.normal(rng, (observations.shape[0], int(count), loc.shape[-1]))
        return self._bounded_chunks(
            jnp.tanh(loc[:, None, :] + scale[:, None, :] * noise)
        )

    @partial(jax.jit, static_argnames=('num_candidates',))
    def sample_proposal_chunks(self, observations, goals, rng, *, num_candidates):
        return self._sample_proposal(observations, goals, rng, int(num_candidates))

    def action_nce_loss(self, batch, params, rng):
        count = int(self.config['num_action_negatives'])
        negatives = jax.lax.stop_gradient(
            self._sample_proposal(
                batch['observations'], batch['goals'], rng, count
            )
        )
        batch_size, _, chunk_dim = negatives.shape
        positive = self.composed_scores(
            batch['observations'], batch['action_chunks'], batch['goals'], params=params
        )
        negative = self.composed_scores(
            jnp.repeat(batch['observations'], count, axis=0),
            negatives.reshape(batch_size * count, chunk_dim),
            jnp.repeat(batch['goals'], count, axis=0),
            params=params,
        ).reshape(batch_size, count)
        logits = jnp.concatenate([positive[:, None], negative], axis=-1)
        logits = logits / float(self.config['contrastive_temperature'])
        labels = jnp.zeros(batch_size, dtype=jnp.int32)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, {
            'action/loss': loss,
            'action/p_positive_gt_q': jnp.mean(positive[:, None] > negative),
            'action/positive_score': jnp.mean(positive),
            'action/q_negative_score': jnp.mean(negative),
        }

    def critic_loss(self, batch, params, rng):
        endpoint, endpoint_info = self.endpoint_loss(batch, params)
        state_goal, sg_info = self.state_goal_loss(batch, params)
        action, action_info = self.action_nce_loss(batch, params, rng)
        loss = endpoint + state_goal + float(self.config['lambda_action']) * action
        return loss, {
            **endpoint_info,
            **sg_info,
            **action_info,
            'critic/loss': loss,
        }

    def awr_distribution(self, observations, goals, params=None):
        return self.network.select('awr_policy')(observations, goals, params=params)

    def awr_log_prob(self, observations, goals, chunks, params=None):
        mean, log_std = self.awr_distribution(observations, goals, params=params)
        normalized = jnp.clip(self._normalize_chunks(chunks), -0.999, 0.999)
        pre_tanh = jnp.arctanh(normalized)
        base = -0.5 * (
            jnp.square((pre_tanh - mean) * jnp.exp(-log_std))
            + 2.0 * log_std
            + _LOG_2PI
        )
        correction = jnp.log(1.0 - jnp.square(normalized) + 1e-6)
        bound_scale = jnp.log(jnp.asarray(self.config['action_half_range']))
        return jnp.sum(base - correction - bound_scale, axis=-1)

    def awr_loss(self, batch, params):
        # Critic evaluation uses the frozen checkpoint parameters, not ``params``.
        # Therefore the AWR objective has no critic-gradient path by construction.
        score = jax.lax.stop_gradient(
            self.composed_scores(
                batch['observations'], batch['action_chunks'], batch['goals']
            )
        )
        value = jax.lax.stop_gradient(
            self.state_goal_values(batch['observations'], batch['goals'])
        )
        advantage = score - value
        weights = jnp.clip(
            jnp.exp(advantage / float(self.config['awr_beta'])),
            0.0,
            float(self.config['awr_weight_max']),
        )
        log_prob = self.awr_log_prob(
            batch['observations'], batch['goals'], batch['action_chunks'], params=params
        )
        loss = -jnp.mean(weights * log_prob)
        ess = jnp.square(jnp.sum(weights)) / (
            jnp.sum(jnp.square(weights)) + 1e-8
        )
        return loss, {
            'awr/loss': loss,
            'awr/chunk_bc_nll': -jnp.mean(log_prob),
            'awr/advantage_mean': jnp.mean(advantage),
            'awr/weight_mean': jnp.mean(weights),
            'awr/weight_std': jnp.std(weights),
            'awr/weight_max': jnp.max(weights),
            'awr/weight_ess': ess,
        }

    def td3bc_chunks(self, observations, goals, params=None):
        raw = self.network.select('td3bc_policy')(observations, goals, params=params)
        return self._bounded_chunks(jnp.tanh(raw))

    def td3bc_loss(self, batch, params):
        predicted = self.td3bc_chunks(batch['observations'], batch['goals'], params=params)
        data_score = jax.lax.stop_gradient(
            self.composed_scores(
                batch['observations'], batch['action_chunks'], batch['goals']
            )
        )
        actor_score = self.composed_scores(
            batch['observations'], predicted, batch['goals'], params=params
        )
        scale = jax.lax.stop_gradient(jnp.mean(jnp.abs(data_score)))
        coefficient = float(self.config['td3bc_alpha']) / (
            scale + float(self.config['score_eps'])
        )
        mse = jnp.mean(jnp.square(predicted - batch['action_chunks']))
        loss = -coefficient * jnp.mean(actor_score) + float(
            self.config['td3bc_bc_coef']
        ) * mse
        return loss, {
            'td3bc/loss': loss,
            'td3bc/chunk_bc_mse': mse,
            'td3bc/actor_score': jnp.mean(actor_score),
            'td3bc/data_score': jnp.mean(data_score),
            'td3bc/score_coefficient': coefficient,
        }

    def total_loss(self, batch, params, rng):
        stage = str(self.config['stage'])
        if stage == 'proposal':
            return self.proposal_loss(batch, params)
        if stage == 'critic':
            return self.critic_loss(batch, params, rng)
        if stage == 'policy_awr':
            return self.awr_loss(batch, params)
        if stage == 'policy_td3bc':
            return self.td3bc_loss(batch, params)
        raise ValueError(f'Unknown stage {stage!r}.')

    @jax.jit
    def update(self, batch):
        rng, loss_rng = jax.random.split(self.rng)
        network, info = self.network.apply_loss_fn(
            lambda params: self.total_loss(batch, params, loss_rng)
        )
        return self.replace(rng=rng, network=network), info

    @jax.jit
    def sample_action_chunks(self, observations, goals):
        if str(self.config['policy_type']) == 'awr':
            mean, _ = self.awr_distribution(observations, goals)
            return self._bounded_chunks(jnp.tanh(mean))
        return self.td3bc_chunks(observations, goals)

    @partial(jax.jit, static_argnames=('num_proposals',))
    def plan_action_chunks(self, observations, goals, rng, *, num_proposals):
        """Rank direct plus q_beta candidates by normalized S + support."""

        if int(num_proposals) < 1:
            raise ValueError('num_proposals must be positive.')
        direct = self.sample_action_chunks(observations, goals)
        proposals = self._sample_proposal(
            observations, goals, rng, int(num_proposals)
        )
        candidates = jnp.concatenate([direct[:, None, :], proposals], axis=1)
        batch_size, count, chunk_dim = candidates.shape
        flat_observations = jnp.repeat(observations, count, axis=0)
        flat_goals = jnp.repeat(goals, count, axis=0)
        flat_chunks = candidates.reshape(batch_size * count, chunk_dim)
        scores = self.composed_scores(
            flat_observations, flat_chunks, flat_goals
        ).reshape(batch_size, count)
        support = self.proposal_log_prob(
            flat_observations, flat_goals, flat_chunks
        ).reshape(batch_size, count)
        normalized = (scores - jnp.mean(scores, axis=1, keepdims=True)) / (
            jnp.std(scores, axis=1, keepdims=True)
            + float(self.config['score_normalization_eps'])
        )
        objective = normalized + float(self.config['lambda_support']) * support
        allowed = support >= float(self.config['support_logprob_threshold'])
        any_allowed = jnp.any(allowed, axis=1, keepdims=True)
        fallback = support == jnp.max(support, axis=1, keepdims=True)
        allowed = jnp.where(any_allowed, allowed, fallback)
        masked = jnp.where(allowed, objective, -jnp.inf)
        indices = jnp.argmax(masked, axis=1)
        selected = candidates[jnp.arange(batch_size), indices]
        selected_scores = scores[jnp.arange(batch_size), indices]
        selected_support = support[jnp.arange(batch_size), indices]
        return selected, candidates, {
            'scores': scores,
            'normalized_scores': normalized,
            'support_logprob': support,
            'objective': objective,
            'allowed': allowed,
        }, {
            'selected_critic_score': selected_scores,
            'selected_support_logprob': selected_support,
            'selected_is_direct': (indices == 0).astype(jnp.float32),
            'direct_critic_score': scores[:, 0],
            'direct_support_logprob': support[:, 0],
            'filtered_fraction': 1.0 - jnp.mean(allowed, axis=1),
            'score_improvement_over_direct': selected_scores - scores[:, 0],
        }

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_action_chunks,
        ex_endpoint_states,
        ex_goals,
        config,
        *,
        stage,
        action_low=None,
        action_high=None,
    ):
        config = dict(config.to_dict() if hasattr(config, 'to_dict') else config)
        variant = str(config.get('variant', 'latent_endpoint_awr'))
        if variant not in VARIANTS:
            raise ValueError(f'variant must be one of {VARIANTS}.')
        if stage not in STAGES:
            raise ValueError(f'stage must be one of {STAGES}.')
        config.update(VARIANT_SETTINGS[variant])
        expected = f"policy_{config['policy_type']}"
        if stage.startswith('policy_') and stage != expected:
            raise ValueError(f'{variant} requires {expected}, not {stage}.')
        config['stage'] = stage

        observations = jnp.asarray(ex_observations, dtype=jnp.float32)
        chunks = jnp.asarray(ex_action_chunks, dtype=jnp.float32)
        endpoints = jnp.asarray(ex_endpoint_states, dtype=jnp.float32)
        goals = jnp.asarray(ex_goals, dtype=jnp.float32)
        horizon = int(config['chunk_horizon'])
        if chunks.shape[-1] % horizon:
            raise ValueError('Chunk dimension must equal H * action_dim.')
        action_dim = int(chunks.shape[-1] // horizon)
        low_one = _bounds(-1.0 if action_low is None else action_low, action_dim, 'action_low')
        high_one = _bounds(1.0 if action_high is None else action_high, action_dim, 'action_high')
        if any(high <= low for low, high in zip(low_one, high_one)):
            raise ValueError('Every action upper bound must exceed its lower bound.')
        low = low_one * horizon
        high = high_one * horizon
        config.update(
            action_dim=action_dim,
            chunk_dim=horizon * action_dim,
            action_low=low,
            action_high=high,
            action_center=tuple((left + right) * 0.5 for left, right in zip(low, high)),
            action_half_range=tuple((right - left) * 0.5 for left, right in zip(low, high)),
        )
        critic_hidden = tuple(int(item) for item in config['contrastive_hidden_dims'])
        policy_hidden = tuple(int(item) for item in config['policy_hidden_dims'])
        repr_dim = int(config['repr_dim'])
        layer_norm = bool(config['policy_layer_norm'])
        modules = {
            'endpoint': EndpointModel(critic_hidden, repr_dim),
            'state': StateEncoder(critic_hidden, repr_dim),
            'goal': GoalEncoder(critic_hidden, repr_dim),
            'awr_policy': GaussianChunkPolicy(chunks.shape[-1], policy_hidden, layer_norm),
            'td3bc_policy': DeterministicChunkPolicy(chunks.shape[-1], policy_hidden, layer_norm),
            'proposal': BehaviorChunkProposal(
                chunks.shape[-1],
                policy_hidden,
                float(config['proposal_min_scale']),
                layer_norm,
            ),
        }
        args = {
            'endpoint': (observations, chunks),
            'state': (endpoints,),
            'goal': (goals,),
            'awr_policy': (observations, goals),
            'td3bc_policy': (observations, goals),
            'proposal': (observations, goals),
        }
        model = ModuleDict(modules)
        rng, init_rng = jax.random.split(jax.random.PRNGKey(int(seed)))
        params = model.init(init_rng, **args)['params']
        optimizer = _stage_optimizer(params, stage, float(config['learning_rate']))
        return cls(
            rng=rng,
            network=TrainState.create(model, params, optimizer),
            config=_freeze_config(config),
        )


def _stage_optimizer(params, stage, learning_rate):
    trainable = _STAGE_TRAINABLE[str(stage)]

    def labels(tree):
        tree = flax.core.unfreeze(tree) if isinstance(tree, flax.core.FrozenDict) else tree
        result = {}
        for key, subtree in tree.items():
            if key not in {f'modules_{name}' for name in _MODULES}:
                raise ValueError(f'Unexpected module subtree {key!r}.')
            label = 'train' if key.removeprefix('modules_') in trainable else 'freeze'
            result[key] = jax.tree_util.tree_map(lambda _: label, subtree)
        return result

    return optax.multi_transform(
        {'train': optax.adam(learning_rate), 'freeze': optax.set_to_zero()},
        labels(params),
    )


def trainable_modules(stage):
    return _STAGE_TRAINABLE[str(stage)]


def restore_latent_endpoint_params(
    agent, restore_path, step=0, *, restore_host_rng=True
):
    path, _ = resolve_checkpoint(restore_path, step)
    with Path(path).open('rb') as file:
        payload = pickle.load(file)
    params = flax.serialization.from_state_dict(
        agent.network.params, payload['agent']['network']['params']
    )
    if restore_host_rng:
        if 'numpy_random_state' in payload:
            np.random.set_state(payload['numpy_random_state'])
        if 'python_random_state' in payload:
            random.setstate(payload['python_random_state'])
    return agent.replace(network=agent.network.replace(params=params))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            env_name='cube-single-play-v0',
            variant='latent_endpoint_awr',
            chunk_horizon=5,
            execute_h=2,
            discount=0.99,
            goal_representation='phi',
            repr_dim=64,
            repr_norm=False,
            contrastive_hidden_dims=(256, 256),
            policy_hidden_dims=(512, 512, 512),
            policy_layer_norm=True,
            learning_rate=3e-4,
            logsumexp_coef=0.01,
            contrastive_temperature=1.0,
            # NEW ABLATION PARAMETERS (not inherited from PathBridger/SGCRL).
            lambda_action=1.0,
            num_action_negatives=16,
            proposal_min_scale=1e-3,
            awr_beta=100.0,
            awr_weight_max=100.0,
            td3bc_alpha=2.5,
            td3bc_bc_coef=1.0,
            score_eps=1e-6,
            lambda_support=0.05,
            support_logprob_threshold=-100.0,
            score_normalization_eps=1e-6,
        )
    )


__all__ = [
    'LatentEndpointChunkAgent',
    'STAGES',
    'VARIANTS',
    'VARIANT_SETTINGS',
    'get_config',
    'restore_latent_endpoint_params',
    'trainable_modules',
]
