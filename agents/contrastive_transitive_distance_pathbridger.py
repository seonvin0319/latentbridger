"""PathBridger with a tied temporal quasimetric in place of the scalar value.

The endpoint proposer, endpoint-pinned bridge, and IDM stay the PathBridger
modules.  The scalar value is ``V_d = gamma ** d_theta``, with the original
self, base, and transitive losses.  Optional InfoNCE reads the same distance.
There is no second critic, no reference bank, and no proposer freeze.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from agents.pathbridger import (
    BridgeResidual,
    FlowEndpointProposer,
    GaussianEndpointProposer,
    InverseDynamics,
    PathBridgerAgent,
    _ACTION_HORIZON,
    _ENDPOINT_WEIGHT_CAP,
    _HIDDEN_DIMS,
    _LAYER_NORM,
    _LEARNING_RATE,
    _VALUE_EPS,
    _replace_module_params,
)
from utils.flax_utils import ModuleDict, TrainState
from utils.goal_representation import (
    assert_phi_goal_obs_indices,
    infer_phi_goal_obs_indices,
)
from utils.networks import MLP

_REPR_DIM = 64
VARIANTS = (
    'dtrl_uniform',
    'dtrl_weighted',
    'ctd_uniform',
    'ctd_weighted',
    'ctd_pathnce_uniform',
    'ctd_pathnce_weighted',
    'ctd_pathnce_uniform_bridgegeo',
    'ctd_pathnce_weighted_bridgegeo',
)


@jax.custom_jvp
def safe_l2_norm(value: jnp.ndarray) -> jnp.ndarray:
    """Exact L2 norm with the zero vector's JVP defined as zero."""

    return jnp.linalg.norm(value, axis=-1)


@safe_l2_norm.defjvp
def _safe_l2_norm_jvp(primals, tangents):
    (value,), (tangent,) = primals, tangents
    norm = safe_l2_norm(value)
    denominator = jnp.where(norm > 0, norm, 1.0)
    derivative = jnp.sum(value * tangent, axis=-1) / denominator
    return norm, jnp.where(norm > 0, derivative, 0.0)


def quasimetric_distance(
    h_left: jnp.ndarray,
    p_left: jnp.ndarray,
    h_right: jnp.ndarray,
    p_right: jnp.ndarray,
) -> jnp.ndarray:
    """Directed distance ``||h_x-h_y||_2 + mean ReLU(p_y-p_x)``."""

    euclidean = safe_l2_norm(h_left - h_right)
    potential = jnp.mean(jax.nn.relu(p_right - p_left), axis=-1)
    return euclidean + potential


def value_from_distance(distance: jnp.ndarray, discount: float) -> jnp.ndarray:
    """``gamma ** d``, clipped away from 0 and 1 for a stable logit."""

    log_value = jnp.log(jnp.asarray(discount, dtype=distance.dtype)) * distance
    value = jnp.exp(log_value)
    return jnp.clip(value, _VALUE_EPS, 1.0 - _VALUE_EPS)


def logits_from_distance(distance: jnp.ndarray, discount: float) -> jnp.ndarray:
    value = value_from_distance(distance, discount)
    return jnp.log(value) - jnp.log1p(-value)


class TemporalQuasimetricValue(nn.Module):
    """One tied encoder, plus an NCE-only goal nuisance-bias head."""

    discount: float
    repr_dim: int = _REPR_DIM

    def setup(self):
        self.trunk = MLP(
            tuple(_HIDDEN_DIMS),
            activate_final=True,
            layer_norm=_LAYER_NORM,
        )
        self.h_head = nn.Dense(self.repr_dim, name='h')
        self.p_head = nn.Dense(self.repr_dim, name='p')
        self.nce_bias_head = nn.Dense(1, name='nce_bias')

    def encode(self, states: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        hidden = self.trunk(states)
        return self.h_head(hidden), self.p_head(hidden)

    def distance(self, left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
        h_left, p_left = self.encode(left)
        h_right, p_right = self.encode(right)
        return quasimetric_distance(h_left, p_left, h_right, p_right)

    def nuisance_bias(self, goals: jnp.ndarray) -> jnp.ndarray:
        h_goal, p_goal = self.encode(goals)
        features = jnp.concatenate([h_goal, p_goal], axis=-1)
        return self.nce_bias_head(features)[..., 0]

    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray) -> jnp.ndarray:
        h_left, p_left = self.encode(observations)
        h_right, p_right = self.encode(goals)
        distance = quasimetric_distance(h_left, p_left, h_right, p_right)
        # Instantiate the nuisance head in the shared module tree while keeping
        # TRL exactly independent of its output and gradients.
        nuisance = self.nce_bias_head(jnp.concatenate([h_right, p_right], axis=-1))[..., 0]
        return logits_from_distance(distance, self.discount) + 0.0 * nuisance


class CTDModuleDict(ModuleDict):
    """ModuleDict plus direct distance and encoder calls for the metric."""

    def metric_distance(self, left, right, *, name: str):
        return self.modules[name].distance(left, right)

    def metric_encode(self, states, *, name: str):
        return self.modules[name].encode(states)

    def metric_bias(self, goals, *, name: str):
        return self.modules[name].nuisance_bias(goals)


def pairwise_distance(
    h_left: jnp.ndarray,
    p_left: jnp.ndarray,
    h_right: jnp.ndarray,
    p_right: jnp.ndarray,
) -> jnp.ndarray:
    euclidean = safe_l2_norm(h_left[:, None, :] - h_right[None, :, :])
    potential = jnp.mean(
        jax.nn.relu(p_right[None, :, :] - p_left[:, None, :]),
        axis=-1,
    )
    return euclidean + potential


def path_candidate_positive_mask(
    start: jnp.ndarray,
    goal: jnp.ndarray,
    anchor_episode: jnp.ndarray,
    candidate_index: jnp.ndarray,
    candidate_episode: jnp.ndarray,
    candidate_valid: jnp.ndarray,
) -> jnp.ndarray:
    """True when a candidate lies on the anchor's ordered trajectory interval."""

    start = jnp.asarray(start)
    goal = jnp.asarray(goal)
    anchor_episode = jnp.asarray(anchor_episode)
    candidate_index = jnp.asarray(candidate_index)
    candidate_episode = jnp.asarray(candidate_episode)
    candidate_valid = jnp.asarray(candidate_valid)
    return (
        candidate_valid[None, :]
        & (candidate_episode[None, :] == anchor_episode[:, None])
        & (candidate_index[None, :] > start[:, None])
        & (candidate_index[None, :] < goal[:, None])
    )


class ContrastiveTransitiveDistanceAgent(PathBridgerAgent):
    """PathBridger whose value is a temporal quasimetric."""

    def _metric_distance(self, left, right, *, name: str, params=None):
        return self.network(
            left,
            right,
            name=name,
            params=params,
            method='metric_distance',
        )

    def _metric_encode(self, states, *, name: str, params=None):
        return self.network(
            states,
            name=name,
            params=params,
            method='metric_encode',
        )

    def _metric_bias(self, goals, *, name: str = 'value', params=None):
        return self.network(
            goals,
            name=name,
            params=params,
            method='metric_bias',
        )

    def nce_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """FutureNCE on ``b(g_j) - d(s_i, g_j) / tau_C``."""

        observations = batch['observations']
        goals = batch['value_goals']
        h_state, p_state = self._metric_encode(
            observations,
            name='value',
            params=grad_params,
        )
        h_goal, p_goal = self._metric_encode(
            goals,
            name='value',
            params=grad_params,
        )
        distance = pairwise_distance(h_state, p_state, h_goal, p_goal)
        temperature = jnp.asarray(
            self.config['contrastive_temperature'],
            dtype=distance.dtype,
        )
        geometry = -distance / temperature
        bias = self._metric_bias(goals, params=grad_params)
        logits = geometry + bias[None, :]
        positive = jnp.diag(logits)
        raw = (jax.nn.logsumexp(logits, axis=1) - positive).mean()
        batch_size = logits.shape[0]
        normalized = raw / jnp.log(jnp.asarray(batch_size, dtype=logits.dtype))
        rank = 1 + (logits > positive[:, None]).sum(axis=1)
        info = {
            'nce/raw': raw,
            'nce/normalized': normalized,
            'nce/recall_at_1': (rank == 1).mean(),
            'nce/recall_at_5': (rank <= 5).mean(),
            'nce/positive_rank': rank.mean(),
            'nce/temperature': temperature,
            'nce/bias_mean': bias.mean(),
            'nce/bias_std': bias.std(),
            'nce/geometry_mean': geometry.mean(),
            'nce/geometry_std': geometry.std(),
        }
        return normalized, info

    def path_nce_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Rank observed path intermediates by composed temporal path cost."""

        states = batch['observations']
        goals = batch['value_goals']
        positives = batch['path_positive_states']
        batch_size, slots, state_dim = positives.shape
        flat_states = positives.reshape(batch_size * slots, state_dim)
        h_state, p_state = self._metric_encode(states, name='value', params=grad_params)
        h_goal, p_goal = self._metric_encode(goals, name='value', params=grad_params)
        h_mid, p_mid = self._metric_encode(flat_states, name='value', params=grad_params)
        direct = quasimetric_distance(h_state, p_state, h_goal, p_goal)
        to_mid = pairwise_distance(h_state, p_state, h_mid, p_mid)
        mid_to_goal = pairwise_distance(h_mid, p_mid, h_goal, p_goal).T
        path_cost = to_mid + mid_to_goal
        residual = path_cost - direct[:, None]
        temperature = jnp.asarray(self.config['tau_path'], dtype=path_cost.dtype)
        # d(s,g) is deliberately absent. It is candidate-independent and
        # PathNCE is a set-level ranking objective, not residual regression.
        scores = -path_cost / temperature
        valid = batch['path_positive_mask'].reshape(-1) > 0
        positive = path_candidate_positive_mask(
            batch['path_start_indices'],
            batch['path_goal_indices'],
            batch['path_anchor_episode'],
            batch['path_positive_indices'].reshape(-1),
            batch['path_positive_episode'].reshape(-1),
            valid,
        )
        masked_scores = jnp.where(valid[None, :], scores, -jnp.inf)
        positive_scores = jnp.where(positive, scores, -jnp.inf)
        row_positive = positive.any(axis=1)
        raw_rows = jax.nn.logsumexp(masked_scores, axis=1) - jax.nn.logsumexp(positive_scores, axis=1)
        raw_rows = jnp.where(row_positive, raw_rows, 0.0)
        raw = raw_rows.sum() / jnp.maximum(row_positive.sum(), 1)
        candidate_count = jnp.maximum(valid.sum(), 1)
        normalized = raw / jnp.log(jnp.maximum(candidate_count, 2).astype(raw.dtype))
        negative = valid[None, :] & ~positive

        def _mean(values, mask):
            return jnp.where(mask, values, 0).sum() / jnp.maximum(mask.sum(), 1)

        def _std(values, mask):
            mean = _mean(values, mask)
            variance = jnp.where(mask, (values - mean) ** 2, 0).sum() / jnp.maximum(mask.sum(), 1)
            return jnp.sqrt(variance)

        positive_cost = _mean(path_cost, positive)
        negative_cost = _mean(path_cost, negative)
        positive_residual = _mean(residual, positive)
        negative_residual = _mean(residual, negative)
        owner = jnp.repeat(jnp.arange(batch_size), slots)
        false_negative_corrections = positive & (owner[None, :] != jnp.arange(batch_size)[:, None])
        info = {
            'path/raw': raw,
            'path/normalized': normalized,
            'path/positive_cost_mean': positive_cost,
            'path/positive_cost_std': _std(path_cost, positive),
            'path/negative_cost_mean': negative_cost,
            'path/negative_cost_std': _std(path_cost, negative),
            'path/cost_margin': negative_cost - positive_cost,
            'path/positive_residual_mean': positive_residual,
            'path/positive_residual_std': _std(residual, positive),
            'path/negative_residual_mean': negative_residual,
            'path/negative_residual_std': _std(residual, negative),
            'path/residual_margin': negative_residual - positive_residual,
            'path/positives_per_anchor': positive.sum(axis=1).mean(),
            'path/false_negative_corrections': false_negative_corrections.sum(axis=1).mean(),
            'path/valid_row_fraction': row_positive.mean(),
            'path/temperature': temperature,
        }
        return normalized, info

    def _frozen_metric_params(self, params: Any) -> Any:
        frozen_value = jax.tree_util.tree_map(jax.lax.stop_gradient, params['modules_value'])
        return flax.core.FrozenDict({**dict(params), 'modules_value': frozen_value})

    def bridge_geometry_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Penalize bridge states against a frozen temporal geodesic.

        Metric parameters are stop-gradient.  The predicted bridge state still
        receives a gradient, so the bridge module is the one that moves.
        """

        observations = batch['observations']
        endpoints = batch['endpoint_targets']
        predicted = self._construct_bridge_prefix(
            observations,
            endpoints,
            params=grad_params,
        )
        states = predicted[:, 1:_ACTION_HORIZON + 1, :]
        batch_size, steps, state_dim = states.shape
        flat_states = states.reshape(batch_size * steps, state_dim)
        frozen = self._frozen_metric_params(grad_params)
        flat_start = jnp.repeat(observations, steps, axis=0)
        flat_end = jnp.repeat(endpoints, steps, axis=0)
        elapsed = self._metric_distance(flat_start, flat_states, name='value', params=frozen)
        remaining = self._metric_distance(flat_states, flat_end, name='value', params=frozen)
        direct = self._metric_distance(observations, endpoints, name='value', params=frozen)
        elapsed = elapsed.reshape(batch_size, steps)
        remaining = remaining.reshape(batch_size, steps)
        horizon = batch['endpoint_offsets'][:, None]
        index = jnp.arange(1, steps + 1, dtype=horizon.dtype)[None, :]
        elapsed_target = jnp.minimum(index, horizon)
        remaining_target = jnp.maximum(horizon - index, 0.0)
        elapsed_error = jnp.abs(elapsed - elapsed_target)
        remaining_error = jnp.abs(remaining - remaining_target)
        path_residual = elapsed + remaining - direct[:, None]
        loss = elapsed_error.mean() + remaining_error.mean() + path_residual.mean()
        info = {
            'bridge/geometry_loss': loss,
            'bridge/elapsed_error': elapsed_error.mean(),
            'bridge/remaining_error': remaining_error.mean(),
            'bridge/path_residual': path_residual.mean(),
        }
        for step in range(steps):
            info[f'bridge/elapsed_error_step{step + 1}'] = elapsed_error[:, step].mean()
            info[f'bridge/remaining_error_step{step + 1}'] = remaining_error[:, step].mean()
            info[f'bridge/path_residual_step{step + 1}'] = path_residual[:, step].mean()
        return loss, info

    def total_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        rng: jax.Array,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        loss, info = super().total_loss(batch, grad_params, rng)
        lambda_nce = jnp.asarray(self.config['lambda_nce'], dtype=jnp.float32)
        lambda_path = jnp.asarray(self.config['lambda_pathnce'], dtype=jnp.float32)
        lambda_geo = jnp.asarray(self.config['lambda_bridge_geo'], dtype=jnp.float32)
        if float(self.config['lambda_nce']) > 0.0:
            normalized, nce_info = self.nce_loss(batch, grad_params)
            loss = loss + lambda_nce * normalized
            info = {**info, **nce_info}
        else:
            nce_info = {
                'nce/raw': jnp.zeros((), dtype=jnp.float32),
                'nce/normalized': jnp.zeros((), dtype=jnp.float32),
            }
            info = {**info, **nce_info}
        if float(self.config['lambda_pathnce']) > 0.0:
            path_loss, path_info = self.path_nce_loss(batch, grad_params)
            loss = loss + lambda_path * path_loss
            info = {**info, **path_info}
        if float(self.config['lambda_bridge_geo']) > 0.0:
            geo_loss, geo_info = self.bridge_geometry_loss(batch, grad_params)
            loss = loss + lambda_geo * geo_loss
            info = {**info, **geo_info}
        info = {
            **info,
            'nce/lambda': lambda_nce,
            'path/lambda': lambda_path,
            'bridge/geometry_lambda': lambda_geo,
            'loss/total': loss,
        }
        return loss, info

    def _endpoint_weights(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        endpoint_targets: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.config['proposer_weighting'] == 'uniform':
            weights = jnp.ones((observations.shape[0],), dtype=observations.dtype)
            gap = jnp.zeros_like(weights)
            return jax.lax.stop_gradient(weights), jax.lax.stop_gradient(gap)
        if self.config['proposer_weighting'] != 'transitive':
            raise ValueError(
                "proposer_weighting must be 'uniform' or 'transitive', got "
                f"{self.config['proposer_weighting']!r}."
            )
        return super()._endpoint_weights(observations, goals, endpoint_targets)

    @partial(
        jax.jit,
        static_argnames=('num_candidates', 'temperature'),
    )
    def sample_action_chunks(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        seed: jax.Array | None = None,
        num_candidates: int | None = None,
        temperature: float | None = None,
    ) -> jnp.ndarray:
        """Sample endpoints, rank by ``d(s,z)+d(z,g)``, and decode with the IDM."""

        if seed is None:
            seed = jax.random.PRNGKey(0)
        if num_candidates is None:
            num_candidates = int(self.config['eval_num_candidates'])
        if temperature is None:
            temperature = float(self.config['eval_temperature'])
        if num_candidates < 1:
            raise ValueError('num_candidates must be at least one.')
        if temperature < 0.0:
            raise ValueError('temperature must be non-negative.')

        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None, :]
            goals = goals[None, :]

        candidates = self._sample_endpoint_candidates(
            observations,
            goals,
            seed,
            num_candidates=num_candidates,
            temperature=temperature,
        )
        batch_size, _, state_dim = candidates.shape
        if num_candidates == 1:
            selected_endpoints = candidates[:, 0, :]
        else:
            flat_observations = jnp.broadcast_to(
                observations[:, None, :],
                candidates.shape,
            ).reshape(batch_size * num_candidates, state_dim)
            flat_candidates = candidates.reshape(batch_size * num_candidates, state_dim)
            flat_goals = jnp.broadcast_to(
                goals[:, None, :],
                (batch_size, num_candidates, goals.shape[-1]),
            ).reshape(batch_size * num_candidates, goals.shape[-1])
            distance_to_endpoint = self._metric_distance(
                flat_observations,
                flat_candidates,
                name='value',
            )
            distance_to_goal = self._metric_distance(
                flat_candidates,
                flat_goals,
                name='value',
            )
            scores = (distance_to_endpoint + distance_to_goal).reshape(
                batch_size,
                num_candidates,
            )
            best_indices = jnp.argmin(scores, axis=1)
            selected_endpoints = jnp.take_along_axis(
                candidates,
                best_indices[:, None, None],
                axis=1,
            )[:, 0, :]

        prefix = self._construct_bridge_prefix(observations, selected_endpoints)
        current_states = prefix[:, :-1, :].reshape(
            batch_size * _ACTION_HORIZON,
            state_dim,
        )
        next_states = prefix[:, 1:, :].reshape(
            batch_size * _ACTION_HORIZON,
            state_dim,
        )
        actions = self.network.select('idm')(current_states, next_states)
        actions = actions.reshape(batch_size, _ACTION_HORIZON, -1)
        return actions[0] if squeeze else actions

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        config: dict[str, Any],
    ) -> 'ContrastiveTransitiveDistanceAgent':
        config = dict(config)
        variant = str(config.get('variant', 'ctd_weighted'))
        if variant not in VARIANTS:
            raise ValueError(f'Unknown CTD variant {variant!r}.')
        weighting = str(config.get('proposer_weighting', ''))
        if weighting not in ('uniform', 'transitive'):
            raise ValueError(
                "proposer_weighting must be 'uniform' or 'transitive', got "
                f'{weighting!r}.'
            )
        lambda_nce = float(config.get('lambda_nce', -1.0))
        if lambda_nce < 0.0:
            raise ValueError('lambda_nce must be non-negative.')
        if float(config.get('lambda_pathnce', 0.0)) < 0.0:
            raise ValueError('lambda_pathnce must be non-negative.')
        if float(config.get('lambda_bridge_geo', 0.0)) < 0.0:
            raise ValueError('lambda_bridge_geo must be non-negative.')
        if float(config.get('tau_path', 5.0)) <= 0.0:
            raise ValueError('tau_path must be positive.')
        horizon = int(config['horizon'])
        if float(config.get('contrastive_temperature', horizon)) <= 0.0:
            raise ValueError('contrastive_temperature must be positive.')
        endpoint_distribution = str(config['endpoint_distribution']).lower()
        if endpoint_distribution not in ('flow', 'gaussian'):
            raise ValueError(
                "endpoint_distribution must be 'flow' or 'gaussian', got "
                f'{endpoint_distribution!r}.'
            )
        config['endpoint_distribution'] = endpoint_distribution
        config['variant'] = variant
        if horizon < _ACTION_HORIZON:
            raise ValueError(f'horizon must be at least {_ACTION_HORIZON}, got {horizon}.')
        discount = float(config['discount'])
        if not 0.0 < discount < 1.0:
            raise ValueError(f'discount must be in (0, 1), got {discount}.')
        if int(config['eval_num_candidates']) < 1:
            raise ValueError('eval_num_candidates must be at least one.')
        if float(config['eval_temperature']) < 0.0:
            raise ValueError('eval_temperature must be non-negative.')
        if float(config['endpoint_value_scale']) <= 0.0:
            raise ValueError('endpoint_value_scale must be positive.')

        observations = jnp.asarray(ex_observations, dtype=jnp.float32)
        actions = jnp.asarray(ex_actions, dtype=jnp.float32)
        state_dim = int(observations.shape[-1])
        action_dim = int(actions.shape[-1])
        env_name = str(config['env_name'])
        phi_goal_obs_indices = infer_phi_goal_obs_indices(env_name, state_dim)
        assert_phi_goal_obs_indices(
            state_dim,
            'phi',
            phi_goal_obs_indices,
            where='ContrastiveTransitiveDistanceAgent.create',
            env_name=env_name,
        )
        value_def = TemporalQuasimetricValue(discount=discount)
        target_value_def = TemporalQuasimetricValue(discount=discount)
        if endpoint_distribution == 'gaussian':
            endpoint_def = GaussianEndpointProposer(
                state_dim=state_dim,
                env_name=env_name,
                phi_goal_obs_indices=phi_goal_obs_indices,
            )
            endpoint_args = (observations, observations)
        else:
            endpoint_def = FlowEndpointProposer(
                state_dim=state_dim,
                env_name=env_name,
                phi_goal_obs_indices=phi_goal_obs_indices,
            )
            endpoint_args = (
                observations,
                observations,
                jnp.zeros_like(observations),
                jnp.zeros((observations.shape[0], 1), dtype=jnp.float32),
            )
        bridge_times = jnp.broadcast_to(
            jnp.linspace(0.0, 1.0, horizon + 1, dtype=jnp.float32)[None, :],
            (observations.shape[0], horizon + 1),
        )
        network_info = {
            'value': (value_def, (observations, observations)),
            'target_value': (target_value_def, (observations, observations)),
            'endpoint': (endpoint_def, endpoint_args),
            'bridge': (
                BridgeResidual(state_dim=state_dim),
                (observations, jnp.zeros_like(observations), bridge_times),
            ),
            'idm': (
                InverseDynamics(action_dim=action_dim),
                (observations, observations),
            ),
        }
        network_def = CTDModuleDict(
            {name: definition for name, (definition, _) in network_info.items()}
        )
        network_args = {name: arguments for name, (_, arguments) in network_info.items()}
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        network_params = network_def.init(init_rng, **network_args)['params']
        network_params = _replace_module_params(
            network_params,
            'target_value',
            network_params['modules_value'],
        )
        network = TrainState.create(
            network_def,
            network_params,
            tx=optax.adam(_LEARNING_RATE),
        )
        config['endpoint_weight_cap'] = _ENDPOINT_WEIGHT_CAP
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(config))


__all__ = [
    'CTDModuleDict',
    'ContrastiveTransitiveDistanceAgent',
    'TemporalQuasimetricValue',
    'VARIANTS',
    'logits_from_distance',
    'path_candidate_positive_mask',
    'pairwise_distance',
    'quasimetric_distance',
    'safe_l2_norm',
    'value_from_distance',
]
