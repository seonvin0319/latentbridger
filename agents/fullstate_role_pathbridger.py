"""PathBridger with frozen, role-specific 16-D value/goal representations.

The endpoint, bridge, and inverse-dynamics execution path remains in the
normalized full-observation space.  Frozen PCA/Eexec maps are used only where
a state occupies the source or goal role in value and endpoint conditioning.
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
    ScalarTransitiveValue,
    _ACTION_HORIZON,
    _BASE_HORIZON,
    _BRIDGE_ALPHA_POWER,
    _ENDPOINT_WEIGHT_CAP,
    _FLOW_STEPS,
    _LEARNING_RATE,
    _TARGET_TAU,
    _VALUE_EPS,
    _expectile_bce,
    _replace_module_params,
)
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP


_REPRESENTATIONS = ('pca', 'eexec')
_ENCODER_HIDDEN = (512, 512, 512)
_LATENT_DIM = 16


class FrozenEexecEncoder(nn.Module):
    """Architecture-compatible copy of the action-pretrain goal encoder."""

    @nn.compact
    def __call__(self, states: jnp.ndarray) -> jnp.ndarray:
        return MLP(
            (*_ENCODER_HIDDEN, _LATENT_DIM),
            activate_final=False,
            layer_norm=True,
        )(states)


def _encode_frozen(
    kind: str,
    states: jnp.ndarray,
    eexec_encoder_params: Any,
    pca_matrix: jnp.ndarray,
) -> jnp.ndarray:
    states = jnp.asarray(states, dtype=jnp.float32)
    if kind == 'pca':
        encoded = states @ pca_matrix
    elif kind == 'eexec':
        encoded = FrozenEexecEncoder().apply(
            {'params': eexec_encoder_params},
            states,
        )
    else:
        raise ValueError(f'unknown frozen representation {kind!r}')
    return jax.lax.stop_gradient(encoded)


class FullStateRolePathBridgerAgent(PathBridgerAgent):
    """Role-separated value/goal features with a full-state execution path."""

    eexec_encoder_params: Any
    pca_matrix: Any
    source_kind: str = nonpytree_field()
    goal_kind: str = nonpytree_field()

    def encode_source(self, states: jnp.ndarray) -> jnp.ndarray:
        return _encode_frozen(
            self.source_kind,
            states,
            self.eexec_encoder_params,
            self.pca_matrix,
        )

    def encode_goal(self, states: jnp.ndarray) -> jnp.ndarray:
        return _encode_frozen(
            self.goal_kind,
            states,
            self.eexec_encoder_params,
            self.pca_matrix,
        )

    def _value_call(
        self,
        module: str,
        source_states: jnp.ndarray,
        goal_states: jnp.ndarray,
        *,
        params: Any | None = None,
    ) -> jnp.ndarray:
        return self.network.select(module)(
            self.encode_source(source_states),
            self.encode_goal(goal_states),
            params=params,
        )

    def value_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        observations = batch['observations']
        discount = jnp.asarray(self.config['discount'], dtype=jnp.float32)

        online_logits = self._value_call(
            'value',
            jnp.concatenate([observations, observations, observations], axis=0),
            jnp.concatenate(
                [observations, batch['base_goals'], batch['value_goals']],
                axis=0,
            ),
            params=grad_params,
        )
        self_logits, base_logits, transitive_logits = jnp.split(
            online_logits, 3, axis=0,
        )
        self_loss = optax.sigmoid_binary_cross_entropy(
            self_logits, jnp.ones_like(self_logits),
        ).mean()

        base_offsets = jnp.asarray(batch['base_offsets'], dtype=jnp.float32)
        base_targets = jnp.power(discount, base_offsets)
        base_bce = optax.sigmoid_binary_cross_entropy(base_logits, base_targets)

        goals = batch['value_goals']
        subgoals = batch['transitive_subgoals']
        left_offsets = jnp.asarray(batch['transitive_offsets'], dtype=jnp.float32)
        total_offsets = jnp.asarray(batch['value_offsets'], dtype=jnp.float32)
        right_offsets = total_offsets - left_offsets

        use_distance_weights = float(self.config['value_distance_weight_power']) != 0.0
        target_states = [observations, subgoals]
        target_goals = [subgoals, goals]
        if use_distance_weights:
            target_states.extend([observations, observations])
            target_goals.extend([batch['base_goals'], goals])
        target_logits = self._value_call(
            'target_value',
            jnp.concatenate(target_states, axis=0),
            jnp.concatenate(target_goals, axis=0),
        )
        target_values = jnp.split(
            jax.nn.sigmoid(target_logits), len(target_states), axis=0,
        )
        target_left, target_right = target_values[:2]
        if use_distance_weights:
            base_distance_weights, base_target_values = (
                self._distance_weight_from_values(target_values[2])
            )
            transitive_distance_weights, transitive_target_values = (
                self._distance_weight_from_values(target_values[3])
            )
        else:
            base_distance_weights = jnp.ones_like(base_bce)
            transitive_distance_weights = jnp.ones_like(base_bce)
        base_loss = jnp.mean(base_distance_weights * base_bce)

        exact_left = jnp.power(discount, left_offsets)
        exact_right = jnp.power(discount, right_offsets)
        mixed_left = jnp.where(
            left_offsets <= _BASE_HORIZON, exact_left, target_left,
        )
        mixed_right = jnp.where(
            right_offsets <= _BASE_HORIZON, exact_right, target_right,
        )
        transitive_targets = jax.lax.stop_gradient(mixed_left * mixed_right)
        transitive_bce = _expectile_bce(transitive_logits, transitive_targets)
        transitive_valids = jnp.asarray(
            batch['transitive_valids'], dtype=jnp.float32,
        )
        if transitive_valids.ndim > 1:
            transitive_valids = transitive_valids.reshape(
                transitive_valids.shape[0], -1,
            )[:, 0]
        transitive_loss = jnp.sum(
            transitive_valids * transitive_distance_weights * transitive_bce,
        ) / jnp.maximum(jnp.sum(transitive_valids), 1.0)

        loss = self_loss + base_loss + transitive_loss
        info = {
            'value/loss': loss,
            'value/self_loss': self_loss,
            'value/base_loss': base_loss,
            'value/transitive_loss': transitive_loss,
            'value/self_mean': jax.nn.sigmoid(self_logits).mean(),
            'value/base_mean': jax.nn.sigmoid(base_logits).mean(),
            'value/base_target_mean': base_targets.mean(),
            'value/base_distance_weight_mean': base_distance_weights.mean(),
            'value/transitive_mean': jax.nn.sigmoid(transitive_logits).mean(),
            'value/transitive_target_mean': transitive_targets.mean(),
            'value/transitive_distance_weight_mean': transitive_distance_weights.mean(),
            'value/transitive_valid_fraction': transitive_valids.mean(),
        }
        if use_distance_weights:
            info['value/base_target_value_mean'] = base_target_values.mean()
            info['value/transitive_target_value_mean'] = transitive_target_values.mean()
        return loss, info

    def _endpoint_weights(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        endpoint_targets: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        value_logits = self._value_call(
            'target_value',
            jnp.concatenate([observations, endpoint_targets], axis=0),
            jnp.concatenate([goals, goals], axis=0),
        )
        current_values, endpoint_values = jnp.split(
            jax.nn.sigmoid(value_logits), 2, axis=0,
        )
        value_gap = endpoint_values - current_values
        weights = jnp.minimum(
            _ENDPOINT_WEIGHT_CAP,
            jnp.exp(
                jnp.asarray(
                    self.config['endpoint_value_scale'], dtype=jnp.float32,
                ) * value_gap,
            ),
        )
        return jax.lax.stop_gradient(weights), jax.lax.stop_gradient(value_gap)

    def endpoint_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        rng: jax.Array,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        observations = batch['observations']
        goals = batch['endpoint_goals']
        endpoint_targets = batch['endpoint_targets']
        displacement_targets = endpoint_targets - observations
        weights, value_gap = self._endpoint_weights(
            observations, goals, endpoint_targets,
        )
        source_inputs = self.encode_source(observations)
        goal_inputs = self.encode_goal(goals)

        if self.config['endpoint_distribution'] == 'gaussian':
            means, log_stds = self.network.select('endpoint')(
                source_inputs, goal_inputs, params=grad_params,
            )
            inverse_variances = jnp.exp(-2.0 * log_stds)
            nll = 0.5 * jnp.sum(
                jnp.square(displacement_targets - means) * inverse_variances
                + 2.0 * log_stds
                + jnp.log(2.0 * jnp.pi),
                axis=-1,
            )
            loss = jnp.mean(weights * nll)
            info = {
                'endpoint/loss': loss,
                'endpoint/nll': nll.mean(),
                'endpoint/std_mean': jnp.exp(log_stds).mean(),
            }
        else:
            noise_rng, time_rng = jax.random.split(rng)
            noise = jax.random.normal(
                noise_rng,
                displacement_targets.shape,
                dtype=displacement_targets.dtype,
            )
            times = jax.random.uniform(
                time_rng,
                (displacement_targets.shape[0], 1),
                dtype=displacement_targets.dtype,
            )
            noisy_displacements = (
                (1.0 - times) * noise + times * displacement_targets
            )
            target_velocities = displacement_targets - noise
            predicted_velocities = self.network.select('endpoint')(
                source_inputs,
                goal_inputs,
                noisy_displacements,
                times,
                params=grad_params,
            )
            flow_errors = jnp.sum(
                jnp.square(predicted_velocities - target_velocities), axis=-1,
            )
            loss = jnp.mean(weights * flow_errors)
            info = {
                'endpoint/loss': loss,
                'endpoint/flow_matching_loss': flow_errors.mean(),
                'endpoint/flow_time_mean': times.mean(),
            }
        info.update({
            'endpoint/weight_mean': weights.mean(),
            'endpoint/weight_max': weights.max(),
            'endpoint/value_gap_mean': value_gap.mean(),
        })
        return loss, info

    def _flow_endpoint_samples(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        initial_noise: jnp.ndarray,
    ) -> jnp.ndarray:
        batch_size, num_candidates, state_dim = initial_noise.shape
        flat_observations = jnp.broadcast_to(
            observations[:, None, :],
            (batch_size, num_candidates, state_dim),
        ).reshape(batch_size * num_candidates, state_dim)
        flat_goals = jnp.broadcast_to(
            goals[:, None, :],
            (batch_size, num_candidates, goals.shape[-1]),
        ).reshape(batch_size * num_candidates, goals.shape[-1])
        source_inputs = self.encode_source(flat_observations)
        goal_inputs = self.encode_goal(flat_goals)
        displacements = initial_noise.reshape(batch_size * num_candidates, state_dim)
        step_size = jnp.asarray(1.0 / _FLOW_STEPS, dtype=jnp.float32)
        for step in range(_FLOW_STEPS):
            times = jnp.full(
                (displacements.shape[0], 1),
                step / _FLOW_STEPS,
                dtype=jnp.float32,
            )
            velocities = self.network.select('endpoint')(
                source_inputs, goal_inputs, displacements, times,
            )
            displacements = displacements + step_size * velocities
        return displacements.reshape(batch_size, num_candidates, state_dim)

    def _sample_endpoint_candidates(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        seed: jax.Array,
        *,
        num_candidates: int,
        temperature: float,
    ) -> jnp.ndarray:
        batch_size, state_dim = observations.shape
        noise = jax.random.normal(
            seed,
            (batch_size, num_candidates, state_dim),
            dtype=observations.dtype,
        )
        temperature_array = jnp.asarray(temperature, dtype=observations.dtype)
        if self.config['endpoint_distribution'] == 'gaussian':
            means, log_stds = self.network.select('endpoint')(
                self.encode_source(observations), self.encode_goal(goals),
            )
            displacements = (
                means[:, None, :]
                + temperature_array * jnp.exp(log_stds)[:, None, :] * noise
            )
        else:
            displacements = self._flow_endpoint_samples(
                observations, goals, temperature_array * noise,
            )
        return observations[:, None, :] + displacements

    @partial(jax.jit, static_argnames=('num_candidates', 'temperature'))
    def sample_action_chunks(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        seed: jax.Array | None = None,
        num_candidates: int | None = None,
        temperature: float | None = None,
    ) -> jnp.ndarray:
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
                observations[:, None, :], candidates.shape,
            ).reshape(batch_size * num_candidates, state_dim)
            flat_candidates = candidates.reshape(batch_size * num_candidates, state_dim)
            flat_goals = jnp.broadcast_to(
                goals[:, None, :],
                (batch_size, num_candidates, goals.shape[-1]),
            ).reshape(batch_size * num_candidates, goals.shape[-1])
            value_logits = self._value_call(
                'value',
                jnp.concatenate([flat_observations, flat_candidates], axis=0),
                jnp.concatenate([flat_candidates, flat_goals], axis=0),
            )
            values_to_endpoint, values_to_goal = jnp.split(
                jax.nn.sigmoid(value_logits), 2, axis=0,
            )
            scores = (values_to_endpoint * values_to_goal).reshape(
                batch_size, num_candidates,
            )
            best_indices = jnp.argmax(scores, axis=1)
            selected_endpoints = jnp.take_along_axis(
                candidates, best_indices[:, None, None], axis=1,
            )[:, 0, :]

        prefix = self._construct_bridge_prefix(observations, selected_endpoints)
        current_states = prefix[:, :-1, :].reshape(
            batch_size * _ACTION_HORIZON, state_dim,
        )
        next_states = prefix[:, 1:, :].reshape(
            batch_size * _ACTION_HORIZON, state_dim,
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
        *,
        source_kind: str,
        goal_kind: str,
        eexec_encoder_params: Any,
        pca_matrix: jnp.ndarray,
    ) -> 'FullStateRolePathBridgerAgent':
        config = dict(config)
        source_kind = str(source_kind).lower()
        goal_kind = str(goal_kind).lower()
        if source_kind not in _REPRESENTATIONS or goal_kind not in _REPRESENTATIONS:
            raise ValueError(
                f'source/goal kinds must lie in {_REPRESENTATIONS}, got '
                f'{source_kind!r}/{goal_kind!r}',
            )
        endpoint_distribution = str(config['endpoint_distribution']).lower()
        if endpoint_distribution not in ('flow', 'gaussian'):
            raise ValueError(f'unknown endpoint distribution {endpoint_distribution!r}')
        config['endpoint_distribution'] = endpoint_distribution
        config['value_representation'] = 'role_separated_16d'
        config['endpoint_goal_representation'] = 'role_separated_16d'
        config['source_representation'] = source_kind
        config['goal_representation'] = goal_kind
        config['execution_representation'] = 'full_normalized_observation'

        observations = jnp.asarray(ex_observations, dtype=jnp.float32)
        actions = jnp.asarray(ex_actions, dtype=jnp.float32)
        pca_matrix = jnp.asarray(pca_matrix, dtype=jnp.float32)
        if observations.ndim != 2 or observations.shape[-1] != 55:
            raise ValueError(f'full observations must have shape [B, 55], got {observations.shape}')
        if actions.ndim != 2 or actions.shape[0] != observations.shape[0]:
            raise ValueError(f'invalid example action shape {actions.shape}')
        if pca_matrix.shape != (55, _LATENT_DIM):
            raise ValueError(f'PCA matrix must have shape (55, 16), got {pca_matrix.shape}')
        state_dim = int(observations.shape[-1])
        action_dim = int(actions.shape[-1])
        source_examples = _encode_frozen(
            source_kind, observations, eexec_encoder_params, pca_matrix,
        )
        goal_examples = _encode_frozen(
            goal_kind, observations, eexec_encoder_params, pca_matrix,
        )
        if source_examples.shape[-1] != _LATENT_DIM or goal_examples.shape[-1] != _LATENT_DIM:
            raise ValueError('role encoders must both produce 16-D features')

        value_def = ScalarTransitiveValue()
        target_value_def = ScalarTransitiveValue()
        if endpoint_distribution == 'gaussian':
            endpoint_def = GaussianEndpointProposer(
                state_dim=state_dim,
                env_name=str(config['env_name']),
                phi_goal_obs_indices=(),
                goal_mode='full',
            )
            endpoint_args = (source_examples, goal_examples)
        else:
            endpoint_def = FlowEndpointProposer(
                state_dim=state_dim,
                env_name=str(config['env_name']),
                phi_goal_obs_indices=(),
                goal_mode='full',
            )
            endpoint_args = (
                source_examples,
                goal_examples,
                jnp.zeros_like(observations),
                jnp.zeros((observations.shape[0], 1), dtype=jnp.float32),
            )
        bridge_def = BridgeResidual(state_dim=state_dim)
        idm_def = InverseDynamics(action_dim=action_dim)
        horizon = int(config['horizon'])
        bridge_times = jnp.broadcast_to(
            jnp.linspace(0.0, 1.0, horizon + 1, dtype=jnp.float32)[None, :],
            (observations.shape[0], horizon + 1),
        )
        network_info = {
            'value': (value_def, (source_examples, goal_examples)),
            'target_value': (target_value_def, (source_examples, goal_examples)),
            'endpoint': (endpoint_def, endpoint_args),
            'bridge': (
                bridge_def,
                (observations, jnp.zeros_like(observations), bridge_times),
            ),
            'idm': (idm_def, (observations, observations)),
        }
        network_def = ModuleDict({
            name: definition for name, (definition, _) in network_info.items()
        })
        network_args = {
            name: arguments for name, (_, arguments) in network_info.items()
        }
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
        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(config),
            eexec_encoder_params=eexec_encoder_params,
            pca_matrix=pca_matrix,
            source_kind=source_kind,
            goal_kind=goal_kind,
        )


__all__ = ['FrozenEexecEncoder', 'FullStateRolePathBridgerAgent']
