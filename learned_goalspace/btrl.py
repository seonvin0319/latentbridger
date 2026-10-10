"""Non-oracle Bottleneck TRL (BTRL16) — TRL-trained E, learned-goal proposer."""

from __future__ import annotations

from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import optax

from agents.learned_goal_proposer import (
    LEARNED_GOAL_DIM,
    LearnedGoalFlowEndpointProposer,
    LearnedGoalGaussianEndpointProposer,
)
from agents.pathbridger import (
    BridgeResidual,
    InverseDynamics,
    PathBridgerAgent,
    _ACTION_HORIZON,
    _ENDPOINT_WEIGHT_CAP,
    _FLOW_STEPS,
    _LEARNING_RATE,
    _replace_module_params,
)
from learned_goalspace.downstream import (
    LatentScalarValue,
    LearnedGoalspaceModuleDict,
)
from learned_goalspace.pretrain import GoalEncoder, LATENT_DIM
from utils.flax_utils import TrainState

assert LATENT_DIM == LEARNED_GOAL_DIM

METHOD = 'BTRL16'


class BottleneckTRLAgent(PathBridgerAgent):
    """NonOracle-GS-TPB with jointly trained 16-D goal encoder.

    - ``V(E(s), E(g))`` receives gradients into ``E`` (TRL only).
    - Proposer consumes ``stop_gradient(E(g))`` and full-state ``s``.
    - Bridge / IDM stay full-state and do not update ``E``.
    - No authoritative ``phi`` in training or action selection.
    """

    def _encode(self, observations: jnp.ndarray, *, params: Any | None = None) -> jnp.ndarray:
        return self.network(observations, params=params, method='goal_encode')

    def _encode_stop(self, observations: jnp.ndarray, *, params: Any | None = None) -> jnp.ndarray:
        return jax.lax.stop_gradient(self._encode(observations, params=params))

    def value_loss(self, batch, grad_params):
        encoded = dict(batch)
        for key in (
            'observations',
            'base_goals',
            'value_goals',
            'transitive_subgoals',
        ):
            # TRL path: gradients flow into E.
            encoded[key] = self._encode(batch[key], params=grad_params)
        return super().value_loss(encoded, grad_params)

    def _endpoint_weights(self, observations, goals, endpoint_targets):
        # Weighting uses E; stop-grad so proposer weighting does not reshape E.
        encoded_observations = self._encode_stop(observations)
        encoded_goals = self._encode_stop(goals)
        encoded_targets = self._encode_stop(endpoint_targets)
        logits = self.network.select('target_value')(
            jnp.concatenate([encoded_observations, encoded_targets], axis=0),
            jnp.concatenate([encoded_goals, encoded_goals], axis=0),
        )
        current, endpoint = jnp.split(jax.nn.sigmoid(logits), 2, axis=0)
        gap = endpoint - current
        weights = jnp.minimum(
            _ENDPOINT_WEIGHT_CAP,
            jnp.exp(jnp.asarray(self.config['endpoint_value_scale'], jnp.float32) * gap),
        )
        return jax.lax.stop_gradient(weights), jax.lax.stop_gradient(gap)

    def endpoint_loss(self, batch, grad_params, rng):
        """Same PB endpoint objectives, but proposer goals are stopgrad(E(g))."""

        observations = batch['observations']
        goals = batch['endpoint_goals']
        endpoint_targets = batch['endpoint_targets']
        displacement_targets = endpoint_targets - observations
        weights, value_gap = self._endpoint_weights(observations, goals, endpoint_targets)
        goal_embeddings = self._encode_stop(goals, params=grad_params)

        if self.config['endpoint_distribution'] == 'gaussian':
            means, log_stds = self.network.select('endpoint')(
                observations,
                goal_embeddings,
                params=grad_params,
            )
            inverse_variances = jnp.exp(-2.0 * log_stds)
            nll = 0.5 * jnp.sum(
                jnp.square(displacement_targets - means) * inverse_variances + 2.0 * log_stds + jnp.log(2.0 * jnp.pi),
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
            noise = jax.random.normal(noise_rng, displacement_targets.shape, dtype=displacement_targets.dtype)
            times = jax.random.uniform(time_rng, (displacement_targets.shape[0], 1), dtype=displacement_targets.dtype)
            noisy = (1.0 - times) * noise + times * displacement_targets
            target_velocities = displacement_targets - noise
            predicted = self.network.select('endpoint')(
                observations,
                goal_embeddings,
                noisy,
                times,
                params=grad_params,
            )
            flow_errors = jnp.sum(jnp.square(predicted - target_velocities), axis=-1)
            loss = jnp.mean(weights * flow_errors)
            info = {
                'endpoint/loss': loss,
                'endpoint/flow_matching_loss': flow_errors.mean(),
                'endpoint/flow_time_mean': times.mean(),
            }
        info.update(
            {
                'endpoint/weight_mean': weights.mean(),
                'endpoint/weight_max': weights.max(),
                'endpoint/value_gap_mean': value_gap.mean(),
            }
        )
        return loss, info

    def _flow_endpoint_samples(self, observations, goal_embeddings, initial_noise):
        batch_size, num_candidates, state_dim = initial_noise.shape
        flat_obs = jnp.broadcast_to(observations[:, None, :], initial_noise.shape).reshape(
            batch_size * num_candidates, state_dim
        )
        flat_goals = jnp.broadcast_to(
            goal_embeddings[:, None, :],
            (batch_size, num_candidates, goal_embeddings.shape[-1]),
        ).reshape(batch_size * num_candidates, goal_embeddings.shape[-1])
        displacements = initial_noise.reshape(batch_size * num_candidates, state_dim)
        step_size = jnp.asarray(1.0 / _FLOW_STEPS, dtype=jnp.float32)
        for step in range(_FLOW_STEPS):
            times = jnp.full((displacements.shape[0], 1), step / _FLOW_STEPS, dtype=jnp.float32)
            velocities = self.network.select('endpoint')(flat_obs, flat_goals, displacements, times)
            displacements = displacements + step_size * velocities
        return displacements.reshape(batch_size, num_candidates, state_dim)

    def _sample_endpoint_candidates(
        self,
        observations,
        goals,
        seed,
        *,
        num_candidates,
        temperature,
    ):
        goal_embeddings = self._encode_stop(goals)
        batch_size, state_dim = observations.shape
        noise = jax.random.normal(seed, (batch_size, num_candidates, state_dim), dtype=observations.dtype)
        temperature_array = jnp.asarray(temperature, dtype=observations.dtype)
        if self.config['endpoint_distribution'] == 'gaussian':
            means, log_stds = self.network.select('endpoint')(observations, goal_embeddings)
            displacements = means[:, None, :] + temperature_array * jnp.exp(log_stds)[:, None, :] * noise
        else:
            displacements = self._flow_endpoint_samples(observations, goal_embeddings, temperature_array * noise)
        return observations[:, None, :] + displacements

    @partial(jax.jit, static_argnames=('num_candidates', 'temperature'))
    def sample_action_chunks(
        self,
        observations,
        goals,
        seed=None,
        num_candidates=None,
        temperature=None,
    ):
        if seed is None:
            seed = jax.random.PRNGKey(0)
        if num_candidates is None:
            num_candidates = int(self.config['eval_num_candidates'])
        if temperature is None:
            temperature = float(self.config['eval_temperature'])
        if num_candidates < 1 or temperature < 0:
            raise ValueError('Invalid candidate count or temperature.')
        squeeze = observations.ndim == 1
        if squeeze:
            observations, goals = observations[None], goals[None]

        candidates = self._sample_endpoint_candidates(
            observations,
            goals,
            seed,
            num_candidates=num_candidates,
            temperature=temperature,
        )
        batch_size, _, state_dim = candidates.shape
        if num_candidates == 1:
            selected = candidates[:, 0]
        else:
            flat_s = jnp.broadcast_to(observations[:, None, :], candidates.shape).reshape(
                batch_size * num_candidates, state_dim
            )
            flat_z = candidates.reshape(batch_size * num_candidates, state_dim)
            flat_g = jnp.broadcast_to(goals[:, None, :], (batch_size, num_candidates, goals.shape[-1])).reshape(
                batch_size * num_candidates, goals.shape[-1]
            )
            latent_s = self._encode_stop(flat_s)
            latent_z = self._encode_stop(flat_z)
            latent_g = self._encode_stop(flat_g)
            logits = self.network.select('value')(
                jnp.concatenate([latent_s, latent_z], axis=0),
                jnp.concatenate([latent_z, latent_g], axis=0),
            )
            left, right = jnp.split(jax.nn.sigmoid(logits), 2)
            scores = (left * right).reshape(batch_size, num_candidates)
            best = jnp.argmax(scores, axis=1)
            selected = jnp.take_along_axis(candidates, best[:, None, None], axis=1)[:, 0]

        prefix = self._construct_bridge_prefix(observations, selected)
        current = prefix[:, :-1].reshape(batch_size * _ACTION_HORIZON, state_dim)
        following = prefix[:, 1:].reshape(batch_size * _ACTION_HORIZON, state_dim)
        actions = self.network.select('idm')(current, following)
        actions = actions.reshape(batch_size, _ACTION_HORIZON, -1)
        return actions[0] if squeeze else actions

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        config: dict[str, Any],
    ) -> 'BottleneckTRLAgent':
        config = dict(config)
        config['method'] = METHOD
        config['high_level_oracle_phi'] = False
        config['proposer_oracle_phi'] = False
        observations = jnp.asarray(ex_observations, dtype=jnp.float32)
        actions = jnp.asarray(ex_actions, dtype=jnp.float32)
        state_dim, action_dim = int(observations.shape[-1]), int(actions.shape[-1])
        horizon = int(config['horizon'])
        if horizon < _ACTION_HORIZON:
            raise ValueError('horizon must be at least five.')
        endpoint_distribution = str(config['endpoint_distribution']).lower()
        config['endpoint_distribution'] = endpoint_distribution
        if endpoint_distribution == 'gaussian':
            endpoint = LearnedGoalGaussianEndpointProposer(state_dim=state_dim)
            endpoint_args = (
                observations,
                jnp.zeros((len(observations), LATENT_DIM), dtype=jnp.float32),
            )
        elif endpoint_distribution == 'flow':
            endpoint = LearnedGoalFlowEndpointProposer(state_dim=state_dim)
            endpoint_args = (
                observations,
                jnp.zeros((len(observations), LATENT_DIM), dtype=jnp.float32),
                jnp.zeros_like(observations),
                jnp.zeros((len(observations), 1), dtype=jnp.float32),
            )
        else:
            raise ValueError('endpoint_distribution must be flow or gaussian.')

        value = LatentScalarValue()
        target_value = LatentScalarValue()
        latent = jnp.zeros((len(observations), LATENT_DIM), dtype=jnp.float32)
        times = jnp.broadcast_to(
            jnp.linspace(0, 1, horizon + 1)[None],
            (len(observations), horizon + 1),
        )
        definitions = {
            'goal_encoder': (GoalEncoder(), (observations,)),
            'value': (value, (latent, latent)),
            'target_value': (target_value, (latent, latent)),
            'endpoint': (endpoint, endpoint_args),
            'bridge': (
                BridgeResidual(state_dim),
                (observations, jnp.zeros_like(observations), times),
            ),
            'idm': (InverseDynamics(action_dim), (observations, observations)),
        }
        model = LearnedGoalspaceModuleDict({name: definition for name, (definition, _) in definitions.items()})
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        params = model.init(
            init_rng,
            **{name: args for name, (_, args) in definitions.items()},
        )['params']
        params = _replace_module_params(params, 'target_value', params['modules_value'])
        tx = optax.adam(_LEARNING_RATE)
        network = TrainState.create(model, params, tx=tx)
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(config))


BTRL16 = BottleneckTRLAgent

__all__ = ['BTRL16', 'BottleneckTRLAgent', 'METHOD']
