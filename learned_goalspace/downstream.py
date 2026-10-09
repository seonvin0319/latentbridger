"""PathBridger transfer with one explicitly frozen learned goal encoder."""

from __future__ import annotations

from functools import partial
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from agents.contrastive_transitive_distance_pathbridger import (
    logits_from_distance,
    quasimetric_distance,
)
from agents.pathbridger import (
    BridgeResidual,
    FlowEndpointProposer,
    GaussianEndpointProposer,
    InverseDynamics,
    PathBridgerAgent,
    _ACTION_HORIZON,
    _HIDDEN_DIMS,
    _LAYER_NORM,
    _LEARNING_RATE,
    _replace_module_params,
)
from learned_goalspace.fixed_representations import FixedLinearEncoder
from learned_goalspace.pretrain import GoalEncoder, LATENT_DIM
from utils.flax_utils import ModuleDict, TrainState
from utils.goal_representation import (
    assert_phi_goal_obs_indices,
    infer_phi_goal_obs_indices,
)
from utils.networks import MLP

METHODS = (
    'LGS_TRL_W_FROZEN',
    'LGSDTRL_W_FROZEN',
    'PCA16_GS_TRL_W',
    'RANDOM16_GS_TRL_W',
    'MH_LGS_TRL_W_FROZEN',
)
SCALAR_TRL_METHODS = (
    'LGS_TRL_W_FROZEN',
    'PCA16_GS_TRL_W',
    'RANDOM16_GS_TRL_W',
    'MH_LGS_TRL_W_FROZEN',
)
FIXED_ENCODER_METHODS = ('PCA16_GS_TRL_W', 'RANDOM16_GS_TRL_W')


class LatentScalarValue(nn.Module):
    hidden_dims: Sequence[int] = _HIDDEN_DIMS

    @nn.compact
    def __call__(self, left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
        return MLP(
            (*self.hidden_dims, 1),
            activate_final=False,
            layer_norm=_LAYER_NORM,
        )(jnp.concatenate([left, right], axis=-1)).squeeze(-1)


class LatentTemporalQuasimetricValue(nn.Module):
    discount: float
    repr_dim: int = 64

    def setup(self):
        self.trunk = MLP(tuple(_HIDDEN_DIMS), activate_final=True, layer_norm=_LAYER_NORM)
        self.h_head = nn.Dense(self.repr_dim, name='h')
        self.p_head = nn.Dense(self.repr_dim, name='p')

    def encode(self, latent: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        hidden = self.trunk(latent)
        return self.h_head(hidden), self.p_head(hidden)

    def distance(self, left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
        h_left, p_left = self.encode(left)
        h_right, p_right = self.encode(right)
        return quasimetric_distance(h_left, p_left, h_right, p_right)

    def __call__(self, left: jnp.ndarray, right: jnp.ndarray) -> jnp.ndarray:
        return logits_from_distance(self.distance(left, right), self.discount)


class LearnedGoalspaceModuleDict(ModuleDict):
    def goal_encode(self, observations):
        return self.modules['goal_encoder'](observations, normalize=False)

    def value_distance(self, left, right, *, name: str):
        return self.modules[name].distance(left, right)


def _optimizer_labels(params: Any) -> Any:
    mutable = flax.core.unfreeze(params)
    labels = {}
    for name, subtree in mutable.items():
        label = 'frozen' if name == 'modules_goal_encoder' else 'train'
        labels[name] = jax.tree_util.tree_map(lambda _: label, subtree)
    return flax.core.freeze(labels) if isinstance(params, flax.core.FrozenDict) else labels


class FrozenLearnedGoalspaceAgent(PathBridgerAgent):
    """Only high-level value/ranking sees E; policy-side modules stay original."""

    def _encode(self, observations: jnp.ndarray, *, params: Any | None = None):
        encoded = self.network(
            observations,
            params=params,
            method='goal_encode',
        )
        return jax.lax.stop_gradient(encoded)

    def _distance(
        self,
        left: jnp.ndarray,
        right: jnp.ndarray,
        *,
        name: str = 'value',
        params: Any | None = None,
    ):
        return self.network(
            left,
            right,
            name=name,
            params=params,
            method='value_distance',
        )

    def value_loss(self, batch, grad_params):
        # Every state role in self/base/transitive/target paths is encoded by E.
        encoded = dict(batch)
        for key in (
            'observations',
            'base_goals',
            'value_goals',
            'transitive_subgoals',
        ):
            encoded[key] = self._encode(batch[key], params=grad_params)
        return super().value_loss(encoded, grad_params)

    def _endpoint_weights(self, observations, goals, endpoint_targets):
        encoded_observations = self._encode(observations)
        encoded_goals = self._encode(goals)
        encoded_targets = self._encode(endpoint_targets)
        logits = self.network.select('target_value')(
            jnp.concatenate([encoded_observations, encoded_targets], axis=0),
            jnp.concatenate([encoded_goals, encoded_goals], axis=0),
        )
        current, endpoint = jnp.split(jax.nn.sigmoid(logits), 2, axis=0)
        gap = endpoint - current
        weights = jnp.minimum(
            5.0,
            jnp.exp(jnp.asarray(self.config['endpoint_value_scale'], jnp.float32) * gap),
        )
        return jax.lax.stop_gradient(weights), jax.lax.stop_gradient(gap)

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
            flat_g = jnp.broadcast_to(
                goals[:, None, :],
                (batch_size, num_candidates, goals.shape[-1]),
            ).reshape(batch_size * num_candidates, goals.shape[-1])
            latent_s = self._encode(flat_s)
            latent_z = self._encode(flat_z)
            latent_g = self._encode(flat_g)
            if self.config['method'] == 'LGSDTRL_W_FROZEN':
                scores = self._distance(latent_s, latent_z) + self._distance(latent_z, latent_g)
                scores = scores.reshape(batch_size, num_candidates)
                best = jnp.argmin(scores, axis=1)
            else:
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
        goal_encoder_params: Any,
        *,
        encoder_module: nn.Module | None = None,
    ) -> 'FrozenLearnedGoalspaceAgent':
        config = dict(config)
        method = str(config['method']).upper()
        if method not in METHODS:
            raise ValueError(f'Unknown frozen learned-goalspace method {method!r}.')
        config['method'] = method
        observations = jnp.asarray(ex_observations, dtype=jnp.float32)
        actions = jnp.asarray(ex_actions, dtype=jnp.float32)
        state_dim, action_dim = observations.shape[-1], actions.shape[-1]
        horizon = int(config['horizon'])
        if horizon < _ACTION_HORIZON:
            raise ValueError('horizon must be at least five.')
        env_name = str(config['env_name'])
        phi_indices = infer_phi_goal_obs_indices(env_name, state_dim)
        assert_phi_goal_obs_indices(
            state_dim,
            'phi',
            phi_indices,
            where='FrozenLearnedGoalspaceAgent endpoint proposer',
            env_name=env_name,
        )
        endpoint_distribution = str(config['endpoint_distribution']).lower()
        config['endpoint_distribution'] = endpoint_distribution
        if endpoint_distribution == 'gaussian':
            endpoint = GaussianEndpointProposer(state_dim, env_name, phi_indices)
            endpoint_args = (observations, observations)
        elif endpoint_distribution == 'flow':
            endpoint = FlowEndpointProposer(state_dim, env_name, phi_indices)
            endpoint_args = (
                observations,
                observations,
                jnp.zeros_like(observations),
                jnp.zeros((len(observations), 1), jnp.float32),
            )
        else:
            raise ValueError('endpoint_distribution must be flow or gaussian.')
        if method == 'LGSDTRL_W_FROZEN':
            value = LatentTemporalQuasimetricValue(float(config['discount']))
            target_value = LatentTemporalQuasimetricValue(float(config['discount']))
        elif method in SCALAR_TRL_METHODS:
            value, target_value = LatentScalarValue(), LatentScalarValue()
        else:
            raise ValueError(f'Unsupported value head for method {method!r}.')
        if encoder_module is None:
            encoder_module = FixedLinearEncoder() if method in FIXED_ENCODER_METHODS else GoalEncoder()
        latent = jnp.zeros((len(observations), LATENT_DIM), dtype=jnp.float32)
        times = jnp.broadcast_to(
            jnp.linspace(0, 1, horizon + 1)[None],
            (len(observations), horizon + 1),
        )
        definitions = {
            'goal_encoder': (encoder_module, (observations,)),
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
        params = _replace_module_params(params, 'goal_encoder', goal_encoder_params)
        params = _replace_module_params(params, 'target_value', params['modules_value'])
        tx = optax.multi_transform(
            {'train': optax.adam(_LEARNING_RATE), 'frozen': optax.set_to_zero()},
            _optimizer_labels(params),
        )
        network = TrainState.create(model, params, tx=tx)
        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(config),
        )


LGS_TRL_W_FROZEN = FrozenLearnedGoalspaceAgent
LGSDTRL_W_FROZEN = FrozenLearnedGoalspaceAgent
PCA16_GS_TRL_W = FrozenLearnedGoalspaceAgent
RANDOM16_GS_TRL_W = FrozenLearnedGoalspaceAgent
MH_LGS_TRL_W_FROZEN = FrozenLearnedGoalspaceAgent

__all__ = [
    'FIXED_ENCODER_METHODS',
    'FrozenLearnedGoalspaceAgent',
    'LGSDTRL_W_FROZEN',
    'LGS_TRL_W_FROZEN',
    'MH_LGS_TRL_W_FROZEN',
    'METHODS',
    'PCA16_GS_TRL_W',
    'RANDOM16_GS_TRL_W',
    'SCALAR_TRL_METHODS',
    'LatentScalarValue',
    'LatentTemporalQuasimetricValue',
]
