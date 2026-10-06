"""Contrastive reachability with PathBridger's unchanged explicit policy modules.

TRL distance/value scales are deliberately unused. No actor, rewards or returns.
"""
from functools import partial
from typing import Any
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from agents.pathbridger import (
    PathBridgerAgent, FlowEndpointProposer, BridgeResidual, InverseDynamics,
    _replace_module_params, _HIDDEN_DIMS, _LAYER_NORM, _LEARNING_RATE,
    _ACTION_HORIZON,
)
from utils.flax_utils import ModuleDict, TrainState
from utils.goal_representation import goal_representation, infer_phi_goal_obs_indices, assert_phi_goal_obs_indices
from utils.networks import MLP

class ReachabilityEncoder(nn.Module):
    env_name: str
    goal: bool = False
    repr_dim: int = 64
    repr_norm: bool = False

    @nn.compact
    def __call__(self, states):
        inputs = goal_representation(states, 'phi', env_name=self.env_name) if self.goal else states
        out = MLP((*_HIDDEN_DIMS, self.repr_dim), activate_final=False, layer_norm=True)(inputs)
        if self.repr_norm:
            out = out / jnp.maximum(jnp.linalg.norm(out, axis=-1, keepdims=True), 1e-8)
        return out


def infonce(u, v, temperature=1.0, repr_norm=False, logsumexp_coef=0.01):
    logits = u @ v.T / temperature
    positive = jnp.diag(logits)
    lse = jax.scipy.special.logsumexp(logits, axis=1)
    loss = (lse - positive).mean()
    if not repr_norm:
        loss = loss + logsumexp_coef * jnp.square(lse).mean()
    # Stable ordinal rank: ties are resolved by candidate index, not all rank 1.
    ids = jnp.arange(logits.shape[0])
    ahead = (logits > positive[:, None]) | ((logits == positive[:, None]) & (ids[None, :] < ids[:, None]))
    rank = 1 + ahead.sum(axis=1)
    negative = (logits.sum() - positive.sum()) / jnp.maximum(logits.size - len(u), 1)
    return loss, {
        'critic/loss': loss, 'critic/recall_at_1': (rank == 1).mean(),
        'critic/recall_at_5': (rank <= 5).mean(), 'critic/positive_rank': rank.mean(),
        'critic/positive_score': positive.mean(), 'critic/negative_score': negative,
        'critic/score_gap': positive.mean() - negative,
    }


def progress_weights(delta, step, progress_scale=1.0, enabled=True):
    delta = jax.lax.stop_gradient(delta)
    raw_norm = delta / (jax.lax.stop_gradient(delta.std()) + 1e-6)
    norm = jnp.clip(raw_norm, -2., 2.)
    uncapped = jnp.exp(progress_scale * norm)
    weight = jnp.minimum(5., uncapped)
    lam = jnp.clip((jnp.asarray(step, dtype=jnp.float32) - 100000.) / 100000., 0., 1.)
    lam = lam if enabled else jnp.zeros_like(lam)
    weight = jax.lax.stop_gradient((1. - lam) + lam * weight)
    return weight, {
        'progress/lambda_C': lam, 'progress/delta_mean': delta.mean(),
        'progress/delta_std': delta.std(), 'progress/delta_norm_mean': norm.mean(),
        'progress/delta_norm_std': norm.std(), 'progress/weight_mean': weight.mean(),
        'progress/weight_std': weight.std(), 'progress/weight_max': weight.max(),
        'progress/clipped_fraction': (uncapped > 5.).mean(),
        'progress/norm_clipped_fraction': (jnp.abs(raw_norm) > 2.).mean(),
    }


class ContrastivePathBridgerAgent(PathBridgerAgent):
    def score(self, states, goals, *, target=False, params=None):
        prefix = 'target_' if target else ''
        u = self.network.select(prefix + 'phi')(states, params=params)
        v = self.network.select(prefix + 'psi')(goals, params=params)
        return jnp.sum(u * v, axis=-1)

    def value_loss(self, batch, grad_params):
        u = self.network.select('phi')(batch['observations'], params=grad_params)
        v = self.network.select('psi')(batch['value_goals'], params=grad_params)
        loss, info = infonce(u, v, self.config['contrastive_temperature'],
                             self.config['repr_norm'], self.config['logsumexp_coef'])
        return self.config['lambda_CR'] * loss, info

    def _progress(self, observations, goals, endpoints):
        delta = self.score(endpoints, goals, target=True) - self.score(observations, goals, target=True)
        weights, info = progress_weights(delta, self.network.step, self.config['progress_scale'],
                                         self.config['variant'] == 'cpb_full')
        return weights, jax.lax.stop_gradient(delta), info

    def _endpoint_weights(self, observations, goals, endpoint_targets):
        weights, delta, _ = self._progress(observations, goals, endpoint_targets)
        return weights, delta

    def endpoint_loss(self, batch, grad_params, rng):
        loss, info = super().endpoint_loss(batch, grad_params, rng)
        _, _, progress = self._progress(batch['observations'], batch['endpoint_goals'], batch['endpoint_targets'])
        info.update(progress)
        info['endpoint/target_displacement_norm'] = jnp.linalg.norm(
            batch['endpoint_targets'] - batch['observations'], axis=-1).mean()
        return loss, info

    def _ema_target_value(self, network):
        for name in ('phi', 'psi'):
            target = jax.tree_util.tree_map(
                lambda online, old: .005 * online + .995 * old,
                network.params['modules_' + name], network.params['modules_target_' + name])
            network = network.replace(params=_replace_module_params(network.params, 'target_' + name, target))
        return network

    def update(self, batch):
        keys = ('observations', 'next_observations', 'actions', 'value_goals',
                'endpoint_goals', 'endpoint_targets', 'bridge_targets')
        missing = set(keys) - batch.keys()
        if missing:
            raise KeyError(f'Missing CPB fields: {missing}')
        if batch['bridge_targets'].shape[1] != 5:
            raise ValueError('Bridge prefix must remain five states')
        return self._update_impl({key: batch[key] for key in keys})

    def rank_candidates(self, candidates, goals, *, target=False):
        batch_size, n, dim = candidates.shape
        # N=1 bypasses the critic entirely.
        if n == 1:
            return candidates[:, 0], jnp.zeros((batch_size,), dtype=jnp.int32)
        flat_goals = jnp.broadcast_to(goals[:, None], (batch_size, n, goals.shape[-1])).reshape(-1, goals.shape[-1])
        scores = self.score(candidates.reshape(-1, dim), flat_goals, target=target).reshape(batch_size, n)
        best = jnp.argmax(scores, axis=1)
        return jnp.take_along_axis(candidates, best[:, None, None], axis=1)[:, 0], best

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
        """Sample endpoints, select only by online C(z,g), and decode actions."""

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
        selected_endpoints, _ = self.rank_candidates(candidates, goals)

        prefix = self._construct_bridge_prefix(
            observations,
            selected_endpoints,
        )
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
    ) -> 'ContrastivePathBridgerAgent':
        """Initialize all PathBridger modules and the joint optimizer."""

        config = dict(config)
        if config.get('variant') == 'pathbridger_original':
            from configs.pbf import cube_single, cube_double, puzzle_3x3, antmaze_medium
            originals = {m.get_config().env_name: m.get_config for m in (cube_single, cube_double, puzzle_3x3, antmaze_medium)}
            return PathBridgerAgent.create(seed, ex_observations, ex_actions, originals[config['env_name']]())
        if config.get('variant') not in ('cpb_full', 'cpb_rank_only'):
            raise ValueError('Unknown CPB variant')
        if config['endpoint_distribution'] != 'flow':
            raise ValueError('CPB requires the original PBF flow proposer')
        if config['contrastive_temperature'] <= 0 or config['repr_dim'] < 1:
            raise ValueError('Temperature and repr_dim must be positive')
        if tuple(config['critic_p']) != (0., 1., 0., 0.):
            raise ValueError('CPB requires geometric future positives')
        endpoint_distribution = str(
            config['endpoint_distribution']
        ).lower()
        if endpoint_distribution not in ('flow', 'gaussian'):
            raise ValueError(
                "endpoint_distribution must be 'flow' or 'gaussian', got "
                f'{endpoint_distribution!r}.'
            )
        config['endpoint_distribution'] = endpoint_distribution

        horizon = int(config['horizon'])
        if horizon < _ACTION_HORIZON:
            raise ValueError(
                f'horizon must be at least {_ACTION_HORIZON}, got {horizon}.'
            )
        discount = float(config['discount'])
        if not 0.0 < discount < 1.0:
            raise ValueError(f'discount must be in (0, 1), got {discount}.')
        if int(config['eval_num_candidates']) < 1:
            raise ValueError('eval_num_candidates must be at least one.')
        if float(config['eval_temperature']) < 0.0:
            raise ValueError('eval_temperature must be non-negative.')

        observations = jnp.asarray(ex_observations, dtype=jnp.float32)
        actions = jnp.asarray(ex_actions, dtype=jnp.float32)
        if observations.ndim != 2 or actions.ndim != 2:
            raise ValueError(
                'ex_observations and ex_actions must be batched rank-2 arrays.'
            )
        if observations.shape[0] != actions.shape[0]:
            raise ValueError(
                'ex_observations and ex_actions must have equal batch sizes.'
            )
        state_dim = int(observations.shape[-1])
        action_dim = int(actions.shape[-1])
        env_name = str(config['env_name'])
        phi_goal_obs_indices = infer_phi_goal_obs_indices(env_name, state_dim)
        assert_phi_goal_obs_indices(
            state_dim,
            'phi',
            phi_goal_obs_indices,
            where='PathBridgerAgent.create (endpoint goal representation)',
            env_name=env_name,
        )
        encoders = {name: ReachabilityEncoder(env_name, goal=('psi' in name),
                     repr_dim=config['repr_dim'], repr_norm=config['repr_norm'])
                    for name in ('phi', 'psi', 'target_phi', 'target_psi')}
        endpoint_def = FlowEndpointProposer(state_dim=state_dim, env_name=env_name,
                                            phi_goal_obs_indices=phi_goal_obs_indices)
        endpoint_args = (observations, observations, jnp.zeros_like(observations),
                         jnp.zeros((observations.shape[0], 1), dtype=jnp.float32))
        bridge_def = BridgeResidual(state_dim=state_dim)
        idm_def = InverseDynamics(action_dim=action_dim)

        bridge_times = jnp.broadcast_to(
            jnp.linspace(
                0.0,
                1.0,
                horizon + 1,
                dtype=jnp.float32,
            )[None, :],
            (observations.shape[0], horizon + 1),
        )
        network_info = {
            **{name: (encoder, (observations,)) for name, encoder in encoders.items()},
            'endpoint': (endpoint_def, endpoint_args),
            'bridge': (
                bridge_def,
                (observations, jnp.zeros_like(observations), bridge_times),
            ),
            'idm': (idm_def, (observations, observations)),
        }
        network_def = ModuleDict(
            {name: definition for name, (definition, _) in network_info.items()}
        )
        network_args = {
            name: arguments for name, (_, arguments) in network_info.items()
        }

        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        network_params = network_def.init(
            init_rng,
            **network_args,
        )['params']
        for name in ('phi', 'psi'):
            network_params = _replace_module_params(network_params, 'target_' + name, network_params['modules_' + name])
        network = TrainState.create(
            network_def,
            network_params,
            tx=optax.adam(_LEARNING_RATE),
        )
        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(config),
        )


def get_config():
    from agents.pathbridger import get_config as original_config
    config = original_config()
    # TRL-only fields are intentionally absent, not remapped.
    del config['endpoint_value_scale']
    del config['value_distance_weight_power']
    config.update(dict(variant='cpb_full', repr_dim=64, repr_norm=False,
                       contrastive_temperature=1.0, logsumexp_coef=0.01,
                       progress_scale=1.0, lambda_CR=1.0))
    return config
