"""PCA-initialized Bottleneck TRL (PCA-BTRL16) — non-oracle."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from agents.learned_goal_proposer import (
    LEARNED_GOAL_DIM,
    LearnedGoalFlowEndpointProposer,
    LearnedGoalGaussianEndpointProposer,
)
from agents.pathbridger import (
    BridgeResidual,
    InverseDynamics,
    _ACTION_HORIZON,
    _LEARNING_RATE,
    _replace_module_params,
)
from learned_goalspace.btrl import BottleneckTRLAgent
from learned_goalspace.downstream import LatentScalarValue, LearnedGoalspaceModuleDict
from learned_goalspace.fixed_representations import LATENT_DIM as PCA_DIM
from learned_goalspace.fixed_representations import load_fixed_representation
from learned_goalspace.pretrain import HIDDEN_DIMS, LATENT_DIM, _HiddenMLP
from utils.flax_utils import TrainState

assert LATENT_DIM == LEARNED_GOAL_DIM == PCA_DIM

METHOD = 'PCA_BTRL16'
DEFAULT_BETA = 0.1


class ResidualAdapter(nn.Module):
    """MLP residual with zero-initialized final projection."""

    widths: Sequence[int] = HIDDEN_DIMS

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        hidden = _HiddenMLP(self.widths, name='trunk')(observations)
        return nn.Dense(
            LATENT_DIM,
            name='latent',
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(hidden)


class PCAResidualGoalEncoder(nn.Module):
    """E(o) = PCA16(o) + beta * delta_psi(o); no L2 normalize by default."""

    beta: float = DEFAULT_BETA

    @nn.compact
    def __call__(self, observations: jnp.ndarray, *, normalize: bool = False) -> jnp.ndarray:
        obs_dim = observations.shape[-1]
        mean = self.param('mean', nn.initializers.zeros, (obs_dim,), jnp.float32)
        kernel = self.param(
            'kernel',
            nn.initializers.normal(stddev=1.0 / np.sqrt(LATENT_DIM)),
            (obs_dim, LATENT_DIM),
            jnp.float32,
        )
        pca = (observations - mean) @ kernel
        delta = ResidualAdapter(name='residual')(observations)
        latent = pca + jnp.asarray(self.beta, dtype=observations.dtype) * delta
        if normalize:
            return latent / jnp.maximum(jnp.linalg.norm(latent, axis=-1, keepdims=True), 1e-8)
        return latent


def _pca_btrl_optimizer_labels(params: Any) -> Any:
    """Freeze PCA mean/kernel; train residual adapter and all other modules."""

    mutable = flax.core.unfreeze(params)
    labels: dict[str, Any] = {}
    for name, subtree in mutable.items():
        if name == 'modules_goal_encoder':
            enc_labels = {}
            for key, value in subtree.items():
                tag = 'frozen' if key in ('mean', 'kernel') else 'train'
                enc_labels[key] = jax.tree_util.tree_map(lambda _: tag, value)
            labels[name] = enc_labels
        else:
            labels[name] = jax.tree_util.tree_map(lambda _: 'train', subtree)
    return flax.core.freeze(labels) if isinstance(params, flax.core.FrozenDict) else labels


def _apply_residual(params: Any, observations: jnp.ndarray) -> jnp.ndarray:
    """Apply residual adapter params under modules_goal_encoder/residual."""

    residual_params = {'params': params['modules_goal_encoder']['residual']}
    return ResidualAdapter().apply(residual_params, observations)


def representation_diagnostics(
    agent: 'PCABTRLAgent',
    observations: np.ndarray,
) -> dict[str, float]:
    """Post-hoc PCA vs residual scale diagnostics (no oracle)."""

    observations = jnp.asarray(observations, dtype=jnp.float32)
    enc = agent.network.params['modules_goal_encoder']
    pca = (observations - enc['mean']) @ enc['kernel']
    delta = _apply_residual(agent.network.params, observations)
    beta = float(agent.config.get('pca_beta', DEFAULT_BETA))
    latent = pca + beta * delta
    pca_norm = jnp.linalg.norm(pca, axis=-1)
    delta_norm = jnp.linalg.norm(delta, axis=-1)
    latent_norm = jnp.linalg.norm(latent, axis=-1)
    drift = jnp.linalg.norm(latent - pca, axis=-1)
    return {
        'diag/pca_norm_mean': float(np.asarray(pca_norm.mean())),
        'diag/residual_norm_mean': float(np.asarray(delta_norm.mean())),
        'diag/latent_norm_mean': float(np.asarray(latent_norm.mean())),
        'diag/residual_over_pca_norm': float(np.asarray(delta_norm.mean() / jnp.maximum(pca_norm.mean(), 1e-8))),
        'diag/drift_from_pca_mean': float(np.asarray(drift.mean())),
        'diag/beta': beta,
    }


class PCABTRLAgent(BottleneckTRLAgent):
    """Non-oracle GS-TPB with frozen PCA16 + TRL-trained residual adapter."""

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        config: dict[str, Any],
        *,
        pca_payload: dict[str, Any],
    ) -> 'PCABTRLAgent':
        config = dict(config)
        config['method'] = METHOD
        config['high_level_oracle_phi'] = False
        config['proposer_oracle_phi'] = False
        beta = float(config.get('pca_beta', DEFAULT_BETA))
        config['pca_beta'] = beta
        if pca_payload.get('kind') != 'pca16':
            raise ValueError(f'Expected pca16 payload, got kind={pca_payload.get("kind")!r}')
        observations = jnp.asarray(ex_observations, dtype=jnp.float32)
        actions = jnp.asarray(ex_actions, dtype=jnp.float32)
        state_dim, action_dim = int(observations.shape[-1]), int(actions.shape[-1])
        if int(pca_payload['obs_dim']) != state_dim:
            raise ValueError(f'PCA obs_dim {pca_payload["obs_dim"]} != env state_dim {state_dim}')
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
            'goal_encoder': (PCAResidualGoalEncoder(beta=beta), (observations,)),
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
        pca_params = {
            **flax.core.unfreeze(params['modules_goal_encoder']),
            'mean': np.asarray(pca_payload['mean'], dtype=np.float32),
            'kernel': np.asarray(pca_payload['kernel'], dtype=np.float32),
        }
        params = _replace_module_params(params, 'goal_encoder', pca_params)
        params = _replace_module_params(params, 'target_value', params['modules_value'])
        tx = optax.multi_transform(
            {'train': optax.adam(_LEARNING_RATE), 'frozen': optax.set_to_zero()},
            _pca_btrl_optimizer_labels(params),
        )
        network = TrainState.create(model, params, tx=tx)
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(config))


def load_pca_payload(path: str) -> dict[str, Any]:
    return load_fixed_representation(path)


PCA_BTRL16 = PCABTRLAgent

__all__ = [
    'DEFAULT_BETA',
    'METHOD',
    'PCAResidualGoalEncoder',
    'PCA_BTRL16',
    'PCABTRLAgent',
    'ResidualAdapter',
    'load_pca_payload',
    'representation_diagnostics',
]
