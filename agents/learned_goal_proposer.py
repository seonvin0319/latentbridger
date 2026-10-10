"""Endpoint proposers conditioned on a learned goal embedding (no oracle phi)."""

from __future__ import annotations

from collections.abc import Sequence

import flax.linen as nn
import jax.numpy as jnp

from agents.pathbridger import (
    _GAUSSIAN_LOG_STD_MAX,
    _GAUSSIAN_LOG_STD_MIN,
    _HIDDEN_DIMS,
    _LAYER_NORM,
)
from utils.networks import MLP

LEARNED_GOAL_DIM = 16


class LearnedGoalFlowEndpointProposer(nn.Module):
    """Rectified-flow endpoint velocity using full state + learned E(g).

    Capacity matches ``FlowEndpointProposer``. Oracle task features are never
    read inside this module; the goal argument is already a 16-D embedding.
    """

    state_dim: int
    goal_dim: int = LEARNED_GOAL_DIM
    hidden_dims: Sequence[int] = _HIDDEN_DIMS

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        goal_embeddings: jnp.ndarray,
        noisy_displacements: jnp.ndarray,
        times: jnp.ndarray,
    ) -> jnp.ndarray:
        if goal_embeddings.shape[-1] != self.goal_dim:
            raise ValueError(f'Expected goal embedding dim {self.goal_dim}, got {goal_embeddings.shape[-1]}.')
        times = jnp.asarray(times, dtype=jnp.float32)
        if times.ndim == noisy_displacements.ndim - 1:
            times = times[..., None]
        inputs = jnp.concatenate(
            [observations, goal_embeddings, noisy_displacements, times],
            axis=-1,
        )
        return MLP(
            (*self.hidden_dims, self.state_dim),
            activate_final=False,
            layer_norm=_LAYER_NORM,
        )(inputs)


class LearnedGoalGaussianEndpointProposer(nn.Module):
    """Gaussian endpoint displacement using full state + learned E(g)."""

    state_dim: int
    goal_dim: int = LEARNED_GOAL_DIM
    hidden_dims: Sequence[int] = _HIDDEN_DIMS

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        goal_embeddings: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        if goal_embeddings.shape[-1] != self.goal_dim:
            raise ValueError(f'Expected goal embedding dim {self.goal_dim}, got {goal_embeddings.shape[-1]}.')
        inputs = jnp.concatenate([observations, goal_embeddings], axis=-1)
        hidden = MLP(
            tuple(self.hidden_dims),
            activate_final=True,
            layer_norm=_LAYER_NORM,
        )(inputs)
        mean = nn.Dense(self.state_dim, name='mean')(hidden)
        log_std = nn.Dense(self.state_dim, name='log_std')(hidden)
        log_std = jnp.clip(log_std, _GAUSSIAN_LOG_STD_MIN, _GAUSSIAN_LOG_STD_MAX)
        return mean, log_std


__all__ = [
    'LEARNED_GOAL_DIM',
    'LearnedGoalFlowEndpointProposer',
    'LearnedGoalGaussianEndpointProposer',
]
