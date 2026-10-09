"""Oracle-free fixed linear goal representations (PCA-16 and random-16)."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

LATENT_DIM = 16
FORMAT = 'learned_goalspace_fixed_v1'


def fit_pca16(observations: np.ndarray) -> dict[str, Any]:
    """Fit PCA-16 on TRAIN observations only (center with train mean, SVD)."""

    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim != 2:
        raise ValueError(f'PCA expects [N, D] observations, got {observations.shape}.')
    if len(observations) < 2:
        raise ValueError('PCA requires at least two train observations.')
    mean = observations.mean(axis=0).astype(np.float32)
    centered = observations - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rank = min(LATENT_DIM, vt.shape[0], observations.shape[1])
    components = vt[:rank].astype(np.float32)
    if rank < LATENT_DIM:
        pad = np.zeros((LATENT_DIM - rank, observations.shape[1]), dtype=np.float32)
        components = np.concatenate([components, pad], axis=0)
    kernel = components.T.astype(np.float32)  # (D, 16): (obs - mean) @ kernel
    return {
        'kind': 'pca16',
        'mean': mean,
        'kernel': kernel,
        'latent_dim': LATENT_DIM,
        'obs_dim': int(observations.shape[1]),
        'n_train': int(len(observations)),
        'explained_variance_ratio': _explained_variance_ratio(centered, components),
    }


def _explained_variance_ratio(centered: np.ndarray, components: np.ndarray) -> np.ndarray:
    total = float(np.sum(np.square(centered)))
    if total <= 0.0:
        return np.zeros(LATENT_DIM, dtype=np.float32)
    projected = centered @ components.T
    per_component = np.sum(np.square(projected), axis=0) / total
    return per_component.astype(np.float32)


def make_random16(obs_dim: int, seed: int) -> dict[str, Any]:
    """Deterministic Gaussian projection R (D x 16) scaled by 1/sqrt(16)."""

    obs_dim = int(obs_dim)
    if obs_dim < 1:
        raise ValueError('obs_dim must be positive.')
    rng = np.random.default_rng(int(seed))
    kernel = rng.normal(size=(obs_dim, LATENT_DIM)).astype(np.float32)
    kernel /= np.sqrt(float(LATENT_DIM)).astype(np.float32)
    return {
        'kind': 'random16',
        'mean': np.zeros(obs_dim, dtype=np.float32),
        'kernel': kernel,
        'latent_dim': LATENT_DIM,
        'obs_dim': obs_dim,
        'seed': int(seed),
    }


class FixedLinearEncoder(nn.Module):
    """Frozen linear encoder: (obs - mean) @ kernel with optional L2 normalize."""

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
        latent = (observations - mean) @ kernel
        if normalize:
            return latent / jnp.maximum(jnp.linalg.norm(latent, axis=-1, keepdims=True), 1e-8)
        return latent


def encoder_params(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    """Return a modules_goal_encoder-compatible params dict."""

    return {
        'mean': np.asarray(payload['mean'], dtype=np.float32),
        'kernel': np.asarray(payload['kernel'], dtype=np.float32),
    }


def save_fixed_representation(path: str | os.PathLike[str], payload: dict[str, Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        'format': FORMAT,
        'payload': {
            key: (np.asarray(value) if isinstance(value, np.ndarray) else value) for key, value in payload.items()
        },
    }
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as file:
        pickle.dump(body, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    return str(path)


def load_fixed_representation(path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(path)
    with path.open('rb') as file:
        body = pickle.load(file)
    if body.get('format') != FORMAT:
        raise ValueError(f'Invalid fixed-representation file: {path}')
    payload = dict(body['payload'])
    for key in ('mean', 'kernel'):
        if key in payload:
            payload[key] = np.asarray(payload[key], dtype=np.float32)
    return payload


__all__ = [
    'FORMAT',
    'FixedLinearEncoder',
    'LATENT_DIM',
    'encoder_params',
    'fit_pca16',
    'load_fixed_representation',
    'make_random16',
    'save_fixed_representation',
]
