"""Shared constants and frozen representations for the full-state cross."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/home/choi/latentbridger-goalspace')

from agents.fullstate_role_pathbridger import FrozenEexecEncoder  # noqa: E402
from eexec_lib import (  # noqa: E402
    DATASET,
    ENV_NAME,
    LATENT_DIM,
    OUT as PILOT_OUT,
    PRETRAIN_STEPS,
    TASKS,
    load_norm,
    load_split,
    normalize,
)


ROOT = Path('/home/choi/latentbridger-goalspace')
OUT = ROOT / 'exp/eexec_fullstate_cross/puzzle_3x3'
ARMS = {
    'pca_pca': ('pca', 'pca'),
    'eexec_eexec': ('eexec', 'eexec'),
    'pca_eexec': ('pca', 'eexec'),
    'eexec_pca': ('eexec', 'pca'),
}
DOWNSTREAM_STEPS = 400_000
DOWNSTREAM_SEED = 0
BATCH_SIZE = 1024
DEV_BASE = 7_000_000
FINAL_BASE = 8_000_000
DEV_EPISODES = 10
FINAL_EPISODES = 50


def load_frozen_representations() -> tuple[Any, np.ndarray]:
    """Load the 16-D Eexec encoder and PCA map in normalized-state coordinates."""

    mean, std, components = load_norm()
    checkpoint = PILOT_OUT / 'pretrain/eexec' / f'params_{PRETRAIN_STEPS}.pkl'
    payload = pickle.loads(checkpoint.read_bytes())
    encoder = FrozenEexecEncoder()
    template = encoder.init(
        jax.random.PRNGKey(0),
        jnp.zeros((2, int(mean.shape[0])), dtype=jnp.float32),
    )['params']
    encoder_params = flax.serialization.from_state_dict(
        template, payload['params']['encoder'],
    )
    pca_matrix = (
        np.asarray(std, dtype=np.float32)[:, None]
        * np.asarray(components, dtype=np.float32)
    )
    if pca_matrix.shape != (55, LATENT_DIM):
        raise RuntimeError(f'unexpected PCA matrix shape {pca_matrix.shape}')
    return encoder_params, pca_matrix


def load_full_train() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train episodes in normalized full-observation coordinates."""

    mean, std, _components = load_norm()
    split = load_split()
    with np.load(DATASET) as data:
        observations = np.asarray(data['observations'], dtype=np.float32)
        actions = np.asarray(data['actions'], dtype=np.float32)
        terminals = np.asarray(data['terminals'])
    pieces_obs = []
    pieces_act = []
    pieces_term = []
    for episode in split['train']:
        start = int(episode) * 1001
        stop = start + 1001
        pieces_obs.append(normalize(observations[start:stop], mean, std))
        pieces_act.append(actions[start:stop])
        pieces_term.append(terminals[start:stop])
    return (
        np.concatenate(pieces_obs).astype(np.float32),
        np.concatenate(pieces_act).astype(np.float32),
        np.concatenate(pieces_term),
    )


def reset_manifest(base: int, episodes: int, name: str) -> list[dict]:
    rows = []
    for task_id in TASKS:
        for episode in range(episodes):
            env_seed = int(base) + int(task_id) * 10_000 + int(episode)
            rows.append({
                'list': name,
                'task_id': int(task_id),
                'episode': int(episode),
                'env_seed': env_seed,
                'numpy_reset_seed': env_seed,
            })
    return rows


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n')


__all__ = [
    'ARMS',
    'BATCH_SIZE',
    'DATASET',
    'DEV_BASE',
    'DEV_EPISODES',
    'DOWNSTREAM_SEED',
    'DOWNSTREAM_STEPS',
    'ENV_NAME',
    'FINAL_BASE',
    'FINAL_EPISODES',
    'OUT',
    'PILOT_OUT',
    'ROOT',
    'TASKS',
    'load_frozen_representations',
    'load_full_train',
    'load_norm',
    'normalize',
    'reset_manifest',
    'write_json',
]
