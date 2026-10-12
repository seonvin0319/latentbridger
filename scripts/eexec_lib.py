"""Shared Eexec-pilot data, sampling, and action-prediction modules."""

from __future__ import annotations

import json
from pathlib import Path

import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from utils.networks import MLP

ROOT = Path('/home/choi/latentbridger-goalspace')
OUT = ROOT / 'exp/eexec_pilot/puzzle_3x3'
DATASET = Path('/home/choi/.ogbench/data/puzzle-3x3-play-v0.npz')
ENV_NAME = 'puzzle-3x3-play-v0'
EPISODE_LENGTH = 1001
N_EPISODES = 1000
SPLIT_SEED = 20261011
GOAL_OFFSETS = (5, 20, 40)
ACTION_CHUNK = 5
LATENT_DIM = 16
BATCH_SIZE = 1024
ABORTED_PRETRAIN_UPDATES = 200_000
PRETRAIN_BUDGET = 600_000
PRETRAIN_STEPS = (PRETRAIN_BUDGET - ABORTED_PRETRAIN_UPDATES) // 3
DOWNSTREAM_STEPS = 400_000
LEARNING_RATE = 3e-4
HIDDEN = (512, 512, 512)
DEV_BASE = 5_000_000
FINAL_BASE = 6_000_000
DEV_EPISODES = 10
FINAL_EPISODES = 50
TASKS = (1, 2, 3, 4, 5)


def episode_start(index: int) -> int:
    return int(index) * EPISODE_LENGTH


def load_split() -> dict[str, np.ndarray]:
    return {
        'train': np.load(OUT / 'train_episodes.npy'),
        'val': np.load(OUT / 'val_episodes.npy'),
        'test': np.load(OUT / 'test_episodes.npy'),
    }


def load_norm() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.load(OUT / 'obs_mean.npy')
    std = np.load(OUT / 'obs_std.npy')
    components = np.load(OUT / 'pca_components.npy')[:, :LATENT_DIM]
    return mean, std, components


def normalize(observations: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((observations - mean) / std).astype(np.float32)


def project_pca(observations: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return ((observations - mean) @ components).astype(np.float32)


def pretrain_inputs(
    kind: str,
    state: np.ndarray,
    goal: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    components: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full current state plus the variant-specific goal slot."""

    full_state = normalize(state, mean, std)
    if kind == 'eexec':
        goal_slot = normalize(goal, mean, std)
    elif kind == 'frozen_pca':
        goal_slot = project_pca(goal, mean, components)
    elif kind == 'state_only':
        goal_slot = np.zeros((*goal.shape[:-1], LATENT_DIM), dtype=np.float32)
    else:
        raise ValueError(kind)
    return full_state, goal_slot


def valid_pairs(episode_ids: np.ndarray) -> np.ndarray:
    """Return rows (observation index, goal offset) legal for every model."""

    rows = []
    for episode in np.asarray(episode_ids, dtype=np.int32):
        start = episode_start(int(episode))
        terminal = start + 1000
        for offset in GOAL_OFFSETS:
            last = terminal - max(offset, ACTION_CHUNK)
            if last < start:
                continue
            index = np.arange(start, last + 1, dtype=np.int32)
            rows.append(np.column_stack([index, np.full(index.shape, offset, dtype=np.int32)]))
    if not rows:
        raise RuntimeError('no legal action-prediction pairs')
    return np.concatenate(rows, axis=0)


def batch_at(step: int, pairs: np.ndarray, batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Deterministic shared batch. The same step yields the same rows."""

    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SPLIT_SEED, int(step), 5])))
    choice = rng.integers(0, pairs.shape[0], size=batch_size)
    return pairs[choice]


def gather_batch(observations: np.ndarray, actions: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index = rows[:, 0].astype(np.int64)
    offset = rows[:, 1].astype(np.int64)
    state = observations[index]
    goal = observations[index + offset]
    chunk = np.stack([actions[index + i] for i in range(ACTION_CHUNK)], axis=1)
    return state, goal, chunk


def fixed_eval_rows(pairs_for_split: np.ndarray, tag: str, size: int = 4096) -> np.ndarray:
    tag_id = {'val': 1, 'test': 2}[tag]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([SPLIT_SEED, tag_id])))
    choice = rng.choice(pairs_for_split.shape[0], size=min(size, pairs_for_split.shape[0]), replace=False)
    return pairs_for_split[np.sort(choice)]


def reset_manifest(base: int, episodes: int, name: str) -> list[dict]:
    rows = []
    for task_id in TASKS:
        for episode in range(episodes):
            env_seed = int(base) + int(task_id) * 10_000 + int(episode)
            rows.append(dict(
                list=name,
                task_id=int(task_id),
                episode=int(episode),
                env_seed=env_seed,
                numpy_reset_seed=env_seed,
            ))
    return rows


def manifests_are_separate(dev: list[dict], final: list[dict]) -> bool:
    dev_seeds = {row['env_seed'] for row in dev}
    final_seeds = {row['env_seed'] for row in final}
    bare = {task * 10_000 + episode for task in TASKS for episode in range(100)}
    return dev_seeds.isdisjoint(final_seeds) and dev_seeds.isdisjoint(bare) and final_seeds.isdisjoint(bare)


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2) + '\n')


class Encoder(nn.Module):
    latent_dim: int = LATENT_DIM

    @nn.compact
    def __call__(self, states: jnp.ndarray) -> jnp.ndarray:
        return MLP((*HIDDEN, self.latent_dim), activate_final=False, layer_norm=True)(states)


class ActionHead(nn.Module):
    action_dim: int = 5

    @nn.compact
    def __call__(self, states: jnp.ndarray, goals: jnp.ndarray) -> jnp.ndarray:
        flat = MLP(
            (*HIDDEN, self.action_dim * ACTION_CHUNK),
            activate_final=False,
            layer_norm=True,
        )(jnp.concatenate([states, goals], axis=-1))
        return flat.reshape(*flat.shape[:-1], ACTION_CHUNK, self.action_dim)


class EexecPredictor(nn.Module):
    """Predict from the full current state and a learned goal embedding."""

    @nn.compact
    def __call__(self, states: jnp.ndarray, goals: jnp.ndarray) -> jnp.ndarray:
        encoder = Encoder(name='encoder')
        return ActionHead(name='head')(states, encoder(goals))
