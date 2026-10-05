"""Terminal-safe relabeling for latent endpoint action chunks.

For every sampled anchor ``t`` this module returns the exact H-step action
chunk, the state at ``t + H``, and a later trajectory goal at ``t + Delta``
where ``Delta >= H``.  Only observations, actions, and terminal markers are
consumed; rewards and returns are intentionally absent from the supervision.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np

from utils.datasets import Dataset
from utils.goal_representation import goal_representation


def _get(config: Any, key: str, default: Any = ...) -> Any:
    if isinstance(config, Mapping) and key in config:
        return config[key]
    if not isinstance(config, Mapping) and hasattr(config, key):
        return getattr(config, key)
    if default is ...:
        raise ValueError(f'LatentEndpointChunkDataset config is missing {key!r}.')
    return default


@dataclasses.dataclass(frozen=True)
class ChunkRelabelingConfig:
    chunk_horizon: int = 5
    discount: float = 0.99
    goal_representation: str = 'phi'
    env_name: str = 'cube-single-play-v0'


@dataclasses.dataclass
class LatentEndpointChunkDataset:
    """Sample ``(s_t, A_t, s_{t+H}, s_{t+Delta})`` within one episode."""

    dataset: Dataset
    config: Any

    def __post_init__(self) -> None:
        self.chunk_horizon = int(_get(self.config, 'chunk_horizon', 5))
        self.discount = float(_get(self.config, 'discount', 0.99))
        self.goal_mode = str(_get(self.config, 'goal_representation', 'phi'))
        self.env_name = str(_get(self.config, 'env_name', ''))
        if self.chunk_horizon < 1:
            raise ValueError('chunk_horizon must be positive.')
        if not 0.0 < self.discount < 1.0:
            raise ValueError('discount must lie strictly in (0, 1).')
        for key in ('observations', 'actions', 'terminals'):
            if key not in self.dataset:
                raise ValueError(f'Offline dataset is missing {key!r}.')

        observations = np.asarray(self.dataset['observations'])
        actions = np.asarray(self.dataset['actions'])
        terminals = np.asarray(self.dataset['terminals'])
        if observations.ndim != 2 or actions.ndim != 2:
            raise ValueError('Only vector observations and actions are supported.')
        if terminals.ndim != 1:
            raise ValueError('terminals must be rank one.')
        if not (len(observations) == len(actions) == len(terminals)):
            raise ValueError('observations, actions, and terminals must have equal length.')

        self.size = len(observations)
        self.action_dim = int(actions.shape[-1])
        self.terminal_locs = np.flatnonzero(terminals > 0).astype(np.int64)
        if not len(self.terminal_locs) or int(self.terminal_locs[-1]) != self.size - 1:
            raise ValueError('The final compact-dataset state must be terminal.')
        self.initial_locs = np.concatenate(
            [np.asarray([0], dtype=np.int64), self.terminal_locs[:-1] + 1]
        )
        self._final_for_idx = np.empty(self.size, dtype=np.int64)
        valid: list[np.ndarray] = []
        for start, final in zip(self.initial_locs, self.terminal_locs):
            self._final_for_idx[start : final + 1] = final
            last = int(final) - self.chunk_horizon
            if last >= int(start):
                valid.append(np.arange(start, last + 1, dtype=np.int64))
        if not valid:
            raise ValueError(
                f'No episode contains a complete H={self.chunk_horizon} action chunk.'
            )
        self.valid_starts = np.concatenate(valid)
        self._valid_start_mask = np.zeros(self.size, dtype=bool)
        self._valid_start_mask[self.valid_starts] = True

    @property
    def final_for_idx(self) -> np.ndarray:
        return self._final_for_idx

    def _future_offsets(self, remaining: np.ndarray) -> np.ndarray:
        """Draw Delta proportional to discount**(Delta-H) on [H, remaining]."""

        remaining = np.asarray(remaining, dtype=np.int64)
        if np.any(remaining < self.chunk_horizon):
            raise ValueError('Every anchor must admit a complete action chunk.')
        support = remaining - self.chunk_horizon + 1
        uniform = np.random.random(len(remaining))
        tail = np.power(self.discount, support)
        target = 1.0 - uniform * (1.0 - tail)
        shifted = np.ceil(np.log(target) / np.log(self.discount)).astype(np.int64) - 1
        shifted = np.clip(shifted, 0, support - 1)
        return self.chunk_horizon + shifted

    def _chunks(self, starts: np.ndarray) -> np.ndarray:
        actions = np.asarray(self.dataset['actions'], dtype=np.float32)
        offsets = np.arange(self.chunk_horizon, dtype=np.int64)
        chunks = actions[starts[:, None] + offsets[None, :]]
        return chunks.reshape(len(starts), self.chunk_horizon * self.action_dim)

    def support_chunks(self, max_chunks: int = 4096, seed: int = 0) -> np.ndarray:
        """Return a deterministic offline chunk bank for support diagnostics."""

        max_chunks = int(max_chunks)
        if max_chunks < 1:
            raise ValueError('max_chunks must be positive.')
        starts = self.valid_starts
        if len(starts) > max_chunks:
            starts = np.random.default_rng(int(seed)).choice(
                starts, size=max_chunks, replace=False
            )
        return self._chunks(np.asarray(starts, dtype=np.int64))

    def _represent(self, states: np.ndarray) -> np.ndarray:
        return np.asarray(
            goal_representation(states, self.goal_mode, env_name=self.env_name),
            dtype=np.float32,
        )

    def sample(self, batch_size: int, idxs: Any | None = None) -> dict[str, np.ndarray]:
        batch_size = int(batch_size)
        if batch_size < 2:
            raise ValueError('batch_size must be at least two for InfoNCE.')
        if idxs is None:
            idxs = self.valid_starts[
                np.random.randint(0, len(self.valid_starts), size=batch_size)
            ]
        idxs = np.asarray(idxs, dtype=np.int64)
        if idxs.shape != (batch_size,):
            raise ValueError(f'idxs must have shape ({batch_size},), got {idxs.shape}.')
        if np.any(idxs < 0) or np.any(idxs >= self.size):
            raise ValueError('Sample indices are outside the dataset.')
        if np.any(~self._valid_start_mask[idxs]):
            raise ValueError('Every sample start must admit a full in-episode chunk.')

        finals = self._final_for_idx[idxs]
        deltas = self._future_offsets(finals - idxs)
        endpoint_idxs = idxs + self.chunk_horizon
        goal_idxs = idxs + deltas
        observations = np.asarray(self.dataset['observations'], dtype=np.float32)
        chunks = self._chunks(idxs)
        return {
            'observations': observations[idxs],
            'action_chunks': chunks,
            'actions': chunks,
            'endpoint_states': observations[endpoint_idxs],
            'goals': self._represent(observations[goal_idxs]),
            'anchor_indices': idxs,
            'endpoint_indices': endpoint_idxs,
            'goal_indices': goal_idxs,
            'future_offsets': deltas.astype(np.int32),
        }


__all__ = ['ChunkRelabelingConfig', 'LatentEndpointChunkDataset']
