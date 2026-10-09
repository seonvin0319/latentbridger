"""Offline full-observation FutureNCE sampling.

This module intentionally knows nothing about task IDs, rewards, or oracle goal
representations.  Compact observations and episode terminals are its complete
input contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass
class FutureNCEDataset:
    """Sample anchors and geometric future positives within one episode."""

    dataset: Mapping[str, Any]
    discount: float

    def __post_init__(self) -> None:
        observations = np.asarray(self.dataset['observations'])
        terminals = np.asarray(self.dataset['terminals']).reshape(-1)
        if observations.ndim != 2:
            raise ValueError(
                f'FutureNCE requires compact full observations with shape [N, D], got {observations.shape}.'
            )
        if len(terminals) != len(observations):
            raise ValueError('observations and terminals must have equal lengths.')
        if not 0.0 < float(self.discount) < 1.0:
            raise ValueError('discount must lie in (0, 1).')
        terminal_locs = np.flatnonzero(terminals > 0).astype(np.int64)
        if not len(terminal_locs) or terminal_locs[-1] != len(observations) - 1:
            raise ValueError('The final compact observation must be terminal.')

        self.observations = observations.astype(np.float32, copy=False)
        self.terminals = terminals
        self.terminal_locs = terminal_locs
        self.final_for_index = np.empty(len(observations), dtype=np.int64)
        initial = np.concatenate(([0], terminal_locs[:-1] + 1))
        for start, final in zip(initial, terminal_locs):
            self.final_for_index[start : final + 1] = final
        # A positive is strictly in the future, so terminal states are excluded.
        self.valid_anchors = np.flatnonzero(np.arange(len(observations)) < self.final_for_index).astype(np.int64)
        if not len(self.valid_anchors):
            raise ValueError('Dataset contains no non-terminal anchors.')

    def sample_indices(
        self,
        batch_size: int,
        anchors: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return anchor indices, future indices, and positive offsets."""

        batch_size = int(batch_size)
        if batch_size < 2:
            raise ValueError('FutureNCE batch_size must be at least two.')
        if anchors is None:
            choices = np.random.randint(0, len(self.valid_anchors), size=batch_size)
            anchors = self.valid_anchors[choices]
        else:
            anchors = np.asarray(anchors, dtype=np.int64)
            if anchors.shape != (batch_size,):
                raise ValueError(f'anchors must have shape ({batch_size},).')
            if np.any(anchors < 0) or np.any(anchors >= len(self.observations)):
                raise IndexError('Anchor index out of bounds.')
            if np.any(anchors >= self.final_for_index[anchors]):
                raise ValueError('Every anchor must have a future in its episode.')

        raw_offsets = np.random.geometric(
            p=1.0 - float(self.discount),
            size=batch_size,
        ).astype(np.int64)
        futures = np.minimum(anchors + raw_offsets, self.final_for_index[anchors])
        offsets = futures - anchors
        if np.any(offsets < 1):
            raise AssertionError('FutureNCE produced a non-future positive.')
        return anchors, futures, offsets

    def sample(
        self,
        batch_size: int,
        anchors: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        anchors, futures, offsets = self.sample_indices(batch_size, anchors)
        return {
            'queries': np.asarray(self.observations[anchors], dtype=np.float32),
            'goals': np.asarray(self.observations[futures], dtype=np.float32),
            'anchor_indices': anchors,
            'future_indices': futures,
            'future_offsets': offsets.astype(np.float32),
        }


__all__ = ['FutureNCEDataset']
