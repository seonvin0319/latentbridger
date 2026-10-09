"""Offline full-observation FutureNCE sampling.

This module intentionally knows nothing about task IDs, rewards, or oracle goal
representations.  Compact observations and episode terminals are its complete
input contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

SHORT_BAND = (1, 5)
MEDIUM_BAND = (6, 20)
LONG_BAND_MIN = 21


def _episode_index(dataset: Mapping[str, Any], *, require_discount: float | None = None):
    observations = np.asarray(dataset['observations'])
    terminals = np.asarray(dataset['terminals']).reshape(-1)
    if observations.ndim != 2:
        raise ValueError(f'FutureNCE requires compact full observations with shape [N, D], got {observations.shape}.')
    if len(terminals) != len(observations):
        raise ValueError('observations and terminals must have equal lengths.')
    if require_discount is not None and not 0.0 < float(require_discount) < 1.0:
        raise ValueError('discount must lie in (0, 1).')
    terminal_locs = np.flatnonzero(terminals > 0).astype(np.int64)
    if not len(terminal_locs) or terminal_locs[-1] != len(observations) - 1:
        raise ValueError('The final compact observation must be terminal.')

    observations = observations.astype(np.float32, copy=False)
    final_for_index = np.empty(len(observations), dtype=np.int64)
    initial = np.concatenate(([0], terminal_locs[:-1] + 1))
    for start, final in zip(initial, terminal_locs):
        final_for_index[start : final + 1] = final
    # A positive is strictly in the future, so terminal states are excluded.
    valid_anchors = np.flatnonzero(np.arange(len(observations)) < final_for_index).astype(np.int64)
    if not len(valid_anchors):
        raise ValueError('Dataset contains no non-terminal anchors.')
    return observations, terminals, terminal_locs, final_for_index, valid_anchors


@dataclass
class FutureNCEDataset:
    """Sample anchors and geometric future positives within one episode."""

    dataset: Mapping[str, Any]
    discount: float

    def __post_init__(self) -> None:
        (
            self.observations,
            self.terminals,
            self.terminal_locs,
            self.final_for_index,
            self.valid_anchors,
        ) = _episode_index(self.dataset, require_discount=float(self.discount))

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


def _sample_uniform_band(
    anchors: np.ndarray,
    finals: np.ndarray,
    band_min: int,
    band_max: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample offsets uniformly in [band_min, band_max], clipped to terminal."""

    remaining = finals - anchors
    valid = remaining >= band_min
    high = np.minimum(band_max, remaining)
    # Invalid rows keep a placeholder offset; the mask drops them from InfoNCE.
    safe_high = np.maximum(high, band_min)
    offsets = np.zeros(len(anchors), dtype=np.int64)
    if np.any(valid):
        # Inclusive uniform integers in [band_min, high].
        widths = safe_high[valid] - band_min + 1
        offsets[valid] = band_min + (np.random.random(int(valid.sum())) * widths).astype(np.int64)
    offsets = np.where(valid, offsets, band_min)
    futures = np.minimum(anchors + offsets, finals)
    offsets = futures - anchors
    valid = valid & (offsets >= band_min) & (offsets <= band_max) & (futures <= finals) & (futures > anchors)
    return futures, offsets, valid.astype(np.float32)


def _sample_long_band(
    anchors: np.ndarray,
    finals: np.ndarray,
    discount: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """LONG band: geometric offset floored at 21, never past the terminal."""

    remaining = finals - anchors
    valid = remaining >= LONG_BAND_MIN
    raw = np.random.geometric(p=1.0 - float(discount), size=len(anchors)).astype(np.int64)
    offsets = np.maximum(LONG_BAND_MIN, raw)
    futures = np.minimum(anchors + offsets, finals)
    offsets = futures - anchors
    valid = valid & (offsets >= LONG_BAND_MIN) & (futures <= finals) & (futures > anchors)
    return futures, offsets, valid.astype(np.float32)


@dataclass
class MultiHorizonNCEDataset:
    """Sample one shared anchor with short/medium/long band positives."""

    dataset: Mapping[str, Any]
    discount: float

    def __post_init__(self) -> None:
        (
            self.observations,
            self.terminals,
            self.terminal_locs,
            self.final_for_index,
            self.valid_anchors,
        ) = _episode_index(self.dataset, require_discount=float(self.discount))

    def sample_indices(
        self,
        batch_size: int,
        anchors: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        batch_size = int(batch_size)
        if batch_size < 2:
            raise ValueError('MultiHorizonNCE batch_size must be at least two.')
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

        finals = self.final_for_index[anchors]
        short_futures, short_offsets, short_mask = _sample_uniform_band(anchors, finals, SHORT_BAND[0], SHORT_BAND[1])
        medium_futures, medium_offsets, medium_mask = _sample_uniform_band(
            anchors, finals, MEDIUM_BAND[0], MEDIUM_BAND[1]
        )
        long_futures, long_offsets, long_mask = _sample_long_band(anchors, finals, float(self.discount))
        for futures, mask in (
            (short_futures, short_mask),
            (medium_futures, medium_mask),
            (long_futures, long_mask),
        ):
            assert np.all((mask < 0.5) | ((futures > anchors) & (futures <= finals)))
        return {
            'anchor_indices': anchors,
            'short_indices': short_futures,
            'medium_indices': medium_futures,
            'long_indices': long_futures,
            'short_offsets': short_offsets,
            'medium_offsets': medium_offsets,
            'long_offsets': long_offsets,
            'short_mask': short_mask,
            'medium_mask': medium_mask,
            'long_mask': long_mask,
        }

    def sample(
        self,
        batch_size: int,
        anchors: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        indices = self.sample_indices(batch_size, anchors)
        anchors = indices['anchor_indices']
        return {
            'queries': np.asarray(self.observations[anchors], dtype=np.float32),
            'goals_short': np.asarray(self.observations[indices['short_indices']], dtype=np.float32),
            'goals_medium': np.asarray(self.observations[indices['medium_indices']], dtype=np.float32),
            'goals_long': np.asarray(self.observations[indices['long_indices']], dtype=np.float32),
            'short_mask': indices['short_mask'].astype(np.float32),
            'medium_mask': indices['medium_mask'].astype(np.float32),
            'long_mask': indices['long_mask'].astype(np.float32),
            'short_offsets': indices['short_offsets'].astype(np.float32),
            'medium_offsets': indices['medium_offsets'].astype(np.float32),
            'long_offsets': indices['long_offsets'].astype(np.float32),
            'anchor_indices': anchors,
        }


__all__ = [
    'FutureNCEDataset',
    'LONG_BAND_MIN',
    'MEDIUM_BAND',
    'MultiHorizonNCEDataset',
    'SHORT_BAND',
]
