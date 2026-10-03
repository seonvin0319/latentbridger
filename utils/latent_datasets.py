"""Offline sampler for the experimental LatentBridger ablation branch.

LatentBridger removes explicit state-space subgoals, transitive-relabelling
(TRL) ranking, and the inverse-dynamics model.  Its supervision is therefore
different from :class:`utils.datasets.PathBridgerDataset`, and this module is
kept separate so the released PathBridger sampler is untouched.

Three independent goal streams are produced per anchor transition:

``contrastive_goals``
    A future state ``s_{t+Delta}`` from the same episode, used as the InfoNCE
    positive for ``C(s, a, g)``.
``actor_goals``
    A future state ``s_{t+delta}``.  With ``actor_goal_offsets`` the offset is
    drawn from an explicit multi-horizon set such as ``{1, 2, 4, 8, 16}``;
    otherwise it is uniform in ``[1, actor_goal_max_offset]``, whose default of
    one makes the actor target exactly the next observation.
    ``actor_goal_sampling='geometric'`` replaces both with
    ``Delta ~ Geometric(1 - actor_discount)`` truncated at the terminal.
``bridge_goals`` / ``bridge_targets``
    An ordinary trajectory-future conditioning goal and the state prefix used
    to supervise the latent flow.  ``flow_target_mode='consecutive'`` takes
    ``s_{t+1}, ..., s_{t+H_a}``; ``'sparse'`` takes sparse long-horizon
    waypoints ``s_{t+h_1}, ..., s_{t+h_{H_a}}`` with ``h_k = ceil(k*H/H_a)``.
    Either prefix is clipped at the goal and padded with it, matching
    PathBridger's close-goal behaviour.  ``bridge_target_offsets`` reports the
    realized offsets so diagnostics can exclude clipped rows.

Like the released sampler this module draws from the global NumPy random state
so checkpoint save/restore reproduces the exact batch stream.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np

from utils.datasets import Dataset

_ACTION_HORIZON = 5
FUTURE_SAMPLING_MODES = ('geometric', 'uniform', 'trajectory')
FLOW_TARGET_MODES = ('consecutive', 'sparse')
ACTOR_GOAL_SAMPLING_MODES = ('offsets', 'geometric')


def sparse_prefix_offsets(horizon: int, action_horizon: int) -> tuple[int, ...]:
    """Sparse long-horizon waypoint offsets ``h_k = ceil(k*H/H_a)``.

    With the PathBridger horizon ``H=40`` and ``H_a=5`` this is
    ``(8, 16, 24, 32, 40)``: five waypoints that span the whole planning
    horizon instead of five consecutive steps that span almost nothing.
    """

    horizon = int(horizon)
    action_horizon = int(action_horizon)
    if action_horizon < 1:
        raise ValueError(f'action_horizon must be positive, got {action_horizon}.')
    if horizon < action_horizon:
        raise ValueError(
            'A sparse latent prefix needs horizon >= action_horizon so the '
            f'waypoints stay distinct; got horizon={horizon}, '
            f'action_horizon={action_horizon}.'
        )
    return tuple(
        -(-k * horizon // action_horizon) for k in range(1, action_horizon + 1)
    )


def _config_get(config: Any, key: str, default: Any = ...) -> Any:
    """Read one setting from a mapping, ConfigDict, or dataclass-like object."""

    if isinstance(config, Mapping):
        if key in config:
            return config[key]
    else:
        try:
            return getattr(config, key)
        except AttributeError:
            pass
    if default is ...:
        raise ValueError(f'LatentBridgerDataset config is missing {key!r}.')
    return default


def _validate_sampling_mode(value: Any, *, name: str) -> str:
    mode = str(value).lower()
    if mode not in FUTURE_SAMPLING_MODES:
        raise ValueError(
            f'{name} must be one of {FUTURE_SAMPLING_MODES}, got {value!r}.'
        )
    return mode


@dataclasses.dataclass(frozen=True)
class LatentBridgerDatasetConfig:
    """Standalone sampler configuration for scripted use and tests."""

    discount: float
    future_sampling: str = 'geometric'
    actor_goal_sampling: str = 'offsets'
    actor_goal_max_offset: int = 1
    actor_goal_offsets: tuple[int, ...] = ()
    actor_discount: float = 0.0
    bridge_goal_sampling: str = 'trajectory'
    action_horizon: int = _ACTION_HORIZON
    flow_target_mode: str = 'consecutive'
    horizon: int = 40


@dataclasses.dataclass
class LatentBridgerDataset:
    """Add LatentBridger hindsight supervision to a compact offline dataset."""

    dataset: Dataset
    config: Any

    def __post_init__(self) -> None:
        self.discount = float(_config_get(self.config, 'discount'))
        self.future_sampling = _validate_sampling_mode(
            _config_get(self.config, 'future_sampling', 'geometric'),
            name='future_sampling',
        )
        self.bridge_goal_sampling = _validate_sampling_mode(
            _config_get(self.config, 'bridge_goal_sampling', 'trajectory'),
            name='bridge_goal_sampling',
        )
        self.actor_goal_max_offset = int(
            _config_get(self.config, 'actor_goal_max_offset', 1)
        )
        self.action_horizon = int(
            _config_get(self.config, 'action_horizon', _ACTION_HORIZON)
        )
        self.actor_goal_sampling = str(
            _config_get(self.config, 'actor_goal_sampling', 'offsets')
        ).lower()
        if self.actor_goal_sampling not in ACTOR_GOAL_SAMPLING_MODES:
            raise ValueError(
                'actor_goal_sampling must be one of '
                f'{ACTOR_GOAL_SAMPLING_MODES}, got {self.actor_goal_sampling!r}.'
            )
        # A zero actor_discount means "track the critic's discount", so the
        # geometric actor starts at the horizon the critic was trained for.
        actor_discount = float(_config_get(self.config, 'actor_discount', 0.0))
        self.actor_discount = actor_discount or self.discount
        if not 0.0 < self.actor_discount < 1.0:
            raise ValueError(
                f'actor_discount must lie in (0, 1), got {self.actor_discount}.'
            )
        self.horizon = int(_config_get(self.config, 'horizon', 40))
        self.flow_target_mode = str(
            _config_get(self.config, 'flow_target_mode', 'consecutive')
        ).lower()
        if self.flow_target_mode not in FLOW_TARGET_MODES:
            raise ValueError(
                f'flow_target_mode must be one of {FLOW_TARGET_MODES}, '
                f'got {self.flow_target_mode!r}.'
            )

        if not 0.0 < self.discount < 1.0:
            raise ValueError(f'discount must lie in (0, 1), got {self.discount}.')
        if self.actor_goal_max_offset < 1:
            raise ValueError(
                'actor_goal_max_offset must be at least 1, got '
                f'{self.actor_goal_max_offset}.'
            )
        if self.action_horizon < 1:
            raise ValueError(
                f'action_horizon must be at least 1, got {self.action_horizon}.'
            )

        # An explicit offset set supersedes the ``[1, max_offset]`` draw.  The
        # empty default keeps every released variant's sampling identical.
        raw_actor_offsets = tuple(
            int(offset)
            for offset in _config_get(self.config, 'actor_goal_offsets', ())
        )
        if raw_actor_offsets:
            if any(offset < 1 for offset in raw_actor_offsets):
                raise ValueError(
                    f'actor_goal_offsets must all be >= 1, got {raw_actor_offsets}.'
                )
            if len(set(raw_actor_offsets)) != len(raw_actor_offsets):
                raise ValueError(
                    f'actor_goal_offsets must be unique, got {raw_actor_offsets}.'
                )
            self.actor_goal_offsets = tuple(sorted(raw_actor_offsets))
        else:
            self.actor_goal_offsets = ()
        self._actor_offset_table = np.asarray(
            self.actor_goal_offsets or (1,),
            dtype=np.int64,
        )

        observations = np.asarray(self.dataset['observations'])
        if observations.ndim != 2:
            raise ValueError(
                'LatentBridger supports state-vector observations only; '
                f'expected shape [N, D], got {observations.shape}.'
            )
        if 'actions' not in self.dataset:
            raise ValueError("LatentBridgerDataset requires an 'actions' field.")
        if 'terminals' not in self.dataset:
            raise ValueError(
                "LatentBridgerDataset requires compact OGBench 'terminals' to "
                'preserve episode boundaries.'
            )

        terminals = np.asarray(self.dataset['terminals'])
        if terminals.ndim != 1:
            raise ValueError(f'terminals must have shape [N], got {terminals.shape}.')
        self.size = self.dataset.size
        self.terminal_locs = np.flatnonzero(terminals > 0).astype(np.int64)
        if len(self.terminal_locs) == 0 or int(self.terminal_locs[-1]) != self.size - 1:
            raise ValueError(
                'The final compact-dataset observation must be marked terminal.'
            )
        self.initial_locs = np.concatenate(
            [np.asarray([0], dtype=np.int64), self.terminal_locs[:-1] + 1]
        )

        # Cache the episode terminal for every state.  A LatentBridger anchor
        # only needs one in-episode successor: the five-step bridge prefix is
        # clipped and padded rather than dropped.
        self._final_for_idx = np.empty(self.size, dtype=np.int64)
        valid_parts: list[np.ndarray] = []
        for start, final in zip(self.initial_locs, self.terminal_locs):
            self._final_for_idx[start : final + 1] = final
            if int(final) - 1 >= int(start):
                valid_parts.append(np.arange(start, int(final), dtype=np.int64))
        if not valid_parts:
            raise ValueError('No episode contains a single usable transition.')
        self.valid_starts = np.concatenate(valid_parts)
        if self.flow_target_mode == 'sparse':
            self.prefix_offsets = sparse_prefix_offsets(
                self.horizon,
                self.action_horizon,
            )
        else:
            self.prefix_offsets = tuple(range(1, self.action_horizon + 1))
        self._prefix_offsets = np.asarray(self.prefix_offsets, dtype=np.int64)

    @property
    def final_for_idx(self) -> np.ndarray:
        """Episode terminal index for every transition index."""

        return self._final_for_idx

    def _validate_starts(self, idxs: Any) -> np.ndarray:
        idxs = np.asarray(idxs, dtype=np.int64)
        if idxs.ndim != 1 or len(idxs) == 0:
            raise ValueError(
                f'idxs must be a non-empty 1D array, got shape {idxs.shape}.'
            )
        if np.any(idxs < 0) or np.any(idxs >= self.size):
            raise IndexError(
                f'Sample starts must lie in [0, {self.size}); got {idxs}.'
            )
        finals = self._final_for_idx[idxs]
        if np.any(idxs >= finals):
            row = int(np.flatnonzero(idxs >= finals)[0])
            raise ValueError(
                'A LatentBridger anchor needs one in-episode successor: '
                f'start={int(idxs[row])}, terminal={int(finals[row])}.'
            )
        return idxs

    @staticmethod
    def _uniform_positive_offsets(max_offsets: np.ndarray) -> np.ndarray:
        """Uniformly sample an integer in ``[1, max_offset]`` per row."""

        max_offsets = np.asarray(max_offsets, dtype=np.int64)
        if np.any(max_offsets < 1):
            raise ValueError(
                'Positive future sampling requires at least one remaining state.'
            )
        return 1 + np.floor(
            np.random.random(len(max_offsets)) * max_offsets
        ).astype(np.int64)

    def _sample_actor_offsets(self, remaining: np.ndarray) -> np.ndarray:
        """Draw one actor-goal offset per row.

        With an explicit ``actor_goal_offsets`` set each row draws uniformly
        from the offsets that still fit inside its episode, so a short suffix
        falls back to the nearer horizons instead of being clipped onto the
        terminal state (which would silently over-sample the episode end).
        ``actor_goal_sampling='geometric'`` instead draws
        ``Delta ~ Geometric(1 - actor_discount)``, truncated at the terminal:
        a smooth horizon distribution rather than five discrete rungs.
        """

        if self.actor_goal_sampling == 'geometric':
            offsets = np.random.geometric(
                p=1.0 - self.actor_discount,
                size=len(remaining),
            ).astype(np.int64)
            return np.minimum(offsets, remaining)
        if not self.actor_goal_offsets:
            return self._uniform_positive_offsets(
                np.minimum(self.actor_goal_max_offset, remaining)
            )
        # The table is sorted, so the feasible prefix length is a searchsorted.
        feasible = np.maximum(
            np.searchsorted(self._actor_offset_table, remaining, side='right'),
            1,
        )
        picks = np.floor(np.random.random(len(remaining)) * feasible).astype(np.int64)
        offsets = self._actor_offset_table[picks]
        # Only bites when even the smallest configured offset overshoots the
        # episode end, which cannot happen for a table that starts at 1.
        return np.minimum(offsets, remaining)

    def _sample_future_idxs(
        self,
        idxs: np.ndarray,
        finals: np.ndarray,
        mode: str,
        discount: float | None = None,
    ) -> np.ndarray:
        """Sample ``s_{t+Delta}`` with ``Delta >= 1`` inside the same episode."""

        remaining = finals - idxs
        if mode == 'geometric':
            offsets = np.random.geometric(
                p=1.0 - (self.discount if discount is None else float(discount)),
                size=len(idxs),
            ).astype(np.int64)
            return np.minimum(idxs + offsets, finals)
        if mode == 'uniform':
            return idxs + self._uniform_positive_offsets(remaining)
        # 'trajectory' reproduces the released sampler's ordinary
        # trajectory-future draw, which is uniform over the remaining suffix in
        # continuous time before rounding.
        distances = np.random.random(len(idxs))
        future_idxs = np.round(
            (idxs + 1) * distances + finals * (1.0 - distances)
        ).astype(np.int64)
        return np.clip(future_idxs, idxs + 1, finals)

    def sample(self, batch_size: int, idxs: Any | None = None) -> dict[str, np.ndarray]:
        """Sample one LatentBridger training batch."""

        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f'batch_size must be positive, got {batch_size}.')
        if idxs is None:
            choices = np.random.randint(0, len(self.valid_starts), size=batch_size)
            idxs = self.valid_starts[choices]
        elif len(np.asarray(idxs)) != batch_size:
            raise ValueError(
                f'batch_size={batch_size} does not match the '
                f'{len(np.asarray(idxs))} provided indices.'
            )
        idxs = self._validate_starts(idxs)

        observations = np.asarray(self.dataset['observations'])
        finals = self._final_for_idx[idxs]
        remaining = finals - idxs

        contrastive_idxs = self._sample_future_idxs(idxs, finals, self.future_sampling)
        contrastive_offsets = contrastive_idxs - idxs

        actor_offsets = self._sample_actor_offsets(remaining)
        actor_idxs = idxs + actor_offsets

        bridge_goal_idxs = self._sample_future_idxs(
            idxs,
            finals,
            self.bridge_goal_sampling,
        )
        bridge_goal_offsets = bridge_goal_idxs - idxs
        # Clip the prefix at the conditioning goal and pad the remainder with
        # it, exactly as the released bridge supervision does for close goals.
        bridge_target_idxs = np.minimum(
            idxs[:, None] + self._prefix_offsets[None, :],
            bridge_goal_idxs[:, None],
        )
        bridge_target_offsets = bridge_target_idxs - idxs[:, None]

        return {
            'observations': np.asarray(observations[idxs], dtype=np.float32),
            'next_observations': np.asarray(
                observations[idxs + 1],
                dtype=np.float32,
            ),
            'actions': np.asarray(self.dataset['actions'][idxs], dtype=np.float32),
            'contrastive_goals': np.asarray(
                observations[contrastive_idxs],
                dtype=np.float32,
            ),
            'contrastive_offsets': contrastive_offsets.astype(np.float32),
            'actor_goals': np.asarray(observations[actor_idxs], dtype=np.float32),
            'actor_offsets': actor_offsets.astype(np.float32),
            'bridge_goals': np.asarray(
                observations[bridge_goal_idxs],
                dtype=np.float32,
            ),
            'bridge_goal_offsets': bridge_goal_offsets.astype(np.float32),
            'bridge_targets': np.asarray(
                observations[bridge_target_idxs],
                dtype=np.float32,
            ),
            'bridge_target_offsets': bridge_target_offsets.astype(np.float32),
        }

    def sample_offset_pairs(
        self,
        batch_size: int,
        delta: int,
    ) -> dict[str, np.ndarray]:
        """Sample ``(s_t, s_{t+delta})`` pairs for the latent-geometry probe.

        Anchors are restricted to starts whose ``delta``-step successor stays in
        the same episode, so the probe never measures a cross-episode pair.
        """

        delta = int(delta)
        if delta < 1:
            raise ValueError(f'delta must be at least 1, got {delta}.')
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError(f'batch_size must be positive, got {batch_size}.')

        eligible = self.valid_starts[
            self.valid_starts + delta <= self._final_for_idx[self.valid_starts]
        ]
        if len(eligible) == 0:
            raise ValueError(f'No episode is long enough for delta={delta}.')
        idxs = eligible[np.random.randint(0, len(eligible), size=batch_size)]
        observations = np.asarray(self.dataset['observations'])
        return {
            'observations': np.asarray(observations[idxs], dtype=np.float32),
            'future_observations': np.asarray(
                observations[idxs + delta],
                dtype=np.float32,
            ),
            'random_observations': np.asarray(
                observations[self.dataset.get_random_idxs(batch_size)],
                dtype=np.float32,
            ),
            'delta': np.full(batch_size, delta, dtype=np.float32),
        }


__all__ = [
    'ACTOR_GOAL_SAMPLING_MODES',
    'FLOW_TARGET_MODES',
    'FUTURE_SAMPLING_MODES',
    'LatentBridgerDataset',
    'LatentBridgerDatasetConfig',
    'sparse_prefix_offsets',
]
