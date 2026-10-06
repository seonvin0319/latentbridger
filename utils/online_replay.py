"""Episodic replay with SGCRL's hindsight future-goal relabeling.

SGCRL stores whole episodes and relabels at sample time: for a timestep ``t``
it draws a goal from the *same* episode's future with probability proportional
to ``discount ** (j - t)`` for ``j > t``.  Two properties of the original
pipeline are load-bearing and are reproduced here.

*Goals never cross an episode boundary.*  A goal from another episode is not a
state this trajectory could have reached, so it would be a false positive for
the contrastive critic.

*Rows of a batch come from different episodes.*  The original achieves this
with a transpose-shuffle over the Reverb stream; here it falls out of sampling
one timestep from each of ``batch_size`` distinct episodes.  Without it the
in-batch negatives would be neighbouring states from one trajectory, which are
trivially distinguishable and make the InfoNCE objective far too easy.

See ``docs/sgcrl_online_semantics.md`` for the derivation from the source.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    'EpisodicReplayBuffer',
    'waypoint_index',
]


def waypoint_index(segment_length: np.ndarray, alpha: float) -> np.ndarray:
    """Offset of the intermediate waypoint inside a segment of length ``Delta``.

    The waypoint sits at ``floor(alpha * Delta)`` steps past the anchor, and is
    clipped into ``[1, Delta - 1]`` so that ``t < i < j`` holds strictly.  The
    clip only bites for short segments: with ``Delta = 2`` and ``alpha = 0.1``
    the unclipped index would coincide with the anchor, which is not a
    waypoint at all.
    """

    if not 0.0 < alpha < 1.0:
        raise ValueError(f'alpha must lie strictly in (0, 1), got {alpha}.')
    segment_length = np.asarray(segment_length, dtype=np.int64)
    raw = np.floor(alpha * segment_length).astype(np.int64)
    return np.clip(raw, 1, segment_length - 1)


class EpisodicReplayBuffer:
    """A FIFO buffer of complete episodes with hindsight goal sampling.

    Capacity is counted in transitions to match ``max_replay_size``; whole
    episodes are evicted, as in the original's episode-keyed Reverb table.
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        *,
        discount: float,
        max_size: int = 1_000_000,
        seed: int = 0,
        holdout_every: int = 20,
        goal_slice: tuple[int, int] | None = None,
    ):
        if not 0.0 < discount < 1.0:
            raise ValueError(f'discount must lie in (0, 1), got {discount}.')
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.discount = float(discount)
        self.max_size = int(max_size)
        if int(holdout_every) < 2:
            raise ValueError(f'holdout_every must be >= 2, got {holdout_every}.')
        self.holdout_every = int(holdout_every)
        if goal_slice is None or int(goal_slice[0]) < 0:
            self.goal_slice = None
            self.goal_dim = self.observation_dim
        else:
            start, end = int(goal_slice[0]), int(goal_slice[1])
            if not 0 <= start < end <= self.observation_dim:
                raise ValueError(
                    f'goal_slice {(start, end)} does not fit an observation '
                    f'of dimension {self.observation_dim}.'
                )
            self.goal_slice = (start, end)
            self.goal_dim = end - start
        self._rng = np.random.default_rng(seed)
        # Each episode is (observations[T+1, obs], actions[T, act]); the extra
        # observation is the final state, which is a valid goal but has no
        # action and so is never an anchor.
        self._episodes: list[tuple[np.ndarray, np.ndarray]] = []
        # Insertion ordinal per stored episode.  Every `holdout_every`-th one
        # is withheld from *bridge* training so the bridge can be scored on
        # trajectories it never fit.  The critic and actor still see all of
        # them, which keeps their data distribution exactly SGCRL's.
        self._episode_ids: list[int] = []
        self._episodes_inserted = 0
        self._num_transitions = 0
        self._total_inserted = 0

    def _project_goal(self, states: np.ndarray) -> np.ndarray:
        """Full state, or the oracle slice of it when one is configured."""

        if self.goal_slice is None:
            return states
        start, end = self.goal_slice
        return states[..., start:end]

    # -- writing -------------------------------------------------------
    def add_episode(self, observations: np.ndarray, actions: np.ndarray) -> None:
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if observations.ndim != 2 or actions.ndim != 2:
            raise ValueError('observations and actions must both be 2-D.')
        if observations.shape[0] != actions.shape[0] + 1:
            raise ValueError(
                'An episode needs one more observation than actions, got '
                f'{observations.shape[0]} and {actions.shape[0]}.'
            )
        if actions.shape[0] < 2:
            # A one-step episode has no strictly-future goal for its only
            # anchor, so it can never produce a training row.
            return
        self._episodes.append((observations, actions))
        self._episode_ids.append(self._episodes_inserted)
        self._episodes_inserted += 1
        self._num_transitions += actions.shape[0]
        self._total_inserted += actions.shape[0]
        while self._num_transitions > self.max_size and len(self._episodes) > 1:
            dropped_observations, dropped_actions = self._episodes.pop(0)
            self._episode_ids.pop(0)
            del dropped_observations
            self._num_transitions -= dropped_actions.shape[0]

    # -- reading -------------------------------------------------------
    def __len__(self) -> int:
        return self._num_transitions

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    @property
    def total_inserted(self) -> int:
        """Transitions ever added, ignoring eviction."""

        return self._total_inserted

    def ready(self, min_size: int) -> bool:
        return self._num_transitions >= int(min_size)

    def _episode_pool(self, holdout: bool | None) -> np.ndarray:
        """Indices of the episodes a given consumer is allowed to sample.

        ``None`` means every episode, which is what the critic and actor use.
        """

        if holdout is None:
            return np.arange(len(self._episodes), dtype=np.int64)
        ids = np.asarray(self._episode_ids, dtype=np.int64)
        is_holdout = (ids % self.holdout_every) == 0
        pool = np.flatnonzero(is_holdout if holdout else ~is_holdout)
        if pool.size == 0:
            raise RuntimeError(
                'No '
                + ('held-out' if holdout else 'training')
                + ' episodes are available yet.'
            )
        return pool.astype(np.int64)

    def _sample_anchor_rows(self, batch_size: int, holdout: bool | None = None):
        """Pick one anchor timestep from each of ``batch_size`` episodes."""

        if not self._episodes:
            raise RuntimeError('Cannot sample from an empty replay buffer.')
        pool = self._episode_pool(holdout)
        episode_indices = pool[self._rng.integers(0, pool.size, size=batch_size)]
        lengths = np.array(
            [self._episodes[index][1].shape[0] for index in episode_indices],
            dtype=np.int64,
        )
        # The final action's state has no strictly-future goal inside the
        # episode, so anchors stop one step earlier.
        anchors = (self._rng.random(batch_size) * (lengths - 1)).astype(np.int64)
        return episode_indices, anchors, lengths

    def _sample_future_offsets(
        self, anchors: np.ndarray, lengths: np.ndarray, min_offset: int = 1
    ):
        """Draw ``Delta ~ Geometric(1 - discount)`` truncated at the episode end.

        The original samples a categorical over ``discount ** (j - t)`` for
        ``j > t``, which is a geometric truncated at the episode's last state
        and renormalized.  Inverse-CDF sampling over that truncated support
        reproduces it exactly, rather than drawing an untruncated geometric and
        clipping -- clipping would pile excess mass on the final state.
        """

        # `remaining` is the largest valid Delta: the episode has `lengths`
        # actions and `lengths + 1` observations, so the last goal index is
        # `lengths` and Delta can reach `lengths - anchors`.
        remaining = lengths - anchors
        uniform = self._rng.random(len(anchors))
        if self.discount >= 1.0:
            offsets = np.ceil(uniform * remaining)
        else:
            # Truncated geometric CDF: F(d) = (1 - q^d) / (1 - q^remaining).
            log_q = np.log(self.discount)
            tail = np.exp(log_q * remaining)
            scaled = 1.0 - uniform * (1.0 - tail)
            offsets = np.ceil(np.log(scaled) / log_q)
        # `min_offset` is 2 for bridge segments, which need room for a state
        # strictly between the anchor and the endpoint.
        return np.clip(offsets, min_offset, remaining).astype(np.int64)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """A critic/actor training batch with hindsight future goals."""

        episode_indices, anchors, lengths = self._sample_anchor_rows(batch_size)
        offsets = self._sample_future_offsets(anchors, lengths)

        observations = np.empty((batch_size, self.observation_dim), dtype=np.float32)
        next_observations = np.empty_like(observations)
        goals = np.empty((batch_size, self.goal_dim), dtype=np.float32)
        actions = np.empty((batch_size, self.action_dim), dtype=np.float32)
        for row, (episode_index, anchor, offset) in enumerate(
            zip(episode_indices, anchors, offsets)
        ):
            episode_observations, episode_actions = self._episodes[episode_index]
            observations[row] = episode_observations[anchor]
            next_observations[row] = episode_observations[anchor + 1]
            goals[row] = self._project_goal(episode_observations[anchor + offset])
            actions[row] = episode_actions[anchor]
        return {
            'observations': observations,
            'actions': actions,
            'next_observations': next_observations,
            'goals': goals,
            'future_offsets': offsets.astype(np.float32),
        }

    def sample_bridge(
        self,
        batch_size: int,
        alpha: float = 0.5,
        holdout: bool = False,
    ) -> dict[str, np.ndarray]:
        """Supervision tuples ``(s_t, g = s_j) -> w* = s_i`` for the bridge.

        The waypoint target is a state the trajectory actually visited, not an
        interpolation between the anchor and the goal.  Segments use the same
        within-episode truncated-geometric endpoint distribution as the
        critic's hindsight goals, so the bridge and the critic are fit on the
        same notion of "reachable future" rather than two different ones.

        ``Delta >= 2`` is enforced so that ``t < i < j`` holds strictly.
        """

        episode_indices, anchors, lengths = self._sample_anchor_rows(
            batch_size, holdout=holdout
        )
        segment_lengths = self._sample_future_offsets(anchors, lengths, min_offset=2)
        waypoint_offsets = waypoint_index(segment_lengths, alpha)

        observations = np.empty((batch_size, self.observation_dim), dtype=np.float32)
        goals = np.empty((batch_size, self.goal_dim), dtype=np.float32)
        waypoints = np.empty((batch_size, self.goal_dim), dtype=np.float32)
        for row, (episode_index, anchor, segment, offset) in enumerate(
            zip(episode_indices, anchors, segment_lengths, waypoint_offsets)
        ):
            episode_observations, _ = self._episodes[episode_index]
            observations[row] = episode_observations[anchor]
            goals[row] = self._project_goal(episode_observations[anchor + segment])
            waypoints[row] = self._project_goal(episode_observations[anchor + offset])
        return {
            'observations': observations,
            'goals': goals,
            'waypoints': waypoints,
            'segment_lengths': segment_lengths.astype(np.float32),
            'waypoint_offsets': waypoint_offsets.astype(np.float32),
        }

    # -- persistence ---------------------------------------------------
    def state_dict(self) -> dict:
        """Everything needed to resume this buffer bit-for-bit."""

        return {
            'observations': np.concatenate(
                [episode[0] for episode in self._episodes], axis=0
            )
            if self._episodes
            else np.zeros((0, self.observation_dim), dtype=np.float32),
            'actions': np.concatenate(
                [episode[1] for episode in self._episodes], axis=0
            )
            if self._episodes
            else np.zeros((0, self.action_dim), dtype=np.float32),
            'episode_lengths': np.array(
                [episode[1].shape[0] for episode in self._episodes], dtype=np.int64
            ),
            'episode_ids': np.array(self._episode_ids, dtype=np.int64),
            'episodes_inserted': np.int64(self._episodes_inserted),
            'total_inserted': np.int64(self._total_inserted),
            'rng_state': self._rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict) -> None:
        lengths = np.asarray(state['episode_lengths'], dtype=np.int64)
        observations = np.asarray(state['observations'], dtype=np.float32)
        actions = np.asarray(state['actions'], dtype=np.float32)
        self._episodes = []
        self._num_transitions = 0
        observation_start = 0
        action_start = 0
        for length in lengths:
            length = int(length)
            self._episodes.append(
                (
                    observations[observation_start : observation_start + length + 1],
                    actions[action_start : action_start + length],
                )
            )
            observation_start += length + 1
            action_start += length
            self._num_transitions += length
        self._episode_ids = [int(value) for value in state['episode_ids']]
        self._episodes_inserted = int(state['episodes_inserted'])
        self._total_inserted = int(state['total_inserted'])
        self._rng.bit_generator.state = state['rng_state']
