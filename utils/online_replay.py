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
    'sparse_bridge_offsets',
]


def sparse_bridge_offsets(horizon: int, num_waypoints: int) -> tuple[int, ...]:
    """Evenly spaced waypoint offsets whose last one lands on ``horizon``."""

    if horizon < num_waypoints:
        raise ValueError(
            f'sparse offsets need horizon >= num_waypoints, got {horizon} < '
            f'{num_waypoints}.'
        )
    return tuple(
        -(-step * horizon // num_waypoints) for step in range(1, num_waypoints + 1)
    )


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
    ):
        if not 0.0 < discount < 1.0:
            raise ValueError(f'discount must lie in (0, 1), got {discount}.')
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.discount = float(discount)
        self.max_size = int(max_size)
        self._rng = np.random.default_rng(seed)
        # Each episode is (observations[T+1, obs], actions[T, act]); the extra
        # observation is the final state, which is a valid goal but has no
        # action and so is never an anchor.
        self._episodes: list[tuple[np.ndarray, np.ndarray]] = []
        self._num_transitions = 0
        self._total_inserted = 0

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
        self._num_transitions += actions.shape[0]
        self._total_inserted += actions.shape[0]
        while self._num_transitions > self.max_size and len(self._episodes) > 1:
            dropped_observations, dropped_actions = self._episodes.pop(0)
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

    def _sample_anchor_rows(self, batch_size: int):
        """Pick one anchor timestep from each of ``batch_size`` episodes."""

        if not self._episodes:
            raise RuntimeError('Cannot sample from an empty replay buffer.')
        episode_indices = self._rng.integers(0, len(self._episodes), size=batch_size)
        lengths = np.array(
            [self._episodes[index][1].shape[0] for index in episode_indices],
            dtype=np.int64,
        )
        # The final action's state has no strictly-future goal inside the
        # episode, so anchors stop one step earlier.
        anchors = (self._rng.random(batch_size) * (lengths - 1)).astype(np.int64)
        return episode_indices, anchors, lengths

    def _sample_future_offsets(self, anchors: np.ndarray, lengths: np.ndarray):
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
            return np.maximum(1, np.ceil(uniform * remaining)).astype(np.int64)
        # Truncated geometric CDF: F(d) = (1 - q^d) / (1 - q^remaining).
        log_q = np.log(self.discount)
        tail = np.exp(log_q * remaining)
        scaled = 1.0 - uniform * (1.0 - tail)
        offsets = np.ceil(np.log(scaled) / log_q)
        return np.clip(offsets, 1, remaining).astype(np.int64)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """A critic/actor training batch with hindsight future goals."""

        episode_indices, anchors, lengths = self._sample_anchor_rows(batch_size)
        offsets = self._sample_future_offsets(anchors, lengths)

        observations = np.empty((batch_size, self.observation_dim), dtype=np.float32)
        next_observations = np.empty_like(observations)
        goals = np.empty_like(observations)
        actions = np.empty((batch_size, self.action_dim), dtype=np.float32)
        for row, (episode_index, anchor, offset) in enumerate(
            zip(episode_indices, anchors, offsets)
        ):
            episode_observations, episode_actions = self._episodes[episode_index]
            observations[row] = episode_observations[anchor]
            next_observations[row] = episode_observations[anchor + 1]
            goals[row] = episode_observations[anchor + offset]
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
        offsets: tuple[int, ...],
    ) -> dict[str, np.ndarray]:
        """A batch of sparse waypoint targets for the latent bridge.

        Returns the anchor, the episode's final-reachable goal, and the states
        at each sparse offset, clipped to the episode end.  The targets are
        *states*; the caller encodes them with the current ``psi``, because
        online training moves the representation under the bridge and a cached
        latent target would be stale within a few thousand updates.
        """

        if not offsets:
            raise ValueError('sample_bridge needs at least one offset.')
        episode_indices, anchors, lengths = self._sample_anchor_rows(batch_size)
        horizon = max(offsets)

        observations = np.empty((batch_size, self.observation_dim), dtype=np.float32)
        goals = np.empty_like(observations)
        targets = np.empty(
            (batch_size, len(offsets), self.observation_dim), dtype=np.float32
        )
        valid = np.empty((batch_size, len(offsets)), dtype=np.float32)
        for row, (episode_index, anchor, length) in enumerate(
            zip(episode_indices, anchors, lengths)
        ):
            episode_observations, _ = self._episodes[episode_index]
            observations[row] = episode_observations[anchor]
            # The bridge's goal is the state the prefix is aimed at: the
            # horizon-th future state, or the episode end if it comes first.
            goal_index = min(anchor + horizon, length)
            goals[row] = episode_observations[goal_index]
            for column, offset in enumerate(offsets):
                index = min(anchor + offset, length)
                targets[row, column] = episode_observations[index]
                valid[row, column] = float(anchor + offset <= length)
        return {
            'observations': observations,
            'goals': goals,
            'bridge_targets': targets,
            'bridge_valid': valid,
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
        self._total_inserted = int(state['total_inserted'])
        self._rng.bit_generator.state = state['rng_state']
