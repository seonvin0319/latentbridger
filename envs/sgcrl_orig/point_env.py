"""2D point maze used by the original SGCRL launcher.

The map and the transition are the ``point_Spiral11x11`` environment from
``graliuce/sgcrl`` ``point_env.py`` at ``2d1b59d``.  Only that map is launched
by ``lp_contrastive.py``.
"""

from __future__ import annotations

import gym
import numpy as np

WALLS = {
    'Spiral11x11': np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0],
            [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        ]
    ),
}


class PointEnv(gym.Env):
    """Point mass on a binary wall grid.  Observation is ``[state, goal]``."""

    def __init__(self, walls: str = 'Spiral11x11', fixed_start_end=None):
        self._walls = WALLS[walls]
        self._height, self._width = self._walls.shape
        self._action_noise = 0.01
        self._fixed_start_end = fixed_start_end
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 0], dtype=np.float32),
            high=np.array(
                [self._height, self._width, self._height, self._width],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self._timestep = 0
        self.state = np.zeros(2)
        self.goal = np.zeros(2)
        self.reset()

    def _sample_empty_state(self) -> np.ndarray:
        candidate_states = np.where(self._walls == 0)
        state_index = np.random.choice(len(candidate_states[0]))
        state = np.array(
            [
                candidate_states[0][state_index],
                candidate_states[1][state_index],
            ],
            dtype=float,
        )
        state += np.random.uniform(size=2)
        return state

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.state, self.goal]).astype(np.float32)

    def reset(self):
        self._timestep = 0
        if self._fixed_start_end is not None:
            self.state = np.asarray(self._fixed_start_end[0], dtype=float).copy()
            self.goal = np.asarray(self._fixed_start_end[1], dtype=float).copy()
        else:
            self.goal = self._sample_empty_state()
            self.state = self._sample_empty_state()
        return self._get_obs()

    def _discretize_state(self, state: np.ndarray) -> np.ndarray:
        ij = np.floor(state).astype(int)
        return np.clip(ij, np.zeros(2, dtype=int), np.array(self._walls.shape) - 1)

    def _is_blocked(self, state: np.ndarray) -> bool:
        if np.any(state < self.observation_space.low[:2]) or np.any(
            state > self.observation_space.high[:2]
        ):
            return True
        i, j = self._discretize_state(state)
        return bool(self._walls[i, j] == 1)

    def step(self, action):
        action = np.array(action, dtype=np.float64, copy=True)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        if self._action_noise > 0:
            action = action + np.random.normal(0, self._action_noise, size=2)
            action = np.clip(action, self.action_space.low, self.action_space.high)
        num_substeps = 10
        dt = 1.0 / num_substeps
        for _ in range(num_substeps):
            for axis in range(len(action)):
                new_state = self.state.copy()
                new_state[axis] += dt * action[axis]
                if not self._is_blocked(new_state):
                    self.state = new_state
        self._timestep += 1
        dist = np.linalg.norm(self.goal - self.state)
        return self._get_obs(), float(dist < 1.0), False, {}
