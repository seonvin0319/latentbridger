"""Original SGCRL environments, exposed as state plus a fixed goal.

The launcher's wrapped observation is ``concat(state, goal)``.  This adapter
splits that vector so the online loop can keep doing what it does for
cube-single: store states, relabel goals from future states, and act toward
the goal returned at reset.  With ``start_index=0`` and ``end_index=-1`` the
original goal extractor is the whole state, so the two halves have equal
dimension.
"""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium
import numpy as np

from envs.sgcrl_orig.point_env import PointEnv
from envs.sgcrl_orig.sawyer import SawyerBin, SawyerBox, SawyerPeg

# lp_contrastive.py fixed_goal_dict, used because --sample_goals defaults False.
FIXED_GOALS = {
    'sawyer_bin': np.array([0.12, 0.7, 0.02]),
    'sawyer_box': np.array([0.0, 0.75, 0.133]),
    'sawyer_peg': np.array([-0.3, 0.6, 0.0]),
    'point_Spiral11x11': (
        np.array([5.0, 5.0]),
        np.array([10.0, 10.0]),
    ),
}

HORIZONS = {
    'sawyer_bin': 150,
    'sawyer_box': 150,
    'sawyer_peg': 150,
    'point_Spiral11x11': 100,
}


class SplitGoalEnv(gymnasium.Env):
    """Turn ``concat(state, goal)`` into an observation and ``info['goal']``."""

    def __init__(self, env, max_episode_steps: int):
        self._env = env
        obs_dim = int(env.observation_space.shape[0]) // 2
        self._obs_dim = obs_dim
        self._max_episode_steps = int(max_episode_steps)
        self.action_space = env.action_space
        low = np.asarray(env.observation_space.low[:obs_dim], dtype=np.float32)
        high = np.asarray(env.observation_space.high[:obs_dim], dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low, high, dtype=np.float32)
        self.spec = SimpleNamespace(max_episode_steps=self._max_episode_steps)
        self._steps = 0

    def _split(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        state = observation[: self._obs_dim]
        goal = observation[self._obs_dim :]
        if state.shape != goal.shape:
            raise RuntimeError(
                f'State and goal halves differ: {state.shape} vs {goal.shape}.'
            )
        return state, goal

    def reset(self, *, seed=None, options=None):
        del options
        if seed is not None:
            np.random.seed(int(seed))
            if hasattr(self._env, 'seed'):
                self._env.seed(int(seed))
        state, goal = self._split(self._env.reset())
        self._steps = 0
        return state, {'goal': goal, 'success': False}

    def step(self, action):
        observation, reward, _done, _info = self._env.step(np.asarray(action))
        self._steps += 1
        state, goal = self._split(observation)
        # The original step returns done=False and lets the step limit end the
        # episode, including after the sparse reward has already fired.
        truncated = self._steps >= self._max_episode_steps
        return (
            state,
            float(reward),
            False,
            truncated,
            {'goal': goal, 'success': float(reward) >= 1.0},
        )

    def close(self):
        closer = getattr(self._env, 'close', None)
        if closer is not None:
            closer()


def make_sgcrl_env(env_name: str) -> SplitGoalEnv:
    """Build one original SGCRL environment with the launcher's fixed goal."""

    if env_name not in FIXED_GOALS:
        raise ValueError(
            f'Unknown SGCRL environment {env_name!r}. '
            f'Known: {tuple(FIXED_GOALS)}.'
        )
    goal = FIXED_GOALS[env_name]
    if env_name == 'sawyer_bin':
        env = SawyerBin(fixed_start_end=goal)
    elif env_name == 'sawyer_box':
        env = SawyerBox(fixed_start_end=goal)
    elif env_name == 'sawyer_peg':
        env = SawyerPeg(fixed_start_end=goal)
    else:
        env = PointEnv(walls='Spiral11x11', fixed_start_end=goal)
    return SplitGoalEnv(env, HORIZONS[env_name])
