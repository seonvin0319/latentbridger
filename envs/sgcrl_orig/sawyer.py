"""Sawyer tasks wrapped the way ``graliuce/sgcrl`` ``env_utils.py`` wraps them.

Each observation is ``concat(state, goal)`` and the reward is the original
sparse 0/1 success signal.  ``fix_goals=True`` in the launcher, so the goal
coordinate is the fixed vector from ``lp_contrastive.py``.
"""

from __future__ import annotations

import gym
import numpy as np
from metaworld.envs.mujoco.env_dict import ALL_V2_ENVIRONMENTS


def _gripper_opening(env) -> float:
    finger_right = env._get_site_pos('rightEndEffector')
    finger_left = env._get_site_pos('leftEndEffector')
    opening = np.linalg.norm(finger_right - finger_left)
    return float(np.clip(opening / 0.1, 0.0, 1.0))


class SawyerBin(ALL_V2_ENVIRONMENTS['bin-picking-v2']):
    """Bin picking.  State is hand, gripper, object.  Horizon 150."""

    def __init__(self, fixed_start_end=None):
        self._goal = np.zeros(3)
        super().__init__()
        self._partially_observable = False
        self._freeze_rand_vec = False
        self._set_task_called = True
        self._fixed_start_end = fixed_start_end
        self.reset()

    def reset(self):
        super().reset()
        body_id = self.model.body_name2id('bin_goal')
        pos1 = self.sim.data.body_xpos[body_id].copy()
        pos1 += np.random.uniform(-0.05, 0.05, 3)
        pos2 = self._get_pos_objects().copy()
        if self._fixed_start_end is not None:
            self._goal = np.asarray(self._fixed_start_end, dtype=np.float64).copy()
        else:
            t = np.random.random()
            self._goal = t * pos1 + (1 - t) * pos2
            self._goal[2] = np.random.uniform(0.03, 0.12)
        self._target_pos = self._goal
        return self._get_obs()

    def step(self, action):
        super().step(action)
        dist = np.linalg.norm(self._goal - self._get_pos_objects())
        return self._get_obs(), float(dist < 0.05), False, {}

    def _get_obs(self):
        obs = np.concatenate(
            (self.get_endeff_pos(), [_gripper_opening(self)], self._get_pos_objects())
        )
        goal = np.concatenate(
            [self._goal + np.array([0.0, 0.0, 0.03]), [0.4], self._goal]
        )
        return np.concatenate([obs, goal]).astype(np.float32)

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=np.full(14, -np.inf, dtype=np.float32),
            high=np.full(14, np.inf, dtype=np.float32),
            dtype=np.float32,
        )


class SawyerBox(ALL_V2_ENVIRONMENTS['box-close-v2']):
    """Box closing.  State adds the lid quaternion.  Horizon 150."""

    def __init__(self, fixed_start_end=None):
        self._goal_pos = np.zeros(3)
        self._goal_quat = np.zeros(4)
        super().__init__()
        self._fixed_start_end = fixed_start_end
        self._set_task_called = True
        self._partially_observable = False
        self._freeze_rand_vec = False
        self.reset()

    def reset(self):
        super().reset()
        pos1 = self._target_pos.copy()
        pos2 = self._get_pos_objects().copy()
        if self._fixed_start_end is not None:
            self._goal_pos = pos1.copy()
        else:
            t = np.random.random()
            self._goal_pos = t * pos1 + (1 - t) * pos2
        self._goal_quat = np.array([0.707, 0, 0, 0.707])
        self._target_pos = self._goal_pos
        return self._get_obs()

    def step(self, action):
        super().step(action)
        dist_pos = np.linalg.norm(self._goal_pos - self._get_pos_objects())
        dist_quat = np.linalg.norm(self._goal_quat - self._get_quat_objects())
        reward = float(dist_pos < 0.08 and dist_quat < 0.08)
        return self._get_obs(), reward, False, {}

    def _get_obs(self):
        obs = np.concatenate(
            (
                self.get_endeff_pos(),
                [_gripper_opening(self)],
                self._get_pos_objects(),
                self._get_quat_objects(),
            )
        )
        goal = np.concatenate(
            [
                self._goal_pos + np.array([0.0, 0.0, 0.03]),
                [0.4],
                self._goal_pos,
                self._goal_quat,
            ]
        )
        return np.concatenate([obs, goal]).astype(np.float32)

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=np.full(22, -np.inf, dtype=np.float32),
            high=np.full(22, np.inf, dtype=np.float32),
            dtype=np.float32,
        )


class SawyerPeg(ALL_V2_ENVIRONMENTS['peg-insert-side-v2']):
    """Side peg insertion.  Horizon 150."""

    def __init__(self, fixed_start_end=None):
        self._goal_pos = np.zeros(3)
        super().__init__()
        self._fixed_start_end = fixed_start_end
        self._set_task_called = True
        self._partially_observable = False
        self._freeze_rand_vec = False
        self.reset()

    def reset(self):
        super().reset()
        pos1 = self._target_pos.copy()
        pos2 = self._get_site_pos('pegHead')
        if self._fixed_start_end is not None:
            self._goal_pos = pos1.copy()
        else:
            t = np.random.random()
            self._goal_pos = t * pos1 + (1 - t) * pos2
        self._target_pos = self._goal_pos
        return self._get_obs()

    def step(self, action):
        super().step(action)
        obj_head = self._get_site_pos('pegHead')
        scale = np.array([1.0, 2.0, 2.0])
        dist_pos = float(np.linalg.norm((obj_head - self._goal_pos) * scale))
        return self._get_obs(), float(dist_pos < 0.07), False, {}

    def _get_obs(self):
        obs = np.concatenate(
            (
                self.get_endeff_pos(),
                [_gripper_opening(self)],
                self._get_site_pos('pegHead'),
            )
        )
        goal = np.concatenate(
            [self._goal_pos + np.array([0.13, 0.0, 0.03]), [0.4], self._goal_pos]
        )
        return np.concatenate([obs, goal]).astype(np.float32)

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=np.full(14, -np.inf, dtype=np.float32),
            high=np.full(14, np.inf, dtype=np.float32),
            dtype=np.float32,
        )
