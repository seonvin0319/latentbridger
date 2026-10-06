"""Cube-single task 1, with a goal-abstraction actor on the SGCRL critic."""

from agents.goal_abstraction import get_config as _defaults


def get_config(variant='sgcrl_psi_goal'):
    config = _defaults(variant)
    config.env_name = 'cube-single-play-v0'
    config.task_id = 1
    return config
