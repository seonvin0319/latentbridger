"""Cube-single with OGBench's oracle goal instead of the full goal observation.

The oracle goal is the scaled cube position, three numbers, not the 28-dimensional
goal observation that includes a fictitious arm pose.  Hindsight goals and bridge
waypoints are that same slice of a visited state.
"""

from configs.online._base import DEFAULT_VARIANT, online_config


def get_config(variant=DEFAULT_VARIANT):
    return online_config(
        env_name='cube-single-play-v0',
        task_id=1,
        variant=variant,
        bridge_alpha=0.5,
        oracle_goal_slice=(19, 22),
    )
