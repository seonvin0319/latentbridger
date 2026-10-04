from configs.online._base import DEFAULT_VARIANT, online_config


def get_config(variant=DEFAULT_VARIANT):
    return online_config(
        env_name='cube-single-play-v0',
        task_id=1,
        variant=variant,
        # alpha=0.5 puts the waypoint target at the segment midpoint.  It is
        # held fixed across the three-way comparison and only swept later.
        bridge_alpha=0.5,
    )
