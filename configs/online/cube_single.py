from configs.online._base import DEFAULT_VARIANT, online_config


def get_config(variant=DEFAULT_VARIANT):
    return online_config(
        env_name='cube-single-play-v0',
        task_id=1,
        variant=variant,
        # cube-single episodes are 200 steps, so a 40-step bridge horizon
        # spans a fifth of an episode, matching the offline sparse scheme.
        bridge_horizon=40,
    )
