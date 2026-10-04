"""Shared config assembly for the online SGCRL raw-bridge comparison."""

from agents.online_sgcrl import VARIANT_SETTINGS, get_config

DEFAULT_VARIANT = 'online_sgcrl'


def apply_variant(config, variant: str):
    """Write the variant's structural choice onto ``config``.

    The variants differ only in which bridge drives behaviour, so every other
    field stays exactly where the shared default put it.
    """

    if variant not in VARIANT_SETTINGS:
        raise ValueError(
            f'variant must be one of {tuple(VARIANT_SETTINGS)}, got {variant!r}.'
        )
    config.variant = variant
    for key, value in VARIANT_SETTINGS[variant].items():
        config[key] = value
    return config


def online_config(*, env_name: str, task_id: int, variant: str, **overrides):
    config = get_config()
    config.env_name = env_name
    config.task_id = task_id
    for key, value in overrides.items():
        config[key] = value
    return apply_variant(config, variant)
