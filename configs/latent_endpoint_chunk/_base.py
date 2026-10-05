"""Environment-specific configs for latent endpoint action chunks."""

from agents.latent_endpoint_chunk import VARIANT_SETTINGS, VARIANTS
from agents.latent_endpoint_chunk import get_config as get_agent_config

DEFAULT_VARIANT = 'latent_endpoint_awr'


def endpoint_chunk_config(*, env_name: str, discount: float, variant: str):
    variant = str(variant or DEFAULT_VARIANT)
    if variant not in VARIANTS:
        raise ValueError(f'variant must be one of {VARIANTS}, got {variant!r}.')
    config = get_agent_config()
    config.env_name = env_name
    config.discount = float(discount)
    config.variant = variant
    for key, value in VARIANT_SETTINGS[variant].items():
        setattr(config, key, value)
    return config


__all__ = ['DEFAULT_VARIANT', 'endpoint_chunk_config']
