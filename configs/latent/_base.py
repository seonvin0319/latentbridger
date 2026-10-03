"""Build the LatentBridger ablation configurations.

Each environment config reuses only the PathBridger quantities that still have
the same meaning -- ``env_name``, ``horizon``, and ``discount``.  The released
endpoint-selection knobs (``endpoint_value_scale``,
``value_distance_weight_power``, ``eval_num_candidates``, ``eval_temperature``)
are deliberately absent: LatentBridger has no endpoint proposer and no TRL
candidate ranking to tune.

Variants are selected through the ml_collections config-file argument, e.g.::

    --agent=configs/latent/cube_single.py:sa_cl_bc

so the structural choices are baked in before any ``--agent.*`` override is
applied.
"""

from agents.latentbridger import VARIANT_SETTINGS, VARIANTS
from agents.latentbridger import get_config as get_agent_config

DEFAULT_VARIANT = 'sa_cl_bc'


def apply_variant(config, variant: str):
    """Write one named variant's structure and coefficients into ``config``."""

    variant = str(variant or DEFAULT_VARIANT)
    if variant not in VARIANTS:
        raise ValueError(f'variant must be one of {VARIANTS}, got {variant!r}.')
    config.variant = variant
    for key, value in VARIANT_SETTINGS[variant].items():
        setattr(config, key, value)
    return config


def latent_config(
    *,
    env_name: str,
    horizon: int,
    discount: float,
    variant: str = DEFAULT_VARIANT,
):
    """Return a LatentBridger config for one environment and variant."""

    config = get_agent_config()
    config.env_name = env_name
    config.horizon = horizon
    config.discount = discount
    return apply_variant(config, variant)


__all__ = ['DEFAULT_VARIANT', 'apply_variant', 'latent_config']
