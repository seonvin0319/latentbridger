"""Locked variants for the seed-0 goal-space pilot."""


VARIANTS = {
    'gsdtrl_weighted': {
        'lambda_nce': 0.0,
        'nce_temperature_mode': 'fixed',
    },
    'gsctd_learned_temp': {
        'lambda_nce': 1.0,
        'nce_temperature_mode': 'learned',
    },
    'gsctd_fixed': {
        'lambda_nce': 1.0,
        'nce_temperature_mode': 'fixed',
    },
}


def with_variant(config, variant='gsdtrl_weighted'):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown goal-space variant {variant!r}.')
    config = config.copy_and_resolve_references()
    config.update(VARIANTS[variant])
    config.variant = variant
    config.proposer_weighting = 'transitive'
    config.metric_representation = 'phi'
    config.contrastive_temperature = float(config.horizon)
    config.lambda_pathnce = 0.0
    config.lambda_bridge_geo = 0.0
    config.tau_path = 5.0
    config.num_path_positives = 4
    return config
