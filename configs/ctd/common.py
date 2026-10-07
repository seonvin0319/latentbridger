"""Seed-0 CTD method settings on top of each environment's PBF config."""

def _variant(lambda_nce, weighting, lambda_pathnce=0.0, lambda_bridge_geo=0.0):
    return {
        'lambda_nce': lambda_nce,
        'proposer_weighting': weighting,
        'lambda_pathnce': lambda_pathnce,
        'lambda_bridge_geo': lambda_bridge_geo,
        'tau_path': 5.0,
        'num_path_positives': 4,
    }


VARIANTS = {
    'dtrl_uniform': _variant(0.0, 'uniform'),
    'dtrl_weighted': _variant(0.0, 'transitive'),
    'ctd_uniform': _variant(1.0, 'uniform'),
    'ctd_weighted': _variant(1.0, 'transitive'),
    'ctd_pathnce_uniform': _variant(1.0, 'uniform', lambda_pathnce=1.0),
    'ctd_pathnce_weighted': _variant(1.0, 'transitive', lambda_pathnce=1.0),
    'ctd_pathnce_uniform_bridgegeo': _variant(1.0, 'uniform', lambda_pathnce=1.0, lambda_bridge_geo=0.1),
    'ctd_pathnce_weighted_bridgegeo': _variant(1.0, 'transitive', lambda_pathnce=1.0, lambda_bridge_geo=0.1),
}

METHOD_ORDER = (
    'ctd_weighted',
    'dtrl_weighted',
    'ctd_pathnce_weighted',
    'ctd_uniform',
    'dtrl_uniform',
    'ctd_pathnce_uniform',
    'ctd_pathnce_weighted_bridgegeo',
    'ctd_pathnce_uniform_bridgegeo',
)
def with_variant(config, variant='ctd_weighted'):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown CTD variant {variant!r}.')
    config = config.copy_and_resolve_references()
    config.update(VARIANTS[variant])
    config.variant = variant
    # tau_C is the local bridge horizon. tau_path stays 5. Do not tune either.
    config.contrastive_temperature = float(config.horizon)
    config.tau_path = 5.0
    return config
