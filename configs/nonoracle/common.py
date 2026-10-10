"""Non-oracle GS-TPB method configs."""

METHODS = {
    'BTRL16': {
        'method': 'BTRL16',
        'high_level_oracle_phi': False,
        'proposer_oracle_phi': False,
    },
    'PCA_BTRL16': {
        'method': 'PCA_BTRL16',
        'high_level_oracle_phi': False,
        'proposer_oracle_phi': False,
        'pca_beta': 0.1,
    },
}


def with_method(config, method='BTRL16'):
    if method not in METHODS:
        raise ValueError(f'Unknown non-oracle method {method!r}.')
    config = config.copy_and_resolve_references()
    config.update(METHODS[method])
    # Keep PB-style transitive proposer weighting from the GS-TRL base.
    config.proposer_weighting = 'transitive'
    config.metric_representation = 'learned'
    config.lambda_nce = 0.0
    config.lambda_pathnce = 0.0
    config.lambda_bridge_geo = 0.0
    return config
