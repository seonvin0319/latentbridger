from learned_goalspace.downstream import FIXED_ENCODER_METHODS, METHODS

_PRETRAIN_VARIANT = {
    'LGS_TRL_W_FROZEN': 'fullobs_future_nce',
    'LGSDTRL_W_FROZEN': 'fullobs_future_nce',
    'MH_LGS_TRL_W_FROZEN': 'fullobs_multihorizon_nce',
    'PCA16_GS_TRL_W': 'pca16',
    'RANDOM16_GS_TRL_W': 'random16',
}


def with_method(config, method='LGS_TRL_W_FROZEN'):
    method = str(method).upper()
    if method not in METHODS:
        raise ValueError(f'Unknown learned-goalspace method {method!r}.')
    config = config.copy_and_resolve_references()
    # Environment, endpoint, bridge, IDM, TRL and evaluation values are copied
    # unchanged from the authoritative gsctd configuration.
    config.method = method
    config.pretrain_variant = _PRETRAIN_VARIANT[method]
    config.pretrain_step = 0 if method in FIXED_ENCODER_METHODS else 500_000
    return config
