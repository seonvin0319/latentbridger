from learned_goalspace.downstream import METHODS


def with_method(config, method='LGS_TRL_W_FROZEN'):
    method = str(method).upper()
    if method not in METHODS:
        raise ValueError(f'Unknown learned-goalspace method {method!r}.')
    config = config.copy_and_resolve_references()
    # Environment, endpoint, bridge, IDM, TRL and evaluation values are copied
    # unchanged from the authoritative gsctd configuration.
    config.method = method
    config.pretrain_variant = 'fullobs_future_nce'
    config.pretrain_step = 500_000
    return config
