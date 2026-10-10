from configs.gsctd.puzzle_3x3 import get_config as gs_config
from configs.nonoracle.common import with_method


def get_config(method='BTRL16'):
    # Hyperparameters (horizon/discount/candidates/endpoint) from authoritative
    # puzzle GS-TPB config; oracle metric flags are cleared in with_method.
    return with_method(gs_config('gsdtrl_weighted'), method)
