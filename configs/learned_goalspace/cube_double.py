from configs.gsctd.cube_double import get_config as authoritative_config
from configs.learned_goalspace.common import with_method


def get_config(method='LGS_TRL_W_FROZEN'):
    return with_method(authoritative_config('gsdtrl_weighted'), method)
