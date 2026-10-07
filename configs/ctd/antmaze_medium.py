from configs.ctd.common import with_variant
from configs.pbf.antmaze_medium import get_config as pbf_config


def get_config(variant='ctd_weighted'):
    return with_variant(pbf_config(), variant)
