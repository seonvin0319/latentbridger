from configs.gsctd.common import with_variant
from configs.pbf.cube_single import get_config as pbf_config


def get_config(variant='gsdtrl_weighted'):
    return with_variant(pbf_config(), variant)
