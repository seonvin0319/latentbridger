from agents.contrastive_pathbridger import get_config as cpb_config
from configs.pbf.cube_single import get_config as pbf_config


def get_config():
    config = cpb_config()
    settings = pbf_config().to_dict()
    # Unused/deprecated for CPB: TRL distance weighting and bounded value scale.
    settings.pop('endpoint_value_scale')
    settings.pop('value_distance_weight_power')
    config.update(settings)
    return config
