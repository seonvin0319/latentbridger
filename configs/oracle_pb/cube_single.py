from configs._base import best_config


def get_config():
    config = best_config(
        env_name='cube-single-play-v0',
        endpoint_distribution='flow',
        horizon=40,
        discount=0.99,
        endpoint_value_scale=5.0,
        value_distance_weight_power=0.7,
        eval_num_candidates=1,
        eval_temperature=0.0,
    )
    config.pathbridger_mode = 'oracle_pb'
    return config
