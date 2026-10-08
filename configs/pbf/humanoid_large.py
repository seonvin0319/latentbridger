from configs._base import best_config


def get_config():
    # No released PBF row. Large-maze PBF eval (N,T)=(32,0.5), K=25, discount 0.995.
    # Appendix lambda=0.1 is stored then dropped by the CPB wrapper.
    return best_config(
        env_name='humanoidmaze-large-navigate-v0',
        endpoint_distribution='flow',
        horizon=25,
        discount=0.995,
        endpoint_value_scale=10.0,
        value_distance_weight_power=0.1,
        eval_num_candidates=32,
        eval_temperature=0.5,
    )
