from configs._base import best_config


def get_config():
    # No released PBF row. Maze-medium eval (N,T)=(8,0.25), humanoid discount 0.995.
    # Appendix lambda=0 is stored then dropped by the CPB wrapper.
    return best_config(
        env_name='humanoidmaze-medium-navigate-v0',
        endpoint_distribution='flow',
        horizon=25,
        discount=0.995,
        endpoint_value_scale=10.0,
        value_distance_weight_power=0.0,
        eval_num_candidates=8,
        eval_temperature=0.25,
    )
