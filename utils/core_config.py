"""Translate the public config without changing original defaults or arithmetic."""
import numpy as np
from pathbridger_core.agents.dynamics import get_dynamics_config
from pathbridger_core.agents.critic import get_config as get_critic_config
from pathbridger_core.utils.goal_representation import infer_phi_goal_obs_indices
from utils.datasets import _validate_goal_mix


def legacy_configs(config, obs_dim, action_dim):
    cfg = dict(config)
    if cfg['endpoint_distribution'] not in ('flow', 'gaussian'):
        raise ValueError('endpoint_distribution must be flow or gaussian.')
    if int(cfg['horizon']) < 5 or not 0 < float(cfg['discount']) < 1:
        raise ValueError('Require horizon >= 5 and 0 < discount < 1.')
    if not np.isfinite(cfg['endpoint_value_scale']) or cfg['endpoint_value_scale'] < 0:
        raise ValueError('endpoint_value_scale must be finite and nonnegative.')
    mode = str(cfg.get('pathbridger_mode', 'pb')).lower()
    if mode not in ('pb', 'oracle_pb'):
        raise ValueError(f"pathbridger_mode must be 'pb' or 'oracle_pb', got {mode!r}.")
    dc, cc = get_dynamics_config().to_dict(), get_critic_config().to_dict()
    phi = infer_phi_goal_obs_indices(cfg['env_name'], obs_dim)
    # Dynamics' own discount retains the original .99 default; the public
    # discount controls the critic, as in the original YAML/main.py interface.
    # Oracle PB keeps the full-state proposer (phi goal conditioning) and only
    # switches the value head + value-based endpoint weights to (phi(s), phi(g)).
    value_state_rep = 'phi' if mode == 'oracle_pb' else 'full'
    value_goal_rep = 'phi' if mode == 'oracle_pb' else 'full'
    dc.update(env_name=cfg['env_name'], phi_goal_obs_indices=phi,
              pathbridger_mode=mode,
              dynamics_N=int(cfg['horizon']), subgoal_steps=int(cfg['horizon']),
              planner_type='forward_bridge_residual', progress_alpha=0.8,
              residual_target_mode='displacement', subgoal_target_mode='displacement',
              subgoal_goal_representation='phi',
              subgoal_value_goal_representation=value_goal_rep,
              subgoal_value_state_representation=value_state_rep,
              subgoal_distribution='flow' if cfg['endpoint_distribution'] == 'flow' else 'diag_gaussian',
              subgoal_stochastic_loss='mse' if cfg['endpoint_distribution'] == 'flow' else 'nll',
              subgoal_value_gap_scale=float(cfg['endpoint_value_scale']),
              subgoal_eval_num_samples=int(cfg['eval_num_candidates']),
              subgoal_temperature=float(cfg['eval_temperature']))
    cc.update(env_name=cfg['env_name'], phi_goal_obs_indices=phi,
              pathbridger_mode=mode,
              batch_size=1024, action_dim=int(action_dim),
              algorithm='trl', critic_type='trl', num_qs=2, q_agg='mean', target_tau=0.005,
              full_chunk_horizon=int(cfg['horizon']), discount=float(cfg['discount']),
              value_distance_weight_power=float(cfg['value_distance_weight_power']))
    for key, out, prefix in [('dynamics_p', dc, 'actor'), ('critic_p', cc, 'value')]:
        cur, geom, traj, rand = _validate_goal_mix(cfg[key], name=key)
        if key == 'critic_p' and (cur > 0 or rand > 0):
            raise ValueError('critic_p requires an ordered future goal distribution.')
        out.update({prefix+'_p_curgoal': cur, prefix+'_p_trajgoal': geom+traj,
                    prefix+'_p_randomgoal': rand, prefix+'_geom_sample': geom > 0})
    return dc, cc
