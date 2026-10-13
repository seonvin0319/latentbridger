"""Original 3999f37 evaluation functions (main.py); shared by both implementations."""
from __future__ import annotations
from typing import Any
import numpy as np
import jax
import jax.numpy as jnp
from pathbridger_core.agents.dynamics import DynamicsAgent
from pathbridger_core.utils.ogbench_eval_rollout import rollout_chunked_eval_episode
def _idm_actions_from_trajectories(dynamics_agent: DynamicsAgent, trajectories: np.ndarray, horizon: int) -> jnp.ndarray:
    if trajectories.shape[1] <= horizon:
        raise ValueError(
            f'Dynamics trajectory length {trajectories.shape[1]} is too short for horizon={horizon}. '
            'Increase dynamics_N / subgoal_steps or reduce chunk horizons.'
        )
    trajectories = jnp.asarray(trajectories, dtype=jnp.float32)
    return dynamics_agent._idm_actions_from_trajectories(trajectories, horizon)

def _idm_action_chunk(
    dynamics_agent: DynamicsAgent,
    obs: np.ndarray,
    predicted_subgoal: np.ndarray,
    horizon: int,
) -> np.ndarray:
    traj = np.asarray(dynamics_agent.plan(obs, predicted_subgoal)['trajectory'], dtype=np.float32)
    if traj.ndim != 2:
        raise RuntimeError(f'Expected single-trajectory plan with rank 2, got shape={traj.shape}.')
    action_chunk = np.asarray(_idm_actions_from_trajectories(dynamics_agent, traj[None, ...], horizon), dtype=np.float32)
    return action_chunk[0]

def _evaluate_env_tasks(
    env,
    dynamics_agent: DynamicsAgent,
    actor_agent: Any,
    actor_config: Any,
    critic_config: Any,
    *,
    critic_agent: Any | None = None,
    task_ids: tuple[int, ...],
    episodes_per_task: int,
    subgoal_override_goal: bool = False,
    idm_only: bool = False,
) -> dict[str, Any]:
    """OGBench-style eval: success is decided **only** by ``info['success']`` (any step). No tolerance diagnostic.

    ``idm_only`` restricts evaluation to the IDM policy (flow subgoal + inverse dynamics)
    and skips the SPI-actor rollout. Use when the actor has not been trained (see
    ``--train_actor_spi``): the actor policy is fine-tuned separately afterwards.
    """
    if not task_ids:
        return {}

    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    actor_horizon = int(actor_config['actor_chunk_horizon'])
    idm_horizon = int(critic_config['action_chunk_horizon'])
    metrics: dict[str, Any] = {}

    num_eval = max(0, int(episodes_per_task))
    if num_eval < 1:
        raise ValueError('episodes_per_task (stat eval) must be >= 1')

    # Eval selection is fixed to best_of_n_value (critic-scored best-of-N).
    use_eval_bon = True
    eval_seed_base = int(dynamics_agent.config.get('subgoal_eval_seed', 0))
    if idm_only:
        variants = [('flow', 'idm', 'eval_flow_idm')]
    else:
        variants = [
            ('flow', 'idm', 'eval_flow_idm'),
            ('flow', 'actor', 'eval_flow_actor'),
        ]
    task_successes_by_key: dict[str, list[float]] = {metric_key: [] for _, _, metric_key in variants}

    def _eval_subgoal(obs: np.ndarray, goal: np.ndarray, *, ep_ix: int, source: str) -> np.ndarray:
        del source
        if use_eval_bon:
            rng = jax.random.PRNGKey(eval_seed_base + int(ep_ix))
            pred = dynamics_agent.infer_subgoal_for_eval(
                jnp.asarray(obs, dtype=jnp.float32),
                jnp.asarray(goal, dtype=jnp.float32),
                critic_agent=critic_agent,
                rng=rng,
            )
            return np.asarray(pred, dtype=np.float32).reshape(-1)
        return np.asarray(
            dynamics_agent.infer_subgoal(obs, goal),
            dtype=np.float32,
        ).reshape(-1)

    def _actor_chunk(obs: np.ndarray, goal: np.ndarray, *, ep_ix: int, source: str) -> np.ndarray:
        if bool(subgoal_override_goal):
            pred = np.asarray(goal, dtype=np.float32).reshape(-1)
        else:
            pred = _eval_subgoal(obs, goal, ep_ix=ep_ix, source=source)
        return np.asarray(actor_agent.sample_actions(obs, pred), dtype=np.float32).reshape(actor_horizon, -1)

    def _idm_chunk(obs: np.ndarray, goal: np.ndarray, *, ep_ix: int, source: str) -> np.ndarray:
        if bool(subgoal_override_goal):
            pred = np.asarray(goal, dtype=np.float32).reshape(-1)
        else:
            pred = _eval_subgoal(obs, goal, ep_ix=ep_ix, source=source)
        return _idm_action_chunk(dynamics_agent, obs, pred, idm_horizon)

    for task_id in task_ids:
        episode_successes_by_key: dict[str, list[float]] = {metric_key: [] for _, _, metric_key in variants}

        for ep_ix in range(num_eval):
            for subgoal_source, policy, metric_key in variants:
                ob, info = env.reset(options=dict(task_id=int(task_id), render_goal=False))
                if 'goal' not in info:
                    raise RuntimeError(f'Env reset(task_id={task_id}) did not provide info["goal"].')
                obs = np.asarray(ob, dtype=np.float32).reshape(-1)
                goal = np.asarray(info['goal'], dtype=np.float32).reshape(-1)

                if policy == 'actor':
                    sampler = lambda o, g, source=subgoal_source: _actor_chunk(o, g, ep_ix=ep_ix, source=source)
                else:
                    sampler = lambda o, g, source=subgoal_source: _idm_chunk(o, g, ep_ix=ep_ix, source=source)
                ok_env = rollout_chunked_eval_episode(
                    env,
                    obs,
                    goal,
                    low,
                    high,
                    sample_action_chunk=sampler,
                )
                episode_successes_by_key[metric_key].append(1.0 if ok_env else 0.0)

        for _source, _policy, metric_key in variants:
            task_success_rate = float(np.mean(episode_successes_by_key[metric_key]))
            metrics[f'{metric_key}/task_{task_id}/success_rate'] = task_success_rate
            task_successes_by_key[metric_key].append(task_success_rate)

        primary_idm_key = 'eval_flow_idm'
        idm_task_success_rate = metrics[f'{primary_idm_key}/task_{task_id}/success_rate']
        metrics[f'eval_idm/task_{task_id}/success_rate'] = idm_task_success_rate
        metrics[f'evaluation/idm_task_{task_id}_success'] = idm_task_success_rate
        if not idm_only:
            primary_actor_key = 'eval_flow_actor'
            actor_task_success_rate = metrics[f'{primary_actor_key}/task_{task_id}/success_rate']
            metrics[f'eval/task_{task_id}/success_rate'] = actor_task_success_rate
            metrics[f'evaluation/task_{task_id}_success'] = actor_task_success_rate

    for _source, _policy, metric_key in variants:
        metrics[f'{metric_key}/success_rate_mean'] = float(np.mean(task_successes_by_key[metric_key]))
    metrics['eval_idm/success_rate_mean'] = metrics['eval_flow_idm/success_rate_mean']
    metrics['evaluation/overall_idm_success'] = metrics['eval_idm/success_rate_mean']
    if not idm_only:
        metrics['eval/success_rate_mean'] = metrics['eval_flow_actor/success_rate_mean']
        metrics['evaluation/overall_success'] = metrics['eval/success_rate_mean']
    metrics['eval/num_tasks'] = float(len(task_ids))
    metrics['eval/episodes_per_task'] = float(num_eval)

    return metrics
