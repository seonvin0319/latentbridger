"""Paired evaluation for direct and support-guided chunk inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import numpy as np

from utils.goal_representation import goal_representation

DEFAULT_TASK_IDS = (1, 2, 3, 4, 5)
INFERENCE_MODES = (
    'direct',
    'support_guided_N2',
    'support_guided_N4',
    'support_guided_N8',
)


def episode_manifest(
    task_ids: Sequence[int] = DEFAULT_TASK_IDS,
    episodes_per_task: int = 50,
    seed: int = 0,
) -> list[dict[str, int]]:
    """Create the required 250-episode manifest by default."""

    return [
        {
            'task_id': int(task_id),
            'env_seed': int(seed) * 1_000_000 + int(task_id) * 10_000 + episode,
            'action_space_seed': int(seed) * 1_000_000
            + int(task_id) * 10_000
            + episode
            + 1,
        }
        for task_id in task_ids
        for episode in range(int(episodes_per_task))
    ]


def chunk_prefix(flat_chunk, *, horizon: int, action_dim: int, execute_h: int):
    if not 1 <= int(execute_h) <= int(horizon):
        raise ValueError('execute_h must lie in [1, chunk_horizon].')
    chunk = np.asarray(flat_chunk, dtype=np.float32)
    expected = int(horizon) * int(action_dim)
    if chunk.shape != (expected,):
        raise ValueError(f'Expected a flat chunk of shape ({expected},), got {chunk.shape}.')
    return chunk.reshape(int(horizon), int(action_dim))[: int(execute_h)]


def _max_steps(env: Any) -> int:
    value = getattr(getattr(env, 'spec', None), 'max_episode_steps', None)
    if value is None:
        value = getattr(env, '_max_episode_steps', None)
    if value is None:
        raise ValueError('Environment does not expose a maximum episode length.')
    return int(value)


def _success(info: Any) -> bool:
    if not isinstance(info, Mapping):
        return False
    for key in ('success', 'is_success', 'goal_achieved'):
        if key in info:
            return bool(np.asarray(info[key]).reshape(()))
    return False


def evaluate_latent_endpoint_chunk(
    agent,
    env,
    *,
    manifest: Sequence[Mapping[str, int]] | None = None,
    task_ids: Sequence[int] = DEFAULT_TASK_IDS,
    episodes_per_task: int = 50,
    seed: int = 0,
    execute_h: int | None = None,
    inference_mode: str = 'direct',
    support_chunks: np.ndarray | None = None,
) -> dict[str, Any]:
    """Execute only the chosen prefix, then replan from the resulting state."""

    if inference_mode not in INFERENCE_MODES:
        raise ValueError(f'inference_mode must be one of {INFERENCE_MODES}.')
    if manifest is None:
        manifest = episode_manifest(task_ids, episodes_per_task, seed)
    horizon = int(agent.config['chunk_horizon'])
    action_dim = int(agent.config['action_dim'])
    execute_h = int(agent.config['execute_h'] if execute_h is None else execute_h)
    if not 1 <= execute_h <= horizon:
        raise ValueError('execute_h must lie in [1, chunk_horizon].')
    num_proposals = 0 if inference_mode == 'direct' else int(inference_mode.rsplit('N', 1)[1])
    low = np.asarray(env.action_space.low)
    high = np.asarray(env.action_space.high)
    outcomes: list[int] = []
    per_task: dict[int, list[int]] = {}
    replans: list[int] = []
    metrics: dict[str, list[float]] = {}
    planner_rng = jax.random.PRNGKey(int(seed) + 72_331)

    for row in manifest:
        task_id = int(row['task_id'])
        per_task.setdefault(task_id, [])
        env.action_space.seed(int(row['action_space_seed']))
        observation, info = env.reset(
            seed=int(row['env_seed']),
            options={'task_id': task_id, 'render_goal': False},
        )
        if not isinstance(info, Mapping) or 'goal' not in info:
            raise RuntimeError('OGBench reset must return info["goal"].')
        represented_goal = np.asarray(
            goal_representation(
                np.asarray(info['goal'], dtype=np.float32)[None],
                str(agent.config['goal_representation']),
                env_name=str(agent.config['env_name']),
            ),
            dtype=np.float32,
        )
        succeeded = False
        steps = 0
        episode_replans = 0
        max_steps = _max_steps(env)
        while steps < max_steps:
            observation_batch = np.asarray(observation, dtype=np.float32)[None]
            if inference_mode == 'direct':
                flat = np.asarray(
                    agent.sample_action_chunks(observation_batch, represented_goal)[0]
                )
                score = float(
                    np.asarray(
                        agent.composed_scores(
                            observation_batch, flat[None], represented_goal
                        )
                    )[0]
                )
                support = float(
                    np.asarray(
                        agent.proposal_log_prob(
                            observation_batch, represented_goal, flat[None]
                        )
                    )[0]
                )
                metrics.setdefault('selected_critic_score', []).append(score)
                metrics.setdefault('selected_support_logprob', []).append(support)
            else:
                planner_rng, selection_rng = jax.random.split(planner_rng)
                selected, _candidates, _details, planner = agent.plan_action_chunks(
                    observation_batch,
                    represented_goal,
                    selection_rng,
                    num_proposals=num_proposals,
                )
                flat = np.asarray(selected[0])
                for key, value in planner.items():
                    metrics.setdefault(key, []).append(float(np.asarray(value)[0]))
            if support_chunks is not None and len(support_chunks):
                distance = np.linalg.norm(
                    np.asarray(support_chunks, dtype=np.float32) - flat[None], axis=-1
                )
                metrics.setdefault('selected_support_distance', []).append(
                    float(np.min(distance))
                )
            actions = chunk_prefix(
                flat, horizon=horizon, action_dim=action_dim, execute_h=execute_h
            )
            episode_replans += 1
            done = False
            for action in actions:
                observation, _reward, terminated, truncated, info = env.step(
                    np.clip(action, low, high)
                )
                steps += 1
                succeeded = succeeded or _success(info)
                done = bool(terminated or truncated or steps >= max_steps)
                if done:
                    break
            if done:
                break
        outcome = int(succeeded)
        outcomes.append(outcome)
        per_task[task_id].append(outcome)
        replans.append(episode_replans)

    result: dict[str, Any] = {
        'episodes': outcomes,
        'num_episodes': len(outcomes),
        'num_successes': int(sum(outcomes)),
        'overall_success': float(np.mean(outcomes)),
        'success_std': float(np.std(outcomes)),
        'execute_h': execute_h,
        'chunk_horizon': horizon,
        'mean_replans': float(np.mean(replans)),
        'inference_mode': inference_mode,
        'num_proposals': num_proposals,
    }
    for key, values in sorted(metrics.items()):
        result[f'planner/{key}'] = float(np.mean(values))
    for task_id, values in sorted(per_task.items()):
        result[f'task_{task_id}_success'] = float(np.mean(values))
        result[f'task_{task_id}_episodes'] = len(values)
    return result


__all__ = [
    'DEFAULT_TASK_IDS',
    'INFERENCE_MODES',
    'chunk_prefix',
    'episode_manifest',
    'evaluate_latent_endpoint_chunk',
]
