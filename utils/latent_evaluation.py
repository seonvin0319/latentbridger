"""State-based multitask OGBench evaluation for LatentBridger.

The released :mod:`utils.evaluation` assumes the agent emits a whole action
chunk from one observation, which is exactly what LatentBridger must not do.
Both modes here are closed-loop at every environment step:

``direct_goal``
    ``a_t = pi(s_t, psi(g))`` for the final task goal ``g``.  This isolates
    Module A -- the contrastive representation and its latent-conditioned
    controller -- with no generated subtarget anywhere in the loop.
``latent_flow``
    Generate a five-step latent prefix ``z_1..z_5`` from the *current* state,
    then execute ``a_i = pi(s_i, z_i)`` against the *actual* updated state
    ``s_i``, and replan a fresh prefix after at most ``replan_interval``
    actions.  ``replan_interval=5`` consumes the whole prefix before replanning;
    ``replan_interval=1`` keeps only ``z_1`` from each generated prefix, which
    isolates how much of the flow's value comes from the first latent step
    versus the rest of the prefix.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import jax
import numpy as np

DEFAULT_TASK_IDS = (1, 2, 3, 4, 5)
EVAL_MODES = ('direct_goal', 'latent_flow')


def _max_episode_steps(env: Any) -> int:
    spec = getattr(env, 'spec', None)
    max_steps = getattr(spec, 'max_episode_steps', None)
    if max_steps is None:
        max_steps = getattr(env, '_max_episode_steps', None)
    if max_steps is None:
        raise ValueError(
            'Evaluation environment must expose `spec.max_episode_steps`.'
        )
    max_steps = int(max_steps)
    if max_steps < 1:
        raise ValueError(
            f'Environment max episode length must be positive, got {max_steps}.'
        )
    return max_steps


def _info_success(info: Any) -> bool:
    if not isinstance(info, Mapping):
        return False
    value = np.asarray(info.get('success', False))
    return bool(np.any(value))


def _action_from_conditioning(
    agent: Any,
    observation: np.ndarray,
    conditioning: np.ndarray,
) -> np.ndarray:
    action = agent.sample_actions(
        observation.reshape(1, -1),
        conditioning.reshape(1, -1),
    )
    action = np.asarray(jax.device_get(action), dtype=np.float32)
    if action.ndim != 2 or action.shape[0] != 1:
        raise ValueError(
            f'agent.sample_actions must return shape [1, action_dim], got {action.shape}.'
        )
    return action[0]


def _direct_goal_episode(
    agent: Any,
    env: Any,
    observation: np.ndarray,
    goal: np.ndarray,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
    max_episode_steps: int,
    rng,
) -> tuple[bool, int, Any]:
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    goal = np.asarray(goal, dtype=np.float32).reshape(-1)
    conditioning = np.asarray(
        jax.device_get(agent.actor_conditioning(goal.reshape(1, -1))),
        dtype=np.float32,
    )[0]

    success = False
    terminated = False
    truncated = False
    step = 0
    while step < max_episode_steps and not (terminated or truncated):
        action = _action_from_conditioning(agent, observation, conditioning)
        action = np.clip(action, action_low, action_high)
        observation, _, terminated, truncated, info = env.step(action)
        observation = np.asarray(observation, dtype=np.float32).reshape(-1)
        success = success or _info_success(info)
        terminated = bool(terminated)
        truncated = bool(truncated)
        step += 1
    return success, step, rng


def _latent_flow_episode(
    agent: Any,
    env: Any,
    observation: np.ndarray,
    goal: np.ndarray,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
    max_episode_steps: int,
    rng,
    replan_interval: int,
) -> tuple[bool, int, Any]:
    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    goal = np.asarray(goal, dtype=np.float32).reshape(-1)

    success = False
    terminated = False
    truncated = False
    step = 0
    replans = 0
    while step < max_episode_steps and not (terminated or truncated):
        rng, prefix_seed = jax.random.split(rng)
        # Replanning always starts from the real current observation.
        prefix = agent.sample_latent_prefix(
            observation.reshape(1, -1),
            goal.reshape(1, -1),
            prefix_seed,
        )
        prefix = np.asarray(jax.device_get(prefix), dtype=np.float32)
        if prefix.ndim != 3 or prefix.shape[0] != 1:
            raise ValueError(
                'agent.sample_latent_prefix must return shape [1, H, d], '
                f'got {prefix.shape}.'
            )
        replans += 1
        for latent in prefix[0, :replan_interval]:
            if step >= max_episode_steps or terminated or truncated:
                break
            # The latent target is open-loop within the chunk, but the state
            # fed to the actor is the actual environment state at this step.
            action = _action_from_conditioning(agent, observation, latent)
            action = np.clip(action, action_low, action_high)
            observation, _, terminated, truncated, info = env.step(action)
            observation = np.asarray(observation, dtype=np.float32).reshape(-1)
            success = success or _info_success(info)
            terminated = bool(terminated)
            truncated = bool(truncated)
            step += 1
    return success, replans, rng


def evaluate_latent(
    agent: Any,
    env: Any,
    *,
    mode: str = 'direct_goal',
    task_ids: Sequence[int] = DEFAULT_TASK_IDS,
    episodes_per_task: int = 10,
    seed: int = 0,
    replan_interval: int | None = None,
) -> dict[str, float | int | str]:
    """Evaluate LatentBridger on the five OGBench tasks."""

    mode = str(mode).lower()
    if mode not in EVAL_MODES:
        raise ValueError(f'mode must be one of {EVAL_MODES}, got {mode!r}.')
    task_ids = tuple(int(task_id) for task_id in task_ids)
    if not task_ids:
        raise ValueError('task_ids must contain at least one task.')
    if int(episodes_per_task) < 1:
        raise ValueError('episodes_per_task must be at least 1.')

    action_horizon = int(agent.config['action_horizon'])
    if replan_interval is None:
        replan_interval = int(agent.config['replan_interval'])
    replan_interval = int(replan_interval)
    if mode == 'latent_flow':
        if not bool(agent.config['use_flow']):
            raise ValueError(
                "mode='latent_flow' requires a variant with use_flow=True."
            )
        if str(agent.config['actor_goal_input']) != 'latent':
            raise ValueError(
                "mode='latent_flow' requires a latent-conditioned actor."
            )
        if not 1 <= replan_interval <= action_horizon:
            raise ValueError(
                'replan_interval must lie in [1, action_horizon] = '
                f'[1, {action_horizon}], got {replan_interval}.'
            )

    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    max_episode_steps = _max_episode_steps(env)
    rng = jax.random.PRNGKey(int(seed))
    episode_fn = (
        _direct_goal_episode
        if mode == 'direct_goal'
        else partial(_latent_flow_episode, replan_interval=replan_interval)
    )

    metrics: dict[str, float | int | str] = {}
    task_success_rates = []
    for task_id in task_ids:
        successes = []
        for _ in range(int(episodes_per_task)):
            observation, info = env.reset(
                options={'task_id': task_id, 'render_goal': False}
            )
            if not isinstance(info, Mapping) or 'goal' not in info:
                raise RuntimeError(
                    f'Environment reset for task {task_id} did not return info["goal"].'
                )
            success, _, rng = episode_fn(
                agent,
                env,
                observation,
                info['goal'],
                action_low=action_low,
                action_high=action_high,
                max_episode_steps=max_episode_steps,
                rng=rng,
            )
            successes.append(float(success))

        task_success = float(np.mean(successes))
        metrics[f'task_{task_id}_success'] = task_success
        task_success_rates.append(task_success)

    metrics['overall_success'] = float(np.mean(task_success_rates))
    metrics['num_tasks'] = len(task_ids)
    metrics['episodes_per_task'] = int(episodes_per_task)
    metrics['mode'] = mode
    if mode == 'latent_flow':
        metrics['replan_interval'] = replan_interval
    return metrics


__all__ = ['DEFAULT_TASK_IDS', 'EVAL_MODES', 'evaluate_latent']
