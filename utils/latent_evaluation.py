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
EVAL_MODES = ('direct_goal', 'latent_flow', 'slerp_bridge')


def episode_manifest(
    task_ids: Sequence[int] = DEFAULT_TASK_IDS,
    episodes_per_task: int = 10,
    seed: int = 0,
) -> list[dict[str, int]]:
    """Enumerate the exact episodes an evaluation will run.

    Every entry pins both of the environment's random sources, so any two
    variants handed the same manifest face the identical task, the identical
    initial state, and the identical goal.  Success can then be compared
    episode by episode instead of only in aggregate.
    """

    task_ids = tuple(int(task_id) for task_id in task_ids)
    if not task_ids:
        raise ValueError('task_ids must contain at least one task.')
    if int(episodes_per_task) < 1:
        raise ValueError('episodes_per_task must be at least 1.')

    manifest: list[dict[str, int]] = []
    for task_id in task_ids:
        for episode in range(int(episodes_per_task)):
            # Distinct per (seed, task, episode), and stable across processes.
            entry_seed = (int(seed) * 1_000_000) + (task_id * 10_000) + episode
            manifest.append(
                {
                    'task_id': int(task_id),
                    'episode': int(episode),
                    'env_seed': int(entry_seed),
                    'action_space_seed': int(entry_seed) + 5,
                }
            )
    return manifest


def slerp(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation between two unit vectors.

    ``alpha=0`` returns ``start`` and ``alpha=1`` returns ``end`` exactly, so
    a SLERP bridge at ``alpha=1`` is bit-for-bit the same conditioning vector
    that ``direct_goal`` uses.  Nearly parallel or antiparallel inputs fall
    back to a normalized linear blend, where the great circle is ill-defined.
    """

    alpha = float(alpha)
    start = np.asarray(start, dtype=np.float32).reshape(-1)
    end = np.asarray(end, dtype=np.float32).reshape(-1)
    if alpha == 0.0:
        return start
    if alpha == 1.0:
        return end

    start_unit = start / max(float(np.linalg.norm(start)), 1e-8)
    end_unit = end / max(float(np.linalg.norm(end)), 1e-8)
    cosine = float(np.clip(np.dot(start_unit, end_unit), -1.0, 1.0))
    omega = float(np.arccos(cosine))
    sin_omega = float(np.sin(omega))
    if sin_omega < 1e-6:
        blended = (1.0 - alpha) * start_unit + alpha * end_unit
        norm = float(np.linalg.norm(blended))
        return (
            end_unit if norm < 1e-8 else (blended / norm).astype(np.float32)
        )
    weight_start = float(np.sin((1.0 - alpha) * omega)) / sin_omega
    weight_end = float(np.sin(alpha * omega)) / sin_omega
    return (weight_start * start_unit + weight_end * end_unit).astype(np.float32)


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


def _slerp_bridge_episode(
    agent: Any,
    env: Any,
    observation: np.ndarray,
    goal: np.ndarray,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
    max_episode_steps: int,
    rng,
    alpha: float,
) -> tuple[bool, int, Any]:
    """Deterministic latent bridge: no flow, just a point on the geodesic.

    At every environment step the waypoint is recomputed from the *actual*
    state, so this is the same receding-horizon control loop the rectified
    flow uses, with the generated waypoint replaced by a fixed interpolation.
    It is the baseline a learned bridge has to beat: if no ``alpha`` improves
    on ``alpha=1`` (which is exactly ``direct_goal``), then moving the actor's
    conditioning vector along the geodesic is not useful, and a flow that
    produces points near that geodesic cannot help either.
    """

    observation = np.asarray(observation, dtype=np.float32).reshape(-1)
    goal = np.asarray(goal, dtype=np.float32).reshape(-1)
    goal_latent = np.asarray(
        jax.device_get(agent.goal_latents(goal.reshape(1, -1))),
        dtype=np.float32,
    )[0]

    success = False
    terminated = False
    truncated = False
    step = 0
    while step < max_episode_steps and not (terminated or truncated):
        state_latent = np.asarray(
            jax.device_get(agent.goal_latents(observation.reshape(1, -1))),
            dtype=np.float32,
        )[0]
        waypoint = slerp(state_latent, goal_latent, alpha)
        action = _action_from_conditioning(agent, observation, waypoint)
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
    alpha: float = 1.0,
    manifest: Sequence[Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """Evaluate LatentBridger on the five OGBench tasks.

    ``manifest`` pins the exact episode list (see :func:`episode_manifest`);
    omit it to generate the default one for ``seed``.  Per-episode outcomes
    are returned under ``'episodes'`` so two variants run on the same manifest
    can be compared episode by episode.
    """

    mode = str(mode).lower()
    if mode not in EVAL_MODES:
        raise ValueError(f'mode must be one of {EVAL_MODES}, got {mode!r}.')

    if manifest is None:
        manifest = episode_manifest(task_ids, episodes_per_task, seed)
    manifest = [dict(entry) for entry in manifest]
    if not manifest:
        raise ValueError('The episode manifest is empty.')
    required = {'task_id', 'env_seed', 'action_space_seed'}
    for entry in manifest:
        missing = required - set(entry)
        if missing:
            raise ValueError(
                f'Manifest entry {entry} is missing {sorted(missing)}.'
            )
    task_ids = tuple(dict.fromkeys(int(entry['task_id']) for entry in manifest))

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
    if mode == 'slerp_bridge':
        if str(agent.config['actor_goal_input']) != 'latent':
            raise ValueError(
                "mode='slerp_bridge' requires a latent-conditioned actor."
            )
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f'alpha must lie in [0, 1], got {alpha}.')

    action_low = np.asarray(env.action_space.low, dtype=np.float32)
    action_high = np.asarray(env.action_space.high, dtype=np.float32)
    max_episode_steps = _max_episode_steps(env)
    rng = jax.random.PRNGKey(int(seed))
    if mode == 'direct_goal':
        episode_fn = _direct_goal_episode
    elif mode == 'latent_flow':
        episode_fn = partial(_latent_flow_episode, replan_interval=replan_interval)
    else:
        episode_fn = partial(_slerp_bridge_episode, alpha=float(alpha))

    metrics: dict[str, Any] = {}
    per_task: dict[int, list[float]] = {task_id: [] for task_id in task_ids}
    outcomes: list[int] = []
    for entry in manifest:
        task_id = int(entry['task_id'])
        # Pin both of the environment's random sources per episode.
        # `reset(seed=...)` only covers `env.np_random` (the initial state);
        # OGBench builds the goal observation by stepping
        # `action_space.sample()` twice, and the action space carries its own
        # generator.  Leaving either unseeded means two variants see different
        # episodes, which makes a paired comparison meaningless.
        env.action_space.seed(int(entry['action_space_seed']))
        observation, info = env.reset(
            seed=int(entry['env_seed']),
            options={'task_id': task_id, 'render_goal': False},
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
        per_task[task_id].append(float(success))
        outcomes.append(int(bool(success)))

    task_success_rates = []
    for task_id in task_ids:
        task_success = float(np.mean(per_task[task_id]))
        metrics[f'task_{task_id}_success'] = task_success
        task_success_rates.append(task_success)

    metrics['overall_success'] = float(np.mean(task_success_rates))
    metrics['num_tasks'] = len(task_ids)
    metrics['num_episodes'] = len(manifest)
    metrics['episodes_per_task'] = len(manifest) // len(task_ids)
    metrics['num_successes'] = int(sum(outcomes))
    metrics['mode'] = mode
    # Per-episode outcomes, aligned with the manifest, enable paired tests.
    metrics['episodes'] = outcomes
    if mode == 'latent_flow':
        metrics['replan_interval'] = replan_interval
    if mode == 'slerp_bridge':
        metrics['alpha'] = float(alpha)
    return metrics


__all__ = [
    'DEFAULT_TASK_IDS',
    'EVAL_MODES',
    'episode_manifest',
    'evaluate_latent',
    'slerp',
]
