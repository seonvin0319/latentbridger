"""Paired evaluation and episode collection for the online SGCRL agents.

Both of the environment's random sources are pinned per episode.
``reset(seed=...)`` only covers ``env.np_random``, which fixes the initial
state; OGBench builds the goal observation by stepping
``env.action_space.sample()`` twice, and a Gymnasium action space carries its
own generator.  Seeding only the first leaves each variant facing different
goals, which is the bug that made the offline sweep's cross-variant
comparisons meaningless until it was found.

The online setting adds a second requirement: collection and evaluation must
not share a goal distribution by accident.  Behaviour collection uses the
single fixed task goal (SGCRL's ``fix_goals=True``), and evaluation uses the
same task, so the manifest below pins both.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import numpy as np

__all__ = [
    'collect_episode',
    'evaluate_online',
    'make_online_env',
    'online_episode_manifest',
]


def make_online_env(env_name: str, oracle_goals: bool = False):
    """Create the training environment without loading an offline dataset.

    OGBench names go through ``ogbench`` with ``env_only=True``.  The four
    names the original SGCRL launcher actually runs (``sawyer_bin``,
    ``sawyer_box``, ``sawyer_peg``, ``point_Spiral11x11``) are built by
    :func:`envs.sgcrl_orig.make_sgcrl_env`.

    ``oracle_goals`` asks OGBench for its oracle goal, which on cube-single
    is the scaled cube position rather than a full goal observation.  The
    matching slice of the state observation is stored on the env so hindsight
    relabeling can recover that same vector from a stored state.
    """

    name = str(env_name)
    if name.startswith('sawyer_') or name.startswith('point_'):
        if oracle_goals:
            raise ValueError('Oracle goals are only defined for OGBench cubes.')
        from envs.sgcrl_orig import make_sgcrl_env

        env = make_sgcrl_env(name)
        env.oracle_goal_slice = (-1, -1)
        return env

    import ogbench

    env = ogbench.make_env_and_datasets(
        name, env_only=True, use_oracle_rep=bool(oracle_goals)
    )
    if not oracle_goals:
        env.oracle_goal_slice = (-1, -1)
        return env
    env.action_space.seed(0)
    observation, info = env.reset(
        seed=0, options={'task_id': 1, 'render_goal': False}
    )
    observation = np.asarray(observation, dtype=np.float32)
    oracle_now = np.asarray(
        env.unwrapped.compute_oracle_observation(), dtype=np.float32
    )
    goal = np.asarray(info['goal'], dtype=np.float32)
    if oracle_now.shape != goal.shape:
        raise RuntimeError(
            f'Oracle state {oracle_now.shape} and goal {goal.shape} differ.'
        )
    width = int(oracle_now.shape[0])
    found = None
    for start in range(observation.shape[0] - width + 1):
        if np.allclose(observation[start : start + width], oracle_now):
            found = (start, start + width)
            break
    if found is None:
        raise RuntimeError('The oracle cube position is not inside the state.')
    env.oracle_goal_slice = found
    return env


def _goal_distance(observation: np.ndarray, goal: np.ndarray, env: Any) -> float:
    """Distance in the space the actor's goal slot uses."""

    sl = getattr(env, 'oracle_goal_slice', None)
    if sl is not None and int(sl[0]) >= 0:
        observation = np.asarray(observation)[int(sl[0]) : int(sl[1])]
    return float(np.linalg.norm(np.asarray(observation) - np.asarray(goal)))


def online_episode_manifest(
    task_id: int = 1,
    num_episodes: int = 50,
    seed: int = 0,
) -> list[dict[str, int]]:
    """Enumerate the exact episodes an evaluation will run.

    Every entry pins the task, the initial state, and the goal, so two
    variants handed the same manifest are compared episode by episode.
    """

    if int(num_episodes) < 1:
        raise ValueError('num_episodes must be at least 1.')
    manifest = []
    for episode in range(int(num_episodes)):
        entry_seed = (int(seed) * 1_000_000) + (int(task_id) * 10_000) + episode
        manifest.append(
            {
                'task_id': int(task_id),
                'episode': int(episode),
                'env_seed': int(entry_seed),
                'action_space_seed': int(entry_seed) + 5,
            }
        )
    return manifest


def _info_success(info: Any) -> bool:
    if isinstance(info, Mapping) and 'success' in info:
        return bool(np.asarray(info['success']).reshape(-1)[0] > 0.5)
    return False


def _max_episode_steps(env: Any) -> int:
    spec = getattr(env, 'spec', None)
    steps = getattr(spec, 'max_episode_steps', None)
    if steps is None:
        raise RuntimeError('The environment does not declare max_episode_steps.')
    return int(steps)


def _reset(env, entry: Mapping[str, int]):
    env.action_space.seed(int(entry['action_space_seed']))
    observation, info = env.reset(
        seed=int(entry['env_seed']),
        options={'task_id': int(entry['task_id']), 'render_goal': False},
    )
    if not isinstance(info, Mapping) or 'goal' not in info:
        raise RuntimeError('Environment reset did not return info["goal"].')
    return (
        np.asarray(observation, dtype=np.float32),
        np.asarray(info['goal'], dtype=np.float32),
    )


def collect_episode(
    agent: Any,
    env: Any,
    entry: Mapping[str, int],
    rng,
    *,
    random_actions: bool = False,
    bridge_mode: str = 'none',
) -> tuple[np.ndarray, np.ndarray, dict[str, float], Any]:
    """Run one behaviour episode toward the task's fixed goal.

    Returns ``(observations[T+1, obs], actions[T, act], metrics, rng)``.  The
    goal is the task goal for the whole episode: SGCRL collects against one
    goal and relabels in hindsight at training time, so the collector must not
    sneak hindsight goals into the behaviour policy.

    A bridge variant replans its waypoint from the state reached after every
    environment step, so ``agent.act`` is called with the live observation and
    the unchanged final goal.
    """

    observation, goal = _reset(env, entry)
    max_steps = _max_episode_steps(env)
    goal_batch = goal.reshape(1, -1)

    observations = [observation]
    actions = []
    success = False
    terminated = False
    truncated = False
    while len(actions) < max_steps and not (terminated or truncated):
        rng, action_rng = jax.random.split(rng)
        if random_actions:
            action = np.asarray(
                jax.random.uniform(
                    action_rng,
                    (env.action_space.shape[0],),
                    minval=-1.0,
                    maxval=1.0,
                ),
                dtype=np.float32,
            )
        else:
            action = np.asarray(
                jax.device_get(
                    agent.act(
                        observation.reshape(1, -1),
                        goal_batch,
                        action_rng,
                        deterministic=False,
                        bridge_mode=bridge_mode,
                    )
                ),
                dtype=np.float32,
            ).reshape(-1)
        action = np.clip(action, env.action_space.low, env.action_space.high)
        observation, _, terminated, truncated, info = env.step(action)
        observation = np.asarray(observation, dtype=np.float32)
        observations.append(observation)
        actions.append(action)
        success = success or _info_success(info)
        terminated = bool(terminated)
        truncated = bool(truncated)

    metrics = {
        'success': float(success),
        'length': float(len(actions)),
        'final_distance': _goal_distance(observations[-1], goal, env),
    }
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        metrics,
        rng,
    )


def evaluate_online(
    agent: Any,
    env: Any,
    *,
    manifest: Sequence[Mapping[str, int]],
    rng,
    bridge_mode: str = 'none',
) -> dict[str, Any]:
    """Deterministic (mode-action) evaluation over a paired episode manifest."""

    if not manifest:
        raise ValueError('The episode manifest is empty.')
    required = {'task_id', 'env_seed', 'action_space_seed'}
    for entry in manifest:
        missing = required - set(entry)
        if missing:
            raise ValueError(f'Manifest entry {entry} is missing {sorted(missing)}.')

    max_steps = _max_episode_steps(env)
    outcomes: list[int] = []
    final_distances: list[float] = []
    lengths: list[float] = []
    for entry in manifest:
        observation, goal = _reset(env, entry)
        goal_batch = goal.reshape(1, -1)
        success = False
        terminated = False
        truncated = False
        steps = 0
        while steps < max_steps and not (terminated or truncated):
            rng, action_rng = jax.random.split(rng)
            action = np.asarray(
                jax.device_get(
                    agent.act(
                        observation.reshape(1, -1),
                        goal_batch,
                        action_rng,
                        deterministic=True,
                        bridge_mode=bridge_mode,
                    )
                ),
                dtype=np.float32,
            ).reshape(-1)
            action = np.clip(action, env.action_space.low, env.action_space.high)
            observation, _, terminated, truncated, info = env.step(action)
            observation = np.asarray(observation, dtype=np.float32)
            success = success or _info_success(info)
            terminated = bool(terminated)
            truncated = bool(truncated)
            steps += 1
        outcomes.append(int(success))
        final_distances.append(_goal_distance(observation, goal, env))
        lengths.append(float(steps))

    return {
        'success': float(np.mean(outcomes)),
        'num_successes': int(sum(outcomes)),
        'num_episodes': len(outcomes),
        'mean_final_distance': float(np.mean(final_distances)),
        'mean_episode_length': float(np.mean(lengths)),
        # Per-episode outcomes, aligned with the manifest, enable paired tests.
        'episodes': outcomes,
    }
