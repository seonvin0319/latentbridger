"""Shared strict-reset CPU evaluation used in the controlled comparisons."""
from contextlib import contextmanager
import hashlib
import jax
import numpy as np
from utils.reference_evaluation import _evaluate_env_tasks

DEFAULT_TASK_IDS = (1, 2, 3, 4, 5)
PROTOCOL = '3999-idm-strict-reset-v2'

# Preserve the agreed evaluator's individual compilation boundaries.
_infer = jax.jit(lambda d, c, o, g, k: d.infer_subgoal_for_eval(o, g, critic_agent=c, rng=k))
_idm = jax.jit(lambda d, t: d._idm_actions_from_trajectories(t, 5))


class EvalDynamics:
    def __init__(self, dynamics, num_candidates, temperature):
        cfg = dynamics.config.copy(add_or_replace={
            'subgoal_eval_num_samples': int(num_candidates),
            'subgoal_temperature': float(temperature), 'subgoal_eval_seed': 0})
        self.d = dynamics.replace(config=cfg)
        self.config = self.d.config

    def infer_subgoal_for_eval(self, o, g, critic_agent=None, rng=None):
        return _infer(self.d, critic_agent, o, g, rng)

    def plan(self, *args, **kwargs):
        return self.d.plan(*args, **kwargs)

    def _idm_actions_from_trajectories(self, trajectory, horizon):
        if horizon != 5:
            raise ValueError('The released policy executes five-step IDM chunks.')
        return _idm(self.d, trajectory)


@contextmanager
def stable_action_space():
    # OGBench ManipSpace creates a new unseeded Box on every property access.
    # Scope the cache patch to this single-process evaluation and restore it.
    from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv
    original = ManipSpaceEnv.action_space
    def cached(env):
        if '_pb_eval_action_space' not in env.__dict__:
            env.__dict__['_pb_eval_action_space'] = original.fget(env)
        return env.__dict__['_pb_eval_action_space']
    ManipSpaceEnv.action_space = property(cached)
    try:
        yield
    finally:
        ManipSpaceEnv.action_space = original


class RecordedEnv:
    def __init__(self, env):
        self.env = env
        self.records = []

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        index = len(self.records)
        self.env.action_space.seed(930000 + index)
        kwargs['seed'] = 730000 + index
        obs, info = self.env.reset(**kwargs)
        self.records.append(dict(episode=index, task_id=int(kwargs['options']['task_id']),
            reset_seed=730000+index, action_space_seed=930000+index,
            observation_sha256=hashlib.sha256(np.asarray(obs).tobytes()).hexdigest(),
            goal_sha256=hashlib.sha256(np.asarray(info['goal']).tobytes()).hexdigest(),
            success=False, steps=0))
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        self.records[-1]['steps'] += 1
        self.records[-1]['success'] |= bool(np.any(result[4].get('success', False)))
        return result


def evaluate(agent, env, *, task_ids=DEFAULT_TASK_IDS, episodes_per_task=25,
             num_candidates=32, temperature=1.0, seed=0, records=None):
    # A training seed must not change paired evaluation inputs. As before,
    # candidate key = episode index within each task, reused when replanning.
    del seed
    if episodes_per_task < 1 or num_candidates < 1 or not np.isfinite(temperature) or temperature < 0:
        raise ValueError('Require positive episode/candidate counts and finite nonnegative temperature.')
    if not task_ids:
        raise ValueError('task_ids cannot be empty.')
    recorded = RecordedEnv(env)
    with stable_action_space():
        result = _evaluate_env_tasks(recorded, EvalDynamics(agent.dynamics, num_candidates, temperature),
                                    None, {'actor_chunk_horizon': 5}, {'action_chunk_horizon': 5},
                                    critic_agent=agent.critic, task_ids=tuple(task_ids),
                                    episodes_per_task=episodes_per_task, idm_only=True)
    if records is not None:
        records.extend(recorded.records)
    return {**{f'task_{t}_success': result[f'eval_idm/task_{t}/success_rate'] for t in task_ids},
            'overall_success': result['eval_idm/success_rate_mean'],
            'num_tasks': len(task_ids), 'episodes_per_task': episodes_per_task}
