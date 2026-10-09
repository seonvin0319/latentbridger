"""Planners for PB / PB-ref / I-SG / Shared-I / Shuffled-I sharing the frozen PB critic score and IDM."""

from __future__ import annotations

import flax
import jax
import jax.numpy as jnp

from intention_pb.common import INTENTION_METHODS, NUM_CANDIDATES, REF_METHOD


def shuffled_code(c, num_codes: int):
    """Deterministic non-matching intention ``c' = (c + 1) mod K``."""
    return (jnp.asarray(c, jnp.int32) + 1) % int(num_codes)


def pb_with_temperature(dyn, temperature: float):
    """PB dynamics agent whose flow-noise temperature is overridden (same as the eval-time YAML override)."""
    cfg = dict(dyn.config)
    cfg['subgoal_temperature'] = float(temperature)
    return dyn.replace(config=flax.core.FrozenDict(cfg))


def make_planner(method: str, pb, *, temperature: float, num_candidates: int = NUM_CANDIDATES,
                 cond_model=None, cond_params=None):
    """Return a jitted ``plan(obs[D], goal[D], rng) -> dict`` for one method.

    Output keys: ``actions`` [h, A] (PB IDM chunk), ``code`` (selected candidate code, -1 for PB),
    ``exec_code`` (code fed to the bridge, -1 if the PB bridge is used), ``score`` (selected PB score),
    ``subgoal`` [D], ``scores`` [N], ``cand_codes`` [N], ``trajectory`` [K+1, D].
    """
    dyn = pb_with_temperature(pb.dynamics, temperature)
    if bool(dyn.config.get('subgoal_eval_include_zero_candidate', False)):
        raise ValueError('PB config includes a zero-noise candidate; equal-budget accounting assumes it does not.')
    critic = pb.critic
    critic_params = critic.network.params
    idm_h = int(pb.idm_horizon)
    n = int(num_candidates)
    if method in INTENTION_METHODS:
        if cond_model is None or cond_params is None:
            raise ValueError(f'{method} requires a conditioned model.')
        K = cond_model.num_codes
        if n % K != 0:
            raise ValueError(f'Candidate budget {n} not divisible by K={K}.')
        per_code = n // K
    elif method not in ('PB', REF_METHOD):
        raise ValueError(f'Unknown method {method!r}')

    def plan(obs, goal, rng):
        obs1 = jnp.asarray(obs, jnp.float32)[None]
        goal1 = jnp.asarray(goal, jnp.float32)[None]
        if method in ('PB', REF_METHOD):
            cands, _ = dyn.sample_subgoal_candidates(obs1, goal1, rng, num_candidates=n, include_mean=False)
            cand_codes = -jnp.ones((n,), jnp.int32)
        else:
            cands, cand_codes = cond_model.sample_candidates(
                cond_params['subgoal'], obs1, goal1, rng, temperature=float(temperature), per_code=per_code,
            )
        scores = critic.score_transitive_subgoals(obs1, cands, goal1, network_params=critic_params)[0]
        j = jnp.argmax(scores)
        z = cands[0, j]
        code = cand_codes[j]
        if method in ('PB', REF_METHOD, 'I-SG'):
            traj = dyn.plan(obs1, z[None])['trajectory']
            exec_code = -jnp.ones((), jnp.int32)
        else:
            exec_code = code if method == 'Shared-I' else shuffled_code(code, cond_model.num_codes)
            traj = cond_model.bridge(cond_params['residual'], obs1, z[None], exec_code[None])
        actions = dyn._idm_actions_from_trajectories(traj, idm_h)[0]
        return dict(actions=actions, code=code, exec_code=exec_code, score=scores[j], subgoal=z,
                    scores=scores, cand_codes=cand_codes, trajectory=traj[0])

    return jax.jit(plan)
