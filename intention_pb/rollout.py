"""Closed-loop OGBench evaluation for the intention experiment (success = any-step ``info['success']``)."""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import jax
import numpy as np

from intention_pb import ensure_pb_code_path

ensure_pb_code_path()

from utils.ogbench_eval_helpers import update_episode_env_success
from utils.ogbench_eval_rollout import _env_max_episode_steps

from intention_pb.common import (
    EVAL_TASK_IDS,
    INTENTION_METHODS,
    NUM_CANDIDATES,
    NUM_CODES,
    REF_METHOD,
    atomic_write_bytes,
    atomic_write_json,
    cond_ckpt,
    find_pb_run_dir,
    refuse_overwrite,
    task_info,
)


def env_reset_seed(task_id: int, ep_ix: int) -> int:
    return 1000 * int(task_id) + int(ep_ix)


def run_episode(env, planner, *, task_id: int, ep_ix: int, h_exec: int, eval_seed_base: int = 0) -> dict:
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    ob, info = env.reset(seed=env_reset_seed(task_id, ep_ix), options=dict(task_id=int(task_id), render_goal=False))
    if 'goal' not in info:
        raise RuntimeError(f'env.reset(task_id={task_id}) did not provide info["goal"].')
    obs = np.asarray(ob, np.float32).reshape(-1)
    goal = np.asarray(info['goal'], np.float32).reshape(-1)
    max_steps = _env_max_episode_steps(env)
    rng = jax.random.PRNGKey(int(eval_seed_base) + int(ep_ix))
    steps, ret, success = 0, 0.0, False
    terminated = truncated = False
    codes, exec_codes, scores = [], [], []
    while not (terminated or truncated) and steps < max_steps:
        out = jax.device_get({k: v for k, v in planner(obs, goal, rng).items() if k in ('actions', 'code', 'exec_code', 'score')})
        acts = np.asarray(out['actions'], np.float32)[: int(h_exec)]
        if not np.all(np.isfinite(acts)):
            raise FloatingPointError('Non-finite planned actions.')
        codes.append(int(out['code']))
        exec_codes.append(int(out['exec_code']))
        scores.append(float(out['score']))
        for a in acts:
            if terminated or truncated or steps >= max_steps:
                break
            ob, r, term, trunc, info = env.step(np.clip(a, low, high))
            obs = np.asarray(ob, np.float32).reshape(-1)
            steps += 1
            ret += float(r)
            terminated, truncated = bool(term), bool(trunc)
            success = update_episode_env_success(success, info)
    return dict(task_id=int(task_id), ep_ix=int(ep_ix), success=bool(success), episode_return=ret, length=steps,
                replans=len(codes), codes=codes, exec_codes=exec_codes, mean_score=float(np.mean(scores)) if scores else float('nan'))


def _entropy(counts: Counter, K: int) -> float:
    tot = sum(counts.values())
    if tot == 0:
        return float('nan')
    p = np.asarray([counts.get(k, 0) / tot for k in range(K)])
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def summarize_episodes(episodes: list[dict], *, num_codes: int = NUM_CODES, uses_codes: bool) -> dict[str, Any]:
    tasks = sorted({e['task_id'] for e in episodes})
    out: dict[str, Any] = dict(
        num_episodes=len(episodes),
        success_rate=float(np.mean([e['success'] for e in episodes])),
        mean_return=float(np.mean([e['episode_return'] for e in episodes])),
        mean_length=float(np.mean([e['length'] for e in episodes])),
        mean_replans=float(np.mean([e['replans'] for e in episodes])),
        mean_selected_score=float(np.nanmean([e['mean_score'] for e in episodes])),
        per_task_success={str(t): float(np.mean([e['success'] for e in episodes if e['task_id'] == t])) for t in tasks},
    )
    if uses_codes:
        all_codes = Counter(c for e in episodes for c in e['codes'])
        tot = sum(all_codes.values())
        out['code_histogram'] = [all_codes.get(k, 0) / tot for k in range(num_codes)]
        out['code_entropy_steps'] = _entropy(all_codes, num_codes)
        majority = [Counter(e['codes']).most_common(1)[0][0] for e in episodes if e['codes']]
        maj_c = Counter(majority)
        out['code_entropy_episodes'] = _entropy(maj_c, num_codes)
        out['code_entropy_max'] = math.log(num_codes)
        switches = [np.mean(np.diff(e['codes']) != 0) for e in episodes if len(e['codes']) > 1]
        out['switch_frequency'] = float(np.mean(switches)) if switches else float('nan')
        per_code = {}
        for k in range(num_codes):
            eps = [e for e, m in zip([e for e in episodes if e['codes']], majority) if m == k]
            per_code[str(k)] = dict(n=len(eps), success=float(np.mean([e['success'] for e in eps])) if eps else float('nan'))
        out['per_code_success_majority'] = per_code
        out['dominant_code_share'] = float(max(out['code_histogram']))
        per_task_hist = {}
        for t in tasks:
            c = Counter(cc for e in episodes if e['task_id'] == t for cc in e['codes'])
            s = sum(c.values())
            per_task_hist[str(t)] = [c.get(k, 0) / s for k in range(num_codes)]
        out['per_task_code_histogram'] = per_task_hist
    return out


def run_eval_job(*, task: str, seed: int, step: int, method: str, h_exec: int, episodes_per_task: int, out_dir: Path,
                 task_ids: tuple[int, ...] = EVAL_TASK_IDS, pb_step: int = 1_000_000) -> dict:
    from intention_pb.conditioned import load_conditioned
    from intention_pb.pb_io import load_pb
    from intention_pb.policy import make_planner

    out_dir = Path(out_dir)
    refuse_overwrite(out_dir / 'summary.json')
    info = task_info(task)
    run_dir = find_pb_run_dir(task, seed)
    pb = load_pb(run_dir, pb_step, need_train=False, need_env=True)
    if pb.env_name != info['env_name']:
        raise ValueError(f'PB run env {pb.env_name} != task env {info["env_name"]}')
    cond_model = cond_params = None
    ckpt = None
    if method in INTENTION_METHODS:
        ckpt = cond_ckpt(task, seed, step)
        cond_model, state, meta = load_conditioned(ckpt, pb.dynamics)
        if Path(meta['pb_run_dir']).resolve() != Path(run_dir).resolve():
            raise ValueError(f'Conditioned checkpoint was trained against {meta["pb_run_dir"]}, not {run_dir}')
        cond_params = state['params']
    if method == REF_METHOD:
        temperature, n = float(info['ref_temperature']), int(info['ref_num_candidates'])
    else:
        temperature, n = float(info['temperature']), NUM_CANDIDATES
    planner = make_planner(method, pb, temperature=temperature, num_candidates=n, cond_model=cond_model, cond_params=cond_params)
    eval_seed_base = int(pb.dynamics.config.get('subgoal_eval_seed', 0))
    t0 = time.time()
    episodes = []
    for tid in task_ids:
        for ep_ix in range(int(episodes_per_task)):
            ep = run_episode(pb.env, planner, task_id=tid, ep_ix=ep_ix, h_exec=h_exec, eval_seed_base=eval_seed_base)
            episodes.append(ep)
            print(f'[eval] {task} s{seed} step{step} {method} h{h_exec} task{tid} ep{ep_ix} success={ep["success"]} '
                  f'len={ep["length"]} replans={ep["replans"]}', flush=True)
    summary = summarize_episodes(episodes, uses_codes=method in INTENTION_METHODS)
    summary.update(task=task, seed=int(seed), step=int(step), method=method, h_exec=int(h_exec), temperature=temperature,
                   num_candidates=n, episodes_per_task=int(episodes_per_task), task_ids=list(task_ids),
                   pb_run_dir=str(run_dir), conditioned_ckpt=str(ckpt) if ckpt else None, runtime_s=time.time() - t0,
                   env_name=pb.env_name)
    atomic_write_bytes(out_dir / 'episodes.jsonl', ''.join(json.dumps(e) + '\n' for e in episodes).encode())
    atomic_write_json(out_dir / 'summary.json', summary)
    return summary
