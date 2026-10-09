"""Held-out subgoal variance reduction for a frozen route tokenizer.

``z`` is PathBridger's own subgoal target (``high_actor_targets``), and ``g`` is the goal
PathBridger samples. The route code is assigned from the longer window that starts at the
same ``t`` and does not cross the episode boundary.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from intention_pb.common import find_pb_run_dir
from intention_pb.pb_io import load_pb

from route_intention.chunks import valid_route_starts
from route_intention.common import (
    H_ROUTE,
    KNN_DEFAULT,
    KNN_SENSITIVITY,
    LOCAL_SUBGOAL_VARIANCE_RATIO,
    TOKENIZER_STEPS,
    atomic_write_json,
    diagnostic_dir,
    tokenizer_dir,
)
from route_intention.tokenizer import assign_codes, load_route_tokenizer
from route_intention.variance import between_within, variance_report


def _sample_pb(pb, num: int, seed: int) -> dict:
    from utils.datasets import Dataset, PathHGCDataset

    ds = PathHGCDataset(Dataset.create(**pb.val), pb.dynamics_config)
    rs = np.random.get_state()
    np.random.seed(int(seed) + 12345)
    batch = ds.sample(int(num))
    np.random.set_state(rs)
    return dict(
        obs=np.asarray(batch['observations'], np.float32),
        goal=np.asarray(batch['high_actor_goals'], np.float32),
        target=np.asarray(batch['high_actor_targets'], np.float32),
        starts=np.asarray(batch['trajectory_start_indices'], np.int64),
    )


def run_variance_diagnostic(*, task: str, seed: int, num_samples: int = 4096, step: int = TOKENIZER_STEPS) -> dict:
    out_dir = diagnostic_dir(task, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = tokenizer_dir(task, seed) / 'checkpoints' / f'tokenizer_{int(step)}.pkl'
    tok, state, stats, _ = load_route_tokenizer(ckpt)
    if int(tok.cfg.horizon) != H_ROUTE:
        raise ValueError(f'Route tokenizer horizon {tok.cfg.horizon} != H_ROUTE {H_ROUTE}')
    pb = load_pb(find_pb_run_dir(task, seed), need_train=False, need_env=False)
    sampled = _sample_pb(pb, num_samples * 2, seed)
    ok = np.zeros(len(pb.val['terminals']), dtype=bool)
    ok[valid_route_starts(pb.val['terminals'], tok.cfg.horizon)] = True
    keep = ok[sampled['starts']]
    if int(keep.sum()) < 256:
        raise RuntimeError(f'Only {int(keep.sum())} held-out windows fit H_route={tok.cfg.horizon}')
    for key in ('obs', 'goal', 'target', 'starts'):
        sampled[key] = sampled[key][keep][:num_samples]
    obs_all = np.asarray(pb.val['observations'], np.float32)
    act_all = np.asarray(pb.val['actions'], np.float32)
    codes = assign_codes(tok, state['params'], state['codebook'], obs_all, act_all, sampled['starts'], stats)
    rel = sampled['target'] - sampled['obs']
    z_std = np.maximum(rel.std(0), 1e-3)
    values = rel / z_std
    feats = np.concatenate([sampled['obs'], sampled['goal']], 1)
    feats = (feats - feats.mean(0)) / np.maximum(feats.std(0), 1e-3)
    rows = variance_report(values, feats, codes, KNN_SENSITIVITY)
    by_k = {int(r['knn_k']): r for r in rows}
    decision = by_k[KNN_DEFAULT]
    # Trajectory geometry used for separability: the route endpoint, not the subgoal.
    from route_intention.chunks import window_views

    obs_win, _act = window_views(obs_all, act_all, sampled['starts'], tok.cfg.horizon)
    endpoint = obs_win[:, -1] - obs_win[:, 0]
    sep = between_within(endpoint, codes)
    # Mean pairwise subgoal distance inside vs across codes (global, not kNN).
    z_sep = between_within(values, codes)
    usage = np.bincount(codes, minlength=tok.cfg.num_codes).astype(np.float64)
    usage /= max(len(codes), 1)
    result = dict(
        task=task,
        seed=int(seed),
        step=int(step),
        num_samples=int(len(codes)),
        horizon=int(tok.cfg.horizon),
        knn_default=KNN_DEFAULT,
        variance_ratio=decision['variance_ratio'],
        var_uncond=decision['var_uncond'],
        var_cond=decision['var_cond'],
        local_variance_ratio=LOCAL_SUBGOAL_VARIANCE_RATIO.get(task),
        sensitivity=rows,
        between_route_endpoint=sep['between'],
        within_route_endpoint=sep['within'],
        between_subgoal=z_sep['between'],
        within_subgoal=z_sep['within'],
        usage=[float(u) for u in usage],
        dominant_code=float(usage.max()),
    )
    atomic_write_json(out_dir / 'variance.json', result)
    _rewrite_variance_csv()
    print(f"[route-diag] {task} seed{seed} ratio@{KNN_DEFAULT}={result['variance_ratio']:.3f} "
          f"(local seed0 {result['local_variance_ratio']})", flush=True)
    return result


def _rewrite_variance_csv() -> None:
    from route_intention.common import EXP_ROOT

    root = EXP_ROOT / 'diagnostics'
    rows = []
    for path in sorted(root.glob('*/variance.json')):
        rows.append(__import__('json').loads(path.read_text()))
    dest = root / 'subgoal_variance.csv'
    dest.parent.mkdir(parents=True, exist_ok=True)
    fields = ['task', 'seed', 'knn_k', 'variance_ratio', 'var_uncond', 'var_cond', 'local_variance_ratio',
              'between_route_endpoint', 'within_route_endpoint', 'between_subgoal', 'within_subgoal', 'dominant_code']
    with dest.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            for sens in row['sensitivity']:
                w.writerow(dict(
                    task=row['task'], seed=row['seed'], knn_k=sens['knn_k'],
                    variance_ratio=sens['variance_ratio'], var_uncond=sens['var_uncond'], var_cond=sens['var_cond'],
                    local_variance_ratio=row['local_variance_ratio'],
                    between_route_endpoint=row['between_route_endpoint'],
                    within_route_endpoint=row['within_route_endpoint'],
                    between_subgoal=row['between_subgoal'], within_subgoal=row['within_subgoal'],
                    dominant_code=row['dominant_code'],
                ))
