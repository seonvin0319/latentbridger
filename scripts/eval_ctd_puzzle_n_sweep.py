"""Candidate-count sweep on finished puzzle-3x3 1M CTD checkpoints."""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import os

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MUJOCO_GL', 'egl')

import jax
import jax.numpy as jnp
import numpy as np

from agents.contrastive_transitive_distance_pathbridger import ContrastiveTransitiveDistanceAgent
from envs.env_utils import make_env_and_datasets
from utils.contrastive_pathbridger_evaluation import evaluate
from utils.flax_utils import restore_agent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'exp' / 'ctd_pathbridger'
METHODS = ('dtrl_weighted', 'ctd_weighted', 'ctd_pathnce_weighted')
NS = (1, 4, 8, 16, 32)
STEP = 1_000_000


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def candidate_stats(agent, batch, num_candidates: int):
    observations = jnp.asarray(batch['observations'])
    goals = jnp.asarray(batch['value_goals'])
    reference = np.asarray(batch['reference_states'])
    temperature = float(agent.config['eval_temperature'])
    candidates = np.asarray(agent._sample_endpoint_candidates(
        observations,
        goals,
        jax.random.PRNGKey(73521),
        num_candidates=num_candidates,
        temperature=temperature,
    ))
    n = num_candidates
    if n == 1:
        selected = candidates[:, 0]
        pairwise = np.zeros(len(candidates), dtype=np.float32)
        scores = np.asarray(
            agent._metric_distance(observations, jnp.asarray(selected), name='value')
            + agent._metric_distance(jnp.asarray(selected), goals, name='value')
        )
        margin = np.zeros(len(candidates), dtype=np.float32)
    else:
        diff = candidates[:, :, None, :] - candidates[:, None, :, :]
        pairwise = np.linalg.norm(diff, axis=-1).sum(axis=(1, 2)) / (n * (n - 1))
        flat_s = jnp.broadcast_to(observations[:, None, :], candidates.shape).reshape(-1, candidates.shape[-1])
        flat_z = jnp.asarray(candidates.reshape(-1, candidates.shape[-1]))
        flat_g = jnp.broadcast_to(goals[:, None, :], (len(goals), n, goals.shape[-1])).reshape(-1, goals.shape[-1])
        scores = np.asarray(
            agent._metric_distance(flat_s, flat_z, name='value')
            + agent._metric_distance(flat_z, flat_g, name='value')
        ).reshape(len(goals), n)
        order = np.argsort(scores, axis=1)
        selected = candidates[np.arange(len(candidates)), order[:, 0]]
        margin = scores[np.arange(len(scores)), order[:, 1]] - scores[np.arange(len(scores)), order[:, 0]]
        scores = scores[np.arange(len(scores)), order[:, 0]]
    nearest = []
    chunk = 256
    flat_sel = selected.astype(np.float32)
    for start in range(0, len(flat_sel), chunk):
        piece = flat_sel[start:start + chunk]
        distances = np.linalg.norm(piece[:, None, :] - reference[None, :, :], axis=-1)
        nearest.append(distances.min(axis=1))
    nearest = np.concatenate(nearest)
    return {
        'candidate_diversity': float(np.mean(pairwise)),
        'selected_path_cost': float(np.mean(scores)),
        'top1_top2_path_cost_margin': float(np.mean(margin)),
        'selected_nearest_data_distance': float(np.mean(nearest)),
    }


def main():
    config = importlib.import_module('configs.ctd.puzzle_3x3').get_config('ctd_weighted')
    env, _, val = make_env_and_datasets(config.env_name)
    rows = []
    try:
        for method in METHODS:
            run = OUT / 'puzzle_3x3' / method / 'seed0'
            checkpoint = run / 'checkpoints' / f'params_{STEP}.pkl'
            complete = run / 'complete.json'
            if not checkpoint.exists() or not complete.exists():
                print(f'skip missing {method}', flush=True)
                continue
            diagnostic_path = run / 'diagnostic_batch_seed92831.npz'
            if not diagnostic_path.exists():
                raise FileNotFoundError(diagnostic_path)
            batch = {key: np.asarray(value) for key, value in np.load(diagnostic_path).items()}
            agent = ContrastiveTransitiveDistanceAgent.create(
                0,
                batch['observations'][:2],
                batch['actions'][:2],
                importlib.import_module('configs.ctd.puzzle_3x3').get_config(method).to_dict(),
            )
            agent = restore_agent(agent, checkpoint)
            manifest = json.loads((run / 'evaluation_manifest.json').read_text())
            for n in NS:
                out_path = run / f'evaluation_{STEP}_h5_N{n}.json'
                if out_path.exists():
                    result = json.loads(out_path.read_text())
                else:
                    result = evaluate(
                        agent,
                        env,
                        episodes_per_task=50,
                        num_candidates=n,
                        temperature=float(agent.config['eval_temperature']),
                        seed=0,
                        execute_h=5,
                    )
                    result.update(
                        env='puzzle_3x3',
                        variant=method,
                        seed=0,
                        checkpoint=STEP,
                        N=n,
                        h=5,
                        paired_manifest=manifest.get('commit') or manifest.get('seed'),
                        method='ctd_pathbridger_n_sweep',
                    )
                    write_json(out_path, result)
                stats = candidate_stats(agent, batch, n)
                row = {
                    'method': method,
                    'N': n,
                    'success': result['overall_success'],
                    **{f'task{i}': result[f'task_{i}_success'] for i in range(1, 6)},
                    **stats,
                    'provenance': str(out_path),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        env.close()

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / 'puzzle_n_sweep.csv'
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ['method'])
        writer.writeheader()
        writer.writerows(rows)
    write_json(OUT / 'puzzle_n_sweep.json', rows)
    print(f'wrote {csv_path}', flush=True)


if __name__ == '__main__':
    main()
