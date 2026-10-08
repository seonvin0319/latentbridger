"""Evaluate N=1,4,8,16,32 for every finished 1M puzzle goal-space model."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MUJOCO_GL', 'egl')

import jax
import jax.numpy as jnp
import numpy as np

from agents.contrastive_transitive_distance_pathbridger import ContrastiveTransitiveDistanceAgent
from configs.gsctd.puzzle_3x3 import get_config
from envs.env_utils import make_env_and_datasets
from utils.contrastive_pathbridger_evaluation import evaluate
from utils.ctd_diagnostics import _nearest
from utils.flax_utils import restore_agent

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'exp' / 'goalspace_transitive_distance'
METHODS = ('gsdtrl_weighted', 'gsctd_learned_temp', 'gsctd_fixed')
NS = (1, 4, 8, 16, 32)
STEP = 1_000_000


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def candidate_stats(agent, batch, num_candidates: int):
    observations = jnp.asarray(batch['observations'])
    goals = jnp.asarray(batch['value_goals'])
    candidates = np.asarray(agent._sample_endpoint_candidates(
        observations,
        goals,
        jax.random.PRNGKey(73521),
        num_candidates=num_candidates,
        temperature=float(agent.config['eval_temperature']),
    ))
    count = num_candidates
    if count == 1:
        selected = candidates[:, 0]
        pairwise = np.zeros(len(candidates), dtype=np.float32)
        scores = np.asarray(
            agent._metric_distance(observations, jnp.asarray(selected), name='value')
            + agent._metric_distance(jnp.asarray(selected), goals, name='value')
        )
        margin = np.zeros(len(candidates), dtype=np.float32)
    else:
        diff = candidates[:, :, None, :] - candidates[:, None, :, :]
        pairwise = np.linalg.norm(diff, axis=-1).sum(axis=(1, 2)) / (count * (count - 1))
        flat_s = jnp.broadcast_to(observations[:, None, :], candidates.shape).reshape(-1, candidates.shape[-1])
        flat_z = jnp.asarray(candidates.reshape(-1, candidates.shape[-1]))
        flat_g = jnp.broadcast_to(
            goals[:, None, :], (len(goals), count, goals.shape[-1])
        ).reshape(-1, goals.shape[-1])
        all_scores = np.asarray(
            agent._metric_distance(flat_s, flat_z, name='value')
            + agent._metric_distance(flat_z, flat_g, name='value')
        ).reshape(len(goals), count)
        order = np.argsort(all_scores, axis=1)
        selected = candidates[np.arange(len(candidates)), order[:, 0]]
        scores = all_scores[np.arange(len(all_scores)), order[:, 0]]
        margin = (
            all_scores[np.arange(len(all_scores)), order[:, 1]]
            - all_scores[np.arange(len(all_scores)), order[:, 0]]
        )
    nearest = _nearest(selected, np.asarray(batch['reference_states']))
    return {
        'candidate_diversity': float(np.mean(pairwise)),
        'selected_path_cost': float(np.mean(scores)),
        'top1_top2_path_cost_margin': float(np.mean(margin)),
        'selected_nearest_data_distance': float(np.mean(nearest)),
    }


def main():
    config = get_config('gsdtrl_weighted')
    env, _, _ = make_env_and_datasets(config.env_name)
    rows = []
    try:
        for method in METHODS:
            run = OUT / 'puzzle_3x3' / method / 'seed0'
            checkpoint = run / 'checkpoints' / f'params_{STEP}.pkl'
            if not checkpoint.exists() or not (run / 'complete.json').exists():
                raise FileNotFoundError(f'Incomplete required puzzle model: {run}')
            batch = {
                key: np.asarray(value)
                for key, value in np.load(run / 'diagnostic_batch_seed92831.npz').items()
            }
            agent = ContrastiveTransitiveDistanceAgent.create(
                0,
                batch['observations'][:2],
                batch['actions'][:2],
                get_config(method).to_dict(),
            )
            agent = restore_agent(agent, checkpoint)
            for count in NS:
                path = run / f'evaluation_{STEP}_h5_N{count}.json'
                if path.exists():
                    result = json.loads(path.read_text())
                else:
                    result = evaluate(
                        agent,
                        env,
                        episodes_per_task=50,
                        num_candidates=count,
                        temperature=float(agent.config['eval_temperature']),
                        seed=0,
                        execute_h=5,
                    )
                    result.update(
                        env='puzzle_3x3', variant=method, seed=0,
                        checkpoint=STEP, N=count, h=5,
                        method='goalspace_puzzle_n_sweep',
                    )
                    write_json(path, result)
                row = {
                    'method': method,
                    'N': count,
                    'success': result['overall_success'],
                    **{f'task{i}': result[f'task_{i}_success'] for i in range(1, 6)},
                    **candidate_stats(agent, batch, count),
                    'provenance': str(path.relative_to(ROOT)),
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    finally:
        env.close()

    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / 'puzzle_n_sweep.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    write_json(OUT / 'puzzle_n_sweep.json', rows)


if __name__ == '__main__':
    main()
