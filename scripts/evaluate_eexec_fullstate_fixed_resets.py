"""Evaluate frozen 400k full-state role-cross checkpoints with fixed resets.

This script performs no training.  It fixes PuzzleEnv's otherwise freshly
constructed/unseeded action space so that the five goal-stabilization actions
inside reset are a deterministic function of the manifest ``env_seed``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import ogbench

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))

from agents.fullstate_role_pathbridger import FullStateRolePathBridgerAgent  # noqa: E402
from configs.pbf.puzzle_3x3 import get_config  # noqa: E402
from eexec_fullstate_lib import (  # noqa: E402
    ARMS,
    ENV_NAME,
    OUT,
    TASKS,
    load_frozen_representations,
    load_full_train,
    load_norm,
    normalize,
    write_json,
)
from utils.datasets import Dataset, PathBridgerDataset  # noqa: E402
from utils.flax_utils import restore_agent  # noqa: E402


STEP = 400_000
HORIZON = 5
REUSED_PCA_SUMMARY = (
    OUT / 'diagnostics/pca_pca_400k_seeded_reset_counterfactual/summary.json'
)


def observation_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    return hashlib.sha256(array.tobytes()).hexdigest()


def make_env():
    env = ogbench.make_env_and_datasets(ENV_NAME, env_only=True)
    fixed_action_space = gym.spaces.Box(
        low=-np.ones(5), high=np.ones(5), shape=(5,), dtype=np.float32,
    )
    env.unwrapped._fixed_eval_action_space = fixed_action_space
    type(env.unwrapped).action_space = property(
        lambda self: self._fixed_eval_action_space,
    )
    return env, fixed_action_space


def make_agent(arm: str, example, encoder_params, pca_matrix):
    source_kind, goal_kind = ARMS[arm]
    agent = FullStateRolePathBridgerAgent.create(
        0,
        example['observations'],
        example['actions'],
        get_config().to_dict(),
        source_kind=source_kind,
        goal_kind=goal_kind,
        eexec_encoder_params=encoder_params,
        pca_matrix=pca_matrix,
    )
    checkpoint = OUT / 'downstream' / arm / 'checkpoints' / f'params_{STEP}.pkl'
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return restore_agent(agent, checkpoint), checkpoint


def evaluate(arm, agent, checkpoint, manifest, mean, std, env, fixed_action_space):
    low = np.asarray(env.action_space.low, dtype=np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    successes = {task: [] for task in TASKS}
    goal_rows = []
    episode_rows = []
    started = time.time()
    for index, row in enumerate(manifest):
        task_id = int(row['task_id'])
        env_seed = int(row['env_seed'])
        np.random.seed(env_seed)
        fixed_action_space.seed(env_seed)
        observation, info = env.reset(
            seed=env_seed,
            options={'task_id': task_id, 'render_goal': False},
        )
        observation = np.asarray(observation, dtype=np.float32)
        goal = np.asarray(info['goal'], dtype=np.float32)
        goal_rows.append({
            'manifest_index': index,
            'task_id': task_id,
            'episode': int(row['episode']),
            'env_seed': env_seed,
            'initial_observation_sha256': observation_hash(observation),
            'goal_observation_sha256': observation_hash(goal),
        })
        normalized_goal = normalize(goal[None, :], mean, std)
        rng = jax.random.PRNGKey(HORIZON * 1_000_003 + env_seed)
        success = False
        step = 0
        max_steps = int(env.spec.max_episode_steps)
        while step < max_steps:
            rng, sample_seed = jax.random.split(rng)
            normalized_observation = normalize(observation[None, :], mean, std)
            chunk = np.asarray(agent.sample_action_chunks(
                observations=jnp.asarray(normalized_observation),
                goals=jnp.asarray(normalized_goal),
                num_candidates=int(agent.config['eval_num_candidates']),
                temperature=float(agent.config['eval_temperature']),
                seed=sample_seed,
            ))[0]
            terminated = truncated = False
            for action in chunk[:HORIZON]:
                if step >= max_steps:
                    break
                observation, _reward, terminated, truncated, info = env.step(
                    np.clip(action, low, high),
                )
                observation = np.asarray(observation, dtype=np.float32)
                success = success or bool(
                    np.any(np.asarray(info.get('success', False)))
                )
                step += 1
                if terminated or truncated:
                    break
            if terminated or truncated:
                break
        successes[task_id].append(float(success))
        episode_rows.append({
            'manifest_index': index,
            'task_id': task_id,
            'episode': int(row['episode']),
            'env_seed': env_seed,
            'success': bool(success),
            'steps': step,
        })
        if (index + 1) % 25 == 0:
            print(json.dumps({
                'arm': arm,
                'episodes_complete': index + 1,
                'successes': int(sum(sum(v) for v in successes.values())),
                'wall_seconds': time.time() - started,
            }), flush=True)

    task_rates = {
        str(task): float(np.mean(successes[task])) for task in TASKS
    }
    result = {
        'status': 'complete',
        'arm': arm,
        'source_representation': ARMS[arm][0],
        'goal_representation': ARMS[arm][1],
        'execution_representation': 'full_state',
        'step': STEP,
        'training_updates': 0,
        'checkpoint': str(checkpoint),
        'reset_manifest': str(OUT / 'eval_resets_final.json'),
        'reset_protocol': 'persistent_action_space_seeded_with_env_seed',
        'same_planner_rng_schedule_as_primary': True,
        'h': HORIZON,
        'episodes_per_task': len(manifest) // len(TASKS),
        'task_success': task_rates,
        'overall_success': float(np.mean(list(task_rates.values()))),
        'success_count': int(sum(sum(v) for v in successes.values())),
        'goal_observation_manifest_sha256': observation_hashes_digest(goal_rows),
        'episodes': episode_rows,
    }
    return result, goal_rows


def observation_hashes_digest(rows: list[dict]) -> str:
    payload = '\n'.join(
        f"{row['env_seed']}:{row['initial_observation_sha256']}:{row['goal_observation_sha256']}"
        for row in rows
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def reuse_pca(manifest: list[dict]) -> dict:
    payload = json.loads(REUSED_PCA_SUMMARY.read_text())
    metrics = payload['metrics']
    task_rates = {
        str(task): float(metrics[f'task_{task}']['success_rate']) for task in TASKS
    }
    result = {
        'status': 'complete',
        'arm': 'pca_pca',
        'source_representation': 'pca',
        'goal_representation': 'pca',
        'execution_representation': 'full_state',
        'step': STEP,
        'training_updates': 0,
        'checkpoint': payload['checkpoint'],
        'reset_manifest': str(OUT / 'eval_resets_final.json'),
        'reset_protocol': 'persistent_action_space_seeded_with_env_seed',
        'same_planner_rng_schedule_as_primary': True,
        'h': HORIZON,
        'episodes_per_task': len(manifest) // len(TASKS),
        'task_success': task_rates,
        'overall_success': float(metrics['overall']['success_rate']),
        'success_count': int(round(
            float(metrics['overall']['success_rate']) * len(manifest)
        )),
        'reused_from': str(REUSED_PCA_SUMMARY),
        'reuse_basis': {
            'same_checkpoint': True,
            'same_reset_manifest': bool(payload['same_primary_resets']),
            'same_planner_rng_schedule': bool(payload['same_planner_rng_schedule']),
            'same_goal_stabilization_rng': (
                payload['goal_stabilization_action_rng']
                == 'deterministically seeded with env_seed'
            ),
        },
    }
    if not all(result['reuse_basis'].values()):
        raise RuntimeError('PCA diagnostic does not satisfy fixed-reset manifest')
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--arms', nargs='+', default=['eexec_eexec', 'pca_eexec', 'eexec_pca'],
        choices=tuple(ARMS),
    )
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    manifest = json.loads((OUT / 'eval_resets_final.json').read_text())
    pca_result = reuse_pca(manifest)
    pca_path = OUT / 'downstream/pca_pca/evaluation_400000_h5_fixed_reset.json'
    write_json(pca_path, pca_result)

    observations, actions, terminals = load_full_train()
    dataset = PathBridgerDataset(
        Dataset.create(observations=observations, actions=actions, terminals=terminals),
        get_config(),
    )
    np.random.seed(0)
    example = dataset.sample(2)
    encoder_params, pca_matrix = load_frozen_representations()
    mean, std, _ = load_norm()
    expected_digest = None
    completed = {'pca_pca': pca_result}
    shared_goal_rows = None
    env, fixed_action_space = make_env()
    try:
        for arm in args.arms:
            if arm == 'pca_pca':
                continue
            output = OUT / 'downstream' / arm / 'evaluation_400000_h5_fixed_reset.json'
            hashes_path = OUT / 'downstream' / arm / 'fixed_reset_observation_hashes.json'
            if output.is_file() and hashes_path.is_file() and not args.force:
                result = json.loads(output.read_text())
                goal_rows = json.loads(hashes_path.read_text())
                print(f'reuse {output}', flush=True)
            else:
                agent, checkpoint = make_agent(arm, example, encoder_params, pca_matrix)
                result, goal_rows = evaluate(
                    arm, agent, checkpoint, manifest, mean, std, env,
                    fixed_action_space,
                )
                write_json(output, result)
                write_json(hashes_path, goal_rows)
            digest = observation_hashes_digest(goal_rows)
            if digest != result['goal_observation_manifest_sha256']:
                raise RuntimeError(f'{arm}: stored goal hash digest mismatch')
            if expected_digest is None:
                expected_digest = digest
                shared_goal_rows = goal_rows
            elif digest != expected_digest:
                raise RuntimeError(f'{arm}: goal observations differ across arms')
            completed[arm] = result
    finally:
        env.close()

    if shared_goal_rows is not None:
        shared = {
            'reset_manifest': str(OUT / 'eval_resets_final.json'),
            'reset_protocol': 'persistent_action_space_seeded_with_env_seed',
            'episodes': len(shared_goal_rows),
            'observation_manifest_sha256': expected_digest,
            'verified_arms': [arm for arm in completed if arm != 'pca_pca'],
            'pca_reused_under_identical_protocol': True,
        }
        write_json(OUT / 'fixed_reset_observation_manifest.json', shared)

    if set(completed) == set(ARMS):
        comparison = {
            'status': 'complete',
            'training_updates': 0,
            'episodes_total': len(manifest) * len(ARMS),
            'episodes_new': len(manifest) * (len(ARMS) - 1),
            'pca_episodes_reused': len(manifest),
            'primary_historical_pca_success': 0.692,
            'fixed_reset_success': {
                arm: completed[arm]['overall_success'] for arm in ARMS
            },
            'goal_observation_manifest_sha256': expected_digest,
        }
        write_json(OUT / 'fixed_reset_400k_h5_comparison.json', comparison)
        print(json.dumps(comparison, indent=2), flush=True)


if __name__ == '__main__':
    main()
