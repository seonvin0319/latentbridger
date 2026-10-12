"""Train one fresh full-state PathBridger role-cross arm through 400k."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import ogbench

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/home/choi/latentbridger-goalspace')

from agents.fullstate_role_pathbridger import FullStateRolePathBridgerAgent  # noqa: E402
from configs.pbf.puzzle_3x3 import get_config  # noqa: E402
from eexec_fullstate_lib import (  # noqa: E402
    ARMS,
    BATCH_SIZE,
    DOWNSTREAM_SEED,
    DOWNSTREAM_STEPS,
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
from utils.flax_utils import restore_agent, save_agent  # noqa: E402

os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')


def evaluate(
    agent: FullStateRolePathBridgerAgent,
    env,
    manifest: list[dict],
    horizon: int,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
) -> dict:
    if mean is None or std is None:
        mean, std, _components = load_norm()
    successes = {task: [] for task in TASKS}
    for row in manifest:
        env_seed = int(row['env_seed'])
        np.random.seed(env_seed)
        observation, info = env.reset(
            seed=env_seed,
            options={'task_id': int(row['task_id']), 'render_goal': False},
        )
        goal = np.asarray(info['goal'], dtype=np.float32)
        observation = np.asarray(observation, dtype=np.float32)
        normalized_goal = normalize(goal.reshape(1, -1), mean, std)
        rng = jax.random.PRNGKey(horizon * 1_000_003 + env_seed)
        success = False
        step = 0
        max_steps = int(env.spec.max_episode_steps)
        low = np.asarray(env.action_space.low, dtype=np.float32)
        high = np.asarray(env.action_space.high, dtype=np.float32)
        while step < max_steps:
            rng, sample_seed = jax.random.split(rng)
            normalized_observation = normalize(
                observation.reshape(1, -1), mean, std,
            )
            chunk = agent.sample_action_chunks(
                observations=jnp.asarray(normalized_observation),
                goals=jnp.asarray(normalized_goal),
                num_candidates=int(agent.config['eval_num_candidates']),
                temperature=float(agent.config['eval_temperature']),
                seed=sample_seed,
            )
            chunk = np.asarray(chunk)[0]
            for action in chunk[:horizon]:
                if step >= max_steps:
                    break
                observation, _reward, terminated, truncated, info = env.step(
                    np.clip(action, low, high),
                )
                observation = np.asarray(observation, dtype=np.float32)
                success = success or bool(np.any(np.asarray(info.get('success', False))))
                step += 1
                if terminated or truncated:
                    break
            if terminated or truncated:
                break
        successes[int(row['task_id'])].append(float(success))
    metrics = {'h': horizon, 'episodes_per_task': len(manifest) // len(TASKS)}
    rates = []
    for task in TASKS:
        rate = float(np.mean(successes[task]))
        metrics[f'task_{task}_success'] = rate
        rates.append(rate)
    metrics['overall_success'] = float(np.mean(rates))
    metrics['success_count'] = int(sum(sum(values) for values in successes.values()))
    return metrics


def train_arm(
    arm: str,
    *,
    seed: int = DOWNSTREAM_SEED,
    max_updates: int = DOWNSTREAM_STEPS,
    checkpoint_steps: tuple[int, ...] = (100_000, 200_000, 400_000),
    output_dir: Path | None = None,
    observations: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    terminals: np.ndarray | None = None,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    pca_matrix: np.ndarray | None = None,
    eexec_encoder_params=None,
    dev_manifest: list[dict] | None = None,
    final_manifest: list[dict] | None = None,
    require_protocol: bool = True,
) -> None:
    """Train one full-state role arm. Defaults reproduce the seed-0 400k run."""

    if require_protocol and not (OUT / 'protocol.json').is_file():
        raise RuntimeError('full-state cross requires a prepared protocol')
    source_kind, goal_kind = ARMS[arm]
    run_dir = Path(output_dir) if output_dir is not None else OUT / 'downstream' / arm
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'complete.json').is_file():
        print(f'skip {arm}', flush=True)
        return

    if observations is None:
        observations, actions, terminals = load_full_train()
    if pca_matrix is None or eexec_encoder_params is None:
        loaded_encoder, loaded_pca = load_frozen_representations()
        if eexec_encoder_params is None:
            eexec_encoder_params = loaded_encoder
        if pca_matrix is None:
            pca_matrix = loaded_pca
    dataset = PathBridgerDataset(
        Dataset.create(
            observations=observations,
            actions=actions,
            terminals=terminals,
        ),
        get_config(),
    )
    config = get_config().to_dict()
    if int(config['eval_num_candidates']) != 32:
        raise RuntimeError('full-state PCA/PCA requires candidate_count 32')
    random.seed(int(seed))
    np.random.seed(int(seed))
    example = dataset.sample(2)
    agent = FullStateRolePathBridgerAgent.create(
        int(seed),
        example['observations'],
        example['actions'],
        config,
        source_kind=source_kind,
        goal_kind=goal_kind,
        eexec_encoder_params=eexec_encoder_params,
        pca_matrix=pca_matrix,
    )
    checkpoints = sorted(
        (run_dir / 'checkpoints').glob('params_*.pkl'),
    ) if (run_dir / 'checkpoints').is_dir() else []
    start = 0
    if checkpoints:
        agent = restore_agent(agent, checkpoints[-1])
        start = int(agent.network.step) - 1

    bridge_kernel = agent.network.params['modules_bridge']['MLP_0']['Dense_0']['kernel']
    bridge_output = agent.network.params['modules_bridge']['MLP_0']['Dense_3']['kernel']
    idm_kernel = agent.network.params['modules_idm']['MLP_0']['Dense_0']['kernel']
    endpoint_output = agent.network.params['modules_endpoint']['MLP_0']['Dense_3']['kernel']
    write_json(run_dir / 'config.json', {
        'arm': arm,
        'source_representation': source_kind,
        'goal_representation': goal_kind,
        'role_width': 16,
        'execution_representation': 'normalized_full_observation',
        'execution_width': 55,
        'bridge_first_kernel': list(bridge_kernel.shape),
        'bridge_output_kernel': list(bridge_output.shape),
        'idm_first_kernel': list(idm_kernel.shape),
        'endpoint_output_kernel': list(endpoint_output.shape),
        'frozen_representations_outside_optimizer': True,
        'fresh_downstream_initialization': start == 0,
        'training_seed': int(seed),
        'batch_size': BATCH_SIZE,
        'steps': int(max_updates),
        'stop': int(max_updates),
        'automatic_1m_extension': False,
    })

    if dev_manifest is None:
        dev_manifest = json.loads((OUT / 'eval_resets_dev.json').read_text())
    if final_manifest is None:
        final_manifest = json.loads((OUT / 'eval_resets_final.json').read_text())
    env = ogbench.make_env_and_datasets(ENV_NAME, env_only=True)
    checks = {
        int(step): ('final' if int(step) == int(max_updates) else 'dev',
                    final_manifest if int(step) == int(max_updates) else dev_manifest)
        for step in checkpoint_steps
    }
    log_path = run_dir / 'train.jsonl'
    started = time.time()
    try:
        with log_path.open('a') as log:
            for step in range(start + 1, int(max_updates) + 1):
                batch = dataset.sample(BATCH_SIZE)
                agent, info = agent.update(batch)
                if step % 1000 == 0 or step in checks:
                    finite = {}
                    for key, value in info.items():
                        array = np.asarray(value)
                        if array.shape == ():
                            finite[key] = float(array)
                    finite.update(step=step, wall_seconds=time.time() - started)
                    log.write(json.dumps(finite) + '\n')
                    log.flush()
                    print(json.dumps({'step': step, 'wall_seconds': finite['wall_seconds']}), flush=True)
                if step in checks:
                    save_agent(agent, run_dir / 'checkpoints', step)
                    list_name, manifest = checks[step]
                    path = run_dir / f'evaluation_{step}_h5_{list_name}.json'
                    if not path.is_file():
                        result = evaluate(agent, env, manifest, 5, mean=mean, std=std)
                        result.update({
                            'arm': arm,
                            'source_representation': source_kind,
                            'goal_representation': goal_kind,
                            'execution_representation': 'full_state',
                            'step': step,
                            'list': list_name,
                            'checkpoint': f'params_{step}.pkl',
                        })
                        write_json(path, result)
                        print(json.dumps(result), flush=True)
        write_json(run_dir / 'complete.json', {
            'steps': int(max_updates),
            'arm': arm,
            'stopped_for_judgment': True,
            'automatic_1m_extension': False,
        })
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=tuple(ARMS))
    args = parser.parse_args()
    train_arm(args.arm)


if __name__ == '__main__':
    main()
