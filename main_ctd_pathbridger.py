"""Train one Contrastive-Transitive Distance PathBridger run."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MUJOCO_GL', 'egl')
import jax
import numpy as np

from agents.contrastive_transitive_distance_pathbridger import (
    VARIANTS,
    ContrastiveTransitiveDistanceAgent,
)
from envs.env_utils import make_env_and_datasets
from utils.ctd_diagnostics import diagnostics
from utils.datasets import PathBridgerDataset
from utils.flax_utils import restore_agent, save_agent
from utils.contrastive_pathbridger_evaluation import evaluate

CHECKPOINTS = (100000, 300000, 500000, 800000, 1000000)
EXECUTE_H = (5, 2, 1)
ENVS = ('cube_single', 'cube_double', 'puzzle_3x3', 'antmaze_medium',
        'puzzle_4x4', 'cube_triple', 'antmaze_large', 'scene')
DIAGNOSTIC_SEED = 92831
DIAGNOSTIC_PAIRS = 1024


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def config_for(env, variant):
    name = Path(env).stem if str(env).endswith('.py') else env
    if name not in ENVS:
        raise ValueError(f'Unknown CTD config: {env}')
    if variant not in VARIANTS:
        raise ValueError(f'Unknown CTD variant: {variant}')
    family = 'gsctd' if variant.startswith('gs') else 'ctd'
    return importlib.import_module(f'configs.{family}.' + name).get_config(variant)


def latest_checkpoint(run_dir):
    paths = list((Path(run_dir) / 'checkpoints').glob('params_*.pkl'))
    return max(paths, key=lambda item: int(item.stem.split('_')[-1])) if paths else None


def finite_metrics(info):
    return {key: float(np.asarray(value)) for key, value in info.items()}


def fixed_diagnostic_batch(dataset, path):
    """Validation batch reused at every checkpoint of this run."""

    if path.exists():
        return {key: np.asarray(value) for key, value in np.load(path).items()}
    observations = np.asarray(dataset.dataset['observations'])
    finals = dataset._final_for_idx
    pool = np.flatnonzero(finals - np.arange(len(observations)) >= dataset.horizon)
    if len(pool) < DIAGNOSTIC_PAIRS:
        raise RuntimeError(f'Only {len(pool)} validation starts cover K={dataset.horizon}')
    rng = np.random.default_rng(DIAGNOSTIC_SEED)
    chosen = np.sort(rng.choice(pool, size=DIAGNOSTIC_PAIRS, replace=False))
    numpy_state = np.random.get_state()
    np.random.seed(DIAGNOSTIC_SEED)
    try:
        batch = dataset.sample(DIAGNOSTIC_PAIRS, idxs=chosen)
    finally:
        np.random.set_state(numpy_state)
    batch['z_true'] = observations[chosen + dataset.horizon].astype(np.float32)
    from utils.goal_representation import goal_representation
    indices = []
    obs_dim = observations.shape[-1]
    for dim in range(obs_dim):
        probe = np.zeros((1, obs_dim), dtype=np.float32)
        probe[0, dim] = 1.0
        if np.any(np.asarray(goal_representation(probe, 'phi', env_name=dataset.config.env_name)) != 0):
            indices.append(dim)
    batch['goal_indices'] = np.asarray(indices, dtype=np.int32)
    ref = rng.choice(len(observations), size=min(20000, len(observations)), replace=False)
    batch['reference_states'] = observations[ref].astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **batch)
    return batch


@jax.jit
def _all_finite(info):
    import jax.numpy as jnp
    return jnp.all(jnp.stack([jnp.isfinite(value) for value in info.values()]))


def run(args):
    config = config_for(args.env, args.variant)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = run_dir / 'config.json'
    if existing.exists():
        saved = json.loads(existing.read_text())
        if saved['agent'] != json.loads(json.dumps(config.to_dict())):
            raise ValueError('Resume config does not match this CTD run')
    else:
        write_json(existing, dict(agent=config.to_dict(), runtime=vars(args)))
    restore_path = args.restore or (str(latest_checkpoint(run_dir) or '') if args.resume else '')
    if latest_checkpoint(run_dir) and not restore_path:
        raise ValueError('Existing checkpoint: use --resume instead of overwriting')
    random.seed(args.seed)
    np.random.seed(args.seed)
    env, train, val = make_env_and_datasets(config.env_name, dataset_dir=args.dataset_dir or None)
    dataset = PathBridgerDataset(train, config)
    validation = PathBridgerDataset(val, config)
    example = dataset.sample(2)
    agent = ContrastiveTransitiveDistanceAgent.create(
        args.seed,
        example['observations'],
        example['actions'],
        config.to_dict(),
    )
    start = 0
    if restore_path:
        agent = restore_agent(agent, restore_path)
        start = int(agent.network.step) - 1
    final_step = min(args.steps, args.stop_after or args.steps)
    git = shutil.which('git') or str(Path.home() / 'miniconda3/bin/git')
    commit = subprocess.check_output([git, 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent, text=True).strip()
    write_json(run_dir / f'provenance_from_{start}.json', dict(
        commit=commit,
        variant=args.variant,
        seed=args.seed,
        packages={name: importlib.metadata.version(name) for name in ('jax', 'flax', 'optax', 'mujoco', 'ogbench', 'gymnasium')},
        devices=[str(device) for device in jax.devices()],
        restored_from=restore_path,
    ))
    manifest = [
        dict(task_id=task, episode=episode, env_seed=args.seed * 1_000_000 + task * 10_000 + episode)
        for task in (1, 2, 3, 4, 5) for episode in range(args.episodes)
    ]
    write_json(run_dir / 'evaluation_manifest.json', manifest)
    diagnostic = fixed_diagnostic_batch(validation, run_dir / 'diagnostic_batch_seed92831.npz')
    log_path = run_dir / 'train.jsonl'
    history = []
    if log_path.exists():
        history = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        history = [row for row in history if row['step'] <= start]
        log_path.write_text(''.join(json.dumps(row) + '\n' for row in history))
    previous_wall = history[-1]['wall_seconds'] if history else 0.0
    started = time.time()
    metrics = history[-1] if history else {}

    def process_checkpoint(agent, step):
        state = np.random.get_state()
        python_state = random.getstate()
        try:
            diag_path = run_dir / f'diagnostics_{step}.json'
            if not diag_path.exists():
                diag = diagnostics(agent, diagnostic, diagnostic['reference_states'])
                if diag.get('triangle/max_violation', 0.0) > 1e-3:
                    raise RuntimeError(f'Triangle violation {diag["triangle/max_violation"]} at step {step}')
                if not all(np.isfinite(value) for value in diag.values()):
                    raise FloatingPointError(f'Nonfinite diagnostic at step {step}')
                diag.update(step=step, variant=args.variant, seed=args.seed)
                write_json(diag_path, diag)
            if args.smoke:
                result = evaluate(
                    agent, env, episodes_per_task=1, execute_h=5,
                    num_candidates=int(config.eval_num_candidates),
                    temperature=float(config.eval_temperature), seed=args.seed,
                )
                write_json(run_dir / 'smoke_evaluation.json', result)
            else:
                for horizon in EXECUTE_H:
                    path = run_dir / f'evaluation_{step}_h{horizon}.json'
                    if path.exists():
                        continue
                    result = evaluate(
                        agent, env, episodes_per_task=args.episodes,
                        num_candidates=int(config.eval_num_candidates),
                        temperature=float(config.eval_temperature),
                        seed=args.seed, execute_h=horizon,
                    )
                    result.update(
                        env=config.env_name, variant=args.variant, seed=args.seed,
                        checkpoint=step,
                        method=(
                            'goalspace_transitive_distance'
                            if args.variant.startswith('gs')
                            else 'ctd_pathbridger'
                        ),
                    )
                    write_json(path, result)
            if 'goalspace_ablation' in run_dir.parts:
                subprocess.run(
                    [os.sys.executable, str(Path(__file__).resolve().parent / 'scripts/summarize_goalspace_ablation.py')],
                    check=True,
                )
            return agent
        finally:
            np.random.set_state(state)
            random.setstate(python_state)

    if start:
        agent = process_checkpoint(agent, start)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool, log_path.open('a') as log:
            future = pool.submit(dataset.sample, args.batch_size) if start < final_step else None
            for step in range(start + 1, final_step + 1):
                batch = future.result()
                save = step in CHECKPOINTS or step == final_step
                if not save:
                    future = pool.submit(dataset.sample, args.batch_size)
                agent, info = agent.update(batch)
                if not bool(np.asarray(_all_finite(info))):
                    raise FloatingPointError(f'Nonfinite training metric at update {step}')
                if step % args.log_interval == 0 or save:
                    metrics = finite_metrics(info)
                    metrics.update(step=step, wall_seconds=previous_wall + time.time() - started)
                    log.write(json.dumps(metrics, allow_nan=False) + '\n')
                    log.flush()
                    print(json.dumps(metrics), flush=True)
                if save:
                    checkpoint = save_agent(agent, run_dir / 'checkpoints', step)
                    restored = restore_agent(agent, checkpoint)
                    for left, right in zip(
                        jax.tree_util.tree_leaves(agent),
                        jax.tree_util.tree_leaves(restored),
                    ):
                        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
                    agent = process_checkpoint(agent, step)
                    if step < final_step:
                        future = pool.submit(dataset.sample, args.batch_size)
        name = 'complete.json' if final_step == args.steps else 'paused.json'
        write_json(run_dir / name, dict(
            steps=final_step,
            wall_seconds=previous_wall + time.time() - started,
            smoke=args.smoke,
            variant=args.variant,
            commit=commit,
        ))
    finally:
        env.close()


def parser():
    parsed = argparse.ArgumentParser()
    parsed.add_argument('--env', default='cube_double', choices=ENVS)
    parsed.add_argument('--variant', default='ctd_weighted', choices=VARIANTS)
    parsed.add_argument('--seed', type=int, default=0)
    parsed.add_argument('--steps', '--train_steps', type=int, default=1_000_000)
    parsed.add_argument('--stop_after', type=int, default=0)
    parsed.add_argument('--batch-size', type=int, default=1024)
    parsed.add_argument('--episodes', type=int, default=50)
    parsed.add_argument('--log-interval', type=int, default=1000)
    parsed.add_argument('--run-dir', '--save_dir', required=True)
    parsed.add_argument('--dataset_dir', default='')
    parsed.add_argument('--restore', default='')
    parsed.add_argument('--resume', action='store_true')
    parsed.add_argument('--smoke', action='store_true')
    return parsed


if __name__ == '__main__':
    args = parser().parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.episodes < 1:
        raise ValueError('Invalid run size')
    run(args)
