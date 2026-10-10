"""Train NonOracle BTRL16 (Bottleneck TRL from scratch)."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MUJOCO_GL', 'egl')

import jax
import numpy as np

from envs.env_utils import make_env_and_datasets
from learned_goalspace.btrl import BottleneckTRLAgent, METHOD
from utils.contrastive_pathbridger_evaluation import evaluate
from utils.datasets import PathBridgerDataset
from utils.flax_utils import restore_agent, save_agent

CHECKPOINTS = (100_000, 300_000, 500_000, 800_000, 1_000_000)
EXECUTE_H = (5, 2, 1)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def config_for(env: str, method: str = METHOD):
    return importlib.import_module(f'configs.nonoracle.{env}').get_config(method)


def latest_checkpoint(run_dir):
    paths = list((Path(run_dir) / 'checkpoints').glob('params_*.pkl'))
    return max(paths, key=lambda item: int(item.stem.split('_')[-1])) if paths else None


def finite_metrics(info):
    return {key: float(np.asarray(value)) for key, value in info.items()}


@jax.jit
def _all_finite(info):
    import jax.numpy as jnp

    return jnp.all(jnp.stack([jnp.isfinite(value) for value in info.values()]))


def count_params(params) -> int:
    return int(sum(np.asarray(x).size for x in jax.tree_util.tree_leaves(params)))


def run(args):
    if args.method != METHOD:
        raise ValueError(f'This entrypoint trains {METHOD} only, got {args.method!r}')
    config = config_for(args.env, args.method)
    if config.get('high_level_oracle_phi', True) or config.get('proposer_oracle_phi', True):
        raise RuntimeError('BTRL16 config must declare non-oracle flags.')
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = run_dir / 'config.json'
    if existing.exists():
        saved = json.loads(existing.read_text())
        if saved['agent'] != json.loads(json.dumps(config.to_dict())):
            raise ValueError('Resume config does not match this BTRL16 run')
    else:
        write_json(existing, dict(agent=config.to_dict(), runtime=vars(args)))

    restore_path = args.restore or (str(latest_checkpoint(run_dir) or '') if args.resume else '')
    if latest_checkpoint(run_dir) and not restore_path:
        raise ValueError('Existing checkpoint: use --resume instead of overwriting')

    random.seed(args.seed)
    np.random.seed(args.seed)
    env, train, _val = make_env_and_datasets(config.env_name, dataset_dir=args.dataset_dir or None)
    dataset = PathBridgerDataset(train, config)
    example = dataset.sample(2)
    agent = BottleneckTRLAgent.create(
        args.seed,
        example['observations'],
        example['actions'],
        config.to_dict(),
    )
    start = 0
    if restore_path:
        agent = restore_agent(agent, restore_path)
        start = int(agent.network.step) - 1

    git = shutil.which('git') or 'git'
    commit = subprocess.check_output([git, 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parent, text=True).strip()
    param_counts = {name: count_params(subtree) for name, subtree in flax_unfreeze(agent.network.params).items()}
    write_json(
        run_dir / f'provenance_from_{start}.json',
        dict(
            commit=commit,
            method=METHOD,
            seed=args.seed,
            high_level_oracle_phi=False,
            proposer_oracle_phi=False,
            param_counts=param_counts,
            param_total=sum(param_counts.values()),
            packages={
                name: importlib.metadata.version(name)
                for name in ('jax', 'flax', 'optax', 'mujoco', 'ogbench', 'gymnasium')
            },
            devices=[str(device) for device in jax.devices()],
            restored_from=restore_path,
        ),
    )

    final_step = min(args.steps, args.stop_after or args.steps)
    log_path = run_dir / 'train.jsonl'
    history = []
    if log_path.exists():
        history = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        history = [row for row in history if row['step'] <= start]
        log_path.write_text(''.join(json.dumps(row) + '\n' for row in history))
    previous_wall = history[-1]['wall_seconds'] if history else 0.0
    started = time.time()

    def process_checkpoint(agent, step):
        state = np.random.get_state()
        python_state = random.getstate()
        try:
            if args.smoke:
                result = evaluate(
                    agent,
                    env,
                    episodes_per_task=1,
                    execute_h=5,
                    num_candidates=int(config.eval_num_candidates),
                    temperature=float(config.eval_temperature),
                    seed=args.seed,
                )
                write_json(run_dir / 'smoke_evaluation.json', result)
            else:
                for horizon in EXECUTE_H:
                    path = run_dir / f'evaluation_{step}_h{horizon}.json'
                    if path.exists():
                        continue
                    result = evaluate(
                        agent,
                        env,
                        episodes_per_task=args.episodes,
                        num_candidates=int(config.eval_num_candidates),
                        temperature=float(config.eval_temperature),
                        seed=args.seed,
                        execute_h=horizon,
                    )
                    result.update(
                        env=config.env_name,
                        method=METHOD,
                        seed=args.seed,
                        checkpoint=step,
                        high_level_oracle_phi=False,
                        proposer_oracle_phi=False,
                    )
                    write_json(path, result)
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
        write_json(
            run_dir / name,
            dict(
                steps=final_step,
                wall_seconds=previous_wall + time.time() - started,
                smoke=args.smoke,
                method=METHOD,
                commit=commit,
                high_level_oracle_phi=False,
                proposer_oracle_phi=False,
            ),
        )
    finally:
        env.close()


def flax_unfreeze(params):
    import flax

    return flax.core.unfreeze(params) if isinstance(params, flax.core.FrozenDict) else dict(params)


def parser():
    parsed = argparse.ArgumentParser()
    parsed.add_argument('--env', default='puzzle_3x3', choices=('puzzle_3x3',))
    parsed.add_argument('--method', default=METHOD)
    parsed.add_argument('--seed', type=int, default=0)
    parsed.add_argument('--steps', type=int, default=1_000_000)
    parsed.add_argument('--stop_after', type=int, default=0)
    parsed.add_argument('--batch-size', type=int, default=1024)
    parsed.add_argument('--episodes', type=int, default=50)
    parsed.add_argument('--log-interval', type=int, default=1000)
    parsed.add_argument('--run-dir', required=True)
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
