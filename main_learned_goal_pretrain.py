"""Train the oracle-free full-observation FutureNCE encoder."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import jax
import numpy as np

from configs.gsctd.cube_double import get_config as cube_config
from configs.gsctd.puzzle_3x3 import get_config as puzzle_config
from envs.env_utils import make_env_and_datasets
from learned_goalspace.checkpoints import (
    restore_pretrainer,
    save_pretrain_checkpoint,
)
from learned_goalspace.dataset import FutureNCEDataset
from learned_goalspace.pretrain import FutureNCEPretrainer, VARIANT

CHECKPOINTS = (100_000, 300_000, 500_000)
CONFIGS = {'puzzle_3x3': puzzle_config, 'cube_double': cube_config}


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _latest(run_dir: Path) -> tuple[Path | None, int]:
    checkpoints = list((run_dir / 'checkpoints').glob('params_*.pkl'))
    if not checkpoints:
        return None, 0
    path = max(checkpoints, key=lambda value: int(value.stem.split('_')[-1]))
    return path, int(path.stem.split('_')[-1])


def default_run_dir(env: str, seed: int) -> Path:
    return Path('exp/learned_goalspace/pretrain') / env / VARIANT / f'seed{seed}'


def run(args) -> None:
    config = CONFIGS[args.env]('gsdtrl_weighted')
    run_dir = Path(args.run_dir) if args.run_dir else default_run_dir(args.env, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / 'config.json'
    identity = {
        'env': args.env,
        'env_name': config.env_name,
        'variant': VARIANT,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'dataset_dir': (str(Path(args.dataset_dir).resolve()) if args.dataset_dir else None),
        'steps': args.steps,
    }
    if config_path.exists() and json.loads(config_path.read_text()) != identity:
        raise ValueError('Resume configuration does not match this pretrain run.')
    _write_json(config_path, identity)

    random.seed(args.seed)
    np.random.seed(args.seed)
    env, train, _ = make_env_and_datasets(config.env_name, dataset_dir=args.dataset_dir or None)
    try:
        sampler = FutureNCEDataset(train, float(config.discount))
        std = np.asarray(train['observations'], dtype=np.float32).std(axis=0)
        # Constant features receive no artificial noise.
        example = np.asarray(train['observations'][:2], dtype=np.float32)
        agent = FutureNCEPretrainer.create(args.seed, example, std, env_name=config.env_name)
        latest, start = _latest(run_dir)
        if latest is not None:
            if not args.resume:
                raise ValueError('Existing checkpoint found; pass --resume.')
            agent = restore_pretrainer(agent, latest, step=start)
        final_step = min(args.steps, args.stop_after or args.steps)
        if start > final_step:
            raise ValueError(f'Checkpoint step {start} exceeds requested {final_step}.')

        log_path = run_dir / 'train.jsonl'
        if start and log_path.exists():
            rows = []
            for line in log_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # The atomic checkpoint remains authoritative if the last
                    # JSONL write was interrupted.
                    continue
            log_path.write_text(''.join(json.dumps(row) + '\n' for row in rows if int(row['step']) <= start))
        with log_path.open('a') as log:
            for step in range(start + 1, final_step + 1):
                batch = {
                    key: jax.numpy.asarray(value)
                    for key, value in sampler.sample(args.batch_size).items()
                    if key in ('queries', 'goals')
                }
                agent, info = agent.update(batch)
                save = step in CHECKPOINTS or step == final_step
                if step % args.log_interval == 0 or save:
                    row = {
                        'step': step,
                        **{key: float(np.asarray(value)) for key, value in info.items()},
                    }
                    log.write(json.dumps(row, allow_nan=False) + '\n')
                    log.flush()
                    print(json.dumps(row), flush=True)
                if save:
                    checkpoint_path = save_pretrain_checkpoint(agent, run_dir / 'checkpoints', step)
                    restored = restore_pretrainer(agent, checkpoint_path, step=step)
                    for left, right in zip(
                        jax.tree_util.tree_leaves(agent),
                        jax.tree_util.tree_leaves(restored),
                    ):
                        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
        marker = 'complete.json' if final_step == args.steps else 'paused.json'
        _write_json(
            run_dir / marker,
            {'steps': final_step, 'variant': VARIANT, 'seed': args.seed},
        )
    finally:
        env.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument('--env', choices=tuple(CONFIGS), required=True)
    result.add_argument('--seed', type=int, default=0)
    result.add_argument('--steps', type=int, default=500_000)
    result.add_argument('--stop-after', type=int, default=0)
    result.add_argument('--batch-size', type=int, choices=(512, 1024), default=1024)
    result.add_argument('--log-interval', type=int, default=1000)
    result.add_argument('--dataset-dir', default='')
    result.add_argument('--run-dir', default='')
    result.add_argument('--resume', action='store_true')
    return result


if __name__ == '__main__':
    arguments = parser().parse_args()
    if arguments.steps < 1:
        raise ValueError('--steps must be positive.')
    run(arguments)
