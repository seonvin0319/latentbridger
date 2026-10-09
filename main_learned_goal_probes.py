"""Run deterministic frozen post-hoc probes for all pretrain checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from configs.gsctd.cube_double import get_config as cube_config
from configs.gsctd.puzzle_3x3 import get_config as puzzle_config
from envs.env_utils import make_env_and_datasets
from learned_goalspace.multihorizon import VARIANT as MULTIHORIZON_VARIANT
from learned_goalspace.pretrain import VARIANT as FUTURE_VARIANT
from learned_goalspace.probes import run_checkpoint_probes, write_metric_csv

CONFIGS = {'puzzle_3x3': puzzle_config, 'cube_double': cube_config}
CHECKPOINTS = (100_000, 300_000, 500_000)
VARIANTS = (FUTURE_VARIANT, MULTIHORIZON_VARIANT)


def default_pretrain_dir(env: str, variant: str, seed: int) -> Path:
    if variant == MULTIHORIZON_VARIANT:
        return Path('exp/learned_goalspace/multihorizon/pretrain') / env / variant / f'seed{seed}'
    return Path('exp/learned_goalspace/pretrain') / env / variant / f'seed{seed}'


def default_probe_dir(env: str, variant: str, seed: int) -> Path:
    if variant == MULTIHORIZON_VARIANT:
        return Path('exp/learned_goalspace/multihorizon/probes') / env / variant / f'seed{seed}'
    return Path('exp/learned_goalspace/probes') / env / variant / f'seed{seed}'


def run(args) -> None:
    variant = args.variant
    config = CONFIGS[args.env]('gsdtrl_weighted')
    pretrain_dir = Path(args.pretrain_dir) if args.pretrain_dir else default_pretrain_dir(args.env, variant, args.seed)
    output_dir = Path(args.output_dir) if args.output_dir else default_probe_dir(args.env, variant, args.seed)
    env, _, validation = make_env_and_datasets(config.env_name, dataset_dir=args.dataset_dir or None)
    try:
        observations = np.asarray(validation['observations'], dtype=np.float32)
        if args.max_samples and len(observations) > args.max_samples:
            chosen = np.random.default_rng(41719).choice(len(observations), args.max_samples, replace=False)
            observations = observations[np.sort(chosen)]
        representation_rows, probe_rows = [], []
        steps = tuple(args.checkpoints) if args.checkpoints else CHECKPOINTS
        for step in steps:
            checkpoint = pretrain_dir / 'checkpoints' / f'params_{step}.pkl'
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            representations, probes = run_checkpoint_probes(
                checkpoint=checkpoint,
                step=step,
                env_name=config.env_name,
                observations=observations,
                expected_variant=variant,
            )
            context = {
                'env': config.env_name,
                'variant': variant,
                'seed': args.seed,
            }
            representation_rows.extend(context | row for row in representations)
            probe_rows.extend(context | row for row in probes)
        write_metric_csv(output_dir / 'representation_metrics.csv', representation_rows)
        write_metric_csv(output_dir / 'probe_metrics.csv', probe_rows)
    finally:
        env.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument('--env', choices=tuple(CONFIGS), required=True)
    result.add_argument('--variant', choices=VARIANTS, default=FUTURE_VARIANT)
    result.add_argument('--seed', type=int, default=0)
    result.add_argument('--dataset-dir', default='')
    result.add_argument('--pretrain-dir', default='')
    result.add_argument('--output-dir', default='')
    result.add_argument('--max-samples', type=int, default=20_000)
    result.add_argument('--checkpoints', type=int, nargs='*')
    return result


if __name__ == '__main__':
    run(parser().parse_args())
