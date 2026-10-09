"""Fit and save PCA-16 / random-16 fixed goal representations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from configs.gsctd.cube_double import get_config as cube_config
from configs.gsctd.puzzle_3x3 import get_config as puzzle_config
from envs.env_utils import make_env_and_datasets
from learned_goalspace.fixed_representations import (
    fit_pca16,
    make_random16,
    save_fixed_representation,
)

CONFIGS = {'puzzle_3x3': puzzle_config, 'cube_double': cube_config}


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def default_dir(kind: str, env: str, seed: int) -> Path:
    return Path('exp/learned_goalspace') / kind / env / f'seed{seed}'


def run(args) -> None:
    config = CONFIGS[args.env]('gsdtrl_weighted')
    kind = args.kind
    run_dir = Path(args.output_dir) if args.output_dir else default_dir(kind, args.env, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    env, train, _ = make_env_and_datasets(config.env_name, dataset_dir=args.dataset_dir or None)
    try:
        observations = np.asarray(train['observations'], dtype=np.float32)
        if kind == 'pca16':
            payload = fit_pca16(observations)
        else:
            payload = make_random16(observations.shape[-1], seed=args.seed)
        payload = {
            **payload,
            'env': args.env,
            'env_name': config.env_name,
            'seed': int(args.seed),
        }
        path = save_fixed_representation(run_dir / 'representation.pkl', payload)
        meta = {
            key: (value.tolist() if isinstance(value, np.ndarray) and value.ndim <= 1 else value)
            for key, value in payload.items()
            if key not in ('mean', 'kernel')
        }
        meta['path'] = path
        meta['mean_norm'] = float(np.linalg.norm(payload['mean']))
        meta['kernel_shape'] = list(payload['kernel'].shape)
        _write_json(run_dir / 'metadata.json', meta)
        print(json.dumps(meta, indent=2), flush=True)
    finally:
        env.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument('--env', choices=tuple(CONFIGS), required=True)
    result.add_argument('--kind', choices=('pca16', 'random16'), required=True)
    result.add_argument('--seed', type=int, default=0)
    result.add_argument('--dataset-dir', default='')
    result.add_argument('--output-dir', default='')
    return result


if __name__ == '__main__':
    run(parser().parse_args())
