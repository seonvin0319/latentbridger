#!/usr/bin/env python
"""Run one seed of the cube-single online bridge pilot.

A seed is four stages: a shared warm start to 100k environment steps that
collects with SGCRL behaviour while training both bridges as auxiliaries,
then three branches to 1M that each restore the identical snapshot and
differ only in what the actor is pointed at.

Each stage is skipped if its final checkpoint already exists, so an
interrupted seed can be relaunched with the same command.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = os.environ.get('PILOT_PYTHON', sys.executable)

BRANCH_VARIANTS = (
    'online_sgcrl',
    'online_sgcrl_det_bridge',
    'online_sgcrl_rf_bridge',
)
WARMUP_STEPS = 100_000
TOTAL_STEPS = 1_000_000
EVAL_POINTS = '10000,50000,100000,200000,300000,500000,800000,1000000'
EVAL_EPISODES = 100


def stage_command(
    *,
    variant: str,
    seed: int,
    output_dir: Path,
    total_env_steps: int,
    train_bridges: str,
    snapshot_at: str = '',
    resume_snapshot: Path | None = None,
) -> list[str]:
    command = [
        PYTHON,
        str(REPO / 'main_online.py'),
        f'--agent={REPO / "configs" / "online" / "cube_single.py"}:{variant}',
        f'--seed={seed}',
        f'--total_env_steps={total_env_steps}',
        f'--eval_points={EVAL_POINTS}',
        f'--eval_episodes={EVAL_EPISODES}',
        f'--train_bridges={train_bridges}',
        f'--output_dir={output_dir}',
    ]
    if snapshot_at:
        command.append(f'--snapshot_at={snapshot_at}')
    if resume_snapshot is not None:
        command.append(f'--resume_snapshot={resume_snapshot}')
    return command


def run_stage(name: str, command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print(f'[pilot] {name}: starting', flush=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f'\n$ {" ".join(command)}\n')
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    minutes = (time.time() - started) / 60.0
    if result.returncode != 0:
        raise RuntimeError(
            f'{name} failed with exit code {result.returncode}; see {log_path}'
        )
    print(f'[pilot] {name}: done in {minutes:.1f} min', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--root', type=Path, default=REPO / 'exp' / 'online_pilot')
    arguments = parser.parse_args()

    seed = int(arguments.seed)
    root = Path(arguments.root).resolve() / f'seed{seed}'
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / 'pilot.log'

    warmup_dir = root / 'warmup'
    snapshot = warmup_dir / f'snapshot_{WARMUP_STEPS}.pkl'
    if snapshot.exists():
        print(f'[pilot] seed {seed}: reusing warm start at {snapshot}', flush=True)
    else:
        run_stage(
            f'seed{seed} warmup',
            stage_command(
                # The warm start collects with the baseline's behaviour; both
                # bridges train on the same replay so every branch inherits
                # equally warm bridge parameters.
                variant='online_sgcrl',
                seed=seed,
                output_dir=warmup_dir,
                total_env_steps=WARMUP_STEPS,
                train_bridges='both',
                snapshot_at=str(WARMUP_STEPS),
            ),
            log_path,
        )

    for variant in BRANCH_VARIANTS:
        branch_dir = root / variant
        if (branch_dir / f'final_{TOTAL_STEPS}.pkl').exists():
            print(f'[pilot] seed {seed} {variant}: already complete', flush=True)
            continue
        run_stage(
            f'seed{seed} {variant}',
            stage_command(
                variant=variant,
                seed=seed,
                output_dir=branch_dir,
                total_env_steps=TOTAL_STEPS,
                train_bridges='auto',
                resume_snapshot=snapshot,
            ),
            log_path,
        )

    summary = {
        'seed': seed,
        'warmup_steps': WARMUP_STEPS,
        'total_steps': TOTAL_STEPS,
        'variants': list(BRANCH_VARIANTS),
        'eval_points': [int(p) for p in EVAL_POINTS.split(',')],
        'eval_episodes': EVAL_EPISODES,
    }
    with (root / 'pilot_manifest.json').open('w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)
        file.write('\n')
    print(f'[pilot] seed {seed}: all stages complete', flush=True)


if __name__ == '__main__':
    main()
