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


def _python() -> str:
    return os.environ.get('PILOT_PYTHON', sys.executable)

BRANCH_VARIANTS = (
    'online_sgcrl',
    'online_sgcrl_det_bridge',
    'online_sgcrl_rf_bridge',
)
WARMUP_STEPS = 100_000
DEFAULT_TOTAL_STEPS = 8_000_000
EVAL_POINTS = (
    '10000,50000,100000,200000,300000,500000,800000,1000000,'
    '2000000,3000000,4000000,5000000,6000000,7000000,8000000'
)
# Nominal interaction counts at which a resumable snapshot is written.
# The final snapshot is written separately when the run stops.
SNAPSHOT_POINTS = '2000000,3000000,4000000,5000000,6000000,7000000'
EVAL_EPISODES = 100


def checkpoint_step(path: Path) -> int:
    """Step encoded in ``snapshot_<step>.pkl`` or ``final_<step>.pkl``."""

    return int(path.stem.split('_', 1)[1])


def latest_checkpoint(run_dir: Path) -> Path | None:
    """The furthest snapshot in a run directory.

    ``snapshot_<point>`` records the nominal point that was crossed and
    ``final_<step>`` records the exact step the process stopped at, so the
    numeric suffix orders them.
    """

    candidates = [
        path
        for path in run_dir.glob('*.pkl')
        if path.name.startswith(('snapshot_', 'final_'))
    ]
    if not candidates:
        return None
    return max(candidates, key=checkpoint_step)


def stage_command(
    *,
    variant: str,
    seed: int,
    task_id: int,
    output_dir: Path,
    total_env_steps: int,
    train_bridges: str,
    snapshot_at: str = '',
    resume_snapshot: Path | None = None,
) -> list[str]:
    command = [
        _python(),
        str(REPO / 'main_online.py'),
        f'--agent={REPO / "configs" / "online" / "cube_single.py"}:{variant}',
        f'--agent.task_id={task_id}',
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
    parser.add_argument('--task_id', type=int, default=1)
    parser.add_argument('--total_env_steps', type=int, default=DEFAULT_TOTAL_STEPS)
    parser.add_argument('--root', type=Path, default=REPO / 'exp' / 'online_pilot')
    arguments = parser.parse_args()

    seed = int(arguments.seed)
    task_id = int(arguments.task_id)
    total_steps = int(arguments.total_env_steps)
    root = Path(arguments.root).resolve() / f'seed{seed}'
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / 'pilot.log'

    warmup_dir = root / 'warmup'
    snapshot = warmup_dir / f'snapshot_{WARMUP_STEPS}.pkl'
    if snapshot.exists():
        print(f'[pilot] seed {seed} task {task_id}: reusing {snapshot}', flush=True)
    else:
        run_stage(
            f'seed{seed} warmup',
            stage_command(
                # The warm start collects with the baseline's behaviour; both
                # bridges train on the same replay so every branch inherits
                # equally warm bridge parameters.
                variant='online_sgcrl',
                seed=seed,
                task_id=task_id,
                output_dir=warmup_dir,
                total_env_steps=WARMUP_STEPS,
                train_bridges='both',
                snapshot_at=str(WARMUP_STEPS),
            ),
            log_path,
        )

    for variant in BRANCH_VARIANTS:
        branch_dir = root / variant
        latest = latest_checkpoint(branch_dir)
        if latest is not None and checkpoint_step(latest) >= total_steps:
            print(
                f'[pilot] seed {seed} {variant}: already at '
                f'{checkpoint_step(latest)}',
                flush=True,
            )
            continue
        # A branch that already ran (for example to 1M) continues from its own
        # latest snapshot. Starting it again from the warm-up snapshot would
        # discard that training and overwrite the run directory.
        resume_from = latest if latest is not None else snapshot
        run_stage(
            f'seed{seed} {variant}',
            stage_command(
                variant=variant,
                seed=seed,
                task_id=task_id,
                output_dir=branch_dir,
                total_env_steps=total_steps,
                train_bridges='auto',
                snapshot_at=SNAPSHOT_POINTS,
                resume_snapshot=resume_from,
            ),
            log_path,
        )

    summary = {
        'seed': seed,
        'task_id': task_id,
        'warmup_steps': WARMUP_STEPS,
        'total_steps': total_steps,
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
