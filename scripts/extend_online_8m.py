#!/usr/bin/env python
"""Continue every online bridge run from its latest snapshot to 8M steps.

Runs that already reached 1M keep that history: each one resumes from its own
final snapshot, so the critic, actor, bridges, replay and RNG continue instead
of being retrained. A run whose directory is currently being written — the
task 4 and 5 pilots still finishing their 1M budget — is left alone until
that process exits.

One seed runs at a time, its three variants together: SGCRL, the deterministic
bridge, and the rectified-flow bridge. A seed that finishes its baseline
early does not start the next seed while its bridges are still training.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_online_pilot import (  # noqa: E402
    BRANCH_VARIANTS,
    DEFAULT_TOTAL_STEPS,
    EVAL_POINTS,
    SNAPSHOT_POINTS,
    checkpoint_step,
    latest_checkpoint,
    stage_command,
)

REPO = Path(__file__).resolve().parents[1]
MAX_PARALLEL = 3
MAX_ATTEMPTS = 2
POLL_SECONDS = 30

# task 1 lives in the original pilot tree; tasks 2-5 in the follow-up tree.
RUN_ROOTS = {1: REPO / 'exp' / 'online_pilot'}
for _task in (2, 3, 4, 5):
    RUN_ROOTS[_task] = REPO / 'exp' / 'online_tasks' / f'task{_task}'


def run_dir_for(task_id: int, seed: int, variant: str) -> Path:
    return RUN_ROOTS[task_id] / f'seed{seed}' / variant


def _process_lines() -> list[str]:
    result = subprocess.run(
        ['ps', '-eo', 'args'], check=False, capture_output=True, text=True
    )
    return result.stdout.splitlines()


def running_output_dirs() -> set[Path]:
    """Directories a live ``main_online.py`` is writing."""

    found = set()
    for line in _process_lines():
        if 'main_online.py' not in line:
            continue
        for token in line.split():
            if token.startswith('--output_dir='):
                found.add(Path(token.split('=', 1)[1]).resolve())
    return found


def pilots_still_running() -> int:
    """Seed drivers that still own a 1M stage and will launch its successor.

    The match has to be the interpreter actually running the driver. A shell
    whose command line merely quotes the script name would keep this scheduler
    waiting forever.
    """

    return sum(
        'bin/python scripts/run_online_pilot.py' in line for line in _process_lines()
    )


def training_env() -> dict[str, str]:
    env = os.environ.copy()
    library_path = (
        f"{Path.home()}/.mujoco/mujoco210/bin:/usr/lib/nvidia:"
        + env.get('LD_LIBRARY_PATH', '')
    )
    env['LD_LIBRARY_PATH'] = library_path
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.12'
    env['PILOT_PYTHON'] = os.environ.get(
        'PILOT_PYTHON',
        str(Path.home() / 'miniconda3/envs/fql-bootstrap-v1/bin/python'),
    )
    return env


def group_of(directory: Path) -> tuple[int, int] | None:
    """``(task_id, seed)`` encoded in a run directory, if it is one of ours."""

    parts = directory.resolve().parts
    seed = task_id = None
    for part in parts:
        if part.startswith('seed') and part[4:].isdigit():
            seed = int(part[4:])
        elif part.startswith('task') and part[4:].isdigit():
            task_id = int(part[4:])
    if seed is None:
        return None
    # Task 1 lives under exp/online_pilot/seedN, which has no task directory.
    return (1 if task_id is None else task_id, seed)


def jobs() -> list[tuple[int, int, str]]:
    """Task 1 first, so the original comparison reaches 8M before the rest."""

    ordered = []
    for task_id in (1, 2, 3, 4, 5):
        for seed in (0, 1, 2):
            for variant in BRANCH_VARIANTS:
                ordered.append((task_id, seed, variant))
    return ordered


def main() -> None:
    pending = jobs()
    children: dict[tuple[int, int, str], subprocess.Popen] = {}
    attempts: dict[tuple[int, int, str], int] = {}
    last_wait_log = 0.0
    env = training_env()
    log_dir = REPO / 'exp' / 'online_8m'
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / 'driver.log'

    def log(message: str) -> None:
        line = f'[{time.strftime("%H:%M:%S")}] {message}'
        print(line, flush=True)
        with status_path.open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')

    log(
        f'extending {len(pending)} runs to {DEFAULT_TOTAL_STEPS} env steps, '
        f'{MAX_PARALLEL} at a time'
    )

    while pending or children:
        finished = [key for key, process in children.items() if process.poll() is not None]
        for key in finished:
            process = children.pop(key)
            task_id, seed, variant = key
            latest = latest_checkpoint(run_dir_for(task_id, seed, variant))
            finished_budget = (
                latest is not None and checkpoint_step(latest) >= DEFAULT_TOTAL_STEPS
            )
            if process.returncode == 0 and finished_budget:
                log(f'task {task_id} seed {seed} {variant}: reached 8M')
            elif process.returncode == 0:
                # A clean stop below the budget wrote a snapshot and should
                # continue. Treating that exit as success would drop the run.
                log(
                    f'task {task_id} seed {seed} {variant}: stopped at '
                    f'{checkpoint_step(latest) if latest else 0}, requeueing'
                )
                pending.append(key)
            else:
                attempts[key] = attempts.get(key, 0) + 1
                if attempts[key] < MAX_ATTEMPTS:
                    log(
                        f'task {task_id} seed {seed} {variant}: FAILED '
                        f'(exit {process.returncode}), retrying'
                    )
                    pending.append(key)
                else:
                    log(
                        f'task {task_id} seed {seed} {variant}: FAILED '
                        f'(exit {process.returncode}), giving up'
                    )

        occupied = running_output_dirs()
        # A seed driver that is between variants is about to claim a directory.
        # Starting our own run in that window would put two writers on one run.
        holding_for_pilots = pilots_still_running() > 0
        active = None
        for directory in occupied:
            active = group_of(directory)
            if active is not None:
                break
        if active is None and not holding_for_pilots:
            for task_id, seed, variant in pending:
                directory = run_dir_for(task_id, seed, variant)
                latest = latest_checkpoint(directory)
                if latest is None or checkpoint_step(latest) < DEFAULT_TOTAL_STEPS:
                    active = (task_id, seed)
                    break

        launched = 0
        still_waiting = []
        for task_id, seed, variant in pending:
            directory = run_dir_for(task_id, seed, variant)
            if directory.resolve() in occupied or (task_id, seed, variant) in children:
                still_waiting.append((task_id, seed, variant))
                continue
            latest = latest_checkpoint(directory)
            if latest is None:
                still_waiting.append((task_id, seed, variant))
                continue
            if checkpoint_step(latest) >= DEFAULT_TOTAL_STEPS:
                log(f'task {task_id} seed {seed} {variant}: already past 8M')
                continue
            # Only this seed's three variants. The baseline finishing early
            # must not pull the next seed in beside the bridges.
            if active != (task_id, seed) or launched >= MAX_PARALLEL:
                still_waiting.append((task_id, seed, variant))
                continue

            flags = directory / 'flags.json'
            backup = directory / 'flags_before_8m.json'
            if flags.is_file() and not backup.is_file():
                backup.write_bytes(flags.read_bytes())

            command = stage_command(
                variant=variant,
                seed=seed,
                task_id=task_id,
                output_dir=directory,
                total_env_steps=DEFAULT_TOTAL_STEPS,
                train_bridges='auto',
                snapshot_at=SNAPSHOT_POINTS,
                resume_snapshot=latest,
            )
            # stage_command reads PILOT_PYTHON from the environment at call
            # time inside run_online_pilot; set it before calling.
            log_path = directory / 'extend_8m.log'
            handle = log_path.open('a', encoding='utf-8')
            handle.write(f'\n$ {" ".join(command)}\n')
            handle.flush()
            children[(task_id, seed, variant)] = subprocess.Popen(
                command, stdout=handle, stderr=subprocess.STDOUT, env=env
            )
            launched += 1
            log(
                f'task {task_id} seed {seed} {variant}: resuming from '
                f'{latest.name} ({checkpoint_step(latest)} steps)'
            )

        pending = still_waiting
        if pending and not children and time.time() - last_wait_log > 600:
            log(
                f'waiting on {len(pending)} runs'
                + (' while the 1M pilots finish' if holding_for_pilots else '')
            )
            last_wait_log = time.time()
        time.sleep(POLL_SECONDS)

    log('all runs reached 8M')


if __name__ == '__main__':
    main()
