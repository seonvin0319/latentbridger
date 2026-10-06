"""Cube-single online comparison with oracle goals instead of full goals.

Seed 0, the same 100k shared warm start and three branches as the full-goal
pilot, written under exp/cube_oracle so the running full-goal jobs are left
alone.  At most three training processes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))

from run_online_pilot import (  # noqa: E402
    BRANCH_VARIANTS,
    EVAL_POINTS,
    SNAPSHOT_POINTS,
    checkpoint_step,
    latest_checkpoint,
)

WARMUP_STEPS = 100_000
TOTAL_STEPS = 8_000_000
MAX_PARALLEL = 3
MAX_ATTEMPTS = 2
ROOT = REPO / 'exp' / 'cube_oracle' / 'seed0'
CONFIG = REPO / 'configs' / 'online' / 'cube_oracle.py'


def _python() -> str:
    return os.environ.get(
        'PILOT_PYTHON',
        str(Path.home() / 'miniconda3/envs/fql-bootstrap-v1/bin/python'),
    )


def training_env() -> dict[str, str]:
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = (
        f"{Path.home()}/.mujoco/mujoco210/bin:/usr/lib/nvidia:"
        + env.get('LD_LIBRARY_PATH', '')
    )
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.08'
    env['PYTHONPATH'] = str(REPO)
    return env


def command(variant: str, output: Path, total: int, bridges: str, snapshot_at: str, resume: Path | None) -> list[str]:
    argv = [
        _python(),
        str(REPO / 'main_online.py'),
        f'--agent={CONFIG}:{variant}',
        '--agent.task_id=1',
        '--seed=0',
        f'--total_env_steps={total}',
        f'--eval_points={EVAL_POINTS}',
        '--eval_episodes=100',
        f'--train_bridges={bridges}',
        f'--output_dir={output}',
    ]
    if snapshot_at:
        argv.append(f'--snapshot_at={snapshot_at}')
    if resume is not None:
        argv.append(f'--resume_snapshot={resume}')
    return argv


def ready(directory: Path, minimum: int) -> Path | None:
    exact = directory / f'snapshot_{minimum}.pkl'
    if exact.exists():
        return exact
    latest = latest_checkpoint(directory)
    if latest is not None and checkpoint_step(latest) >= minimum:
        return latest
    return None


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    log_path = ROOT.parent / 'driver.log'
    running: dict[str, subprocess.Popen] = {}
    handles: dict[str, object] = {}
    attempts: dict[str, int] = {}

    def log(message: str) -> None:
        line = f'{time.strftime("%H:%M:%S")} {message}'
        print(line, flush=True)
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')

    def start(name: str) -> None:
        if name == 'warmup':
            directory = ROOT / 'warmup'
            argv = command(
                'online_sgcrl',
                directory,
                WARMUP_STEPS,
                'both',
                str(WARMUP_STEPS),
                latest_checkpoint(directory),
            )
        else:
            directory = ROOT / name
            resume = latest_checkpoint(directory) or ready(ROOT / 'warmup', WARMUP_STEPS)
            argv = command(
                name, directory, TOTAL_STEPS, 'auto', SNAPSHOT_POINTS, resume
            )
        directory.mkdir(parents=True, exist_ok=True)
        handle = (directory / 'driver_stage.log').open('a', encoding='utf-8')
        handle.write(f'\n$ {" ".join(argv)}\n')
        handle.flush()
        running[name] = subprocess.Popen(
            argv, stdout=handle, stderr=subprocess.STDOUT, env=training_env()
        )
        handles[name] = handle
        log(f'start {name} pid={running[name].pid}')

    while True:
        for name, process in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handles.pop(name).close()
            running.pop(name)
            directory = ROOT / name
            minimum = WARMUP_STEPS if name == 'warmup' else TOTAL_STEPS
            if ready(directory, minimum) is not None:
                log(f'done {name} exit={code}')
                continue
            attempts[name] = attempts.get(name, 0) + (0 if code == 0 else 1)
            if code != 0 and attempts[name] > MAX_ATTEMPTS:
                log(f'give up {name} exit={code}')
                continue
            log(f'requeue {name} exit={code}')

        while len(running) < MAX_PARALLEL:
            if ready(ROOT / 'warmup', WARMUP_STEPS) is None:
                if 'warmup' not in running and attempts.get('warmup', 0) <= MAX_ATTEMPTS:
                    start('warmup')
                break
            launched = False
            for variant in BRANCH_VARIANTS:
                if (
                    ready(ROOT / variant, TOTAL_STEPS) is None
                    and variant not in running
                    and attempts.get(variant, 0) <= MAX_ATTEMPTS
                ):
                    start(variant)
                    launched = True
                    break
            if not launched:
                break

        pending = ready(ROOT / 'warmup', WARMUP_STEPS) is None and attempts.get('warmup', 0) <= MAX_ATTEMPTS
        for variant in BRANCH_VARIANTS:
            if ready(ROOT / variant, TOTAL_STEPS) is None and attempts.get(variant, 0) <= MAX_ATTEMPTS:
                pending = True
        if not pending and not running:
            log('oracle cube run finished')
            return
        time.sleep(20)


if __name__ == '__main__':
    main()
