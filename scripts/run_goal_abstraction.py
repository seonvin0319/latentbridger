"""Independent 0-to-8M runs of the three learned goal abstractions.

``sgcrl_raw``, the oracle-xyz run, and the two waypoint bridges are not
relaunched.  Their task-1 results are already on disk and the summarizer
reads them in place.  At most three training processes run at once.  A run
that exits before 8M is retried once, then recorded in ``failed_runs.json``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))

from run_online_pilot import checkpoint_step, latest_checkpoint  # noqa: E402

VARIANTS = ('sgcrl_psi_goal', 'sgcrl_state_goal', 'sgcrl_state_mask')
SEEDS = (0, 1, 2)
TOTAL_STEPS = 8_000_000
EVAL_POINTS = '1000000,2000000,3000000,4000000,5000000,6000000,7000000,8000000'
SNAPSHOT_POINTS = '1000000,2000000,3000000,4000000,5000000,6000000,7000000'
MAX_PARALLEL = 3
MAX_ATTEMPTS = 2
ROOT = REPO / 'exp' / 'goal_abstraction_task1'
CONFIG = REPO / 'configs' / 'online' / 'goal_abstraction.py'


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
    env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.12'
    env['PYTHONPATH'] = str(REPO)
    return env


def run_dir(variant: str, seed: int) -> Path:
    return ROOT / variant / f'seed{seed}'


def command(variant: str, seed: int, resume: Path | None) -> list[str]:
    argv = [
        _python(),
        str(REPO / 'main_goal_abstraction.py'),
        f'--agent={CONFIG}:{variant}',
        f'--seed={seed}',
        f'--total_env_steps={TOTAL_STEPS}',
        f'--eval_points={EVAL_POINTS}',
        '--eval_episodes=100',
        f'--snapshot_at={SNAPSHOT_POINTS}',
        f'--output_dir={run_dir(variant, seed)}',
        '--actor_pretrain_updates=0',
    ]
    if resume is not None:
        argv.append(f'--resume_snapshot={resume}')
    return argv


def reached(directory: Path) -> bool:
    latest = latest_checkpoint(directory)
    return latest is not None and checkpoint_step(latest) >= TOTAL_STEPS


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / 'driver.log'
    running: dict[tuple[str, int], subprocess.Popen] = {}
    handles: dict[tuple[str, int], object] = {}
    attempts: dict[tuple[str, int], int] = {}
    failures: list[dict] = []

    def log(message: str) -> None:
        line = f'{time.strftime("%H:%M:%S")} {message}'
        print(line, flush=True)
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')

    def start(variant: str, seed: int) -> None:
        directory = run_dir(variant, seed)
        directory.mkdir(parents=True, exist_ok=True)
        resume = latest_checkpoint(directory)
        if resume is not None and checkpoint_step(resume) >= TOTAL_STEPS:
            return
        argv = command(variant, seed, resume)
        handle = (directory / 'driver_stage.log').open('a', encoding='utf-8')
        handle.write(f'\n$ {" ".join(argv)}\n')
        handle.flush()
        key = (variant, seed)
        running[key] = subprocess.Popen(
            argv, stdout=handle, stderr=subprocess.STDOUT, env=training_env()
        )
        handles[key] = handle
        log(f'start {variant} seed {seed} pid={running[key].pid}')

    jobs = [(variant, seed) for seed in SEEDS for variant in VARIANTS]
    while True:
        for key, process in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handles.pop(key).close()
            running.pop(key)
            variant, seed = key
            if reached(run_dir(variant, seed)):
                log(f'done {variant} seed {seed} exit={code}')
                continue
            attempts[key] = attempts.get(key, 0) + 1
            if attempts[key] >= MAX_ATTEMPTS:
                failure = {
                    'variant': variant,
                    'seed': seed,
                    'exit_code': code,
                    'attempts': attempts[key],
                }
                failures.append(failure)
                (ROOT / 'failed_runs.json').write_text(
                    json.dumps(failures, indent=2) + '\n', encoding='utf-8'
                )
                log(f'failed {variant} seed {seed} exit={code}')
                continue
            log(f'retry {variant} seed {seed} exit={code}')

        while len(running) < MAX_PARALLEL:
            launched = False
            for variant, seed in jobs:
                key = (variant, seed)
                if reached(run_dir(variant, seed)) or key in running:
                    continue
                if attempts.get(key, 0) >= MAX_ATTEMPTS:
                    continue
                start(variant, seed)
                launched = True
                break
            if not launched:
                break

        pending = False
        for variant, seed in jobs:
            key = (variant, seed)
            if not reached(run_dir(variant, seed)) and attempts.get(key, 0) < MAX_ATTEMPTS:
                pending = True
        if not pending and not running:
            log('goal abstraction runs finished')
            return
        time.sleep(20)


if __name__ == '__main__':
    main()
