"""Train the online SGCRL comparison on the original launcher environments.

Environments, in launcher order: sawyer_bin, sawyer_box, sawyer_peg,
point_Spiral11x11.  Each seed is the cube protocol: a 100k warm start that
behaves as SGCRL while both bridges train, then three branches that differ
only in the actor's goal.  Seed 0 is the first pass.  At most three training
processes run, so the cube-single 8M jobs can keep their GPUs.
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

ENVS = (
    'sawyer_bin',
    'sawyer_box',
    'sawyer_peg',
    'point_Spiral11x11',
)
WARMUP_STEPS = 100_000
TOTAL_STEPS = 8_000_000
MAX_PARALLEL = 3
MAX_ATTEMPTS = 2
POLL_SECONDS = 20
ROOT = REPO / 'exp' / 'sgcrl_orig'


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
    env['MUJOCO_PY_MUJOCO_PATH'] = str(Path.home() / '.mujoco' / 'mujoco210')
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    # Cube runs already reserve 0.12 each.  Keep these smaller.
    env['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.08'
    env['PYTHONPATH'] = str(REPO)
    return env


def run_dir(env_name: str, variant: str) -> Path:
    return ROOT / env_name / 'seed0' / variant


def warmup_dir(env_name: str) -> Path:
    return ROOT / env_name / 'seed0' / 'warmup'


def command(
    env_name: str,
    variant: str,
    output_dir: Path,
    total: int,
    train_bridges: str,
    snapshot_at: str,
    resume: Path | None,
) -> list[str]:
    argv = [
        _python(),
        str(REPO / 'main_online.py'),
        f'--agent={REPO / "configs" / "online" / "cube_single.py"}:{variant}',
        '--agent.task_id=1',
        f'--agent.env_name={env_name}',
        '--seed=0',
        f'--total_env_steps={total}',
        f'--eval_points={EVAL_POINTS}',
        '--eval_episodes=100',
        f'--train_bridges={train_bridges}',
        f'--output_dir={output_dir}',
    ]
    if snapshot_at:
        argv.append(f'--snapshot_at={snapshot_at}')
    if resume is not None:
        argv.append(f'--resume_snapshot={resume}')
    return argv


def warmup_ready(env_name: str) -> Path | None:
    exact = warmup_dir(env_name) / f'snapshot_{WARMUP_STEPS}.pkl'
    if exact.exists():
        return exact
    latest = latest_checkpoint(warmup_dir(env_name))
    if latest is not None and checkpoint_step(latest) >= WARMUP_STEPS:
        return latest
    return None


def branch_done(env_name: str, variant: str) -> bool:
    latest = latest_checkpoint(run_dir(env_name, variant))
    return latest is not None and checkpoint_step(latest) >= TOTAL_STEPS


def live_output_dirs() -> set[Path]:
    result = subprocess.run(
        ['ps', '-eo', 'args'], check=False, capture_output=True, text=True
    )
    found = set()
    for line in result.stdout.splitlines():
        if 'main_online.py' not in line:
            continue
        for token in line.split():
            if token.startswith('--output_dir='):
                found.add(Path(token.split('=', 1)[1]).resolve())
    return found


def main() -> None:
    log_path = ROOT / 'driver.log'
    ROOT.mkdir(parents=True, exist_ok=True)
    attempts: dict[tuple[str, str], int] = {}
    running: dict[tuple[str, str], subprocess.Popen] = {}
    logs: dict[tuple[str, str], object] = {}

    def log(message: str) -> None:
        line = f'{time.strftime("%H:%M:%S")} {message}'
        print(line, flush=True)
        with log_path.open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')

    def start(env_name: str, variant: str) -> None:
        key = (env_name, variant)
        if variant == 'warmup':
            directory = warmup_dir(env_name)
            latest = latest_checkpoint(directory)
            resume = latest if latest is not None else None
            argv = command(
                env_name,
                'online_sgcrl',
                directory,
                WARMUP_STEPS,
                'both',
                str(WARMUP_STEPS),
                resume,
            )
        else:
            directory = run_dir(env_name, variant)
            latest = latest_checkpoint(directory)
            resume = latest if latest is not None else warmup_ready(env_name)
            argv = command(
                env_name,
                variant,
                directory,
                TOTAL_STEPS,
                'auto',
                SNAPSHOT_POINTS,
                resume,
            )
        directory.mkdir(parents=True, exist_ok=True)
        handle = (directory / 'driver_stage.log').open('a', encoding='utf-8')
        handle.write(f'\n$ {" ".join(argv)}\n')
        handle.flush()
        running[key] = subprocess.Popen(
            argv, stdout=handle, stderr=subprocess.STDOUT, env=training_env()
        )
        logs[key] = handle
        log(f'start {env_name} {variant} pid={running[key].pid}')

    while True:
        finished = []
        for key, process in running.items():
            code = process.poll()
            if code is None:
                continue
            finished.append((key, code))
        for key, code in finished:
            env_name, variant = key
            process = running.pop(key)
            logs.pop(key).close()
            if variant == 'warmup':
                ready = warmup_ready(env_name) is not None
            else:
                ready = branch_done(env_name, variant)
            if ready:
                log(f'done {env_name} {variant} exit={code}')
                continue
            attempts[key] = attempts.get(key, 0) + (0 if code == 0 else 1)
            if code != 0 and attempts[key] > MAX_ATTEMPTS:
                log(f'give up {env_name} {variant} exit={code}')
                continue
            log(f'requeue {env_name} {variant} exit={code}')

        live = live_output_dirs()
        while len(running) < MAX_PARALLEL:
            launched = False
            for env_name in ENVS:
                key = (env_name, 'warmup')
                if (
                    warmup_ready(env_name) is None
                    and key not in running
                    and attempts.get(key, 0) <= MAX_ATTEMPTS
                    and warmup_dir(env_name).resolve() not in live
                ):
                    start(env_name, 'warmup')
                    launched = True
                    break
            if launched:
                continue
            for env_name in ENVS:
                if warmup_ready(env_name) is None:
                    continue
                for variant in BRANCH_VARIANTS:
                    key = (env_name, variant)
                    if (
                        not branch_done(env_name, variant)
                        and key not in running
                        and attempts.get(key, 0) <= MAX_ATTEMPTS
                        and run_dir(env_name, variant).resolve() not in live
                    ):
                        start(env_name, variant)
                        launched = True
                        break
                if launched:
                    break
            if not launched:
                break

        pending = False
        for env_name in ENVS:
            if warmup_ready(env_name) is None and attempts.get((env_name, 'warmup'), 0) <= MAX_ATTEMPTS:
                pending = True
            for variant in BRANCH_VARIANTS:
                if not branch_done(env_name, variant) and attempts.get((env_name, variant), 0) <= MAX_ATTEMPTS:
                    pending = True
        if not pending and not running:
            log('all original-env runs finished')
            return
        time.sleep(POLL_SECONDS)


if __name__ == '__main__':
    main()
