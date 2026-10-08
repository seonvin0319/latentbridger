"""Priority CTD controller after PathNCE-W completion focus.

Never starts BridgeGeo. Never kills already-running trainers; it only adopts
them by matching command lines and fills free GPU slots from the priority list.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'exp' / 'ctd_pathbridger'
PY = Path('/home/shchoi/latentbridger/.venv/bin/python')

# Finish PathNCE-W remaining envs first (cube_double already done).
PHASE_A = (
    ('puzzle_3x3', 'ctd_pathnce_weighted'),
    ('antmaze_medium', 'ctd_pathnce_weighted'),
    ('cube_single', 'ctd_pathnce_weighted'),
)

# Weighted-vs-uniform decomposition only on the two critical envs.
PHASE_B = (
    ('cube_double', 'dtrl_uniform'),
    ('puzzle_3x3', 'dtrl_uniform'),
    ('cube_double', 'ctd_uniform'),
    ('puzzle_3x3', 'ctd_uniform'),
)

PHASE_N_SWEEP = (
    'dtrl_weighted',
    'ctd_weighted',
    'ctd_pathnce_weighted',
)


def gpu_env():
    env = dict(os.environ)
    env.pop('JAX_PLATFORMS', None)
    if env.get('CUDA_VISIBLE_DEVICES', None) in (None, ''):
        env.pop('CUDA_VISIBLE_DEVICES', None)
    env.update(
        XLA_PYTHON_CLIENT_PREALLOCATE='false',
        MUJOCO_GL='egl',
        OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',
    )
    return env


def run_dir(env: str, variant: str) -> Path:
    return OUT / env / variant / 'seed0'


def is_complete(env: str, variant: str, steps: int) -> bool:
    path = run_dir(env, variant) / 'complete.json'
    if not path.exists():
        return False
    return int(json.loads(path.read_text()).get('steps', 0)) == steps


def live_trainers() -> dict[tuple[str, str], int]:
    found = {}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / 'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace')
        except OSError:
            continue
        if 'main_ctd_pathbridger.py' not in command:
            continue
        env = variant = None
        parts = command.split()
        for index, token in enumerate(parts):
            if token == '--env' and index + 1 < len(parts):
                env = parts[index + 1]
            if token == '--variant' and index + 1 < len(parts):
                variant = parts[index + 1]
        if env and variant:
            found[(env, variant)] = int(entry.name)
    return found


def launch(env_name: str, variant: str, steps: int, env: dict) -> subprocess.Popen:
    destination = run_dir(env_name, variant)
    destination.mkdir(parents=True, exist_ok=True)
    argv = [
        str(PY), 'main_ctd_pathbridger.py',
        '--env', env_name, '--variant', variant, '--seed', '0',
        '--steps', str(steps), '--run-dir', str(destination),
    ]
    if list((destination / 'checkpoints').glob('params_*.pkl')):
        argv.append('--resume')
    log = (destination / 'run.log').open('a')
    proc = subprocess.Popen(
        argv,
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f'start {env_name} {variant} seed0 pid {proc.pid}', flush=True)
    return proc


def wait_slot(parallel: int):
    while len(live_trainers()) >= parallel:
        time.sleep(30)


def ensure_jobs(jobs, steps: int, parallel: int, env: dict):
    for env_name, variant in jobs:
        if is_complete(env_name, variant, steps):
            print(f'skip complete {env_name} {variant}', flush=True)
            continue
        live = live_trainers()
        if (env_name, variant) in live:
            print(
                f'adopt running {env_name} {variant} pid {live[(env_name, variant)]}',
                flush=True,
            )
            continue
        wait_slot(parallel)
        # Re-check after waiting: another controller or residual job may exist.
        if is_complete(env_name, variant, steps):
            print(f'skip complete {env_name} {variant}', flush=True)
            continue
        live = live_trainers()
        if (env_name, variant) in live:
            print(
                f'adopt running {env_name} {variant} pid {live[(env_name, variant)]}',
                flush=True,
            )
            continue
        launch(env_name, variant, steps, env)
    # Drain only the jobs in this phase.
    pending = {(env_name, variant) for env_name, variant in jobs}
    while True:
        remaining = []
        live = live_trainers()
        for env_name, variant in pending:
            if is_complete(env_name, variant, steps):
                continue
            if (env_name, variant) in live:
                remaining.append((env_name, variant))
                continue
            time.sleep(5)
            if is_complete(env_name, variant, steps):
                continue
            failure = run_dir(env_name, variant) / 'failure.json'
            detail = failure.read_text() if failure.exists() else 'missing complete.json'
            raise RuntimeError(f'{env_name} {variant} stopped unexpectedly: {detail}')
        if not remaining:
            break
        print(f'waiting phase jobs: {remaining}', flush=True)
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_steps', type=int, default=1_000_000)
    parser.add_argument('--parallel', type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock_file = (OUT / 'priority.lock').open('w')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = gpu_env()
    plan = {
        'phase_a_pathnce_w': [list(item) for item in PHASE_A],
        'phase_b_uniform': [list(item) for item in PHASE_B],
        'phase_n_sweep': list(PHASE_N_SWEEP),
        'bridgegeo': 'deferred',
    }
    (OUT / 'priority_queue.json').write_text(json.dumps(plan, indent=2) + '\n')
    print('priority controller start', flush=True)
    print(json.dumps(plan), flush=True)

    print('PHASE A: finish PathNCE-W remaining envs', flush=True)
    ensure_jobs(PHASE_A, args.train_steps, args.parallel, env)
    subprocess.run([str(PY), 'scripts/summarize_ctd_seed0.py'], cwd=ROOT, check=False)

    print('PHASE B: DTRL-U / CTD-U on cube-double and puzzle', flush=True)
    ensure_jobs(PHASE_B, args.train_steps, args.parallel, env)
    subprocess.run([str(PY), 'scripts/summarize_ctd_seed0.py'], cwd=ROOT, check=False)

    print('PHASE C: puzzle candidate-count sweep', flush=True)
    # Wait until GPU trainers are idle so eval can use the device cleanly.
    while live_trainers():
        time.sleep(30)
    code = subprocess.run(
        [str(PY), 'scripts/eval_ctd_puzzle_n_sweep.py'],
        cwd=ROOT,
        env=env,
    ).returncode
    if code:
        raise RuntimeError(f'puzzle N sweep failed with code {code}')

    print('PHASE D: write NEXT_DIRECTION.md', flush=True)
    subprocess.run([str(PY), 'scripts/summarize_ctd_seed0.py'], cwd=ROOT, check=False)
    subprocess.run([str(PY), 'scripts/write_ctd_next_direction.py'], cwd=ROOT, check=True)
    print('priority controller finished', flush=True)


if __name__ == '__main__':
    main()
