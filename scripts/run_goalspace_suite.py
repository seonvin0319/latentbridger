"""Run the five-job seed-0 goal-space pilot in its mandated order."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'exp' / 'goalspace_transitive_distance'
QUEUE = (
    ('puzzle_3x3', 'gsdtrl_weighted'),
    ('cube_double', 'gsdtrl_weighted'),
    ('puzzle_3x3', 'gsctd_learned_temp'),
    ('cube_double', 'gsctd_learned_temp'),
    ('puzzle_3x3', 'gsctd_fixed'),
)


def child_env():
    env = dict(os.environ)
    env.pop('JAX_PLATFORMS', None)
    if env.get('CUDA_VISIBLE_DEVICES') == '':
        env.pop('CUDA_VISIBLE_DEVICES')
    env.update(
        XLA_PYTHON_CLIENT_PREALLOCATE='false',
        MUJOCO_GL='egl',
        OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',
    )
    return env


def accelerator_names() -> list[str]:
    import jax

    return [str(device) for device in jax.devices()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_steps', type=int, default=1_000_000)
    parser.add_argument('--dataset_dir', default='')
    parser.add_argument('--allow-cpu', action='store_true')
    parser.add_argument('--skip-n-sweep', action='store_true')
    args = parser.parse_args()
    if args.train_steps < 1:
        raise ValueError('--train_steps must be positive')
    devices = accelerator_names()
    OUT.mkdir(parents=True, exist_ok=True)
    accelerator_visible = any(
        ('gpu' in item.lower() or 'tpu' in item.lower()) for item in devices
    )
    preflight = {
        'observed_at_utc': datetime.now(timezone.utc).isoformat(),
        'devices': devices,
        'train_steps': args.train_steps,
        'accelerator_visible': accelerator_visible,
        'long_cpu_override': args.allow_cpu,
        'status': (
            'blocked_no_accelerator'
            if args.train_steps >= 100_000 and not args.allow_cpu and not accelerator_visible
            else 'passed'
        ),
    }
    (OUT / 'preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
    if args.train_steps >= 100_000 and not args.allow_cpu:
        if not accelerator_visible:
            raise RuntimeError(
                'No GPU/TPU is visible; refusing a long CPU run. '
                'Fix the accelerator or explicitly pass --allow-cpu.'
            )

    lock = (OUT / 'suite.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (OUT / 'queue_seed0.json').write_text(json.dumps({
        'seed': 0,
        'train_steps': args.train_steps,
        'devices': devices,
        'queue': [dict(env=env, variant=variant) for env, variant in QUEUE],
    }, indent=2) + '\n')

    env = child_env()
    for env_name, variant in QUEUE:
        run = OUT / env_name / variant / 'seed0'
        run.mkdir(parents=True, exist_ok=True)
        complete = run / 'complete.json'
        if complete.exists():
            record = json.loads(complete.read_text())
            if int(record.get('steps', 0)) == args.train_steps:
                print(f'skip complete {env_name} {variant}', flush=True)
                continue
        argv = [
            sys.executable,
            'main_ctd_pathbridger.py',
            '--env', env_name,
            '--variant', variant,
            '--seed', '0',
            '--steps', str(args.train_steps),
            '--run-dir', str(run),
        ]
        if args.dataset_dir:
            argv.extend(['--dataset_dir', args.dataset_dir])
        if list((run / 'checkpoints').glob('params_*.pkl')):
            argv.append('--resume')
        print(f'start {env_name} {variant}', flush=True)
        with (run / 'run.log').open('a') as log:
            subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'done {env_name} {variant}', flush=True)

    if args.train_steps == 1_000_000 and not args.skip_n_sweep:
        subprocess.run(
            [sys.executable, 'scripts/eval_goalspace_puzzle_n_sweep.py'],
            cwd=ROOT,
            env=env,
            check=True,
        )
    subprocess.run(
        [sys.executable, 'scripts/summarize_goalspace_seed0.py'],
        cwd=ROOT,
        env=env,
        check=True,
    )
    print('goal-space pilot finished; no additional environments scheduled', flush=True)


if __name__ == '__main__':
    main()
