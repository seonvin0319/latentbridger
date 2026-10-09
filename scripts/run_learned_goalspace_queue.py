#!/usr/bin/env python3
"""Sequential, resumable learned-goalspace queue.  Never parallelizes GPUs."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANT = 'fullobs_future_nce'
METHODS = ('LGS_TRL_W_FROZEN', 'LGSDTRL_W_FROZEN')
ENV_NAMES = {
    'puzzle_3x3': 'puzzle-3x3-play-v0',
    'cube_double': 'cube-double-play-v0',
}
EVALUATION_SETTINGS = {
    'puzzle_3x3': (32, 1.0),
    'cube_double': (8, 0.25),
}
ORDER = (
    ('pretrain', 'puzzle_3x3', ''),
    ('downstream', 'puzzle_3x3', 'LGS_TRL_W_FROZEN'),
    ('downstream', 'puzzle_3x3', 'LGSDTRL_W_FROZEN'),
    ('pretrain', 'cube_double', ''),
    ('downstream', 'cube_double', 'LGS_TRL_W_FROZEN'),
    ('downstream', 'cube_double', 'LGSDTRL_W_FROZEN'),
)


def _nvidia_library_path() -> str:
    names = (
        'cusparse',
        'cublas',
        'cuda_runtime',
        'cudnn',
        'cufft',
        'cusolver',
        'curand',
        'nvjitlink',
    )
    for entry in sys.path:
        base = Path(entry) / 'nvidia'
        if (base / 'cusparse' / 'lib').is_dir():
            return ':'.join(str(path) for name in names if (path := base / name / 'lib').is_dir())
    return ''


def child_environment(gpu: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',
        NUMEXPR_NUM_THREADS='1',
        TF_NUM_INTRAOP_THREADS='1',
        TF_NUM_INTEROP_THREADS='1',
        EIGEN_NUM_THREADS='1',
        CUDA_VISIBLE_DEVICES=str(gpu),
        JAX_PLATFORMS='cuda',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',
    )
    pins = '--xla_cpu_multi_thread_eigen=false --xla_gpu_force_compilation_parallelism=1 --xla_gpu_autotune_level=0'
    current = env.get('XLA_FLAGS', '')
    env['XLA_FLAGS'] = f'{pins} {current}'.strip()
    nvidia_libraries = _nvidia_library_path()
    if nvidia_libraries:
        current_ld = env.get('LD_LIBRARY_PATH', '')
        env['LD_LIBRARY_PATH'] = f'{nvidia_libraries}:{current_ld}' if current_ld else nvidia_libraries
    return env


def _json_steps(path: Path, expected: int) -> bool:
    try:
        return int(json.loads(path.read_text())['steps']) == expected
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False


def _evaluation_json_valid(
    path: Path,
    *,
    checkpoint: int,
    horizon: int,
    env: str,
    method: str,
    episodes: int,
) -> bool:
    try:
        payload = json.loads(path.read_text())
        return (
            int(payload['checkpoint']) == checkpoint
            and int(payload['h']) == horizon
            and payload['env'] == ENV_NAMES[env]
            and payload['method'] == method
            and payload['variant'] == VARIANT
            and int(payload['seed']) == 0
            and int(payload['episodes_per_task']) == episodes
            and int(payload['N']) == EVALUATION_SETTINGS[env][0]
            and float(payload['temperature']) == EVALUATION_SETTINGS[env][1]
        )
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def pretrain_dir(base: Path, env: str) -> Path:
    return base / 'pretrain' / env / VARIANT / 'seed0'


def downstream_dir(base: Path, env: str, method: str) -> Path:
    return base / 'downstream' / env / method / 'seed0'


def pretrain_complete(base: Path, env: str, steps: int) -> bool:
    run = pretrain_dir(base, env)
    return _json_steps(run / 'complete.json', steps) and (run / 'checkpoints' / f'params_{steps}.pkl').is_file()


def downstream_complete(
    base: Path,
    env: str,
    method: str,
    steps: int,
    *,
    smoke: bool,
    episodes: int | None = None,
) -> bool:
    run = downstream_dir(base, env, method)
    base_ok = (
        _json_steps(run / 'complete.json', steps)
        and (run / 'checkpoints' / f'params_{steps}.pkl').is_file()
        and (run / 'downstream_results.csv').is_file()
    )
    if smoke:
        return base_ok and _evaluation_json_valid(
            run / 'smoke_evaluation.json',
            checkpoint=steps,
            horizon=5,
            env=env,
            method=method,
            episodes=episodes if episodes is not None else 1,
        )
    return base_ok and all(
        _evaluation_json_valid(
            run / f'evaluation_{step}_h{horizon}.json',
            checkpoint=step,
            horizon=horizon,
            env=env,
            method=method,
            episodes=episodes if episodes is not None else 50,
        )
        for step in (100_000, 300_000, 500_000, 800_000, 1_000_000)
        for horizon in (5, 2, 1)
    )


def probes_complete(
    directory: Path,
    checkpoints: tuple[int, ...],
    env: str,
) -> bool:
    expected = set(checkpoints)
    for name in ('representation_metrics.csv', 'probe_metrics.csv'):
        try:
            with (directory / name).open(newline='') as file:
                rows = list(csv.DictReader(file))
        except FileNotFoundError:
            return False
        try:
            observed = {int(row['checkpoint']) for row in rows}
        except (KeyError, TypeError, ValueError):
            return False
        if not rows or not expected.issubset(observed):
            return False
        if any(
            row.get('variant') != VARIANT or row.get('seed') != '0' or row.get('env') != ENV_NAMES[env] for row in rows
        ):
            return False
    return True


def _run(command: list[str], args, env) -> None:
    rendered = ' '.join(command)
    print(rendered, flush=True)
    if args.dry_run:
        return
    cpu_set = args.cpu_set
    if not cpu_set:
        allowed = sorted(os.sched_getaffinity(0))
        cpu_set = ','.join(str(cpu) for cpu in allowed[:8])
    if not cpu_set:
        raise RuntimeError('No CPUs are available for the mandatory taskset pin.')
    subprocess.run(
        ['taskset', '-c', cpu_set, *command],
        cwd=ROOT,
        env=env,
        check=True,
    )


def _aggregate_csv(base: Path, name: str) -> None:
    sources = sorted(path for path in base.rglob(name) if path.parent != base and path.is_file())
    rows: list[dict[str, str]] = []
    fields: list[str] = []
    for source in sources:
        with source.open(newline='') as file:
            reader = csv.DictReader(file)
            for field in reader.fieldnames or ():
                if field not in fields:
                    fields.append(field)
            rows.extend(reader)
    if not rows:
        return
    destination = base / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    with temporary.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)


def _aggregate_outputs(base: Path) -> None:
    for name in (
        'representation_metrics.csv',
        'probe_metrics.csv',
        'downstream_results.csv',
    ):
        _aggregate_csv(base, name)


def run(args) -> None:
    base = (
        Path(args.output_root)
        if args.output_root
        else ROOT / ('exp/learned_goalspace-smoke' if args.smoke else 'exp/learned_goalspace')
    )
    queue_lock = None
    if not args.dry_run:
        base.mkdir(parents=True, exist_ok=True)
        queue_lock = (base / '.queue.lock').open('w')
        try:
            fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another learned-goalspace queue owns {base}') from error
        queue_lock.write(f'{os.getpid()}\n')
        queue_lock.flush()
        for marker in ('QUEUE_COMPLETE', 'SMOKE_COMPLETE'):
            (base / marker).unlink(missing_ok=True)
    pretrain_steps = args.smoke_pretrain_steps if args.smoke else 500_000
    downstream_steps = args.smoke_downstream_steps if args.smoke else 1_000_000
    probe_steps = (pretrain_steps,) if args.smoke else (100_000, 300_000, 500_000)
    evaluation_episodes = args.smoke_episodes if args.smoke else 50
    child_env = child_environment(args.gpu)
    python = sys.executable
    _run(
        [
            python,
            '-c',
            (
                'import jax; '
                'assert jax.default_backend() == "gpu", jax.default_backend(); '
                'print("JAX backend:", jax.default_backend(), '
                '"devices:", jax.devices(), flush=True)'
            ),
        ],
        args,
        child_env,
    )

    for kind, env_name, method in ORDER:
        if kind == 'pretrain':
            run_dir = pretrain_dir(base, env_name)
            if pretrain_complete(base, env_name, pretrain_steps):
                print(f'SKIP complete pretrain {env_name}', flush=True)
            else:
                command = [
                    python,
                    'main_learned_goal_pretrain.py',
                    '--env',
                    env_name,
                    '--steps',
                    str(pretrain_steps),
                    '--batch-size',
                    str(args.smoke_batch_size if args.smoke else 1024),
                    '--run-dir',
                    str(run_dir),
                ]
                if list((run_dir / 'checkpoints').glob('params_*.pkl')):
                    command.append('--resume')
                _run(command, args, child_env)
                if not args.dry_run and not pretrain_complete(base, env_name, pretrain_steps):
                    raise RuntimeError(f'Pretraining returned without complete artifacts: {run_dir}')
            probe_dir = base / 'probes' / env_name / VARIANT / 'seed0'
            if not probes_complete(probe_dir, probe_steps, env_name):
                _run(
                    [
                        python,
                        'main_learned_goal_probes.py',
                        '--env',
                        env_name,
                        '--pretrain-dir',
                        str(run_dir),
                        '--output-dir',
                        str(probe_dir),
                        '--checkpoints',
                        *(str(step) for step in probe_steps),
                        '--max-samples',
                        str(args.smoke_probe_samples if args.smoke else 20000),
                    ],
                    args,
                    child_env,
                )
            if not args.dry_run and not probes_complete(probe_dir, probe_steps, env_name):
                raise RuntimeError(f'Probe command returned without required outputs: {probe_dir}')
            if not args.dry_run:
                _aggregate_outputs(base)
            continue

        run_dir = downstream_dir(base, env_name, method)
        if downstream_complete(
            base,
            env_name,
            method,
            downstream_steps,
            smoke=args.smoke,
            episodes=evaluation_episodes,
        ):
            print(f'SKIP complete downstream {env_name} {method}', flush=True)
            continue
        pretrained = pretrain_dir(base, env_name) / 'checkpoints' / f'params_{pretrain_steps}.pkl'
        command = [
            python,
            'main_learned_goalspace.py',
            '--env',
            env_name,
            '--method',
            method,
            '--steps',
            str(downstream_steps),
            '--episodes',
            str(args.smoke_episodes if args.smoke else 50),
            '--batch-size',
            str(args.smoke_batch_size if args.smoke else 1024),
            '--pretrain-checkpoint',
            str(pretrained),
            '--pretrain-step',
            str(pretrain_steps),
            '--run-dir',
            str(run_dir),
        ]
        if list((run_dir / 'checkpoints').glob('params_*.pkl')):
            command.append('--resume')
        if args.smoke:
            command.extend(('--smoke', '--allow-nonproduction-pretrain'))
        _run(command, args, child_env)
        if not args.dry_run and not downstream_complete(
            base,
            env_name,
            method,
            downstream_steps,
            smoke=args.smoke,
            episodes=evaluation_episodes,
        ):
            raise RuntimeError(f'Downstream command returned without all checkpoints and evaluations: {run_dir}')
        if not args.dry_run:
            _aggregate_outputs(base)

    if not args.dry_run:
        incomplete = [
            (kind, env_name, method)
            for kind, env_name, method in ORDER
            if (
                not pretrain_complete(base, env_name, pretrain_steps)
                if kind == 'pretrain'
                else not downstream_complete(
                    base,
                    env_name,
                    method,
                    downstream_steps,
                    smoke=args.smoke,
                    episodes=evaluation_episodes,
                )
            )
        ]
        if incomplete:
            raise RuntimeError(f'Queue postcondition failed: {incomplete}')
        _aggregate_outputs(base)
        (base / ('SMOKE_COMPLETE' if args.smoke else 'QUEUE_COMPLETE')).write_text('complete\n')


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument('--dry-run', action='store_true')
    result.add_argument('--smoke', action='store_true')
    result.add_argument('--gpu', default='0')
    result.add_argument(
        '--cpu-set',
        default='',
        help='taskset CPU list; default is the first eight CPUs allowed to this process',
    )
    result.add_argument('--output-root', default='')
    result.add_argument('--smoke-pretrain-steps', type=int, default=2)
    result.add_argument('--smoke-downstream-steps', type=int, default=2)
    result.add_argument('--smoke-batch-size', type=int, choices=(512, 1024), default=512)
    result.add_argument('--smoke-episodes', type=int, default=1)
    result.add_argument('--smoke-probe-samples', type=int, default=256)
    return result


if __name__ == '__main__':
    run(parser().parse_args())
