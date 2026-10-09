#!/usr/bin/env python3
"""Phase-2 learned-goalspace queue (PCA / random / multihorizon).

Does not touch the live FutureNCE → LGS_TRL_W_FROZEN queue.  Waits/skips until
archival puzzle LGS_TRL_W_FROZEN is complete, then runs the comparison suite.
"""

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
sys.path.insert(0, str(ROOT))

from scripts.run_learned_goalspace_queue import (  # noqa: E402
    EVALUATION_SETTINGS,
    child_environment,
    downstream_complete as archival_downstream_complete,
    downstream_dir,
)

FUTURE_VARIANT = 'fullobs_future_nce'
MH_VARIANT = 'fullobs_multihorizon_nce'
ENV = 'puzzle_3x3'
ENV_NAME = 'puzzle-3x3-play-v0'
METHOD_VARIANT = {
    'LGS_TRL_W_FROZEN': FUTURE_VARIANT,
    'PCA16_GS_TRL_W': 'pca16',
    'RANDOM16_GS_TRL_W': 'random16',
    'MH_LGS_TRL_W_FROZEN': MH_VARIANT,
}
# Gate: only auto-start MH downstream if button_exact_state_accuracy exceeds this.
MH_EXACT_ACC_THRESHOLD = 0.05

PHASE2_ORDER = (
    ('fixed', 'pca16'),
    ('fixed', 'random16'),
    ('downstream', 'PCA16_GS_TRL_W'),
    ('downstream', 'RANDOM16_GS_TRL_W'),
    ('pretrain_mh', ''),
    ('probes_mh', ''),
    ('downstream_mh', 'MH_LGS_TRL_W_FROZEN'),
)


def archival_lgs_complete(base: Path, steps: int, *, smoke: bool, episodes: int) -> bool:
    return archival_downstream_complete(
        base,
        ENV,
        'LGS_TRL_W_FROZEN',
        steps,
        smoke=smoke,
        episodes=episodes,
    )


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
    method: str,
    episodes: int,
    variant: str,
) -> bool:
    try:
        payload = json.loads(path.read_text())
        return (
            int(payload['checkpoint']) == checkpoint
            and int(payload['h']) == horizon
            and payload['env'] == ENV_NAME
            and payload['method'] == method
            and payload['variant'] == variant
            and int(payload['seed']) == 0
            and int(payload['episodes_per_task']) == episodes
            and int(payload['N']) == EVALUATION_SETTINGS[ENV][0]
            and float(payload['temperature']) == EVALUATION_SETTINGS[ENV][1]
        )
    except (
        FileNotFoundError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def downstream_complete(
    base: Path,
    method: str,
    steps: int,
    *,
    smoke: bool,
    episodes: int,
) -> bool:
    run = downstream_dir(base, ENV, method)
    variant = METHOD_VARIANT[method]
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
            method=method,
            episodes=episodes,
            variant=variant,
        )
    return base_ok and all(
        _evaluation_json_valid(
            run / f'evaluation_{step}_h{horizon}.json',
            checkpoint=step,
            horizon=horizon,
            method=method,
            episodes=episodes,
            variant=variant,
        )
        for step in (100_000, 300_000, 500_000, 800_000, 1_000_000)
        for horizon in (5, 2, 1)
    )


def fixed_dir(base: Path, kind: str) -> Path:
    return base / kind / ENV / 'seed0'


def fixed_complete(base: Path, kind: str) -> bool:
    run = fixed_dir(base, kind)
    return (run / 'representation.pkl').is_file() and (run / 'metadata.json').is_file()


def mh_pretrain_dir(base: Path) -> Path:
    return base / 'multihorizon' / 'pretrain' / ENV / MH_VARIANT / 'seed0'


def mh_probe_dir(base: Path) -> Path:
    return base / 'multihorizon' / 'probes' / ENV / MH_VARIANT / 'seed0'


def mh_pretrain_complete(base: Path, steps: int) -> bool:
    run = mh_pretrain_dir(base)
    try:
        complete = int(json.loads((run / 'complete.json').read_text())['steps']) == steps
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
        return False
    return complete and (run / 'checkpoints' / f'params_{steps}.pkl').is_file()


def probes_complete(directory: Path, checkpoints: tuple[int, ...], variant: str) -> bool:
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
        if any(row.get('variant') != variant or row.get('seed') != '0' or row.get('env') != ENV_NAME for row in rows):
            return False
    return True


def _learned_exact_acc(probe_dir: Path, checkpoint: int) -> float | None:
    path = probe_dir / 'probe_metrics.csv'
    try:
        with path.open(newline='') as file:
            rows = list(csv.DictReader(file))
    except FileNotFoundError:
        return None
    values = [
        float(row['value'])
        for row in rows
        if row.get('representation') == 'learned_E'
        and row.get('metric') == 'button_exact_state_accuracy'
        and int(row.get('checkpoint', -1)) == checkpoint
    ]
    if not values:
        return None
    return float(values[-1])


def mh_downstream_allowed(probe_dir: Path, checkpoint: int, *, force: bool, gate_file: Path | None) -> tuple[bool, str]:
    if force:
        return True, 'forced via --force-mh-downstream'
    if gate_file is not None and gate_file.is_file():
        return True, f'gate file present: {gate_file}'
    exact = _learned_exact_acc(probe_dir, checkpoint)
    if exact is not None and exact > MH_EXACT_ACC_THRESHOLD:
        return True, f'learned_E button_exact_state_accuracy={exact:.4f} > {MH_EXACT_ACC_THRESHOLD}'
    reason = (
        f'skip MH downstream: exact_acc={exact!r} (need > {MH_EXACT_ACC_THRESHOLD}), '
        'no gate file, and --force-mh-downstream not set'
    )
    return False, reason


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


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run(args) -> None:
    base = (
        Path(args.output_root)
        if args.output_root
        else ROOT / ('exp/learned_goalspace-smoke' if args.smoke else 'exp/learned_goalspace')
    )
    queue_lock = None
    if not args.dry_run:
        base.mkdir(parents=True, exist_ok=True)
        queue_lock = (base / '.phase2.queue.lock').open('w')
        try:
            fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another phase-2 learned-goalspace queue owns {base}') from error
        queue_lock.write(f'{os.getpid()}\n')
        queue_lock.flush()
        (base / 'PHASE2_COMPLETE').unlink(missing_ok=True)

    archival_steps = args.smoke_downstream_steps if args.smoke else 1_000_000
    pretrain_steps = args.smoke_pretrain_steps if args.smoke else 500_000
    downstream_steps = args.smoke_downstream_steps if args.smoke else 1_000_000
    probe_steps = (pretrain_steps,) if args.smoke else (100_000, 300_000, 500_000)
    evaluation_episodes = args.smoke_episodes if args.smoke else 50
    child_env = child_environment(args.gpu)
    python = sys.executable
    gate_file = Path(args.mh_gate_file) if args.mh_gate_file else base / 'multihorizon' / 'ALLOW_MH_DOWNSTREAM'

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

    # Wait / skip until archival FutureNCE LGS puzzle cell is done.
    if not archival_lgs_complete(
        base,
        archival_steps,
        smoke=args.smoke,
        episodes=evaluation_episodes,
    ):
        if args.wait_for_archival:
            raise RuntimeError(
                f'Archival LGS_TRL_W_FROZEN puzzle not complete under {base}; '
                're-run after the live queue finishes (or pass smoke artifacts).'
            )
        print(
            f'WAIT archival LGS_TRL_W_FROZEN puzzle incomplete under {base}; exiting without GPU work.',
            flush=True,
        )
        return

    print('Archival LGS_TRL_W_FROZEN puzzle complete; starting phase-2 order.', flush=True)

    for kind, name in PHASE2_ORDER:
        if kind == 'fixed':
            run_dir = fixed_dir(base, name)
            if fixed_complete(base, name):
                print(f'SKIP complete fixed {name}', flush=True)
                continue
            _run(
                [
                    python,
                    'main_fit_fixed_representations.py',
                    '--env',
                    ENV,
                    '--kind',
                    name,
                    '--seed',
                    '0',
                    '--output-dir',
                    str(run_dir),
                ],
                args,
                child_env,
            )
            if not args.dry_run and not fixed_complete(base, name):
                raise RuntimeError(f'Fixed representation missing: {run_dir}')
            continue

        if kind == 'downstream':
            method = name
            run_dir = downstream_dir(base, ENV, method)
            if downstream_complete(
                base,
                method,
                downstream_steps,
                smoke=args.smoke,
                episodes=evaluation_episodes,
            ):
                print(f'SKIP complete downstream {method}', flush=True)
                continue
            fixed_kind = 'pca16' if method.startswith('PCA') else 'random16'
            representation = fixed_dir(base, fixed_kind) / 'representation.pkl'
            command = [
                python,
                'main_learned_goalspace.py',
                '--env',
                ENV,
                '--method',
                method,
                '--steps',
                str(downstream_steps),
                '--episodes',
                str(evaluation_episodes),
                '--batch-size',
                str(args.smoke_batch_size if args.smoke else 1024),
                '--pretrain-checkpoint',
                str(representation),
                '--pretrain-step',
                '0',
                '--run-dir',
                str(run_dir),
            ]
            if list((run_dir / 'checkpoints').glob('params_*.pkl')):
                command.append('--resume')
            if args.smoke:
                command.append('--smoke')
            _run(command, args, child_env)
            if not args.dry_run and not downstream_complete(
                base,
                method,
                downstream_steps,
                smoke=args.smoke,
                episodes=evaluation_episodes,
            ):
                raise RuntimeError(f'Downstream incomplete: {run_dir}')
            continue

        if kind == 'pretrain_mh':
            run_dir = mh_pretrain_dir(base)
            if mh_pretrain_complete(base, pretrain_steps):
                print('SKIP complete multihorizon pretrain', flush=True)
                continue
            command = [
                python,
                'main_learned_goal_pretrain.py',
                '--env',
                ENV,
                '--variant',
                MH_VARIANT,
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
            if not args.dry_run and not mh_pretrain_complete(base, pretrain_steps):
                raise RuntimeError(f'Multihorizon pretrain incomplete: {run_dir}')
            continue

        if kind == 'probes_mh':
            probe_dir = mh_probe_dir(base)
            if probes_complete(probe_dir, probe_steps, MH_VARIANT):
                print('SKIP complete multihorizon probes', flush=True)
            else:
                _run(
                    [
                        python,
                        'main_learned_goal_probes.py',
                        '--env',
                        ENV,
                        '--variant',
                        MH_VARIANT,
                        '--pretrain-dir',
                        str(mh_pretrain_dir(base)),
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
            if not args.dry_run and not probes_complete(probe_dir, probe_steps, MH_VARIANT):
                raise RuntimeError(f'Multihorizon probes incomplete: {probe_dir}')
            exact = _learned_exact_acc(probe_dir, pretrain_steps)
            summary = {
                'env': ENV,
                'variant': MH_VARIANT,
                'checkpoint': pretrain_steps,
                'button_exact_state_accuracy': exact,
                'mh_exact_acc_threshold': MH_EXACT_ACC_THRESHOLD,
                'note': (
                    'MH_LGS_TRL_W_FROZEN starts only if --force-mh-downstream, '
                    f'gate file {gate_file}, or exact_acc > {MH_EXACT_ACC_THRESHOLD}.'
                ),
            }
            if not args.dry_run:
                _write_json(base / 'multihorizon' / 'probe_summary.json', summary)
            print(json.dumps(summary), flush=True)
            continue

        if kind == 'downstream_mh':
            method = name
            allowed, reason = mh_downstream_allowed(
                mh_probe_dir(base),
                pretrain_steps,
                force=args.force_mh_downstream,
                gate_file=gate_file,
            )
            print(reason, flush=True)
            if not allowed:
                if not args.dry_run:
                    _write_json(
                        base / 'multihorizon' / 'mh_downstream_skipped.json',
                        {'reason': reason, 'threshold': MH_EXACT_ACC_THRESHOLD},
                    )
                continue
            run_dir = downstream_dir(base, ENV, method)
            if downstream_complete(
                base,
                method,
                downstream_steps,
                smoke=args.smoke,
                episodes=evaluation_episodes,
            ):
                print(f'SKIP complete downstream {method}', flush=True)
                continue
            pretrained = mh_pretrain_dir(base) / 'checkpoints' / f'params_{pretrain_steps}.pkl'
            command = [
                python,
                'main_learned_goalspace.py',
                '--env',
                ENV,
                '--method',
                method,
                '--steps',
                str(downstream_steps),
                '--episodes',
                str(evaluation_episodes),
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
                method,
                downstream_steps,
                smoke=args.smoke,
                episodes=evaluation_episodes,
            ):
                raise RuntimeError(f'MH downstream incomplete: {run_dir}')
            continue

        raise ValueError(f'Unknown phase-2 kind {kind!r}')

    if not args.dry_run:
        (base / 'PHASE2_COMPLETE').write_text('complete\n')
    print('PHASE2 queue finished.', flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument('--dry-run', action='store_true')
    result.add_argument('--smoke', action='store_true')
    result.add_argument('--gpu', default='0')
    result.add_argument('--cpu-set', default='')
    result.add_argument('--output-root', default='')
    result.add_argument('--smoke-pretrain-steps', type=int, default=2)
    result.add_argument('--smoke-downstream-steps', type=int, default=2)
    result.add_argument('--smoke-batch-size', type=int, choices=(512, 1024), default=512)
    result.add_argument('--smoke-episodes', type=int, default=1)
    result.add_argument('--smoke-probe-samples', type=int, default=256)
    result.add_argument(
        '--wait-for-archival',
        action='store_true',
        help='Error instead of soft-exit when archival LGS is incomplete.',
    )
    result.add_argument(
        '--force-mh-downstream',
        action='store_true',
        help='Start MH_LGS_TRL_W_FROZEN even if probe exact-acc gate fails.',
    )
    result.add_argument(
        '--mh-gate-file',
        default='',
        help='Optional path; if present, allows MH downstream (default: <base>/multihorizon/ALLOW_MH_DOWNSTREAM).',
    )
    return result


if __name__ == '__main__':
    run(parser().parse_args())
