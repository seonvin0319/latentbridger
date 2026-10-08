"""Run the seed-0 CTD queue with a fixed GPU worker pool after CPB completes."""

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
CPB_ROOT = Path('/home/shchoi/latentbridger/exp/contrastive_pathbridger')
from configs.ctd.common import METHOD_ORDER
ENV_ORDER = ('cube_double', 'puzzle_3x3', 'antmaze_medium', 'cube_single')
REQUIRED_CPB = ('puzzle_3x3', 'antmaze_medium')


def queue_blockers():
    """Return human-readable reasons the live CPB queue is not finished."""

    reasons = []
    for env in REQUIRED_CPB:
        run = CPB_ROOT / env / 'cpb_rank_only' / 'seed0'
        complete = run / 'complete.json'
        if not complete.exists():
            reasons.append(f'missing {complete}')
            continue
        record = json.loads(complete.read_text())
        if int(record.get('steps', 0)) != 1_000_000:
            reasons.append(f'{complete} steps={record.get("steps")}')
        for name in (
            'evaluation_1000000_h1.json',
            'evaluation_1000000_h2.json',
            'evaluation_1000000_h5.json',
            'checkpoints/params_1000000.pkl',
        ):
            if not (run / name).exists():
                reasons.append(f'missing {run / name}')
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / 'cmdline').read_bytes().replace(b'\x00', b' ').decode(errors='replace')
        except OSError:
            continue
        if 'run_contrastive_pathbridger_suite.py' in command or 'main_contrastive_pathbridger.py' in command:
            reasons.append(f'live CPB process {entry.name}')
    return reasons


def gpu_env():
    """Build a child env that cannot inherit CPU-smoke platform flags."""

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


def build_queue(*, prioritize_puzzle: bool):
    """Method-major queue; optionally finish every puzzle run before other envs."""

    if not prioritize_puzzle:
        return [
            {'variant': variant, 'env': name, 'seed': 0}
            for variant in METHOD_ORDER
            for name in ENV_ORDER
        ]
    other = tuple(env for env in ENV_ORDER if env != 'puzzle_3x3')
    queue = [
        {'variant': variant, 'env': 'puzzle_3x3', 'seed': 0}
        for variant in METHOD_ORDER
    ]
    queue += [
        {'variant': variant, 'env': name, 'seed': 0}
        for variant in METHOD_ORDER
        for name in other
    ]
    return queue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_steps', type=int, default=1_000_000)
    parser.add_argument('--parallel', type=int, default=3)
    parser.add_argument('--allow-before-queue', action='store_true')
    parser.add_argument(
        '--prioritize-puzzle',
        action='store_true',
        help='Run puzzle_3x3 for every unfinished method before other environments.',
    )
    args = parser.parse_args()
    if args.parallel < 1:
        raise ValueError('--parallel must be >= 1')
    output = ROOT / 'exp' / 'ctd_pathbridger'
    output.mkdir(parents=True, exist_ok=True)
    blockers = queue_blockers()
    if blockers and not args.allow_before_queue:
        raise RuntimeError('CPB queue is not finished:\n' + '\n'.join(blockers))
    lock_file = (output / 'suite.lock').open('w')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = gpu_env()
    queue = build_queue(prioritize_puzzle=args.prioritize_puzzle)
    (output / 'queue_seed0.json').write_text(json.dumps({
        'parallel': args.parallel,
        'prioritize_puzzle': args.prioritize_puzzle,
        'queue': queue,
    }, indent=2) + '\n')
    print(
        f'parallel={args.parallel}; prioritize_puzzle={args.prioritize_puzzle}; '
        f'queue={len(queue)} jobs',
        flush=True,
    )

    running = []
    failed = []

    def reap(*, wait_for_slot: bool = False, drain: bool = False):
        while running:
            progressed = False
            for slot in list(running):
                code = slot['proc'].poll()
                if code is None:
                    continue
                progressed = True
                running.remove(slot)
                slot['log'].close()
                label = f"{slot['env']} {slot['variant']} seed0"
                if code == 0:
                    print(f'done {label}', flush=True)
                    continue
                (slot['run'] / 'failure.json').write_text(json.dumps({
                    'returncode': code,
                    'reason': 'implementation_or_runtime_failure',
                }) + '\n')
                print(f'failed {label} code {code}', flush=True)
                failed.append(label)
            if wait_for_slot and len(running) < args.parallel:
                return
            if not drain:
                return
            if not running:
                return
            if not progressed:
                time.sleep(30)

    def launch(name, variant):
        seed = 0
        run = output / name / variant / 'seed0'
        run.mkdir(parents=True, exist_ok=True)
        complete = run / 'complete.json'
        if complete.exists() and int(json.loads(complete.read_text()).get('steps', 0)) == args.train_steps:
            print(f'skip complete {name} {variant} seed {seed}', flush=True)
            return False
        argv = [
            sys.executable, 'main_ctd_pathbridger.py',
            '--env', name, '--variant', variant, '--seed', str(seed),
            '--steps', str(args.train_steps), '--run-dir', str(run),
        ]
        if list((run / 'checkpoints').glob('params_*.pkl')):
            argv.append('--resume')
        log = (run / 'run.log').open('a')
        proc = subprocess.Popen(
            argv,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        running.append(dict(proc=proc, run=run, env=name, variant=variant, log=log))
        print(f'start {name} {variant} seed0 pid {proc.pid}', flush=True)
        return True

    for job in queue:
        if failed:
            break
        while len(running) >= args.parallel:
            reap(wait_for_slot=True, drain=True)
            if failed:
                break
        if failed:
            break
        launch(job['env'], job['variant'])
        reap(wait_for_slot=False, drain=False)

    reap(wait_for_slot=False, drain=True)

    subprocess.run([sys.executable, 'scripts/summarize_ctd_seed0.py'], cwd=ROOT, check=False)
    if failed:
        raise RuntimeError('CTD suite stopped after implementation failure: ' + '; '.join(failed))
    print('ctd suite finished', flush=True)


if __name__ == '__main__':
    main()
