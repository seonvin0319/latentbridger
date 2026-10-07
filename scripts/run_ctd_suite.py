"""Run the seed-0 CTD methods sequentially after the CPB queue is complete."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_steps', type=int, default=1_000_000)
    parser.add_argument('--allow-before-queue', action='store_true')
    args = parser.parse_args()
    output = ROOT / 'exp' / 'ctd_pathbridger'
    output.mkdir(parents=True, exist_ok=True)
    blockers = queue_blockers()
    if blockers and not args.allow_before_queue:
        raise RuntimeError('CPB queue is not finished:\n' + '\n'.join(blockers))
    lock_file = (output / 'suite.lock').open('w')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(
        os.environ,
        XLA_PYTHON_CLIENT_PREALLOCATE='false',
        MUJOCO_GL='egl',
        OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',
    )
    queue = [
        {'position': position, 'variant': variant, 'env': name, 'seed': 0}
        for position, variant in enumerate(METHOD_ORDER, start=1)
        for name in ENV_ORDER
    ]
    (output / 'queue_seed0.json').write_text(json.dumps(queue, indent=2) + '\n')

    def run_one(name, variant):
        seed = 0
        run = output / name / variant / 'seed0'
        run.mkdir(parents=True, exist_ok=True)
        complete = run / 'complete.json'
        if complete.exists() and int(json.loads(complete.read_text()).get('steps', 0)) == args.train_steps:
            print(f'skip complete {name} {variant} seed {seed}', flush=True)
            return
        argv = [
            sys.executable, 'main_ctd_pathbridger.py',
            '--env', name, '--variant', variant, '--seed', str(seed),
            '--steps', str(args.train_steps), '--run-dir', str(run),
        ]
        if list((run / 'checkpoints').glob('params_*.pkl')):
            argv.append('--resume')
        log = (run / 'run.log').open('a')
        print(f'start {name} {variant} seed0', flush=True)
        code = subprocess.run(
            argv,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        ).returncode
        log.close()
        if code:
            (run / 'failure.json').write_text(json.dumps({
                'returncode': code,
                'reason': 'implementation_or_runtime_failure',
            }) + '\n')
            raise RuntimeError(f'{name} {variant} seed0 failed with code {code}')

    for method_index, variant in enumerate(METHOD_ORDER, start=1):
        print(f'method {method_index}/8 {variant}', flush=True)
        for name in ENV_ORDER:
            run_one(name, variant)
        subprocess.run([sys.executable, 'scripts/summarize_ctd_seed0.py'], cwd=ROOT, check=False)
    print('ctd suite finished', flush=True)


if __name__ == '__main__':
    main()
