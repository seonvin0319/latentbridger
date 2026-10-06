"""Durable sequential suite; each subprocess finishes before the next starts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
ENVS = ('cube_single', 'cube_double', 'puzzle_3x3', 'antmaze_medium')


def schedule():
    for env in ENVS:
        if env in ENVS[:2]:
            yield env, 'cpb_rank_only', 0
        for seed in range(3):
            yield env, 'cpb_full', seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='exp/contrastive_pathbridger')
    p.add_argument('--gates-passed', action='store_true', help='Require existing test and smoke evidence instead of rerunning gates')
    args = p.parse_args()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / 'suite.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false', MUJOCO_GL='egl', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    started = time.time()
    def command(argv, log):
        with log.open('a') as file:
            proc = subprocess.Popen([sys.executable, *argv], cwd=ROOT, env=env, stdout=file, stderr=subprocess.STDOUT)
            count = -1
            try:
                while True:
                    try:
                        code = proc.wait(timeout=30)
                        if code:
                            raise subprocess.CalledProcessError(code, proc.args)
                        break
                    except subprocess.TimeoutExpired:
                        current = len(list(output.glob('*/*/seed*/evaluation_*.json')))
                        if current != count:
                            summarize()
                            count = current
            except BaseException:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait()
                raise
    def summarize():
        subprocess.run([sys.executable, 'scripts/summarize_contrastive_pathbridger.py', '--output', str(output)], cwd=ROOT, check=True)
    try:
        if not args.gates_passed:
            command(['-m', 'pytest', '-q'], output / 'tests.log')
            (output / 'tests_passed.json').write_text(json.dumps({'time': time.time()}))
            command(['main_contrastive_pathbridger.py', '--steps', '2000', '--smoke', '--run-dir', str(output / 'smoke')], output / 'smoke.log')
        if not (output / 'tests_passed.json').exists() or not (output / 'smoke/complete.json').exists():
            raise RuntimeError('Tests and 2k smoke evidence required')
        for name, variant, seed in schedule():
            run = output / name / variant / f'seed{seed}'
            run.mkdir(parents=True, exist_ok=True)
            if (run / 'complete.json').exists():
                continue
            (output / 'status.json').write_text(json.dumps(dict(status='running', env=name, variant=variant, seed=seed, started=started)))
            # A failed partial run is explicit: never silently overwrite or skip missing evaluations.
            if list((run / 'checkpoints').glob('params_*.pkl')):
                raise RuntimeError(f'Partial run needs checkpoint/evaluation recovery: {run}')
            command(['main_contrastive_pathbridger.py', '--env', name, '--variant', variant, '--seed', str(seed), '--run-dir', str(run)], run / 'run.log')
            summarize()
        (output / 'status.json').write_text(json.dumps(dict(status='complete', wall_seconds=time.time()-started)))
    except BaseException as exc:
        (output / 'failure.json').write_text(json.dumps(dict(error=str(exc), time=time.time(), wall_seconds=time.time()-started)))
        raise
    finally:
        summarize()

if __name__ == '__main__':
    main()
