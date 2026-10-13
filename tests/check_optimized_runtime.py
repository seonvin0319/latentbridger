"""Compare optimized CLI runs with the installed pre-optimization release.

Check whole checkpoints and every logged loss, exact prefetch resume, the
optimization-off switch, and detached CPU evaluation. Use a new run directory
on every invocation so stale results cannot satisfy the checks.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time

from check_runtime import exact

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline-root', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT/'validation/optimized_runtime.json')
    args = p.parse_args()
    base = ROOT/'validation/optimization_smoke'
    base.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix='cli_', dir=base))
    env = {**os.environ, 'JAX_PLATFORMS': 'cuda', 'CUDA_VISIBLE_DEVICES': '0',
           'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
           'XLA_FLAGS': '--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',
           'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1'}
    records = {}

    def run(name, family, preset, *, baseline=False, evaluate=False, extra=()):
        entry = args.baseline_root/'main.py' if baseline else ROOT/'main.py'
        cmd = [sys.executable, str(entry), '--agent='+str(ROOT/f'configs/{family}/{preset}.py'),
               '--seed=1', '--train_steps=16', '--save_interval=8', '--log_interval=4',
               '--nouse_tqdm', '--noeval_final', '--eval_episodes=1',
               '--eval_interval='+('8' if evaluate else '0'),
               '--save_dir='+str(out), '--run_group='+name, *extra]
        with (out/(name+'.log')).open('w') as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        paths = list((out/'pathbridger'/name).glob('*'))
        assert len(paths) == 1, paths
        return paths[0]

    def checkpoint(run):
        with (run/'checkpoints/params_16.pkl').open('rb') as file:
            return pickle.load(file)

    def losses(run):
        with (run/'train.csv').open() as file:
            return [{key: float(value) for key, value in row.items()
                     if not key.startswith('time/')} for row in csv.DictReader(file)]

    evaluated = []
    for family, preset in [('pbf','puzzle_3x3'), ('pbf','puzzle_4x4'), ('pbf','scene'), ('pbg','scene')]:
        name = family+'_'+preset
        old = run(name+'_baseline', family, preset, baseline=True)
        do_eval = preset=='puzzle_3x3'
        new = run(name+'_optimized', family, preset, evaluate=do_eval)
        exact(checkpoint(old), checkpoint(new))
        exact(losses(old), losses(new))
        records[name] = {'baseline': str(old), 'optimized': str(new),
                         'checkpoint_and_host_rng_exact': True, 'all_logged_losses_exact': True}
        if do_eval:
            evaluated.append(new)
            records[name]['eval_worker_pending_at_training_exit'] = not (new/'cpu_eval/WORKER_DONE.json').exists()
            resumed = run(name+'_resumed', family, preset,
                          extra=['--restore_path='+str(new/'checkpoints/params_8.pkl')])
            off = run(name+'_disabled', family, preset,
                      extra=['--noreuse_train_buffers', '--noasync_prefetch'])
            exact(checkpoint(new), checkpoint(resumed))
            exact(checkpoint(new), checkpoint(off))
            exact(losses(new)[2:], losses(resumed))
            exact(losses(new), losses(off))
            records[name].update(resume_exact=True, disabled_optimizations_exact=True,
                                 resumed=str(resumed), disabled=str(off))
        print('CLI_EXACT', name, flush=True)

    deadline = time.monotonic()+600
    for run_dir in evaluated:
        while not (run_dir/'cpu_eval/WORKER_DONE.json').exists():
            assert time.monotonic()<deadline, 'CPU evaluation timed out'
            time.sleep(2)
        done = json.loads((run_dir/'cpu_eval/WORKER_DONE.json').read_text())
        assert not done['errors'], done
        paths = sorted((run_dir/'cpu_eval').glob('result_*.json'))
        assert len(paths)==2, paths
        for path in paths:
            result = json.loads(path.read_text())
            assert result['backend']=='cpu' and len(result['episodes'])==5
            assert result['protocol']=='3999-idm-strict-reset-v2'
    result = {'passed':True, 'baseline_root':str(args.baseline_root), 'steps':16,
              'runs':records, 'cpu_eval_jobs':2, 'cpu_eval_episodes':10}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('OPTIMIZED_RUNTIME_ALL_PASS',flush=True)


if __name__=='__main__':
    main()
