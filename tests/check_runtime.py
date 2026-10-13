"""Exercise real CLI prefetch/save/resume and the detached CPU eval queue."""
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'validation/runtime_smoke'
OUT.mkdir(parents=True, exist_ok=True)


def exact(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    else:
        assert a == b


def run(name, extra):
    cmd = [sys.executable, 'main.py', '--agent=configs/pbf/puzzle_3x3.py', '--seed=1',
           '--train_steps=6', '--save_interval=3', '--log_interval=3', '--nouse_tqdm',
           '--noeval_final', '--eval_episodes=1', '--save_dir='+str(OUT), '--run_group='+name] + extra
    env = {**os.environ, 'JAX_PLATFORMS': 'cuda', 'CUDA_VISIBLE_DEVICES': '0', 'XLA_PYTHON_CLIENT_MEM_FRACTION': '.65',
           'XLA_FLAGS': '--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',
           'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1'}
    with (OUT/(name+'.log')).open('w') as log:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return next((OUT/'pathbridger'/name).glob('*'))


def main():
    full = run('continuous', ['--eval_interval=3'])
    resumed = run('resumed', ['--eval_interval=0', '--restore_path='+str(full/'checkpoints/params_3.pkl')])
    a = pickle.load((full/'checkpoints/params_6.pkl').open('rb'))
    b = pickle.load((resumed/'checkpoints/params_6.pkl').open('rb'))
    exact(a, b)
    result = dict(prefetch_cli_resume_exact=True, full_checkpoint_including_host_rng_exact=True,
                  run_dir=str(full), resumed_run_dir=str(resumed), training_steps=6)
    (ROOT/'validation/runtime_smoke_training.json').write_text(json.dumps(result, indent=2)+'\n')
    print('CLI_PREFETCH_RESUME_ALL_EXACT', flush=True)
    # Wait in short intervals; outer caller can continue other work meanwhile.
    deadline = time.monotonic()+600
    while not (full/'cpu_eval/WORKER_DONE.json').exists():
        assert time.monotonic() < deadline, 'CPU evaluator timed out'
        time.sleep(2)
    worker = json.loads((full/'cpu_eval/WORKER_DONE.json').read_text())
    assert not worker['errors'], worker
    outputs = sorted((full/'cpu_eval').glob('result_*.json'))
    assert len(outputs) == 2
    for path in outputs:
        record = json.loads(path.read_text())
        assert record['backend'] == 'cpu' and len(record['episodes']) == 5
        assert record['protocol'] == '3999-idm-strict-reset-v2'
    result.update(cpu_evaluation_jobs=2, cpu_evaluation_passed=True, passed=True)
    (ROOT/'validation/runtime_smoke.json').write_text(json.dumps(result, indent=2)+'\n')
    print('CPU_QUEUE_ALL_PASS', flush=True)


if __name__ == '__main__':
    main()
