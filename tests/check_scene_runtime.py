"""Exercise PBF/PBG scene training, save, and actual detached CPU evaluation."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'validation/scene_smoke'
OUT.mkdir(parents=True, exist_ok=True)


def main():
    env = {**os.environ, 'JAX_PLATFORMS': 'cuda', 'CUDA_VISIBLE_DEVICES': '0',
           'XLA_PYTHON_CLIENT_PREALLOCATE': 'false',
           'XLA_FLAGS': '--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true',
           'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1'}
    runs = {}
    for family in ['pbf', 'pbg']:
        command = [sys.executable, 'main.py', '--agent=configs/'+family+'/scene.py',
                   '--seed=1', '--train_steps=16', '--save_interval=8', '--eval_interval=16',
                   '--eval_episodes=1', '--log_interval=4', '--nouse_tqdm', '--run_group='+family,
                   '--save_dir='+str(OUT)]
        with (OUT/(family+'.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        run = sorted((OUT/'pathbridger'/family).glob('*'))[-1]
        rows = list(csv.DictReader((run/'train.csv').open()))
        components = ['value/loss', 'endpoint/loss', 'bridge/loss', 'idm/loss']
        for row in rows:
            assert all(np.isfinite(float(row[key])) for key in components)
        assert (run/'checkpoints/params_8.pkl').exists() and (run/'checkpoints/params_16.pkl').exists()
        provenance = json.loads((run/'provenance.json').read_text())
        assert len(provenance['core_compatibility_patches']) == 1
        runs[family] = {'run_dir': str(run), 'train_steps': 16,
                        'losses': [{key: float(row[key]) for key in ['step']+components} for row in rows]}
        print('TRAIN_PASS', family, flush=True)
    deadline = time.monotonic()+600
    for family, record in runs.items():
        run = Path(record['run_dir'])
        while not (run/'cpu_eval/WORKER_DONE.json').exists():
            assert time.monotonic() < deadline, 'scene CPU eval timed out'
            time.sleep(2)
        done = json.loads((run/'cpu_eval/WORKER_DONE.json').read_text())
        assert not done['errors'], done
        result = json.loads((run/'cpu_eval/result_000000016.json').read_text())
        assert result['backend']=='cpu' and len(result['episodes'])==5
        assert result['protocol']=='3999-idm-strict-reset-v2'
        record.update(cpu_eval_episodes=5, cpu_eval_passed=True, checkpoints_saved=[8,16])
        print('CPU_EVAL_PASS', family, flush=True)
    result = dict(passed=True, runs=runs, purpose='Functionality smoke test, not a success-rate benchmark')
    (ROOT/'validation/scene_runtime.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
