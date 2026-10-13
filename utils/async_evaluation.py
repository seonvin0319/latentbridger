"""A durable queue serviced by one independent CPU-only evaluation process."""
import json
import os
from pathlib import Path
import subprocess
import sys


class CpuEvaluator:
    def __init__(self, run_dir, episodes):
        self.root = Path(run_dir) / 'cpu_eval'
        self.root.mkdir(parents=True, exist_ok=True)
        self.episodes = int(episodes)
        self.process = None

    def submit(self, checkpoint, step):
        job = self.root / f'job_{step:09d}.json'
        temp = job.with_suffix('.tmp')
        temp.write_text(json.dumps({'checkpoint': checkpoint, 'step': step, 'episodes': self.episodes}))
        temp.replace(job)
        if self.process is None:
            worker = Path(__file__).resolve().with_name('eval_worker.py')
            env = {**os.environ, 'JAX_PLATFORMS': 'cpu', 'CUDA_VISIBLE_DEVICES': '',
                   'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
            with (self.root / 'worker.log').open('ab', buffering=0) as log:
                self.process = subprocess.Popen([sys.executable, str(worker), str(self.root)],
                    env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            (self.root / 'worker_pid.json').write_text(json.dumps({'pid': self.process.pid}))
        if self.process.poll() is not None:
            raise RuntimeError(f'CPU evaluator exited; see {self.root / "worker.log"}')

    def finish(self):
        # Worker drains the queue after training ends; no GPU training wait.
        (self.root / 'TRAINING_DONE').touch()
