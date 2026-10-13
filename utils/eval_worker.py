"""CPU queue worker. Run as a separate process, never via a training thread."""
import os
import sys
from pathlib import Path
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:6])
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import time
import traceback
from evaluate import evaluate_checkpoint
from utils.log_utils import CsvLogger


def run(root):
    root = Path(root)
    logger = CsvLogger(root.parent / 'eval.csv', resume=True)
    try:
        while True:
            pending = [p for p in sorted(root.glob('job_*.json'))
                       if not p.with_suffix('.done').exists() and not p.with_suffix('.error').exists()]
            if not pending:
                if (root / 'TRAINING_DONE').exists():
                    break
                time.sleep(2)
                continue
            for path in pending:
                try:
                    job = json.loads(path.read_text())
                    result = evaluate_checkpoint(job['checkpoint'], episodes=job['episodes'])
                    output = root / f"result_{job['step']:09d}.json"
                    tmp = output.with_suffix('.tmp')
                    tmp.write_text(json.dumps(result, indent=2)+'\n')
                    tmp.replace(output)
                    logger.log({k: v for k, v in result.items() if k.startswith('evaluation/')}, step=job['step'])
                    path.with_suffix('.done').touch()
                    print(json.dumps({'event': 'evaluated', 'step': job['step'],
                                      'success': result['evaluation/overall_success']}), flush=True)
                except Exception:
                    path.with_suffix('.error').write_text(traceback.format_exc())
                    traceback.print_exc()
        status = {'finished': True, 'errors': [p.name for p in root.glob('*.error')]}
        (root / 'WORKER_DONE.json').write_text(json.dumps(status, indent=2))
    finally:
        logger.close()


if __name__ == '__main__':
    run(sys.argv[1])
