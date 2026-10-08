"""Priority queue with one local run per physical GPU and opt-in sharing."""
from __future__ import annotations
import argparse
import csv
import fcntl
import importlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT = ROOT/'exp/goalspace_ablation'
BASE = 'da37867d31d47b02a14dcbc56dd8f267062c02aa'
ABLATIONS = ('gs_trl_weighted','gsdtrl_uniform','gsdtrl_no_transitive_weighted','gs_symmetric_weighted')
HARD = ('puzzle_4x4','cube_triple','antmaze_large','scene')
CORE = ('puzzle_3x3','cube_double','antmaze_medium','cube_single')
FIELDS = ('env_name','horizon','discount','endpoint_distribution','eval_num_candidates','eval_temperature','endpoint_value_scale','value_distance_weight_power')


def now():return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(payload,indent=2)+'\n');tmp.replace(path)


def discover():
    """Compare local configs against exact git objects; fail closed on drift."""
    git=shutil.which('git') or str(Path.home()/'miniconda3/bin/git')
    rows=[]
    for env in HARD:
        relative=f'configs/pbf/{env}.py'
        path=ROOT/relative
        try:
            exact=subprocess.check_output([git,'show',f'{BASE}:{relative}'],cwd=ROOT)
            if path.read_bytes()!=exact:raise ValueError('config differs from authoritative base')
            config=importlib.import_module(f'configs.pbf.{env}').get_config()
            row=dict(env=env,status='available',source=relative,source_commit=BASE,
                     **{k:config[k] for k in FIELDS})
            data=Path.home()/'.ogbench/data'
            row['dataset_present']=all((data/f'{config.env_name}{suffix}.npz').exists() for suffix in ('','-val'))
        except (OSError,ValueError,ImportError,subprocess.CalledProcessError) as exc:
            row=dict(env=env,status='config unavailable',reason=str(exc))
        rows.append(row)
    write_json(OUT/'harder_env_provenance.json',rows)
    return rows


def plan(harder):
    jobs=[dict(phase=f'A{i}',env=e,variant=v,seed=0) for i,v in enumerate(ABLATIONS,1) for e in CORE[:2]]
    jobs += [dict(phase='harder',env=r['env'],variant='gsdtrl_weighted',seed=0) for r in harder if r['status']=='available']
    jobs += [dict(phase='extra_seeds',env=e,variant='gsdtrl_weighted',seed=s) for s in (1,2) for e in CORE]
    return jobs


def run_dir(job):return OUT/job['env']/job['variant']/f"seed{job['seed']}"


def completed(job):
    path=run_dir(job)/'complete.json'
    if not path.exists():return False
    record=json.loads(path.read_text())
    if record['steps'] != 1_000_000 or record.get('smoke', False):return False
    for step in (100000,300000,500000,800000,1000000):
        for h in (5,2,1):
            path=run_dir(job)/f'evaluation_{step}_h{h}.json'
            if not path.exists():return False
            row=json.loads(path.read_text())
            if (row.get('checkpoint'),row.get('h'),row.get('num_tasks'),row.get('episodes_per_task'),row.get('seed'),row.get('variant')) != (step,h,5,50,job['seed'],job['variant']):return False
    return True


def gpu_inventory():
    result=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    gpus=[dict(index=int(r[0]),uuid=r[1].strip(),free_mib=int(r[2])) for r in csv.reader(result.splitlines())]
    result=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    busy={r[0].strip() for r in csv.reader(result.splitlines()) if len(r)>=2}
    return gpus,busy


def available_gpus(gpus, busy, reserved, allow_shared=False):
    return [g for g in gpus if (allow_shared or g['uuid'] not in busy)
            and g['uuid'] not in reserved and g['free_mib'] >= 8192]


def repository_trainers():
    found=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if (p/'cwd').resolve()!=ROOT:continue
            args=(p/'cmdline').read_bytes().split(b'\0')
            if b'main_ctd_pathbridger.py' in args:found.append(int(p.name))
        except OSError:pass
    return found


def summarize():
    subprocess.run([sys.executable,str(ROOT/'scripts/summarize_goalspace_ablation.py')],check=True,cwd=ROOT)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--allow-shared-gpu',action='store_true',help='Allow external GPU workloads; still at most one local job per GPU and 8 GiB free at launch.')
    parser.add_argument('--poll-seconds',type=float,default=30)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=plan(discover())
    write_json(OUT/'queue.json',dict(base=BASE,jobs=jobs,steps=1_000_000,episodes=50,horizons=[5,2,1],max_parallel=3,allow_shared_gpu=args.allow_shared_gpu))
    if args.prepare_only:
        summarize();return
    existing=repository_trainers()
    if existing:raise RuntimeError(f'Existing trainers require inspection before restarting controller: {existing}')
    active={};failed=[];stop=False
    def stopping(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    try:
        while not stop:
            for key,item in list(active.items()):
                code=item['process'].poll()
                if code is None:continue
                item['log'].close()
                if code!=0 or not completed(item['job']):
                    failed.append(dict(job=item['job'],returncode=code))
                    write_json(run_dir(item['job'])/'failure.json',dict(returncode=code,time=now()))
                del active[key]
            failed_keys={str(run_dir(r['job'])) for r in failed}
            pending=[j for j in jobs if not completed(j) and str(run_dir(j)) not in active and str(run_dir(j)) not in failed_keys]
            # Fail closed after an experiment failure; do not silently skip to later phases.
            if failed and not active:break
            gpus,busy=gpu_inventory()
            reserved={item['gpu'] for item in active.values()}
            free=available_gpus(gpus,busy,reserved,args.allow_shared_gpu)
            if not failed:
                for gpu in free:
                    if not pending or len(active)>=min(len(gpus),3):break
                    job=pending.pop(0);run=run_dir(job);run.mkdir(parents=True,exist_ok=True)
                    env=dict(os.environ,JAX_PLATFORMS='cuda',CUDA_VISIBLE_DEVICES=gpu['uuid'],
                             XLA_PYTHON_CLIENT_PREALLOCATE='false',MUJOCO_GL='egl',
                             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                             PYTHONPATH=str(ROOT))
                    argv=[sys.executable,'main_ctd_pathbridger.py','--env',job['env'],'--variant',job['variant'],
                          '--seed',str(job['seed']),'--steps','1000000','--episodes','50','--run-dir',str(run)]
                    if list((run/'checkpoints').glob('params_*.pkl')):argv.append('--resume')
                    log=(run/'run.log').open('a')
                    child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    active[str(run)]=dict(job=job,process=child,gpu=gpu['uuid'],log=log)
                    with (OUT/'launch_history.jsonl').open('a') as f:
                        f.write(json.dumps(dict(time=now(),pid=child.pid,gpu=gpu,job=job,argv=argv))+'\n')
                    print('launched',job,'pid',child.pid,'gpu',gpu['index'],flush=True)
            state='failed' if failed else 'running' if active else ('waiting_for_gpu_memory' if args.allow_shared_gpu else 'waiting_for_dedicated_gpu') if pending else 'complete'
            write_json(OUT/'queue_status.json',dict(time=now(),controller_pid=os.getpid(),status=state,allow_shared_gpu=args.allow_shared_gpu,
                active=[dict(job=r['job'],pid=r['process'].pid,gpu=r['gpu']) for r in active.values()],
                pending=len(pending),completed=sum(completed(j) for j in jobs),failed=failed,gpus=gpus,busy_gpu_uuids=sorted(busy)))
            summarize()
            if not pending and not active:break
            time.sleep(args.poll_seconds)
    finally:
        # Signal only children we created; never kill unrelated repository jobs.
        for r in active.values():r['process'].terminate()
        for r in active.values():
            try:r['process'].wait(timeout=30)
            except subprocess.TimeoutExpired:r['process'].kill();r['process'].wait()
            r['log'].close()
        if stop:write_json(OUT/'queue_status.json',dict(time=now(),status='stopped',failed=failed))
        elif failed:write_json(OUT/'queue_status.json',dict(time=now(),status='failed',failed=failed))
        summarize()
    if failed:raise SystemExit(1)


if __name__=='__main__':main()
