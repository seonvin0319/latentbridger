"""Calibrated CPB suite. Every seed-0 case finishes before seed 1, then seed 2."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
ENVS=('cube_single','cube_double','antmaze_medium','antmaze_large','puzzle_3x3','humanoid_medium','humanoid_large')
DEFAULT_ENVS=('cube_double','antmaze_medium','puzzle_3x3')


def schedule(configs=DEFAULT_ENVS, variants=('cpb_rank_only','cpb_full'), seeds=(0,1,2)):
    # Finish every environment at one seed before any run of the next seed.
    for seed in sorted(seeds):
        for env in configs:
            if 'pathbridger_original' in variants:
                yield env,'pathbridger_original',seed
            # Seed-0 paired ablation on every configured environment.
            if seed==0 and 'cpb_rank_only' in variants:
                yield env,'cpb_rank_only',0
            if 'cpb_full' in variants:
                yield env,'cpb_full',seed


def source_digest():
    digest=hashlib.sha256()
    paths=[ROOT/'agents/contrastive_pathbridger.py',ROOT/'main_contrastive_pathbridger.py',ROOT/'utils/cpb_reference_bank.py',ROOT/'tests/test_contrastive_pathbridger.py']
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--configs',nargs='+',default=list(DEFAULT_ENVS))
    p.add_argument('--variants',nargs='+',choices=('cpb_full','cpb_rank_only','pathbridger_original'),default=['cpb_rank_only','cpb_full'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--train_steps',type=int,default=1000000)
    p.add_argument('--dataset_dir',default='')
    p.add_argument('--save_dir','--output',default='exp/contrastive_pathbridger')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--skip_existing',action='store_true')
    p.add_argument('--use_wandb',action='store_true')
    p.add_argument('--gates-passed',action='store_true',help='Validate current-source test/smoke evidence instead of rerunning')
    p.add_argument('--parallel',type=int,default=2,help='Jobs at once inside one seed. The next seed waits for the whole seed.')
    return p


class Slot:
    def __init__(self,name,variant,seed,run,proc=None,pid=None,owned=True):
        self.name=name;self.variant=variant;self.seed=seed;self.run=run
        self.proc=proc;self.pid=pid if pid is not None else proc.pid;self.owned=owned
        self._code=None
    def poll(self):
        if self._code is not None:
            return self._code
        if self.proc is not None:
            code=self.proc.poll()
            if code is not None:
                self._code=code
            return code
        try:
            os.kill(self.pid,0)
        except OSError:
            self._code=0 if (self.run/'complete.json').exists() else 1
            return self._code
        return None
    def stop(self):
        if self.owned and self.proc is not None and self.proc.poll() is None:
            self.proc.terminate();self.proc.wait()


def main():
    args=parser().parse_args()
    configs=[Path(name).stem if name.endswith('.py') else name for name in args.configs]
    if set(configs)-set(ENVS) or args.train_steps<1 or any(seed<0 for seed in args.seeds) or args.parallel<1:
        raise ValueError('Invalid configs, seeds, training steps, or parallel')
    output=(ROOT/args.save_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'suite.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=dict(os.environ,XLA_PYTHON_CLIENT_PREALLOCATE='false',MUJOCO_GL='egl',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    started=time.time()
    def summarize():
        subprocess.run([sys.executable,'scripts/summarize_contrastive_pathbridger.py','--output',str(output)],cwd=ROOT,check=True)
    def command(argv,log):
        with log.open('a') as file:
            proc=subprocess.Popen([sys.executable,*argv],cwd=ROOT,env=env,stdout=file,stderr=subprocess.STDOUT)
            count=-1
            try:
                while True:
                    try:
                        code=proc.wait(timeout=30)
                        if code:
                            raise subprocess.CalledProcessError(code,proc.args)
                        break
                    except subprocess.TimeoutExpired:
                        current=len(list(output.glob('*/*/seed*/evaluation_[0-9]*_h*.json')))
                        if current!=count:
                            summarize();count=current
            except BaseException:
                if proc.poll() is None:
                    proc.terminate();proc.wait()
                raise
    def train(name,variant,seed,*,stop_after=0,prelude=False):
        run=output/name/variant/f'seed{seed}';run.mkdir(parents=True,exist_ok=True)
        completed=run/'complete.json'
        if completed.exists():
            record=json.loads(completed.read_text())
            if record.get('calibration')!='fixed_future_bank_v1' or record['steps']!=args.train_steps:
                raise ValueError(f'Incompatible completed run {run}')
            if args.skip_existing or args.resume:
                return
            raise ValueError(f'Run already complete: {run}; use --skip_existing')
        checkpoints=list((run/'checkpoints').glob('params_*.pkl'))
        # Full seed0's 100k sanity is the beginning of its joint 1M run.
        from_sanity=(run/'sanity_100000.json').exists()
        if checkpoints and not (args.resume or from_sanity):
            raise ValueError(f'Partial run found: {run}; use --resume')
        status=dict(status='running',env=name,variant=variant,seed=seed,phase='100k_sanity' if prelude else 'joint_training',started=started)
        (output/'status.json').write_text(json.dumps(status))
        argv=['main_contrastive_pathbridger.py','--env',name,'--variant',variant,'--seed',str(seed),
              '--steps',str(args.train_steps),'--run-dir',str(run),'--dataset_dir',args.dataset_dir]
        if checkpoints:
            argv+=['--resume']
        if stop_after:
            argv+=['--stop_after',str(stop_after)]
        if args.use_wandb:
            argv+=['--use_wandb']
        command(argv,run/'run.log')
        summarize()
    try:
        if not args.gates_passed:
            command(['-m','pytest','-q'],output/'tests.log')
            (output/'tests_passed.json').write_text(json.dumps(dict(source_digest=source_digest(),time=time.time())))
            smoke=['main_contrastive_pathbridger.py','--steps','2000','--smoke','--run-dir',str(output/'smoke'),'--dataset_dir',args.dataset_dir]
            if (output/'smoke/checkpoints/params_2000.pkl').exists():
                smoke+=['--resume']
            command(smoke,output/'smoke.log')
        evidence=json.loads((output/'tests_passed.json').read_text())
        complete=json.loads((output/'smoke/complete.json').read_text())
        if evidence['source_digest']!=source_digest() or complete.get('calibration')!='fixed_future_bank_v1':
            raise RuntimeError('Current-source tests and calibrated smoke are required')
        # User-requested extra sanity prelude, preserved and resumed rather than retrained.
        if 'cube_single' in configs and 'cpb_full' in args.variants and 0 in args.seeds and args.train_steps>=100000:
            gate=output/'cube_single/cpb_full/seed0/sanity_100000.json'
            if not gate.exists():
                train('cube_single','cpb_full',0,stop_after=100000,prelude=True)
            if not all(json.loads(gate.read_text()).values()):
                raise RuntimeError('100k sanity gate failed')
        owned=[]
        def trainer_pid(run):
            needle=str(run)
            for entry in Path('/proc').iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    cmd=(entry/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
                except OSError:
                    continue
                if 'main_contrastive_pathbridger.py' in cmd and needle in cmd:
                    return int(entry.name)
            return None
        def publish(slots):
            (output/'status.json').write_text(json.dumps(dict(
                status='running',started=started,parallel=args.parallel,order='seed0_then_1_then_2',
                jobs=[dict(env=slot.name,variant=slot.variant,seed=slot.seed,pid=slot.pid,adopted=not slot.owned) for slot in slots])))
        def spawn(name,variant,seed,run,checkpoints):
            argv=['main_contrastive_pathbridger.py','--env',name,'--variant',variant,'--seed',str(seed),
                  '--steps',str(args.train_steps),'--run-dir',str(run),'--dataset_dir',args.dataset_dir]
            if checkpoints:
                argv+=['--resume']
            if args.use_wandb:
                argv+=['--use_wandb']
            with (run/'run.log').open('a') as file:
                proc=subprocess.Popen([sys.executable,*argv],cwd=ROOT,env=env,stdout=file,stderr=subprocess.STDOUT)
            slot=Slot(name,variant,seed,run,proc=proc,owned=True)
            owned.append(slot)
            print(f'start {name} {variant} seed {seed} pid {proc.pid}',flush=True)
            return slot
        def begin(name,variant,seed):
            run=output/name/variant/f'seed{seed}';run.mkdir(parents=True,exist_ok=True)
            completed=run/'complete.json'
            if completed.exists():
                record=json.loads(completed.read_text())
                if record.get('calibration')!='fixed_future_bank_v1' or record['steps']!=args.train_steps:
                    raise ValueError(f'Incompatible completed run {run}')
                if args.skip_existing or args.resume:
                    return None
                raise ValueError(f'Run already complete: {run}; use --skip_existing')
            checkpoints=list((run/'checkpoints').glob('params_*.pkl'))
            from_sanity=(run/'sanity_100000.json').exists()
            if checkpoints and not (args.resume or from_sanity):
                raise ValueError(f'Partial run found: {run}; use --resume')
            pid=trainer_pid(run)
            if pid is not None:
                print(f'adopt {name} {variant} seed {seed} pid {pid}',flush=True)
                return Slot(name,variant,seed,run,pid=pid,owned=False)
            return spawn(name,variant,seed,run,checkpoints)
        try:
            jobs=list(schedule(configs,args.variants,args.seeds))
            seed_order=[]
            for _,_,seed in jobs:
                if seed not in seed_order:
                    seed_order.append(seed)
            eval_count=-1
            for seed in seed_order:
                pending=[job for job in jobs if job[2]==seed]
                active=[]
                while pending or active:
                    while pending and len(active)<args.parallel:
                        slot=begin(*pending.pop(0))
                        if slot is not None:
                            active.append(slot)
                    publish(active)
                    if not active:
                        break
                    time.sleep(5)
                    current=len(list(output.glob('*/*/seed*/evaluation_[0-9]*_h*.json')))
                    if current!=eval_count:
                        try:
                            summarize()
                        except subprocess.CalledProcessError as exc:
                            print(f'summarize failed: {exc}',flush=True)
                        eval_count=current
                    still=[]
                    for slot in active:
                        code=slot.poll()
                        if code is None:
                            still.append(slot)
                            continue
                        if code:
                            raise subprocess.CalledProcessError(code,[slot.name,slot.variant,str(slot.seed)])
                        print(f'done {slot.name} {slot.variant} seed {slot.seed}',flush=True)
                    active=still
                print(f'seed {seed} phase complete',flush=True)
        finally:
            for slot in owned:
                if slot.poll() is None:
                    slot.stop()
        (output/'status.json').write_text(json.dumps(dict(status='complete',wall_seconds=time.time()-started,parallel=args.parallel)))
    except BaseException as exc:
        failure=dict(error=str(exc),time=time.time(),wall_seconds=time.time()-started)
        (output/'failure.json').write_text(json.dumps(failure))
        (output/'status.json').write_text(json.dumps(dict(status='failed',**failure)))
        raise
    finally:
        summarize()

if __name__=='__main__':
    main()
