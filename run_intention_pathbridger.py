#!/usr/bin/env python3
"""Intention-Conditioned PathBridger: job CLI + resumable launcher.

Subcommands (each one job; refuse to overwrite completed outputs):
  pb-train     train the original PB (commit 21a4042 code, seed-0 env-best YAML) for seeds without a checkpoint
  tokenizer    Phase A: VQ intention tokenizer (200k steps; 50k/100k/200k checkpoints)
  conditioned  Phase B: intention-conditioned subgoal flow + bridge (1M steps; frozen tokenizer / IDM / critic)
  diag         offline diagnostics on the held-out split (1M conditioned checkpoint)
  eval         control evaluation of one (task, seed, step, method, h_exec) cell
  aggregate    write exp/intention_pathbridger/aggregate/*
  launch       run the whole DAG (``--dry-run`` prints the plan)
  smoke        tiny end-to-end run on cube-single seed 0 under exp/intention_pathbridger/smoke/
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PYTHON = sys.executable


def _pre_parse_root() -> None:
    for i, a in enumerate(sys.argv):
        if a.startswith('--exp-root='):
            os.environ['IPB_EXP_ROOT'] = str(Path(a.split('=', 1)[1]).resolve())
        elif a == '--exp-root' and i + 1 < len(sys.argv):
            os.environ['IPB_EXP_ROOT'] = str(Path(sys.argv[i + 1]).resolve())


_pre_parse_root()

import intention_pb  # noqa: E402  (installs the PB code path first)
from intention_pb import PB_CODE_DIR  # noqa: E402
from intention_pb import common as C  # noqa: E402


# ============================================================================ job bodies
def cmd_pb_train(a) -> None:
    if int(a.seed) == 0:
        raise SystemExit('Seed 0 reuses checkpoints/1m_env_best; refusing to train it.')
    if C.pb_run_complete(a.task, a.seed):
        print(f'[pb-train] already complete: {C.find_pb_run_dir(a.task, a.seed)}')
        return
    info = C.task_info(a.task)
    src_cfg = C.SEED0_PB_ROOT / info['seed0_label'] / 'config_used.yaml'
    if not src_cfg.is_file():
        raise FileNotFoundError(src_cfg)
    root = C.pb_train_root(a.task, a.seed)
    root.mkdir(parents=True, exist_ok=True)
    cfg_copy = root / 'config_from_seed0.yaml'
    if not cfg_copy.exists():
        shutil.copy2(src_cfg, cfg_copy)
    runs = sorted(p for p in root.iterdir() if p.is_dir() and (p / 'flags.json').is_file())
    if len(runs) > 1:
        raise RuntimeError(f'Multiple PB runs under {root}: {runs}')
    base = [PYTHON, 'main.py']
    if runs:
        run_dir = runs[0]
        steps = sorted(int(p.stem.split('_')[1]) for p in (run_dir / 'checkpoints' / 'dynamics').glob('params_*.pkl'))
        steps = [s for s in steps if all((run_dir / 'checkpoints' / n / f'params_{s}.pkl').is_file() for n in ('critic', 'actor'))]
        if not steps:
            raise RuntimeError(f'PB run {run_dir} has no complete checkpoint to resume from; inspect it manually.')
        args = base + [f'--resume_run_dir={run_dir}', f'--resume_step={steps[-1]}']
    else:
        args = base + [
            f'--run_config={cfg_copy}', f'--seed={a.seed}', f'--runs_root={root}', f'--train_steps={a.steps}',
            f'--save_every_n_steps={a.save_every}', '--eval_freq=0', '--eval_every_n_steps=0',
            '--final_eval_subgoal_eval_num_samples=', '--use_wandb=False',
        ]
    env = dict(os.environ, PYTHONPATH=str(PB_CODE_DIR))
    print('[pb-train]', ' '.join(args), flush=True)
    rc = subprocess.call(args, cwd=str(PB_CODE_DIR), env=env)
    if rc != 0:
        raise SystemExit(rc)
    if a.steps == C.COND_STEPS and not C.pb_run_complete(a.task, a.seed):
        raise SystemExit('PB training exited 0 but the 1M checkpoints are missing.')


def cmd_tokenizer(a) -> None:
    from intention_pb.pb_io import load_plain_datasets
    from intention_pb.tokenizer import train_tokenizer

    out = C.tokenizer_dir(a.task, a.seed)
    C.refuse_overwrite(out / 'COMPLETE.json')
    C.require_gpu()
    train, val = load_plain_datasets(C.task_info(a.task)['env_name'])
    save_steps = tuple(int(s) for s in a.save_steps.split(','))
    t0 = time.time()
    m = train_tokenizer(out_dir=out, train=train, val=val, seed=int(a.seed), total_steps=int(a.steps), save_steps=save_steps)
    C.atomic_write_json(out / 'COMPLETE.json', dict(task=a.task, seed=a.seed, steps=a.steps, collapsed=m['collapsed'],
                                                     perplexity=m['perplexity'], max_usage=m['max_usage'], runtime_s=time.time() - t0))


def cmd_conditioned(a) -> None:
    from intention_pb.conditioned import train_conditioned
    from intention_pb.pb_io import load_pb

    out = C.cond_dir(a.task, a.seed)
    C.refuse_overwrite(out / 'COMPLETE.json')
    C.require_gpu()
    tok_done = C.read_json(C.tokenizer_dir(a.task, a.seed) / 'COMPLETE.json')
    if tok_done['collapsed']:
        raise SystemExit(f'Tokenizer {a.task} seed{a.seed} collapsed; downstream training is skipped by design.')
    pb = load_pb(C.find_pb_run_dir(a.task, a.seed), need_train=True, need_env=False)
    save_steps = tuple(int(s) for s in a.save_steps.split(','))
    t0 = time.time()
    train_conditioned(out_dir=out, pb=pb, tok_path=C.tokenizer_ckpt(a.task, a.seed, a.tokenizer_step), seed=int(a.seed),
                      total_steps=int(a.steps), save_steps=save_steps, log_every=int(a.log_every))
    C.atomic_write_json(out / 'COMPLETE.json', dict(task=a.task, seed=a.seed, steps=a.steps, runtime_s=time.time() - t0))


def cmd_diag(a) -> None:
    from intention_pb.diagnostics import run_diagnostics

    run_diagnostics(task=a.task, seed=int(a.seed), out_dir=C.diag_dir(a.task, a.seed), step=int(a.step),
                    num_samples=int(a.num_samples))


def cmd_eval(a) -> None:
    from intention_pb.rollout import run_eval_job

    s = run_eval_job(task=a.task, seed=int(a.seed), step=int(a.step), method=a.method, h_exec=int(a.h_exec),
                     episodes_per_task=int(a.episodes_per_task), out_dir=C.eval_dir(a.task, a.seed, a.step, a.method, a.h_exec))
    print(json.dumps({k: s[k] for k in ('task', 'seed', 'step', 'method', 'h_exec', 'success_rate', 'runtime_s')}))


def cmd_aggregate(a) -> None:
    from intention_pb.aggregate import aggregate

    print(json.dumps(aggregate(C.EXP_ROOT), indent=2))


# ============================================================================ launcher
@dataclasses.dataclass
class Job:
    name: str
    kind: str  # 'gpu' | 'cpu'
    args: list[str]
    priority: tuple
    threads: int = 1
    task: str = ''
    seed: int = 0

    def done(self) -> bool:
        raise NotImplementedError

    def deps(self) -> str:
        """'ready' | 'wait' | 'skip:<reason>'."""
        return 'ready'


class _J(Job):
    def __init__(self, *, done_fn, deps_fn, **kw):
        super().__init__(**kw)
        self._done = done_fn
        self._deps = deps_fn

    def done(self) -> bool:
        return bool(self._done())

    def deps(self) -> str:
        return self._deps()


def load_hold(state_root: Path) -> dict | None:
    """Optional enqueue hold. Missing or ``active: false`` means the full DAG may start."""
    path = Path(state_root) / 'HOLD_NEW_JOBS.json'
    if not path.is_file():
        return None
    hold = json.loads(path.read_text())
    if not hold.get('active', True):
        return None
    return hold


def hold_decision(job_name: str, hold: dict | None) -> str:
    """``run`` if this job may be started, else ``hold``.

    A hold file stops new enqueue without killing jobs that are already running.
    Allowlisted names (exact or prefix) still start, so an in-flight task can finish its evals.
    """
    if not hold:
        return 'run'
    for prefix in hold.get('allow_prefixes', []):
        if job_name == prefix or job_name.startswith(prefix):
            return 'run'
    return 'hold'


def _tok_state(task, seed) -> str:
    p = C.tokenizer_dir(task, seed) / 'COMPLETE.json'
    if not p.is_file():
        return 'pending'
    return 'collapsed' if C.read_json(p)['collapsed'] else 'ok'


def build_jobs(seeds=C.SEEDS, tasks=C.TASK_ORDER) -> list[Job]:
    jobs: list[Job] = []
    me = [PYTHON, str(REPO / 'run_intention_pathbridger.py')]
    root_arg = [f'--exp-root={C.EXP_ROOT}']
    for seed in seeds:
        grp = 0 if seed == 0 else 1
        for ti, task in enumerate(tasks):
            base = dict(task=task, seed=seed)
            if seed != 0:
                jobs.append(_J(name=f'pb_train/{task}_s{seed}', kind='gpu', priority=(grp, 2, seed, ti),
                               args=me + root_arg + ['pb-train', f'--task={task}', f'--seed={seed}'],
                               done_fn=lambda t=task, s=seed: C.pb_run_complete(t, s), deps_fn=lambda: 'ready', threads=2, **base))
            jobs.append(_J(name=f'tokenizer/{task}_s{seed}', kind='gpu', priority=(grp, 0, seed, ti),
                           args=me + root_arg + ['tokenizer', f'--task={task}', f'--seed={seed}'],
                           done_fn=lambda t=task, s=seed: _tok_state(t, s) != 'pending', deps_fn=lambda: 'ready', threads=1, **base))

            def cond_deps(t=task, s=seed):
                st = _tok_state(t, s)
                if st == 'collapsed':
                    return 'skip:tokenizer collapsed'
                if st == 'pending' or not C.pb_run_complete(t, s):
                    return 'wait'
                return 'ready'

            jobs.append(_J(name=f'conditioned/{task}_s{seed}', kind='gpu', priority=(grp, 1, seed, ti),
                           args=me + root_arg + ['conditioned', f'--task={task}', f'--seed={seed}'],
                           done_fn=lambda t=task, s=seed: (C.cond_dir(t, s) / 'COMPLETE.json').is_file(), deps_fn=cond_deps,
                           threads=2, **base))

            def ckpt_deps(step, t=task, s=seed):
                st = _tok_state(t, s)
                if st == 'collapsed':
                    return 'skip:tokenizer collapsed'
                return 'ready' if C.cond_ckpt(t, s, step).is_file() else 'wait'

            jobs.append(_J(name=f'diag/{task}_s{seed}', kind='cpu', priority=(grp, 3, seed, ti, 0),
                           args=me + root_arg + ['diag', f'--task={task}', f'--seed={seed}'],
                           done_fn=lambda t=task, s=seed: (C.diag_dir(t, s) / 'diagnostics.json').is_file(),
                           deps_fn=lambda t=task, s=seed: ckpt_deps(C.FINAL_STEP, t, s), threads=4, **base))
            pb_methods = ['PB'] + ([C.REF_METHOD] if C.task_info(task)['ref_num_candidates'] != C.NUM_CANDIDATES
                                   and C.task_info(task)['ref_temperature'] != 0.0 else [])
            for h in C.H_EXECS:
                for m in pb_methods:
                    jobs.append(_J(name=f'eval/{task}_s{seed}/step{C.FINAL_STEP}/{m}_h{h}', kind='cpu',
                                   priority=(grp, 3, seed, ti, 1, C.FINAL_STEP),
                                   args=me + root_arg + ['eval', f'--task={task}', f'--seed={seed}', f'--step={C.FINAL_STEP}',
                                                         f'--method={m}', f'--h-exec={h}', f'--episodes-per-task={C.FINAL_EPISODES_PER_TASK}'],
                                   done_fn=lambda t=task, s=seed, m=m, h=h: (C.eval_dir(t, s, C.FINAL_STEP, m, h) / 'summary.json').is_file(),
                                   deps_fn=lambda t=task, s=seed: 'ready' if C.pb_run_complete(t, s) else 'wait', **base))
                for step in (*C.INTERMEDIATE_EVAL_STEPS, C.FINAL_STEP):
                    eps = C.FINAL_EPISODES_PER_TASK if step == C.FINAL_STEP else C.INTERMEDIATE_EPISODES_PER_TASK
                    for m in C.INTENTION_METHODS:
                        jobs.append(_J(name=f'eval/{task}_s{seed}/step{step}/{m}_h{h}', kind='cpu',
                                       priority=(grp, 3, seed, ti, 2, -step),
                                       args=me + root_arg + ['eval', f'--task={task}', f'--seed={seed}', f'--step={step}',
                                                             f'--method={m}', f'--h-exec={h}', f'--episodes-per-task={eps}'],
                                       done_fn=lambda t=task, s=seed, st=step, m=m, h=h: (C.eval_dir(t, s, st, m, h) / 'summary.json').is_file(),
                                       deps_fn=lambda st=step, t=task, s=seed: ckpt_deps(st, t, s), **base))
    return jobs


def discover_gpus() -> list[str]:
    try:
        out = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [str(i) for i, line in enumerate(out.splitlines()) if line.startswith('GPU ')]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    try:
        with open(f'/proc/{int(pid)}/stat') as f:
            return f.read().split()[2] != 'Z'
    except OSError:
        return False


def _job_env(job: Job, gpu: str | None) -> dict:
    env = dict(os.environ)
    env['MUJOCO_GL'] = env.get('MUJOCO_GL', 'egl')
    env['PYTHONUNBUFFERED'] = '1'
    env['IPB_EXP_ROOT'] = str(C.EXP_ROOT)
    if job.kind == 'gpu':
        env['CUDA_VISIBLE_DEVICES'] = gpu or '0'
        env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
        env.pop('JAX_PLATFORMS', None)
    else:
        env['JAX_PLATFORMS'] = 'cpu'
        env['CUDA_VISIBLE_DEVICES'] = ''
        n = str(job.threads)
        env['XLA_FLAGS'] = f'--xla_cpu_multi_thread_eigen={"true" if job.threads > 1 else "false"} intra_op_parallelism_threads={n}'
        for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            env[k] = n
    return env


def _wrap_cuda(args: list[str]) -> list[str]:
    script = PB_CODE_DIR / 'scripts' / 'jax_cuda_env.sh'
    if not script.is_file():
        raise FileNotFoundError(script)
    quoted = ' '.join("'" + x.replace("'", "'\\''") + "'" for x in args)
    return ['bash', '-c', f'export PYTHON={PYTHON}; source {script} && exec {quoted}']


GPU_START_GAP_S = 30.0  # simultaneous CUDA inits failed with CUDA_ERROR_NOT_INITIALIZED on svcho


class Launcher:
    def __init__(self, jobs: list[Job], *, max_gpu_jobs: int, max_cpu_threads: int, rerun_failed: bool, poll: float):
        self.jobs = sorted(jobs, key=lambda j: j.priority)
        self.max_gpu = max_gpu_jobs
        self.max_cpu = max_cpu_threads
        self.rerun_failed = rerun_failed
        self.poll = poll
        self.gpus = discover_gpus()
        self.state_root = C.EXP_ROOT / 'launcher'
        self.procs: dict[str, subprocess.Popen] = {}
        self.adopted: dict[str, int] = {}
        self.gpu_of: dict[str, str] = {}
        self.t0 = time.time()
        self.last_gpu_start = 0.0

    def _dir(self, job: Job) -> Path:
        return self.state_root / 'jobs' / job.name.replace('/', '__')

    def status(self, job: Job) -> dict:
        p = self._dir(job) / 'status.json'
        return C.read_json(p) if p.is_file() else {}

    def _set(self, job: Job, **kw) -> None:
        st = self.status(job)
        st.update(kw, name=job.name, kind=job.kind)
        C.atomic_write_json(self._dir(job) / 'status.json', st)

    def _log(self, msg: str) -> None:
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
        print(line, flush=True)
        self.state_root.mkdir(parents=True, exist_ok=True)
        with open(self.state_root / 'launcher.log', 'a') as f:
            f.write(line + '\n')

    def classify(self) -> dict[str, list[Job]]:
        out: dict[str, list[Job]] = {k: [] for k in ('done', 'running', 'ready', 'wait', 'skipped', 'failed')}
        for j in self.jobs:
            st = self.status(j)
            if j.name in self.procs or j.name in self.adopted:
                out['running'].append(j)
            elif j.done():
                out['done'].append(j)
            elif st.get('state') == 'failed':
                out['failed'].append(j)
            else:
                d = j.deps()
                if d.startswith('skip'):
                    out['skipped'].append(j)
                else:
                    out['ready' if d == 'ready' else 'wait'].append(j)
        return out

    def adopt_running(self) -> None:
        for j in self.jobs:
            st = self.status(j)
            if st.get('state') == 'running':
                pid = int(st.get('pid', -1))
                if _pid_alive(pid):
                    self.adopted[j.name] = pid
                    if j.kind == 'gpu':
                        self.gpu_of[j.name] = st.get('gpu', '0')
                    self._log(f'adopted live job {j.name} pid={pid}')
                else:
                    self._set(j, state='done' if j.done() else 'failed', end=time.time(), exit_code=st.get('exit_code'),
                              note='launcher restarted; process not alive')

    def _gpu_load(self) -> int:
        return sum(1 for n in list(self.procs) + list(self.adopted) if self._kind(n) == 'gpu')

    def _cpu_load(self) -> int:
        return sum(self._threads(n) for n in list(self.procs) + list(self.adopted) if self._kind(n) == 'cpu')

    def _kind(self, name):
        return next(j.kind for j in self.jobs if j.name == name)

    def _threads(self, name):
        return next(j.threads for j in self.jobs if j.name == name)

    def _pick_gpu(self) -> str:
        if not self.gpus:
            raise RuntimeError('No GPU discovered for a GPU job.')
        used = [self.gpu_of.get(n) for n in list(self.procs) + list(self.adopted) if self._kind(n) == 'gpu']
        return min(self.gpus, key=lambda g: used.count(g))

    def start(self, job: Job) -> None:
        d = self._dir(job)
        d.mkdir(parents=True, exist_ok=True)
        st = self.status(job)
        attempt = int(st.get('attempts', 0)) + 1
        gpu = self._pick_gpu() if job.kind == 'gpu' else None
        args = _wrap_cuda(job.args) if job.kind == 'gpu' else job.args
        out = open(d / f'stdout.attempt{attempt}.log', 'w')
        err = open(d / f'stderr.attempt{attempt}.log', 'w')
        p = subprocess.Popen(args, cwd=str(REPO), env=_job_env(job, gpu), stdout=out, stderr=err, start_new_session=True)
        self.procs[job.name] = p
        if gpu is not None:
            self.gpu_of[job.name] = gpu
        self._set(job, state='running', pid=p.pid, start=time.time(), start_str=time.strftime('%Y-%m-%d %H:%M:%S'),
                  attempts=attempt, gpu=gpu, cmd=job.args, end=None, exit_code=None)
        self._log(f'START {job.name} pid={p.pid} gpu={gpu} attempt={attempt}')

    def reap(self) -> None:
        for name, p in list(self.procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            job = next(j for j in self.jobs if j.name == name)
            ok = rc == 0 and job.done()
            note = '' if ok else ('exit 0 but output validation failed' if rc == 0 else f'exit {rc}')
            self._set(job, state='done' if ok else 'failed', end=time.time(), end_str=time.strftime('%Y-%m-%d %H:%M:%S'),
                      exit_code=rc, note=note)
            self._log(f'{"DONE" if ok else "FAIL"} {name} rc={rc} {note}')
            del self.procs[name]
            self.gpu_of.pop(name, None)
        for name, pid in list(self.adopted.items()):
            if not _pid_alive(pid):
                job = next(j for j in self.jobs if j.name == name)
                ok = job.done()
                self._set(job, state='done' if ok else 'failed', end=time.time(), note='adopted process ended')
                self._log(f'{"DONE" if ok else "FAIL"} (adopted) {name}')
                del self.adopted[name]
                self.gpu_of.pop(name, None)

    def snapshot(self) -> dict:
        c = self.classify()
        snap = {k: len(v) for k, v in c.items()}
        snap.update(running_jobs=[j.name for j in c['running']], failed_jobs=[j.name for j in c['failed']],
                    skipped_jobs=[j.name for j in c['skipped']], time=time.strftime('%Y-%m-%d %H:%M:%S'),
                    elapsed_s=time.time() - self.t0, total=len(self.jobs))
        C.atomic_write_json(self.state_root / 'snapshot.json', snap)
        return snap

    def run(self, aggregate_every: float = 1800.0) -> dict:
        if self.max_gpu > 0 and not self.gpus:
            raise RuntimeError('GPU jobs requested but nvidia-smi found no GPU.')
        self.adopt_running()
        if self.rerun_failed:
            for j in self.jobs:
                if self.status(j).get('state') == 'failed':
                    self._set(j, state='pending', note='reset by --rerun-failed')
        self._log(f'launcher start: {len(self.jobs)} jobs gpus={self.gpus} max_gpu_jobs={self.max_gpu} max_cpu_threads={self.max_cpu}')
        last_agg = time.time()
        last_snap = 0.0
        while True:
            self.reap()
            c = self.classify()
            hold = load_hold(self.state_root)
            runnable = [j for j in c['ready'] if hold_decision(j.name, hold) == 'run']
            for j in runnable:
                if j.kind == 'gpu' and self._gpu_load() < self.max_gpu and time.time() - self.last_gpu_start > GPU_START_GAP_S:
                    self.start(j)
                    self.last_gpu_start = time.time()
                elif j.kind == 'cpu' and self._cpu_load() + j.threads <= self.max_cpu:
                    self.start(j)
            if time.time() - last_snap > 60:
                snap = self.snapshot()
                last_snap = time.time()
                held_n = sum(1 for j in c['ready'] if hold_decision(j.name, hold) == 'hold')
                snap['held'] = held_n
                last_snap = time.time()
                self._log(f"status done={snap['done']} running={snap['running']} ready={snap['ready']} wait={snap['wait']} "
                          f"failed={snap['failed']} skipped={snap['skipped']} held={held_n}")
            if time.time() - last_agg > aggregate_every:
                self._aggregate()
                last_agg = time.time()
            if not self.procs and not self.adopted:
                c = self.classify()
                hold = load_hold(self.state_root)
                runnable = [j for j in c['ready'] if hold_decision(j.name, hold) == 'run']
                if not runnable:
                    if hold:
                        self._log(f'hold: not starting {sum(1 for j in c["ready"] if hold_decision(j.name, hold) == "hold")} ready jobs')
                    break
            time.sleep(self.poll)
        self._aggregate()
        snap = self.snapshot()
        self._log(f'launcher finished: {json.dumps({k: snap[k] for k in ("done", "failed", "skipped", "wait", "total")})}')
        return snap

    def _aggregate(self) -> None:
        rc = subprocess.call([PYTHON, str(REPO / 'run_intention_pathbridger.py'), f'--exp-root={C.EXP_ROOT}', 'aggregate'],
                             cwd=str(REPO), env=dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES=''),
                             stdout=subprocess.DEVNULL, stderr=open(self.state_root / 'aggregate.err.log', 'a'))
        self._log(f'aggregate rc={rc}')


def cmd_launch(a) -> None:
    seeds = tuple(int(s) for s in a.seeds.split(','))
    jobs = build_jobs(seeds=seeds)
    L = Launcher(jobs, max_gpu_jobs=int(a.max_gpu_jobs), max_cpu_threads=int(a.max_cpu_threads),
                 rerun_failed=bool(a.rerun_failed), poll=float(a.poll))
    c = L.classify()
    by_kind: dict[str, int] = {}
    for j in jobs:
        key = j.name.split('/')[0]
        by_kind[key] = by_kind.get(key, 0) + 1
    print(f'GPUs discovered: {L.gpus}  max_gpu_jobs={a.max_gpu_jobs}  max_cpu_threads={a.max_cpu_threads}')
    print('Job counts by type:', json.dumps(by_kind))
    train_jobs = by_kind.get('tokenizer', 0) + by_kind.get('conditioned', 0)
    print(f'New intention training jobs: {train_jobs} (spec expects 18 = 9 tokenizer + 9 conditioned); '
          f'additional PB baseline trainings: {by_kind.get("pb_train", 0)} (seeds 1-2 have no PB checkpoint; seed 0 reused).')
    print('State:', json.dumps({k: len(v) for k, v in c.items()}))
    if a.dry_run:
        for j in L.jobs:
            st = 'done' if j in c['done'] else ('failed' if j in c['failed'] else ('skipped' if j in c['skipped'] else j.deps()))
            print(f'  [{j.kind}] {j.name:60s} {st}')
        return
    L.run()


def cmd_smoke(a) -> None:
    """Tiny end-to-end run: real data/env/checkpoints, tiny step counts, separate output root."""
    root = C.EXP_ROOT
    if 'smoke' not in root.parts:
        raise SystemExit('smoke must be run with --exp-root pointing at a */smoke* directory.')
    me = [PYTHON, str(REPO / 'run_intention_pathbridger.py'), f'--exp-root={root}']
    task, seed = 'cube-single', 0
    env_gpu = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE='false', MUJOCO_GL='egl')
    env_cpu = dict(os.environ, JAX_PLATFORMS='cpu', MUJOCO_GL='egl')

    def run(args, env, gpu=False):
        args = _wrap_cuda(args) if gpu else args
        print('[smoke] $', ' '.join(args[-6:]), flush=True)
        rc = subprocess.call(args, cwd=str(REPO), env=env)
        if rc != 0:
            raise SystemExit(f'smoke step failed rc={rc}: {args}')

    run(me + ['tokenizer', f'--task={task}', f'--seed={seed}', '--steps=400', '--save-steps=200,400'], env_gpu, gpu=True)
    run(me + ['conditioned', f'--task={task}', f'--seed={seed}', '--steps=300', '--save-steps=100,200,300',
              '--tokenizer-step=400', '--log-every=100'], env_gpu, gpu=True)
    for m in ('PB', 'I-SG', 'Shared-I', 'Shuffled-I'):
        for h in (1, 5):
            run(me + ['eval', f'--task={task}', f'--seed={seed}', '--step=300', f'--method={m}', f'--h-exec={h}',
                      '--episodes-per-task=1'], env_cpu)
    run(me + ['diag', f'--task={task}', f'--seed={seed}', '--step=300', '--num-samples=512'], env_cpu)
    run(me + ['aggregate'], env_cpu)
    print('[smoke] OK')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--exp-root', default=None, help='Override exp/intention_pathbridger (used by smoke).')
    sub = p.add_subparsers(dest='cmd', required=True)

    def ts(sp):
        sp.add_argument('--task', required=True, choices=list(C.TASKS))
        sp.add_argument('--seed', required=True, type=int)

    sp = sub.add_parser('pb-train'); ts(sp)
    sp.add_argument('--steps', type=int, default=C.COND_STEPS)
    sp.add_argument('--save-every', type=int, default=100_000)
    sp.set_defaults(fn=cmd_pb_train)
    sp = sub.add_parser('tokenizer'); ts(sp)
    sp.add_argument('--steps', type=int, default=C.TOKENIZER_STEPS)
    sp.add_argument('--save-steps', default=','.join(map(str, C.TOKENIZER_SAVE_STEPS)))
    sp.set_defaults(fn=cmd_tokenizer)
    sp = sub.add_parser('conditioned'); ts(sp)
    sp.add_argument('--steps', type=int, default=C.COND_STEPS)
    sp.add_argument('--save-steps', default=','.join(map(str, C.COND_SAVE_STEPS)))
    sp.add_argument('--tokenizer-step', type=int, default=C.TOKENIZER_STEPS)
    sp.add_argument('--log-every', type=int, default=10_000)
    sp.set_defaults(fn=cmd_conditioned)
    sp = sub.add_parser('diag'); ts(sp)
    sp.add_argument('--step', type=int, default=C.FINAL_STEP)
    sp.add_argument('--num-samples', type=int, default=4096)
    sp.set_defaults(fn=cmd_diag)
    sp = sub.add_parser('eval'); ts(sp)
    sp.add_argument('--step', type=int, required=True)
    sp.add_argument('--method', required=True, choices=[*C.METHODS, C.REF_METHOD])
    sp.add_argument('--h-exec', type=int, required=True, choices=list(C.H_EXECS))
    sp.add_argument('--episodes-per-task', type=int, required=True)
    sp.set_defaults(fn=cmd_eval)
    sub.add_parser('aggregate').set_defaults(fn=cmd_aggregate)
    sp = sub.add_parser('launch')
    sp.add_argument('--dry-run', action='store_true')
    sp.add_argument('--max-gpu-jobs', type=int, default=2, help='Concurrent GPU training jobs (user policy: 2 on 1 GPU).')
    sp.add_argument('--max-cpu-threads', type=int, default=12)
    sp.add_argument('--rerun-failed', action='store_true')
    sp.add_argument('--seeds', default=','.join(map(str, C.SEEDS)))
    sp.add_argument('--poll', type=float, default=20.0)
    sp.set_defaults(fn=cmd_launch)
    sub.add_parser('smoke').set_defaults(fn=cmd_smoke)
    a = p.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
