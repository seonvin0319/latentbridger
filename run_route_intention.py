#!/usr/bin/env python3
"""CLI for Route-Level Intention PathBridger.

Stages stop at the representation gate. A 1M control job is not started from this
launcher unless ``aggregate/gate.json`` contains ``PASS`` and ``--run-control`` is set.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PYTHON = sys.executable


def _pre_parse_root() -> None:
    for i, a in enumerate(sys.argv):
        if a.startswith('--exp-root='):
            os.environ['ROUTE_EXP_ROOT'] = str(Path(a.split('=', 1)[1]).resolve())
        elif a == '--exp-root' and i + 1 < len(sys.argv):
            os.environ['ROUTE_EXP_ROOT'] = str(Path(sys.argv[i + 1]).resolve())


_pre_parse_root()

from intention_pb import PB_CODE_DIR  # noqa: E402
from route_intention.common import (  # noqa: E402
    EXP_ROOT,
    H_ROUTE,
    ORACLE_STEPS,
    SEEDS,
    TASK_ORDER,
    TOKENIZER_STEPS,
    atomic_write_json,
    require_gpu,
    task_info,
    tokenizer_dir,
)
from route_intention.gates import representation_gate  # noqa: E402
from intention_pb.pb_io import load_plain_datasets  # noqa: E402


def _foreign_gpu() -> bool:
    """True when a local-intention GPU trainer is still using the device."""
    try:
        out = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                             capture_output=True, text=True, check=False).stdout
    except OSError:
        return False
    pids = [ln.strip() for ln in out.splitlines() if ln.strip().isdigit()]
    for pid in pids:
        cmd = Path(f'/proc/{pid}/cmdline')
        if not cmd.is_file():
            continue
        raw = cmd.read_bytes().replace(b'\x00', b' ').decode(errors='ignore')
        if 'run_intention_pathbridger.py' in raw and 'run_route_intention.py' not in raw:
            return True
    return False


def cmd_tokenizer(a) -> None:
    from route_intention.tokenizer import RouteTokenizerConfig, train_route_tokenizer

    require_gpu()
    out = tokenizer_dir(a.task, a.seed)
    marker = out / 'COMPLETE.json'
    if marker.is_file() and not a.force:
        print(f'[route-tok] already complete: {marker}')
        return
    train, val = load_plain_datasets(task_info(a.task)['env_name'])
    t0 = time.time()
    metrics = train_route_tokenizer(
        out_dir=out, train=train, val=val, seed=int(a.seed), total_steps=int(a.steps),
        save_steps=tuple(int(s) for s in a.save_steps.split(',')),
        cfg=RouteTokenizerConfig(horizon=H_ROUTE, batch_size=int(a.batch_size)),
    )
    atomic_write_json(marker, dict(
        task=a.task, seed=int(a.seed), steps=int(a.steps), horizon=H_ROUTE,
        collapsed=bool(metrics['collapsed']), perplexity=metrics['perplexity'],
        max_usage=metrics['max_usage'], runtime_s=time.time() - t0,
    ))


def cmd_diagnose(a) -> None:
    from route_intention.diagnostics import run_variance_diagnostic

    run_variance_diagnostic(task=a.task, seed=int(a.seed), num_samples=int(a.num_samples))


def cmd_oracle(a) -> None:
    from route_intention.oracle import evaluate_oracle_subgoal, train_oracle_subgoal

    require_gpu()
    train_oracle_subgoal(task=a.task, seed=int(a.seed), total_steps=int(a.steps))
    evaluate_oracle_subgoal(task=a.task, seed=int(a.seed), step=int(a.steps))


def cmd_aggregate(_a) -> None:
    from route_intention.aggregate import write_aggregate

    path = write_aggregate()
    print(path)


def cmd_smoke(a) -> None:
    root = EXP_ROOT
    if 'smoke' not in root.parts:
        raise SystemExit('smoke requires --exp-root to contain a smoke directory.')
    os.environ.setdefault('IPB_ALLOW_CPU', '1')
    from route_intention.diagnostics import run_variance_diagnostic
    from route_intention.tokenizer import RouteTokenizerConfig, train_route_tokenizer

    task, seed = 'cube-single', 0
    train, val = load_plain_datasets(task_info(task)['env_name'])
    out = tokenizer_dir(task, seed)
    train_route_tokenizer(
        out_dir=out, train=train, val=val, seed=seed, total_steps=30, save_steps=(30,),
        log_every=30, cfg=RouteTokenizerConfig(batch_size=64),
    )
    atomic_write_json(out / 'COMPLETE.json', dict(task=task, seed=seed, steps=30, collapsed=False, smoke=True))
    # Diagnostic needs a 200k filename. Alias the smoke checkpoint.
    ck = out / 'checkpoints'
    src = ck / 'tokenizer_30.pkl'
    alias = ck / f'tokenizer_{TOKENIZER_STEPS}.pkl'
    if not alias.is_file():
        alias.write_bytes(src.read_bytes())
    run_variance_diagnostic(task=task, seed=seed, num_samples=512, step=TOKENIZER_STEPS)
    print('[route-smoke] OK', root)


def _gpu_argv(args: list[str]) -> list[str]:
    """Same CUDA setup as the local-intention launcher. A bare python falls back to CPU."""
    script = PB_CODE_DIR / 'scripts' / 'jax_cuda_env.sh'
    if not script.is_file():
        raise FileNotFoundError(script)
    quoted = ' '.join("'" + x.replace("'", "'\\''") + "'" for x in args)
    return ['bash', '-c', f'export PYTHON={PYTHON}; source {script} && exec {quoted}']


def _launch_one(args: list[str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, 'w')
    env = dict(os.environ)
    env['PYTHONUNBUFFERED'] = '1'
    env['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    env.pop('JAX_PLATFORMS', None)
    return subprocess.Popen(_gpu_argv(args), cwd=str(REPO), env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def cmd_screen(a) -> None:
    """Tokenizers, variance diagnostics, oracle models only where the variance gate passes, then stop."""
    from route_intention.aggregate import write_aggregate

    log_root = EXP_ROOT / 'launcher'
    log_root.mkdir(parents=True, exist_ok=True)
    jobs = [(task, seed) for seed in SEEDS for task in TASK_ORDER]
    pending = []
    for task, seed in jobs:
        if not (tokenizer_dir(task, seed) / 'COMPLETE.json').is_file():
            pending.append((task, seed))
    active: list[tuple[subprocess.Popen, str]] = []
    failed: list[tuple[str, int]] = []
    last_start = 0.0
    while pending or active:
        still = []
        for proc, name in active:
            rc = proc.poll()
            if rc is None:
                still.append((proc, name))
            elif rc != 0:
                failed.append((name, rc))
        active = still
        if failed:
            raise SystemExit(f'route jobs failed: {failed}')
        while pending and len(active) < int(a.max_gpu_jobs) and time.time() - last_start > 30:
            if _foreign_gpu():
                print('[route-screen] waiting for local-intention GPU jobs', flush=True)
                time.sleep(30)
                break
            task, seed = pending.pop(0)
            cmd = [PYTHON, str(REPO / 'run_route_intention.py'), f'--exp-root={EXP_ROOT}',
                   'tokenizer', f'--task={task}', f'--seed={seed}', f'--steps={TOKENIZER_STEPS}']
            print('[route-screen] start', task, seed, flush=True)
            active.append((_launch_one(cmd, log_root / f'tokenizer_{task}_s{seed}.log'), f'{task}_s{seed}'))
            last_start = time.time()
        time.sleep(10)
    print('[route-screen] tokenizers finished', flush=True)
    for task, seed in jobs:
        cmd_diagnose_ns = [PYTHON, str(REPO / 'run_route_intention.py'), f'--exp-root={EXP_ROOT}',
                           'diagnose', f'--task={task}', f'--seed={seed}']
        env = dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='')
        rc = subprocess.call(cmd_diagnose_ns, cwd=str(REPO), env=env)
        if rc != 0:
            raise SystemExit(f'diagnose {task} seed{seed} rc={rc}')
    oracle_jobs = []
    for task, seed in jobs:
        metrics = json.loads((tokenizer_dir(task, seed) / 'metrics' / f'step_{TOKENIZER_STEPS}.json').read_text())
        var = json.loads((EXP_ROOT / 'diagnostics' / f'{task}_seed{seed}' / 'variance.json').read_text())
        gate = representation_gate(perplexity=metrics['perplexity'], max_usage=metrics['max_usage'],
                                   variance_ratio=var['variance_ratio'], task=task)
        print(f'[route-screen] {task} s{seed} pre-oracle gate={gate} ratio={var["variance_ratio"]:.3f}', flush=True)
        if gate == 'NEED_ORACLE':
            oracle_jobs.append((task, seed))
    if not oracle_jobs:
        write_aggregate()
        print('[route-screen] variance gate failed for every task/seed. No oracle and no 1M control.', flush=True)
        return
    pending = list(oracle_jobs)
    active = []
    last_start = 0.0
    while pending or active:
        still = []
        for proc, name in active:
            rc = proc.poll()
            if rc is None:
                still.append((proc, name))
            elif rc != 0:
                failed.append((name, rc))
        active = still
        if failed:
            raise SystemExit(f'oracle jobs failed: {failed}')
        while pending and len(active) < int(a.max_gpu_jobs) and time.time() - last_start > 30:
            task, seed = pending.pop(0)
            cmd = [PYTHON, str(REPO / 'run_route_intention.py'), f'--exp-root={EXP_ROOT}',
                   'oracle', f'--task={task}', f'--seed={seed}', f'--steps={ORACLE_STEPS}']
            print('[route-screen] oracle', task, seed, flush=True)
            active.append((_launch_one(cmd, log_root / f'oracle_{task}_s{seed}.log'), f'{task}_s{seed}'))
            last_start = time.time()
        time.sleep(10)
    path = write_aggregate()
    print('[route-screen] gate written', path, flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exp-root', default=None)
    sub = p.add_subparsers(dest='cmd', required=True)

    def ts(sp):
        sp.add_argument('--task', required=True, choices=list(TASK_ORDER))
        sp.add_argument('--seed', required=True, type=int)

    sp = sub.add_parser('tokenizer'); ts(sp)
    sp.add_argument('--steps', type=int, default=TOKENIZER_STEPS)
    sp.add_argument('--save-steps', default='50000,100000,200000')
    sp.add_argument('--batch-size', type=int, default=1024)
    sp.add_argument('--force', action='store_true')
    sp.set_defaults(fn=cmd_tokenizer)
    sp = sub.add_parser('diagnose'); ts(sp)
    sp.add_argument('--num-samples', type=int, default=4096)
    sp.set_defaults(fn=cmd_diagnose)
    sp = sub.add_parser('oracle'); ts(sp)
    sp.add_argument('--steps', type=int, default=ORACLE_STEPS)
    sp.set_defaults(fn=cmd_oracle)
    sub.add_parser('aggregate').set_defaults(fn=cmd_aggregate)
    sub.add_parser('smoke').set_defaults(fn=cmd_smoke)
    sp = sub.add_parser('screen')
    sp.add_argument('--max-gpu-jobs', type=int, default=2)
    sp.set_defaults(fn=cmd_screen)
    a = p.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
