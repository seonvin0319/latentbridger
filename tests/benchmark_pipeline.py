"""Measure sampling + transfer + native updates without changing the core.

Each mode runs in its own process. Final state and NumPy RNG digests also
detect a changed update/batch stream. Compilation, saving and evaluation are
excluded; metrics are collected once per measured block.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flax.serialization
import jax
import numpy as np
import ogbench
from agents import PathBridgerAgent
from utils.core_dataset import CoreDataset
from utils.training import own_training_state, update_training_state


def digest(tree):
    out = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        out.update(str((array.dtype.str, array.shape)).encode())
        out.update(array.tobytes())
    return out.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['serial', 'host', 'device', 'donate'], required=True)
    p.add_argument('--env', default='puzzle_3x3')
    p.add_argument('--distribution', default='pbf')
    p.add_argument('--dataset', required=True)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--trials', type=int, default=5)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cfg = importlib.import_module(f'configs.{args.distribution}.{args.env}').get_config()
    raw = ogbench.load_dataset(args.dataset, compact_dataset=True)
    sampler = CoreDataset(raw, cfg)
    np.random.seed(1)
    db, cb = sampler.sample(1024)
    state = PathBridgerAgent.create(1, db['observations'], db['actions'], cfg, critic_batch=cb)
    if args.mode == 'donate':
        state = own_training_state(state)
        update = update_training_state
    else:
        def update(agent, batch):
            return agent.update(batch)
    start = time.perf_counter()
    for _ in range(10):
        state, info = update(state, sampler.sample(1024))
        jax.block_until_ready((state, info))
    warmup = time.perf_counter() - start

    def sample():
        batch = sampler.sample(1024)
        return jax.device_put(batch) if args.mode == 'device' else batch

    timings = []
    with ThreadPoolExecutor(max_workers=1) as pool:
        for trial in range(args.trials):
            start = time.perf_counter()
            future = pool.submit(sample) if args.mode != 'serial' else None
            for step in range(args.steps):
                batch = sample() if future is None else future.result()
                future = pool.submit(sample) if args.mode != 'serial' and step + 1 < args.steps else None
                state, info = update(state, batch)
            jax.block_until_ready((state, info))
            timings.append(time.perf_counter() - start)
    result = dict(mode=args.mode, env=cfg.env_name, distribution=args.distribution,
                  batch_size=1024, backend=jax.default_backend(), device=str(jax.devices()[0]),
                  measured_updates=args.steps*args.trials, warmup_updates=10,
                  warmup_including_compile_seconds=warmup, trial_seconds=timings,
                  median_updates_per_second=args.steps/float(np.median(timings)),
                  final_state_sha256=digest(flax.serialization.to_state_dict(state)),
                  final_metrics_sha256=digest(info), final_numpy_rng_sha256=digest(np.random.get_state()),
                  memory=jax.devices()[0].memory_stats(),
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  xla_flags=os.environ.get('XLA_FLAGS'), preallocate=os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
