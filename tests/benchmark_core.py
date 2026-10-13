"""Bounded warm GPU timing and memory; use separate processes for each engine."""
import argparse
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from check_core_equivalence import oracle
from agents import PathBridgerAgent
from utils.core_dataset import CoreDataset
from legacy_extract import load_extract_critic_value_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engine', choices=['reference', 'deployment'], required=True)
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--env', default='puzzle_3x3')
    p.add_argument('--output', type=Path, required=True)
    arg = p.parse_args()
    extract_fn, _ = load_extract_critic_value_params()
    with tempfile.TemporaryDirectory() as temp:
        D, C, *_ = oracle(arg.reference_root, temp)
        cfg = importlib.import_module('configs.pbf.'+arg.env).get_config()
        raw = ogbench.load_dataset(arg.dataset, compact_dataset=True)
        sampler = CoreDataset(raw, cfg)
        np.random.seed(1)
        db, cb = sampler.sample(1024)
        if arg.engine == 'reference':
            d = D.create(1, db['observations'], sampler.dynamics.config, ex_actions=db['actions'])
            c = C.create(1, cb['observations'], None, cb['action_chunk_actions'], sampler.critic.config,
                         ex_goals=cb['value_goals'])
            state = (d, c)
            del d, c  # Release initial states just as the native training loop does.
            def update(state, batch):
                d, c = state
                d, di = d.update(batch[0], critic_value_params=extract_fn(c))
                c, ci = c.update(batch[1])
                return (d, c), (di, ci)
        else:
            state = PathBridgerAgent.create(1, db['observations'], db['actions'], cfg, critic_batch=cb)
            def update(state, batch):
                return state.update(batch)
        batches = [jax.tree_util.tree_map(jnp.asarray, sampler.sample(1024)) for _ in range(4)]
        start = time.perf_counter()
        for i in range(10):
            state, info = update(state, batches[i % len(batches)])
            jax.block_until_ready((state, info))
        warmup = time.perf_counter()-start
        periods = []
        for trial in range(5):
            start = time.perf_counter()
            for i in range(100):
                state, info = update(state, batches[i % len(batches)])
            jax.block_until_ready((state, info))
            periods.append(time.perf_counter()-start)
        sampling = []
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(100):
                sampler.sample(1024)
            sampling.append((time.perf_counter()-start)/100)
        result = dict(engine=arg.engine, env=cfg.env_name, backend=jax.default_backend(),
                      device=str(jax.devices()[0]), batch_size=1024, measured_updates=500,
                      warmup_updates=10, warmup_including_compile_seconds=warmup,
                      trial_seconds_per_100_updates=periods, median_updates_per_second=100/float(np.median(periods)),
                      sampler_seconds_per_batch=float(np.median(sampling)),
                      memory=jax.devices()[0].memory_stats(), peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      xla_flags=os.environ.get('XLA_FLAGS'), preallocate=os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE'))
        arg.output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
