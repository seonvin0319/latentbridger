"""Dump N PathBridger updates (params + losses) for cross-tree numerical compare.

Usage:
  PYTHONPATH=<repo> python tests/probe_pb_update_trace.py \\
    --config configs/pbf/puzzle_3x3.py --dataset_dir ~/.ogbench/data \\
    --steps 4 --seed 1 --out /tmp/trace.pkl
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from agents import PathBridgerAgent
from utils.core_dataset import CoreDataset
from envs.env_utils import make_env_and_datasets


def _load_config(path: str):
    p = Path(path).resolve()
    spec = importlib.util.spec_from_file_location('probe_cfg', p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.get_config()


def _tree_nbytes(obj) -> int:
    return int(sum(np.asarray(x).nbytes for x in jax.tree_util.tree_leaves(obj)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--dataset_dir', default='')
    ap.add_argument('--steps', type=int, default=4)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=256)
    ap.add_argument('--out', required=True)
    ap.add_argument('--force_mode', default='', help='Optional override pathbridger_mode')
    args = ap.parse_args()

    cfg = _load_config(args.config)
    if args.force_mode:
        cfg.pathbridger_mode = args.force_mode

    env_name = str(cfg.env_name)
    dataset_dir = args.dataset_dir.strip() or None
    env, train_data, _val = make_env_and_datasets(env_name, dataset_dir=dataset_dir)
    env.close()
    del env, _val

    sampler = CoreDataset(train_data, cfg)
    np.random.seed(args.seed)
    db, cb = sampler.sample(args.batch_size)
    agent = PathBridgerAgent.create(
        args.seed, db['observations'], db['actions'], cfg, critic_batch=cb
    )

    def pack_params(a):
        return {
            'dynamics': jax.tree_util.tree_map(lambda x: np.asarray(x), a.dynamics.network.params),
            'critic': jax.tree_util.tree_map(lambda x: np.asarray(x), a.critic.network.params),
            'd_opt': jax.tree_util.tree_map(lambda x: np.asarray(x), a.dynamics.network.opt_state),
            'c_opt': jax.tree_util.tree_map(lambda x: np.asarray(x), a.critic.network.opt_state),
            'd_rng': np.asarray(a.dynamics.rng),
            'c_rng': np.asarray(a.critic.rng),
        }

    mode = str(cfg.get('pathbridger_mode', 'pb')) if hasattr(cfg, 'get') else str(
        dict(cfg).get('pathbridger_mode', 'pb')
    )
    try:
        mode = str(cfg.pathbridger_mode)
    except Exception:
        mode = 'pb'

    trace = {
        'backend': jax.default_backend(),
        'env_name': env_name,
        'mode': mode,
        'seed': args.seed,
        'batch_size': args.batch_size,
        'steps': args.steps,
        'init': pack_params(agent),
        'updates': [],
    }

    for i in range(args.steps):
        batch = sampler.sample(args.batch_size)
        batch_j = jax.tree_util.tree_map(jnp.asarray, batch)
        agent, info = agent.update(batch_j)
        info_np = {k: float(np.asarray(v)) for k, v in info.items()}
        packed = pack_params(agent)
        trace['updates'].append({'step': i, 'info': info_np, 'params': packed})
        print(
            f"STEP {i} critic_total={info_np.get('critic/total_loss')} "
            f"dyn={info_np.get('dynamics/phase1/loss')} "
            f"nbytes_c={_tree_nbytes(packed['critic'])}",
            flush=True,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('wb') as f:
        pickle.dump(trace, f)
    print('WROTE', out, flush=True)


if __name__ == '__main__':
    main()
