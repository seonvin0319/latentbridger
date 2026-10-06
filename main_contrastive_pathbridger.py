"""One joint CPB run, with exact checkpoints and paired h=1/2/5 evaluation."""
import argparse
import importlib
import json
import os
import random
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MUJOCO_GL', 'egl')
import jax
import numpy as np
from agents.contrastive_pathbridger import ContrastivePathBridgerAgent
from envs.env_utils import make_env_and_datasets
from utils.datasets import PathBridgerDataset
from utils.flax_utils import save_agent, restore_agent
from utils.contrastive_pathbridger_evaluation import evaluate
from utils.cpb_diagnostics import diagnostics

CHECKPOINTS = (100000, 300000, 500000, 800000, 1000000)


def write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def config_for(env, variant):
    config = importlib.import_module('configs.cpb.' + env).get_config()
    config.variant = variant
    return config


def run(args):
    config = config_for(args.env, args.variant)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / 'config.json', dict(agent=config.to_dict(), runtime=vars(args)))
    random.seed(args.seed)
    np.random.seed(args.seed)
    env, train, val = make_env_and_datasets(config.env_name)
    dataset = PathBridgerDataset(train, config)
    validation = PathBridgerDataset(val, config)
    example = dataset.sample(1)
    agent = ContrastivePathBridgerAgent.create(args.seed, example['observations'], example['actions'], config)
    start = 0
    if args.restore:
        agent = restore_agent(agent, args.restore)
        start = int(agent.network.step) - 1
    import importlib.metadata
    import subprocess
    write_json(run_dir / 'provenance.json', dict(
        commit=subprocess.check_output(['/home/shchoi/miniconda3/bin/git', 'rev-parse', 'HEAD'], text=True).strip(),
        packages={name: importlib.metadata.version(name) for name in ('jax', 'flax', 'optax', 'mujoco', 'ogbench', 'gymnasium')},
        devices=[str(device) for device in jax.devices()],
    ))
    started = time.time()
    metrics = {}
    with ThreadPoolExecutor(max_workers=1) as pool, (run_dir / 'train.jsonl').open('a') as log:
        future = pool.submit(dataset.sample, args.batch_size)
        for step in range(start + 1, args.steps + 1):
            batch = future.result()
            save = step in CHECKPOINTS or step == args.steps
            if not save:
                future = pool.submit(dataset.sample, args.batch_size)
            agent, info = agent.update(batch)
            # Check every update for nonfinite losses without transferring the entire tree.
            if not bool(np.asarray(jnp_finite(info))):
                raise FloatingPointError(f'Nonfinite training metric at step {step}')
            if step % args.log_interval == 0 or save:
                metrics = {k: float(np.asarray(v)) for k, v in info.items()}
                metrics.update(step=step, wall_seconds=time.time()-started)
                log.write(json.dumps(metrics, allow_nan=False) + '\n'); log.flush()
                print(json.dumps(metrics), flush=True)
            if save:
                checkpoint = save_agent(agent, run_dir / 'checkpoints', step)
                restored = restore_agent(agent, checkpoint)
                for x, y in zip(jax.tree_util.tree_leaves(agent.network.params), jax.tree_util.tree_leaves(restored.network.params)):
                    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
                # Diagnostics must not alter the subsequent training RNG stream.
                rng_state = np.random.get_state()
                vb = validation.sample(128)
                _, critic_info = agent.value_loss(vb, agent.network.params)
                diag = {k: float(np.asarray(v)) for k, v in critic_info.items()}
                if args.variant != 'pathbridger_original':
                    diag.update(diagnostics(agent, vb, train['observations']))
                diag.update(step=step, **{k: v for k, v in metrics.items() if k != 'step' and not k.startswith('critic/')})
                diag['critic/batch_size'] = len(vb['observations'])
                write_json(run_dir / f'diagnostics_{step}.json', diag)
                if args.smoke:
                    # Independent validation positives, against B=128 chance.
                    if float(np.asarray(critic_info['critic/recall_at_1'])) <= 1 / len(vb['observations']):
                        raise RuntimeError('Smoke critic recall is not above chance')
                    for h in (1, 2, 5):
                        actions = agent.sample_action_chunks(vb['observations'][:1], vb['value_goals'][:1])
                        assert actions.shape[1] == 5 and np.isfinite(np.asarray(actions)).all()
                    smoke_eval = evaluate(agent, env, episodes_per_task=1, execute_h=2,
                            num_candidates=config.eval_num_candidates, temperature=config.eval_temperature, seed=args.seed)
                    write_json(run_dir / 'smoke_evaluation.json', smoke_eval)
                if not args.smoke:
                    for h in (1, 2, 5):
                        result = evaluate(agent, env, episodes_per_task=args.episodes,
                                num_candidates=config.eval_num_candidates, temperature=config.eval_temperature,
                                seed=args.seed, execute_h=h)
                        result.update(env=config.env_name, variant=args.variant, seed=args.seed, checkpoint=step)
                        write_json(run_dir / f'evaluation_{step}_h{h}.json', result)
                np.random.set_state(rng_state)
                if step < args.steps:
                    future = pool.submit(dataset.sample, args.batch_size)
    env.close()
    write_json(run_dir / 'complete.json', dict(steps=args.steps, wall_seconds=time.time()-started, smoke=args.smoke))


@jax.jit
def jnp_finite(info):
    import jax.numpy as jnp
    return jnp.all(jnp.stack([jnp.isfinite(x) for x in info.values()]))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--env', choices=('cube_single', 'cube_double', 'puzzle_3x3', 'antmaze_medium'), default='cube_single')
    p.add_argument('--variant', choices=('cpb_full', 'cpb_rank_only', 'pathbridger_original'), default='cpb_full')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=1000000)
    p.add_argument('--batch-size', type=int, default=1024)
    p.add_argument('--episodes', type=int, default=50)
    p.add_argument('--log-interval', type=int, default=1000)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--restore', default='')
    p.add_argument('--smoke', action='store_true')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    if args.steps < 1 or args.batch_size < 2 or args.episodes < 1:
        raise ValueError('Invalid run size')
    run(args)
