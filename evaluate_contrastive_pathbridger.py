"""Evaluate a saved CPB run without retraining or changing execution settings."""
import argparse
import json
from pathlib import Path
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax.numpy as jnp
from agents.contrastive_pathbridger import ContrastivePathBridgerAgent
from utils.flax_utils import restore_agent, resolve_checkpoint
from utils.contrastive_pathbridger_evaluation import evaluate
from main_contrastive_pathbridger import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--checkpoint', type=int, default=1000000)
    p.add_argument('--h', type=int, choices=(1, 2, 5), default=2)
    p.add_argument('--episodes', type=int, default=50)
    args = p.parse_args()
    run = Path(args.run_dir)
    saved = json.loads((run / 'config.json').read_text())
    config, runtime = saved['agent'], saved['runtime']
    import ogbench
    env = ogbench.make_env_and_datasets(config['env_name'], env_only=True)
    agent = ContrastivePathBridgerAgent.create(runtime['seed'], jnp.zeros((1, *env.observation_space.shape)),
                                               jnp.zeros((1, *env.action_space.shape)), config)
    agent = restore_agent(agent, run / 'checkpoints', args.checkpoint)
    result = evaluate(agent, env, episodes_per_task=args.episodes, execute_h=args.h,
                      num_candidates=config['eval_num_candidates'], temperature=config['eval_temperature'], seed=runtime['seed'])
    result.update(env=config['env_name'], variant=config['variant'], seed=runtime['seed'], checkpoint=args.checkpoint)
    write_json(run / f'evaluation_{args.checkpoint}_h{args.h}.json', result)
    print(json.dumps(result, indent=2))
    env.close()

if __name__ == '__main__':
    main()
