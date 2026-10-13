"""CPU evaluation of a preserved-core checkpoint; no training dataset needed."""
from __future__ import annotations
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
from pathlib import Path
from absl import app, flags
from ml_collections import config_flags
from utils.evaluation import evaluate, PROTOCOL
from utils.flax_utils import load_agent, resolve_checkpoint

FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint_dir', '', 'Directory or exact params_<step>.pkl file.')
flags.DEFINE_integer('checkpoint_step', 0, 'Step, inferred for an exact checkpoint file.')
flags.DEFINE_integer('episodes', 25, 'Episodes per task.')
flags.DEFINE_integer('seed', 0, 'Accepted for compatibility; strict paired evaluation has fixed seeds.')
flags.DEFINE_string('dataset_dir', '', 'Accepted for compatibility; evaluation does not load datasets.')
flags.DEFINE_string('output_path', '', 'Optional result JSON.')
config_flags.DEFINE_config_file('agent', None, 'Optional config; must match checkpoint metadata.', lock_config=False)


def evaluate_checkpoint(path, step=0, episodes=25):
    import jax
    import ogbench
    if jax.default_backend() != 'cpu':
        raise RuntimeError('Evaluation must run in its own JAX_PLATFORMS=cpu process.')
    path, step = resolve_checkpoint(path, step)
    agent = load_agent(path, step)
    cfg = agent.config
    env = ogbench.make_env_and_datasets(cfg['env_name'], env_only=True)
    records = []
    try:
        env.reset()  # Match the agreed evaluator's preliminary environment reset.
        metrics = evaluate(agent, env, episodes_per_task=episodes,
                           num_candidates=cfg['eval_num_candidates'], temperature=cfg['eval_temperature'], records=records)
    finally:
        env.close()
    return dict(checkpoint_step=step, env_name=cfg['env_name'], endpoint_distribution=cfg['endpoint_distribution'],
                eval_num_candidates=cfg['eval_num_candidates'], eval_temperature=cfg['eval_temperature'],
                backend=jax.default_backend(), protocol=PROTOCOL, episodes=records,
                **{'evaluation/'+k: v for k, v in metrics.items()})


def main(_):
    if not FLAGS.checkpoint_dir:
        raise ValueError('checkpoint_dir is required.')
    if FLAGS.agent is not None:
        agent = load_agent(FLAGS.checkpoint_dir, FLAGS.checkpoint_step)
        if json.dumps(FLAGS.agent.to_dict(), sort_keys=True) != json.dumps(dict(agent.config), sort_keys=True):
            raise ValueError('Supplied agent config does not match checkpoint metadata.')
    result = evaluate_checkpoint(FLAGS.checkpoint_dir, FLAGS.checkpoint_step, FLAGS.episodes)
    text = json.dumps(result, indent=2)
    print(text)
    if FLAGS.output_path:
        path = Path(FLAGS.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text+'\n')


def run():
    app.run(main)


if __name__ == '__main__':
    run()
