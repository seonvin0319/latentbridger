"""Evaluate one LatentBridger checkpoint on the five OGBench tasks.

``--mode=direct_goal`` is the Module-A test: the actor is driven by
``psi(final task goal)`` at every environment step, with no generated latent
anywhere in the loop.  ``--mode=latent_flow`` adds Module B's five-step latent
prefix under receding-horizon replanning.
"""

from __future__ import annotations

import json
from pathlib import Path

from absl import app, flags
from ml_collections import config_flags

from agents.latentbridger import LatentBridgerAgent, restore_latent_params
from envs.env_utils import make_env_and_datasets
from utils.flax_utils import resolve_checkpoint
from utils.latent_datasets import LatentBridgerDataset
from utils.latent_evaluation import DEFAULT_TASK_IDS, EVAL_MODES, evaluate_latent

FLAGS = flags.FLAGS
_DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent / 'configs' / 'latent' / 'antmaze_medium.py'
)

flags.DEFINE_string('checkpoint_dir', '', 'Checkpoint directory or exact .pkl file.')
flags.DEFINE_integer(
    'checkpoint_step',
    0,
    'Checkpoint step; required for a directory and inferred from an exact '
    'params_<step>.pkl file.',
)
flags.DEFINE_string('dataset_dir', '', 'Optional OGBench dataset directory.')
flags.DEFINE_string('mode', 'direct_goal', f'One of {EVAL_MODES}.')
flags.DEFINE_integer(
    'replan_interval',
    0,
    'Latent-flow actions executed per generated prefix; 0 uses the config '
    'default (the full action horizon).',
)
flags.DEFINE_float(
    'slerp_alpha',
    1.0,
    'Interpolation fraction for --mode=slerp_bridge; 1.0 is direct_goal.',
)
flags.DEFINE_integer('episodes', 50, 'Episodes for each of the five predefined tasks.')
flags.DEFINE_integer('seed', 0, 'Evaluation seed.')
flags.DEFINE_string(
    'manifest_path',
    '',
    'Optional paired episode manifest JSON; generated from --seed if absent.',
)
flags.DEFINE_string('output_path', '', 'Optional JSON result path.')

config_flags.DEFINE_config_file(
    'agent',
    _DEFAULT_CONFIG,
    'LatentBridger config used by the checkpoint; append :<variant> to select a variant.',
    lock_config=False,
)


def main(_):
    if not FLAGS.checkpoint_dir:
        raise ValueError('checkpoint_dir is required.')
    if FLAGS.checkpoint_step < 0:
        raise ValueError('checkpoint_step cannot be negative.')
    if FLAGS.episodes < 1:
        raise ValueError('episodes must be at least 1.')
    _, checkpoint_step = resolve_checkpoint(
        FLAGS.checkpoint_dir,
        FLAGS.checkpoint_step,
    )

    config = FLAGS.agent
    env, train_data, _ = make_env_and_datasets(
        str(config.env_name),
        dataset_dir=FLAGS.dataset_dir or None,
    )
    dataset = LatentBridgerDataset(train_data, config)
    example_batch = dataset.sample(2)
    agent = LatentBridgerAgent.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
        stage='joint',
        action_low=env.action_space.low,
        action_high=env.action_space.high,
    )
    agent = restore_latent_params(
        agent,
        FLAGS.checkpoint_dir,
        FLAGS.checkpoint_step,
        restore_host_rng=False,
    )

    manifest = None
    if FLAGS.manifest_path:
        with Path(FLAGS.manifest_path).open('r', encoding='utf-8') as file:
            manifest = json.load(file)['episodes']

    metrics = evaluate_latent(
        agent,
        env,
        mode=str(FLAGS.mode),
        task_ids=DEFAULT_TASK_IDS,
        episodes_per_task=FLAGS.episodes,
        seed=FLAGS.seed,
        replan_interval=FLAGS.replan_interval or None,
        alpha=FLAGS.slerp_alpha,
        manifest=manifest,
    )
    result = {
        'checkpoint_step': checkpoint_step,
        'env_name': str(config.env_name),
        'variant': str(config.variant),
        'mode': str(FLAGS.mode),
        'seed': int(FLAGS.seed),
        'manifest_path': str(FLAGS.manifest_path),
        **{f'evaluation/{key}': value for key, value in metrics.items()},
    }
    if 'replan_interval' in metrics:
        result['replan_interval'] = int(metrics['replan_interval'])
    if 'alpha' in metrics:
        result['slerp_alpha'] = float(metrics['alpha'])
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if FLAGS.output_path:
        output_path = Path(FLAGS.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as file:
            file.write(text)
            file.write('\n')


def run():
    """Run the command-line evaluation entry point."""

    app.run(main)


if __name__ == '__main__':
    run()
