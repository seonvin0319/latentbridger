"""Offline diagnostics for a LatentBridger checkpoint.

Diagnostics run on the held-out OGBench validation split by default, because
future-retrieval recall on the training split is not evidence that the
representation generalizes.

The gate for Module B is ``action_sensitivity``: if ``C(s, a, g)`` does not
separate the dataset action from shuffled and uniform alternatives, the actor's
contrastive gradient carries no information about actions and the latent flow
is being built on a critic that cannot grade it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from absl import app, flags
from ml_collections import config_flags

from agents.latentbridger import LatentBridgerAgent, restore_latent_params
from envs.env_utils import make_env_and_datasets
from utils.flax_utils import resolve_checkpoint
from utils.latent_datasets import LatentBridgerDataset
from utils.latent_diagnostics import run_all_diagnostics

FLAGS = flags.FLAGS
_DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent / 'configs' / 'latent' / 'antmaze_medium.py'
)
_DEFAULT_DELTAS = (1, 2, 4, 8, 16, 32)

flags.DEFINE_string('checkpoint_dir', '', 'Checkpoint directory or exact .pkl file.')
flags.DEFINE_integer(
    'checkpoint_step',
    0,
    'Checkpoint step; required for a directory and inferred from an exact '
    'params_<step>.pkl file.',
)
flags.DEFINE_string('dataset_dir', '', 'Optional OGBench dataset directory.')
flags.DEFINE_string('split', 'val', "Diagnostic split: 'val' or 'train'.")
flags.DEFINE_integer('batch_size', 256, 'Diagnostic batch size.')
flags.DEFINE_integer('num_batches', 8, 'Number of diagnostic batches.')
flags.DEFINE_integer('seed', 0, 'Diagnostic seed.')
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
    if FLAGS.batch_size < 2:
        raise ValueError('batch_size must be at least 2 for retrieval metrics.')
    if FLAGS.num_batches < 1:
        raise ValueError('num_batches must be at least 1.')
    split = str(FLAGS.split).lower()
    if split not in ('train', 'val'):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}.")
    _, checkpoint_step = resolve_checkpoint(
        FLAGS.checkpoint_dir,
        FLAGS.checkpoint_step,
    )

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    config = FLAGS.agent
    env, train_data, val_data = make_env_and_datasets(
        str(config.env_name),
        dataset_dir=FLAGS.dataset_dir or None,
    )
    dataset = LatentBridgerDataset(
        val_data if split == 'val' else train_data,
        config,
    )
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

    metrics = run_all_diagnostics(
        agent,
        dataset,
        action_low=np.asarray(env.action_space.low, dtype=np.float32),
        action_high=np.asarray(env.action_space.high, dtype=np.float32),
        batch_size=int(FLAGS.batch_size),
        num_batches=int(FLAGS.num_batches),
        deltas=_DEFAULT_DELTAS,
        seed=int(FLAGS.seed),
    )
    result = {
        'checkpoint_step': checkpoint_step,
        'env_name': str(config.env_name),
        'variant': str(config.variant),
        'split': split,
        'seed': int(FLAGS.seed),
        **{f'diagnostics/{key}': value for key, value in metrics.items()},
    }
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if FLAGS.output_path:
        output_path = Path(FLAGS.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as file:
            file.write(text)
            file.write('\n')


def run():
    """Run the command-line diagnostics entry point."""

    app.run(main)


if __name__ == '__main__':
    run()
