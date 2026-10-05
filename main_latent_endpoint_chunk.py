"""Train one latent endpoint chunk stage on a fixed offline dataset."""

from __future__ import annotations

import json
import os
import random
import signal
import time
from pathlib import Path

import jax
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

from agents.latent_endpoint_chunk import (
    LatentEndpointChunkAgent,
    restore_latent_endpoint_params,
)
from envs.env_utils import make_env_and_datasets
from utils.chunk_relabeling import LatentEndpointChunkDataset
from utils.flax_utils import resolve_checkpoint, restore_agent, save_agent
from utils.log_utils import CsvLogger, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS
_DEFAULT_CONFIG = str(
    Path(__file__).parent / 'configs/latent_endpoint_chunk/cube_single.py'
)

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_enum(
    'stage',
    'proposal',
    ['proposal', 'critic', 'policy_awr', 'policy_td3bc'],
    'Training stage.',
)
flags.DEFINE_integer('train_steps', 1_000_000, 'Absolute updates in this stage.')
flags.DEFINE_integer('batch_size', 1024, 'Offline batch size.')
flags.DEFINE_integer('log_interval', 10_000, 'CSV logging interval.')
flags.DEFINE_string(
    'checkpoint_steps',
    '100000,300000,500000,800000,1000000',
    'Comma-separated absolute checkpoints.',
)
flags.DEFINE_string('output_dir', '', 'Exact stage output directory.')
flags.DEFINE_string('save_dir', 'exp/latent_endpoint_chunk', 'Fallback output root.')
flags.DEFINE_string('dataset_dir', '', 'Optional OGBench dataset directory.')
flags.DEFINE_string('restore_path', '', 'Previous-stage checkpoint.')
flags.DEFINE_integer('restore_step', 0, 'Previous-stage checkpoint step.')
flags.DEFINE_string('resume_path', '', 'Same-stage checkpoint.')
flags.DEFINE_integer('resume_step', 0, 'Same-stage checkpoint step.')
flags.DEFINE_boolean('use_wandb', False, 'Enable W&B.')
flags.DEFINE_boolean('use_tqdm', True, 'Display progress.')
config_flags.DEFINE_config_file(
    'agent', _DEFAULT_CONFIG, 'Latent endpoint chunk config.', lock_config=False
)


def _host(info):
    return {
        key: float(np.asarray(jax.device_get(value)).reshape(()))
        for key, value in info.items()
    }


def main(_):
    if FLAGS.train_steps < 1 or FLAGS.batch_size < 2:
        raise ValueError('train_steps >= 1 and batch_size >= 2 are required.')
    if FLAGS.restore_path and FLAGS.resume_path:
        raise ValueError('restore_path and resume_path are mutually exclusive.')
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    config = FLAGS.agent
    env, train_data, _ = make_env_and_datasets(
        str(config.env_name), dataset_dir=FLAGS.dataset_dir or None
    )
    dataset = LatentEndpointChunkDataset(train_data, config)
    example = dataset.sample(2)
    agent = LatentEndpointChunkAgent.create(
        FLAGS.seed,
        example['observations'],
        example['action_chunks'],
        example['endpoint_states'],
        example['goals'],
        config,
        stage=FLAGS.stage,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
    )
    start_step = 0
    if FLAGS.restore_path:
        agent = restore_latent_endpoint_params(
            agent, FLAGS.restore_path, FLAGS.restore_step, restore_host_rng=False
        )
    elif FLAGS.resume_path:
        _, start_step = resolve_checkpoint(FLAGS.resume_path, FLAGS.resume_step)
        agent = restore_agent(agent, FLAGS.resume_path, FLAGS.resume_step)
    if start_step >= FLAGS.train_steps:
        print(f'Already at step {start_step}; nothing to do.')
        return

    run_dir = os.path.abspath(
        FLAGS.output_dir
        or os.path.join(
            FLAGS.save_dir,
            str(config.env_name),
            str(config.variant),
            f'seed{FLAGS.seed}',
            FLAGS.stage,
        )
    )
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        'flags': get_flag_dict(),
        'agent': config.to_dict(),
        'resolved_agent': dict(agent.config),
        'training_fields': [
            'observations',
            'actions',
            'terminals',
            'endpoint_states',
            'goals',
        ],
        'uses_reward_or_return': False,
    }
    filename = 'flags.json' if not start_step else f'flags_resume_{start_step}.json'
    with open(os.path.join(run_dir, filename), 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, sort_keys=True, default=str)
        file.write('\n')

    checkpoints = {
        int(value) for value in FLAGS.checkpoint_steps.split(',') if value.strip()
    }
    checkpoints.add(int(FLAGS.train_steps))
    stop: list[str] = []

    def request_stop(number, _frame):
        stop.append(signal.Signals(number).name)

    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, request_stop)
    logger = CsvLogger(os.path.join(run_dir, 'train.csv'), resume=start_step > 0)
    wandb_run = None
    if FLAGS.use_wandb:
        wandb_run = setup_wandb(
            project='LatentEndpointChunk',
            group=str(config.env_name),
            name=f'{config.variant}-{FLAGS.stage}-seed{FLAGS.seed}',
            config=payload,
            directory=run_dir,
        )
    steps = range(start_step + 1, FLAGS.train_steps + 1)
    if FLAGS.use_tqdm:
        steps = tqdm.tqdm(steps, dynamic_ncols=True, desc=f'{FLAGS.stage}@{start_step}')
    started = time.time()
    try:
        for step in steps:
            agent, info = agent.update(dataset.sample(FLAGS.batch_size))
            final = step == FLAGS.train_steps or bool(stop)
            if step % FLAGS.log_interval == 0 or final:
                metrics = _host(info)
                metrics['time/total_seconds'] = time.time() - started
                logger.log(metrics, step=step)
                if wandb_run is not None:
                    wandb_run.log(metrics, step=step)
            if step in checkpoints or final:
                save_agent(agent, checkpoint_dir, step)
            if stop:
                print(f'Interrupted by {stop[0]}; resume from step {step}.')
                break
    finally:
        logger.close()
        if wandb_run is not None:
            wandb_run.finish()
    print(f'Run saved to {run_dir}')


def run():
    app.run(main)


if __name__ == '__main__':
    run()
