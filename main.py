"""Train PathBridger on a fixed state-based OGBench dataset."""

from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from utils.runtime import configure_training_runtime

configure_training_runtime()

import jax
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

from agents import PathBridgerAgent
from envs.env_utils import make_env_and_datasets
from utils.core_dataset import CoreDataset
from utils.training import own_training_state, update_training_state
from utils.async_evaluation import CpuEvaluator
from utils.flax_utils import resolve_checkpoint, restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS

_BATCH_SIZE = 1024
_DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent / 'configs' / 'pbf' / 'antmaze_medium.py'
)

flags.DEFINE_string('run_group', 'Debug', 'Experiment group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('save_dir', 'exp/', 'Root output directory.')
flags.DEFINE_string('dataset_dir', '', 'Optional OGBench dataset directory.')
flags.DEFINE_string('restore_path', '', 'Checkpoint directory or exact .pkl file.')
flags.DEFINE_integer(
    'restore_step',
    0,
    'Checkpoint step; required for a directory and inferred from an exact params_<step>.pkl file.',
)
flags.DEFINE_integer('train_steps', 1_000_000, 'Number of gradient updates.')
flags.DEFINE_integer('log_interval', 5_000, 'Training CSV/W&B logging interval.')
flags.DEFINE_integer('eval_interval', 100_000, 'Environment evaluation interval; 0 disables intermediate eval.')
flags.DEFINE_integer('save_interval', 100_000, 'Checkpoint interval; 0 saves only the final checkpoint.')
flags.DEFINE_integer('eval_episodes', 25, 'Evaluation episodes for each of the five OGBench tasks.')
flags.DEFINE_boolean('eval_final', True, 'Queue CPU evaluation at the final step.')
flags.DEFINE_boolean('use_wandb', False, 'Enable optional Weights & Biases logging.')
flags.DEFINE_boolean('use_tqdm', True, 'Show a training progress bar.')
flags.DEFINE_boolean(
    'async_prefetch',
    True,
    'Overlap host batch sampling with the accelerator update.',
)

flags.DEFINE_boolean(
    'reuse_train_buffers', True,
    'Reuse consumed training-state buffers; disable with --noreuse_train_buffers.',
)

config_flags.DEFINE_config_file(
    'agent',
    _DEFAULT_CONFIG,
    'PathBridger agent and paper-reproduction environment config.',
    lock_config=False,
)


def _host_metrics(info) -> dict[str, float]:
    metrics = {}
    for key, value in jax.device_get(info).items():
        array = np.asarray(value)
        if array.size != 1:
            raise ValueError(f'Training metric {key!r} must be scalar, got shape {array.shape}.')
        metrics[str(key)] = float(array.reshape(()))
    # Public plotting names remain available. Alias already-computed host
    # scalars only; never add computations to the original JIT/loss graph.
    for alias, source in {
        'value/loss': 'critic/value/loss',
        'endpoint/loss': 'dynamics/phase1/loss_subgoal',
        'bridge/loss': 'dynamics/forward_bridge/loss_path_interior',
        'idm/loss': 'dynamics/phase1/loss_idm',
    }.items():
        metrics[alias] = metrics[source]
    return metrics


def _write_run_config(run_dir: str, config) -> None:
    payload = {
        'flags': get_flag_dict(),
        'agent': config.to_dict() if hasattr(config, 'to_dict') else dict(config),
    }
    with open(os.path.join(run_dir, 'flags.json'), 'w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write('\n')


def _validate_runtime_flags() -> None:
    if FLAGS.train_steps < 1:
        raise ValueError('train_steps must be at least 1.')
    if FLAGS.restore_step < 0:
        raise ValueError('restore_step cannot be negative.')
    if FLAGS.restore_step and not FLAGS.restore_path:
        raise ValueError('restore_step requires restore_path.')
    if FLAGS.log_interval < 1:
        raise ValueError('log_interval must be at least 1.')
    if FLAGS.eval_interval < 0 or FLAGS.save_interval < 0:
        raise ValueError('eval_interval and save_interval cannot be negative.')
    if FLAGS.eval_episodes < 1:
        raise ValueError('eval_episodes must be at least 1.')


def main(_):
    _validate_runtime_flags()
    config = FLAGS.agent
    restore_step = 0
    if FLAGS.restore_path:
        _, restore_step = resolve_checkpoint(
            FLAGS.restore_path,
            FLAGS.restore_step,
        )
        if restore_step >= FLAGS.train_steps:
            raise ValueError('The restored step must be smaller than train_steps.')

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    env, train_data, validation_data = make_env_and_datasets(
        str(config.env_name),
        dataset_dir=FLAGS.dataset_dir or None,
    )
    dataset = CoreDataset(train_data, config)
    # The detached evaluator creates its own environment and needs no dataset.
    env.close()
    del env, train_data, validation_data
    # Original main.py reseeds immediately before its full-size initialization batch.
    np.random.seed(FLAGS.seed)
    example_batch, example_critic = dataset.sample(_BATCH_SIZE)
    agent = PathBridgerAgent.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
        critic_batch=example_critic,
    )
    if FLAGS.restore_path:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_step)

    if FLAGS.reuse_train_buffers:
        agent = own_training_state(agent)

    exp_name = get_exp_name(
        FLAGS.seed,
        env_name=str(config.env_name),
        agent_name=f'pathbridger_{config.endpoint_distribution}',
    )
    run_dir = os.path.abspath(
        os.path.join(FLAGS.save_dir, 'pathbridger', FLAGS.run_group, exp_name)
    )
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    _write_run_config(run_dir, config)
    from utils.provenance import write_provenance
    write_provenance(run_dir, agent)

    wandb_run = None
    if FLAGS.use_wandb:
        wandb_run = setup_wandb(
            project='PathBridger',
            group=FLAGS.run_group,
            name=exp_name,
            config={'flags': get_flag_dict(), 'agent': config.to_dict()},
            directory=run_dir,
        )

    start_step = restore_step + 1
    steps = range(start_step, FLAGS.train_steps + 1)
    if FLAGS.use_tqdm:
        steps = tqdm.tqdm(steps, smoothing=0.1, dynamic_ncols=True)

    train_logger = CsvLogger(os.path.join(run_dir, 'train.csv'))
    cpu_evaluator = CpuEvaluator(run_dir, FLAGS.eval_episodes)
    start_time = time.time()
    interval_start = start_time
    prefetch_pool = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix='train-prefetch')
        if FLAGS.async_prefetch
        else None
    )

    def _submit_batch() -> Future:
        return prefetch_pool.submit(dataset.sample, _BATCH_SIZE)

    next_batch_future = _submit_batch() if prefetch_pool is not None else None

    try:
        for step in steps:
            if next_batch_future is None:
                batch = dataset.sample(_BATCH_SIZE)
            else:
                batch = next_batch_future.result()
                next_batch_future = None

            is_final = step == FLAGS.train_steps
            do_eval = (is_final and FLAGS.eval_final) or (FLAGS.eval_interval > 0 and step % FLAGS.eval_interval == 0)
            do_save = do_eval or is_final or (
                FLAGS.save_interval > 0
                and step % FLAGS.save_interval == 0
            )
            # Do not sample past a checkpoint before its NumPy RNG state is
            # serialized. This preserves exact batch-stream resume semantics.
            if prefetch_pool is not None and not do_save:
                next_batch_future = _submit_batch()

            if FLAGS.reuse_train_buffers:
                agent, update_info = update_training_state(agent, batch)
            else:
                agent, update_info = agent.update(batch)

            if step % FLAGS.log_interval == 0 or is_final:
                train_metrics = _host_metrics(update_info)
                train_metrics['time/interval_seconds'] = time.time() - interval_start
                train_metrics['time/total_seconds'] = time.time() - start_time
                interval_start = time.time()
                train_logger.log(train_metrics, step=step)
                if wandb_run is not None:
                    wandb_run.log({f'training/{key}': value for key, value in train_metrics.items()}, step=step)

            if do_save:
                checkpoint = save_agent(agent, checkpoint_dir, step)
                if do_eval:
                    cpu_evaluator.submit(checkpoint, step)
                if prefetch_pool is not None and not is_final:
                    next_batch_future = _submit_batch()
    finally:
        if prefetch_pool is not None:
            prefetch_pool.shutdown(wait=True)
        train_logger.close()
        cpu_evaluator.finish()
        if wandb_run is not None:
            wandb_run.finish()

    print(f'Run saved to {run_dir}')


def run():
    """Run the command-line training entry point."""

    app.run(main)


if __name__ == '__main__':
    run()
