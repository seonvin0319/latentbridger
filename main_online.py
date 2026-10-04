"""Train one online SGCRL variant against a single OGBench task goal.

The x axis of every online result is **environment interactions**, not
gradient updates.  The update-to-data ratio is held at SGCRL's value (one
batch-256 update per environment step, derived in
``docs/sgcrl_online_semantics.md``) and logged explicitly, so a variant can
never win by quietly taking more gradient steps.

The latent bridge is trained as a separate module update whose count is logged
on its own.  Activating the bridge therefore changes what the actor is
conditioned on, not how much reinforcement learning it receives.

Two runs can branch from one shared history::

    # 0 -> 100k with the latent actor, writing a branch snapshot
    python main_online.py --agent=configs/online/cube_single.py:online_sgcrl_latent \\
        --total_env_steps=100000 --snapshot_at=100000 --output_dir=exp/online/warmup/seed0

    # 100k -> 1M, continuing without the bridge
    python main_online.py --agent=configs/online/cube_single.py:online_sgcrl_latent \\
        --total_env_steps=1000000 --resume_snapshot=exp/online/warmup/seed0/snapshot_100000.pkl \\
        --output_dir=exp/online/latent/seed0

    # 100k -> 1M, same snapshot, bridge now driving behaviour
    python main_online.py --agent=configs/online/cube_single.py:online_sgcrl_latent_bridge \\
        --total_env_steps=1000000 --resume_snapshot=exp/online/warmup/seed0/snapshot_100000.pkl \\
        --output_dir=exp/online/bridge/seed0

At the branch point the critic parameters, actor parameters, replay buffer,
and RNG state are identical, so the curves after it differ because of the
bridge and nothing else.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import signal
import time
from pathlib import Path

import flax
import jax
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

from agents.online_sgcrl import VARIANT_SETTINGS, OnlineSGCRLAgent
from utils.log_utils import CsvLogger, get_flag_dict
from utils.online_evaluation import (
    collect_episode,
    evaluate_online,
    make_online_env,
    online_episode_manifest,
)
from utils.online_replay import EpisodicReplayBuffer, sparse_bridge_offsets

FLAGS = flags.FLAGS
_DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent / 'configs' / 'online' / 'cube_single.py'
)

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('output_dir', 'exp/online/debug', 'Run directory.')
flags.DEFINE_integer('total_env_steps', 1_000_000, 'Environment interaction budget.')
flags.DEFINE_string(
    'eval_points',
    '10000,50000,100000,200000,500000,1000000',
    'Environment-step counts at which to evaluate.',
)
flags.DEFINE_integer('eval_episodes', 50, 'Paired evaluation episodes per point.')
flags.DEFINE_integer('log_interval', 5_000, 'Environment steps between CSV rows.')
flags.DEFINE_string(
    'snapshot_at',
    '',
    'Comma-separated environment-step counts at which to write a branch snapshot.',
)
flags.DEFINE_string('resume_snapshot', '', 'Branch snapshot to continue from.')
flags.DEFINE_boolean('use_tqdm', False, 'Show a progress bar.')

config_flags.DEFINE_config_file(
    'agent',
    _DEFAULT_CONFIG,
    'Online SGCRL config; append :<variant> to select a variant.',
    lock_config=False,
)


def _parse_steps(text: str) -> tuple[int, ...]:
    return tuple(
        sorted({int(piece) for piece in text.split(',') if piece.strip()})
    )


def _host_metrics(info) -> dict[str, float]:
    return {
        str(key): float(np.asarray(jax.device_get(value)).reshape(()))
        for key, value in info.items()
    }


def _write_snapshot(
    path: Path, agent, replay, rng, env_step, update_count, episode_index
) -> None:
    """Everything needed to branch two behaviours from one shared history."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # The collection seed is derived from this counter, so a branch that
        # restarted it would re-collect the warmup's own episodes.
        'episode_index': int(episode_index),
        'critic': flax.serialization.to_state_dict(agent.critic),
        'actor': flax.serialization.to_state_dict(agent.actor),
        'flow': flax.serialization.to_state_dict(agent.flow),
        'agent_rng': np.asarray(jax.device_get(agent.rng)),
        'collect_rng': np.asarray(jax.device_get(rng)),
        'replay': replay.state_dict(),
        'env_step': int(env_step),
        'update_count': int(update_count),
        'numpy_random_state': np.random.get_state(),
        'python_random_state': random.getstate(),
    }
    temporary = path.with_suffix('.pkl.tmp')
    with temporary.open('wb') as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _load_snapshot(path: Path, agent, replay):
    with Path(path).open('rb') as file:
        payload = pickle.load(file)
    agent = agent.replace(
        critic=flax.serialization.from_state_dict(agent.critic, payload['critic']),
        actor=flax.serialization.from_state_dict(agent.actor, payload['actor']),
        flow=flax.serialization.from_state_dict(agent.flow, payload['flow']),
        rng=jax.numpy.asarray(payload['agent_rng']),
    )
    replay.load_state_dict(payload['replay'])
    np.random.set_state(payload['numpy_random_state'])
    random.setstate(payload['python_random_state'])
    return (
        agent,
        jax.numpy.asarray(payload['collect_rng']),
        int(payload['env_step']),
        int(payload['update_count']),
        int(payload['episode_index']),
    )


def main(_):
    config = FLAGS.agent
    variant = str(config.variant)
    if variant not in VARIANT_SETTINGS:
        raise ValueError(f'Unknown variant {variant!r}.')
    if FLAGS.total_env_steps < 1:
        raise ValueError('total_env_steps must be at least 1.')

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    env = make_online_env(str(config.env_name))
    observation_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])

    agent = OnlineSGCRLAgent.create(
        FLAGS.seed,
        np.zeros((2, observation_dim), dtype=np.float32),
        np.zeros((2, action_dim), dtype=np.float32),
        config,
    )
    replay = EpisodicReplayBuffer(
        observation_dim,
        action_dim,
        discount=float(config.discount),
        max_size=int(config.max_replay_size),
        seed=FLAGS.seed,
    )

    run_dir = Path(FLAGS.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = jax.random.PRNGKey(FLAGS.seed + 10_000)
    start_step = 0
    update_count = 0
    start_episode = 0
    if FLAGS.resume_snapshot:
        agent, rng, start_step, update_count, start_episode = _load_snapshot(
            Path(FLAGS.resume_snapshot), agent, replay
        )
        print(
            f'[main_online] branched from {FLAGS.resume_snapshot} at '
            f'{start_step} env steps with {len(replay)} replay transitions',
            flush=True,
        )

    with (run_dir / 'flags.json').open('w', encoding='utf-8') as file:
        json.dump(
            {'flags': get_flag_dict(), 'agent': config.to_dict()},
            file,
            indent=2,
            sort_keys=True,
            default=str,
        )
        file.write('\n')

    batch_size = int(config.batch_size)
    min_replay_size = int(config.min_replay_size)
    updates_per_env_step = float(config.updates_per_env_step)
    use_bridge = bool(VARIANT_SETTINGS[variant]['use_bridge'])
    bridge_updates_per_env_step = (
        float(config.bridge_updates_per_env_step) if use_bridge else 0.0
    )
    bridge_offsets = sparse_bridge_offsets(
        int(config.bridge_horizon), int(config.num_waypoints)
    )

    eval_points = _parse_steps(FLAGS.eval_points)
    snapshot_points = _parse_steps(FLAGS.snapshot_at) if FLAGS.snapshot_at else ()
    manifest = online_episode_manifest(
        int(config.task_id), int(FLAGS.eval_episodes), int(FLAGS.seed)
    )
    with (run_dir / 'eval_manifest.json').open('w', encoding='utf-8') as file:
        json.dump(
            {
                'task_id': int(config.task_id),
                'num_episodes': int(FLAGS.eval_episodes),
                'seed': int(FLAGS.seed),
                'episodes': manifest,
            },
            file,
            indent=2,
        )
        file.write('\n')

    train_logger = CsvLogger(str(run_dir / 'train.csv'), resume=start_step > 0)
    eval_logger = CsvLogger(str(run_dir / 'eval.csv'), resume=start_step > 0)

    interrupted: list[str] = []

    def _request_stop(signal_number, _frame):
        if interrupted:
            raise KeyboardInterrupt('Second interrupt; aborting.')
        interrupted.append(signal.Signals(signal_number).name)
        print(f'\n[main_online] {interrupted[0]} received; stopping cleanly.', flush=True)

    for signal_number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signal_number, _request_stop)

    env_step = start_step
    training_env_steps = 0
    bridge_update_count = 0
    episode_index = start_episode
    recent_success: list[float] = []
    recent_distance: list[float] = []
    pending_updates = 0.0
    pending_bridge_updates = 0.0
    started = time.time()
    last_info: dict[str, float] = {}
    progress = (
        tqdm.tqdm(total=FLAGS.total_env_steps, initial=start_step, dynamic_ncols=True)
        if FLAGS.use_tqdm
        else None
    )

    while env_step < FLAGS.total_env_steps and not interrupted:
        # -- collect one episode against the task's single fixed goal -----
        # Collection seeds are disjoint from the evaluation manifest's, so
        # training never visits the exact episodes it is scored on.
        collect_seed = 500_000_000 + int(FLAGS.seed) * 10_000_000 + episode_index
        entry = {
            'task_id': int(config.task_id),
            'env_seed': collect_seed,
            'action_space_seed': collect_seed + 5,
        }
        random_actions = not replay.ready(min_replay_size)
        observations, actions, episode_metrics, rng = collect_episode(
            agent, env, entry, rng, random_actions=random_actions
        )
        replay.add_episode(observations, actions)
        episode_index += 1
        env_step += len(actions)
        recent_success.append(episode_metrics['success'])
        recent_distance.append(episode_metrics['final_distance'])

        # -- train at the fixed update-to-data ratio ----------------------
        if replay.ready(min_replay_size):
            training_env_steps += len(actions)
            pending_updates += updates_per_env_step * len(actions)
            while pending_updates >= 1.0:
                agent, last_info = agent.update(replay.sample(batch_size))
                update_count += 1
                pending_updates -= 1.0
            if bridge_updates_per_env_step:
                pending_bridge_updates += bridge_updates_per_env_step * len(actions)
                while pending_bridge_updates >= 1.0:
                    agent, bridge_info = agent.update_bridge(
                        replay.sample_bridge(batch_size, bridge_offsets)
                    )
                    bridge_update_count += 1
                    pending_bridge_updates -= 1.0
                    last_info = {**last_info, **bridge_info}

        if progress is not None:
            progress.update(len(actions))

        # -- log ----------------------------------------------------------
        if last_info and env_step // FLAGS.log_interval != (
            env_step - len(actions)
        ) // FLAGS.log_interval:
            metrics = _host_metrics(last_info)
            metrics.update(
                {
                    'train/env_steps': float(env_step),
                    'train/updates': float(update_count),
                    'train/bridge_updates': float(bridge_update_count),
                    # Logged, not assumed: this is the number that must match
                    # across variants for the comparison to be fair.  The
                    # denominator counts only steps collected after the
                    # buffer became trainable, which is what the ratio means.
                    'train/updates_per_env_step': float(update_count)
                    / max(training_env_steps, 1),
                    'train/replay_transitions': float(len(replay)),
                    'train/replay_episodes': float(replay.num_episodes),
                    'train/behaviour_success_100': float(np.mean(recent_success[-100:])),
                    'train/behaviour_final_distance_100': float(
                        np.mean(recent_distance[-100:])
                    ),
                    'train/successful_trajectory_fraction': float(
                        np.mean(recent_success)
                    ),
                    'train/elapsed_minutes': (time.time() - started) / 60.0,
                }
            )
            train_logger.log(metrics, step=env_step)

        # -- evaluate and snapshot at the configured interaction counts ---
        crossed = [
            point
            for point in eval_points
            if env_step - len(actions) < point <= env_step
        ]
        for point in crossed:
            rng, eval_rng = jax.random.split(rng)
            results = evaluate_online(agent, env, manifest=manifest, rng=eval_rng)
            row = {f'evaluation/{key}': value for key, value in results.items()
                   if not isinstance(value, list)}
            row['train/env_steps'] = float(env_step)
            row['train/updates'] = float(update_count)
            row['train/bridge_updates'] = float(bridge_update_count)
            eval_logger.log(row, step=point)
            with (run_dir / f'eval_{point}.json').open('w', encoding='utf-8') as file:
                json.dump(
                    {
                        'variant': variant,
                        'seed': int(FLAGS.seed),
                        'env_step': int(env_step),
                        'eval_point': int(point),
                        'updates': int(update_count),
                        'bridge_updates': int(bridge_update_count),
                        **{f'evaluation/{k}': v for k, v in results.items()},
                    },
                    file,
                    indent=2,
                    sort_keys=True,
                )
                file.write('\n')
            print(
                f'[main_online] {point} env steps: success '
                f'{results["num_successes"]}/{results["num_episodes"]}',
                flush=True,
            )

        for point in snapshot_points:
            if env_step - len(actions) < point <= env_step:
                _write_snapshot(
                    run_dir / f'snapshot_{point}.pkl',
                    agent,
                    replay,
                    rng,
                    env_step,
                    update_count,
                    episode_index,
                )
                print(f'[main_online] snapshot at {env_step} env steps', flush=True)

    if progress is not None:
        progress.close()
    _write_snapshot(
        run_dir / f'final_{env_step}.pkl',
        agent,
        replay,
        rng,
        env_step,
        update_count,
        episode_index,
    )
    train_logger.close()
    eval_logger.close()
    print(
        f'[main_online] stopped at {env_step} env steps, {update_count} updates, '
        f'{bridge_update_count} bridge updates -> {run_dir}',
        flush=True,
    )


def run():
    """Run the command-line online training entry point."""

    app.run(main)


if __name__ == '__main__':
    run()
