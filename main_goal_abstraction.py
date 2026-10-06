"""Train one goal-abstraction SGCRL variant from scratch.

The critic, replay, and update ratio match ``main_online.py``.  The actor's
goal slot is the abstraction defined by the variant.  There is no waypoint
bridge and no reward.  An optional ``actor_pretrain_updates`` flag runs actor
steps on a restored replay before interaction; the primary experiment leaves
it at zero and trains each variant independently from step 0.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import signal
import time
from pathlib import Path

import jax
import numpy as np
import tqdm
from absl import app, flags
from ml_collections import config_flags

from agents.goal_abstraction import (
    VARIANTS,
    GoalAbstractionAgent,
    export_agent,
    import_agent,
    linear_probe_oracle,
)
from utils.log_utils import CsvLogger, get_flag_dict
from utils.online_evaluation import (
    collect_episode,
    evaluate_online,
    make_online_env,
    online_episode_manifest,
)
from utils.online_replay import EpisodicReplayBuffer

FLAGS = flags.FLAGS
_DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent / 'configs' / 'online' / 'goal_abstraction.py'
)

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('output_dir', 'exp/goal_abstraction_task1/debug', 'Run directory.')
flags.DEFINE_integer('total_env_steps', 8_000_000, 'Environment interaction budget.')
flags.DEFINE_string(
    'eval_points',
    '1000000,2000000,3000000,4000000,5000000,6000000,7000000,8000000',
    'Environment-step counts at which to evaluate.',
)
flags.DEFINE_integer('eval_episodes', 100, 'Paired evaluation episodes per point.')
flags.DEFINE_integer('diagnostic_batch', 1024, 'Rows used for representation diagnostics.')
flags.DEFINE_integer('log_interval', 10_000, 'Environment steps between CSV rows.')
flags.DEFINE_string(
    'snapshot_at',
    '',
    'Comma-separated environment-step counts at which to write a snapshot.',
)
flags.DEFINE_string('resume_snapshot', '', 'Snapshot to continue from.')
flags.DEFINE_integer(
    'actor_pretrain_updates',
    0,
    'Actor-only updates on the current replay before interaction.  The primary '
    'runs leave this at 0.',
)
flags.DEFINE_boolean('use_tqdm', False, 'Show a progress bar.')

config_flags.DEFINE_config_file(
    'agent',
    _DEFAULT_CONFIG,
    'Goal-abstraction config; append :<variant> to select a variant.',
    lock_config=False,
)

# Cube position inside the full observation.  Read only by the held-out probe.
_ORACLE_SLICE = (19, 22)


def _parse_steps(text: str) -> tuple[int, ...]:
    return tuple(sorted({int(piece) for piece in text.split(',') if piece.strip()}))


def _host_metrics(info) -> dict[str, float]:
    scalars = {}
    for key, value in info.items():
        array = np.asarray(jax.device_get(value))
        if array.shape == ():
            scalars[str(key)] = float(array)
    return scalars


def _tree_digest(tree) -> str:
    hasher = hashlib.sha256()
    for path, leaf in sorted(
        jax.tree_util.tree_flatten_with_path(tree)[0],
        key=lambda item: jax.tree_util.keystr(item[0]),
    ):
        hasher.update(jax.tree_util.keystr(path).encode('utf-8'))
        hasher.update(np.ascontiguousarray(jax.device_get(leaf)).tobytes())
    return hasher.hexdigest()


def _write_snapshot(path: Path, agent, replay, rng, counters) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    replay_state = replay.state_dict()
    replay_hasher = hashlib.sha256()
    for key in ('observations', 'actions', 'episode_lengths', 'episode_ids'):
        replay_hasher.update(np.ascontiguousarray(replay_state[key]).tobytes())
    digests = {
        'critic': _tree_digest(agent.critic.params),
        'policy': _tree_digest(agent.policy.params),
        'replay': replay_hasher.hexdigest(),
    }
    payload = {
        'agent': export_agent(agent),
        'collect_rng': np.asarray(jax.device_get(rng)),
        'replay': replay_state,
        'counters': dict(counters),
        'digests': digests,
        'numpy_random_state': np.random.get_state(),
        'python_random_state': random.getstate(),
        'variant': str(agent.config['variant']),
    }
    temporary = path.with_suffix('.pkl.tmp')
    with temporary.open('wb') as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    with path.with_suffix('.hashes.json').open('w', encoding='utf-8') as file:
        json.dump({'counters': dict(counters), 'digests': digests}, file, indent=2)
        file.write('\n')
    return digests


def _load_snapshot(path: Path, agent, replay):
    with Path(path).open('rb') as file:
        payload = pickle.load(file)
    if payload.get('variant') != str(agent.config['variant']):
        raise RuntimeError(
            f'Snapshot variant {payload.get("variant")!r} does not match '
            f'{agent.config["variant"]!r}.'
        )
    agent = import_agent(agent, payload['agent'])
    replay.load_state_dict(payload['replay'])
    np.random.set_state(payload['numpy_random_state'])
    random.setstate(payload['python_random_state'])
    return agent, jax.numpy.asarray(payload['collect_rng']), payload['counters'], payload['digests']


def _representation_report(agent, replay, batch_size: int) -> dict:
    """Diagnostics that must not enter the optimizer, including the oracle probe."""

    try:
        batch = replay.sample(batch_size)
    except RuntimeError:
        return {}
    info = _host_metrics(agent.abstraction_diagnostics(batch))
    _, critic_info = agent.critic_loss(batch, agent.critic.params)
    info.update(_host_metrics(critic_info))
    profile = agent.mask_profile(batch['observations'], batch['goals'])
    if profile['mask_mean'].size:
        info['mask_profile'] = {
            'mean': profile['mask_mean'].tolist(),
            'std': profile['mask_std'].tolist(),
        }
    observations = np.asarray(batch['observations'])
    goals = np.asarray(batch['goals'])
    start, end = _ORACLE_SLICE
    if observations.shape[-1] >= end and goals.shape[-1] >= end:
        features = np.asarray(
            jax.device_get(
                agent.goal_features(batch['observations'], batch['goals'])['goal_input']
            )
        )
        probe = linear_probe_oracle(features, goals[:, start:end])
        info['diagnostics/oracle_probe_mse'] = probe['oracle_probe_mse']
        info['diagnostics/oracle_probe_r2'] = probe['oracle_probe_r2']
    return info


def main(_):
    config = FLAGS.agent
    variant = str(config.variant)
    if variant not in VARIANTS:
        raise ValueError(f'Unknown variant {variant!r}.')
    if FLAGS.total_env_steps < 1:
        raise ValueError('total_env_steps must be at least 1.')
    if int(config.oracle_goal_slice[0]) >= 0 if 'oracle_goal_slice' in config else False:
        raise ValueError('Learned abstraction runs do not take an oracle goal slice.')

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    env = make_online_env(str(config.env_name), oracle_goals=False)
    observation_dim = int(env.observation_space.shape[0])
    action_dim = int(env.action_space.shape[0])
    agent = GoalAbstractionAgent.create(
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
        holdout_every=int(config.holdout_every),
    )
    run_dir = Path(FLAGS.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = jax.random.PRNGKey(FLAGS.seed + 10_000)
    counters = {'env_step': 0, 'update_count': 0, 'episode_index': 0, 'actor_pretrain_updates': 0}
    if FLAGS.resume_snapshot:
        agent, rng, restored, digests = _load_snapshot(Path(FLAGS.resume_snapshot), agent, replay)
        counters.update(restored)
        print(
            f'[goal_abstraction] resumed {FLAGS.resume_snapshot} at '
            f'{counters["env_step"]} env steps',
            flush=True,
        )

    with (run_dir / 'flags.json').open('w', encoding='utf-8') as file:
        json.dump(
            {
                'flags': get_flag_dict(),
                'agent': config.to_dict(),
                'variant': variant,
                'abstraction': str(config.abstraction),
            },
            file,
            indent=2,
            sort_keys=True,
            default=str,
        )
        file.write('\n')

    manifest = online_episode_manifest(int(config.task_id), int(FLAGS.eval_episodes), int(FLAGS.seed))
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

    if FLAGS.actor_pretrain_updates and replay.ready(int(config.min_replay_size)):
        for _ in range(int(FLAGS.actor_pretrain_updates)):
            agent, _ = agent.update_actor_only(replay.sample(int(config.batch_size)))
            counters['actor_pretrain_updates'] += 1

    resumed = bool(FLAGS.resume_snapshot)
    train_logger = CsvLogger(str(run_dir / 'train.csv'), resume=resumed)
    eval_logger = CsvLogger(str(run_dir / 'eval.csv'), resume=resumed)
    interrupted: list[str] = []

    def _request_stop(signal_number, _frame):
        if interrupted:
            raise KeyboardInterrupt('Second interrupt; aborting.')
        interrupted.append(signal.Signals(signal_number).name)
        print(f'\n[goal_abstraction] {interrupted[0]} received; stopping cleanly.', flush=True)

    for signal_number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signal_number, _request_stop)

    batch_size = int(config.batch_size)
    min_replay_size = int(config.min_replay_size)
    updates_per_env_step = float(config.updates_per_env_step)
    eval_points = _parse_steps(FLAGS.eval_points)
    snapshot_points = _parse_steps(FLAGS.snapshot_at) if FLAGS.snapshot_at else ()
    training_env_steps = int(counters['update_count'] / updates_per_env_step)
    recent_success: list[float] = []
    pending_updates = 0.0
    started = time.time()
    last_info: dict = {}
    progress = (
        tqdm.tqdm(total=FLAGS.total_env_steps, initial=counters['env_step'], dynamic_ncols=True)
        if FLAGS.use_tqdm
        else None
    )

    def run_evaluation(point: int) -> None:
        nonlocal rng
        rng, eval_rng = jax.random.split(rng)
        results = evaluate_online(agent, env, manifest=manifest, rng=eval_rng, bridge_mode='none')
        diagnostics = _representation_report(agent, replay, int(FLAGS.diagnostic_batch))
        profile = diagnostics.pop('mask_profile', None)
        row = {f'evaluation/{key}': value for key, value in results.items() if not isinstance(value, list)}
        row.update({key: value for key, value in diagnostics.items() if isinstance(value, float)})
        row['train/env_steps'] = float(counters['env_step'])
        row['train/updates'] = float(counters['update_count'])
        eval_logger.log(row, step=point)
        payload = {
            'variant': variant,
            'seed': int(FLAGS.seed),
            'eval_point': int(point),
            **{f'counters/{k}': int(v) for k, v in counters.items()},
            **{f'evaluation/{k}': v for k, v in results.items()},
            **diagnostics,
        }
        if profile is not None:
            payload['mask_profile'] = profile
        with (run_dir / f'eval_{point}.json').open('w', encoding='utf-8') as file:
            json.dump(payload, file, indent=2, sort_keys=True)
            file.write('\n')
        print(
            f'[goal_abstraction] {point} env steps: success '
            f'{results["num_successes"]}/{results["num_episodes"]}',
            flush=True,
        )

    if counters['env_step'] in eval_points:
        eval_points = tuple(point for point in eval_points if point > counters['env_step'])

    while counters['env_step'] < FLAGS.total_env_steps and not interrupted:
        collect_seed = 500_000_000 + int(FLAGS.seed) * 10_000_000 + counters['episode_index']
        entry = {
            'task_id': int(config.task_id),
            'env_seed': collect_seed,
            'action_space_seed': collect_seed + 5,
        }
        observations, actions, episode_metrics, rng = collect_episode(
            agent,
            env,
            entry,
            rng,
            random_actions=not replay.ready(min_replay_size),
            bridge_mode='none',
        )
        replay.add_episode(observations, actions)
        counters['episode_index'] += 1
        previous_step = counters['env_step']
        counters['env_step'] += len(actions)
        recent_success.append(episode_metrics['success'])

        if replay.ready(min_replay_size):
            training_env_steps += len(actions)
            pending_updates += updates_per_env_step * len(actions)
            while pending_updates >= 1.0:
                agent, last_info = agent.update(replay.sample(batch_size))
                counters['update_count'] += 1
                pending_updates -= 1.0

        if progress is not None:
            progress.update(len(actions))

        if last_info and counters['env_step'] // FLAGS.log_interval != previous_step // FLAGS.log_interval:
            metrics = _host_metrics(last_info)
            metrics.update(
                {
                    'train/env_steps': float(counters['env_step']),
                    'train/updates': float(counters['update_count']),
                    'train/updates_per_env_step': float(counters['update_count']) / max(training_env_steps, 1),
                    'train/replay_transitions': float(len(replay)),
                    'train/behaviour_success_100': float(np.mean(recent_success[-100:])),
                    'train/elapsed_minutes': (time.time() - started) / 60.0,
                }
            )
            train_logger.log(metrics, step=counters['env_step'])

        for point in eval_points:
            if previous_step < point <= counters['env_step']:
                run_evaluation(point)
        for point in snapshot_points:
            if previous_step < point <= counters['env_step']:
                _write_snapshot(run_dir / f'snapshot_{point}.pkl', agent, replay, rng, counters)

    if progress is not None:
        progress.close()
    _write_snapshot(run_dir / f'final_{counters["env_step"]}.pkl', agent, replay, rng, counters)
    train_logger.close()
    eval_logger.close()
    print(
        f'[goal_abstraction] stopped at {counters["env_step"]} env steps, '
        f'{counters["update_count"]} RL updates -> {run_dir}',
        flush=True,
    )


if __name__ == '__main__':
    app.run(main)
