"""Train one online SGCRL variant against a single OGBench task goal.

The x axis of every online result is **environment interactions**, not
gradient updates.  The update-to-data ratio is held at SGCRL's value (one
batch-256 update per environment step, derived in
``docs/sgcrl_online_semantics.md``) and logged explicitly, so a variant can
never win by quietly taking more gradient steps.  The bridges are trained by
separate optimizers whose step counts are logged on their own, so adding a
bridge changes what the actor is pointed at, not how much reinforcement
learning it receives.

The primary comparison is a paired branch from a shared warm start::

    # 0 -> 100k: SGCRL behaviour, both bridges trained as auxiliaries
    python main_online.py --agent=configs/online/cube_single.py:online_sgcrl \\
        --seed=0 --total_env_steps=100000 --train_bridges=both \\
        --snapshot_at=100000 --output_dir=exp/online/warmup/seed0

    # 100k -> 1M: three behaviours from the identical snapshot
    for V in online_sgcrl online_sgcrl_det_bridge online_sgcrl_rf_bridge; do
      python main_online.py --agent=configs/online/cube_single.py:$V --seed=0 \\
          --total_env_steps=1000000 \\
          --resume_snapshot=exp/online/warmup/seed0/snapshot_100000.pkl \\
          --output_dir=exp/online/$V/seed0
    done

At the branch point the critic parameters, actor parameters, both bridges,
the replay buffer, and the RNG state are identical; ``snapshot_hashes.json``
records digests so the three branches can be proven to have started equal.
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
from utils.online_replay import EpisodicReplayBuffer

FLAGS = flags.FLAGS
_DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent / 'configs' / 'online' / 'cube_single.py'
)

flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('output_dir', 'exp/online/debug', 'Run directory.')
flags.DEFINE_integer('total_env_steps', 1_000_000, 'Environment interaction budget.')
flags.DEFINE_string(
    'eval_points',
    '10000,50000,100000,200000,300000,500000,800000,1000000',
    'Environment-step counts at which to evaluate.',
)
flags.DEFINE_integer('eval_episodes', 100, 'Paired evaluation episodes per point.')
flags.DEFINE_integer('diagnostic_batch', 1024, 'Held-out rows per diagnostic.')
flags.DEFINE_integer('log_interval', 10_000, 'Environment steps between CSV rows.')
flags.DEFINE_enum(
    'train_bridges',
    'auto',
    ['auto', 'both', 'none'],
    'Which bridges to train. "auto" follows the variant; the shared warmup '
    'uses "both" so every branch starts from equally warm bridges.',
)
flags.DEFINE_string(
    'snapshot_at',
    '',
    'Comma-separated environment-step counts at which to write a snapshot.',
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
    return tuple(sorted({int(piece) for piece in text.split(',') if piece.strip()}))


def _host_metrics(info) -> dict[str, float]:
    return {
        str(key): float(np.asarray(jax.device_get(value)).reshape(()))
        for key, value in info.items()
    }


def _tree_digest(tree) -> str:
    """A stable digest of a parameter tree, for proving branch equality."""

    hasher = hashlib.sha256()
    for path, leaf in sorted(
        jax.tree_util.tree_flatten_with_path(tree)[0],
        key=lambda item: jax.tree_util.keystr(item[0]),
    ):
        hasher.update(jax.tree_util.keystr(path).encode('utf-8'))
        hasher.update(np.ascontiguousarray(jax.device_get(leaf)).tobytes())
    return hasher.hexdigest()


def _snapshot_digests(agent, replay) -> dict[str, str]:
    replay_state = replay.state_dict()
    replay_hasher = hashlib.sha256()
    for key in ('observations', 'actions', 'episode_lengths', 'episode_ids'):
        replay_hasher.update(np.ascontiguousarray(replay_state[key]).tobytes())
    return {
        'critic': _tree_digest(agent.critic.params),
        'actor': _tree_digest(agent.actor.params),
        'det_bridge': _tree_digest(agent.det_bridge.params),
        'rf_bridge': _tree_digest(agent.rf_bridge.params),
        'replay': replay_hasher.hexdigest(),
    }


def _write_snapshot(path: Path, agent, replay, rng, counters) -> dict[str, str]:
    """Everything needed to branch three behaviours from one shared history."""

    path.parent.mkdir(parents=True, exist_ok=True)
    digests = _snapshot_digests(agent, replay)
    payload = {
        'critic': flax.serialization.to_state_dict(agent.critic),
        'actor': flax.serialization.to_state_dict(agent.actor),
        'det_bridge': flax.serialization.to_state_dict(agent.det_bridge),
        'rf_bridge': flax.serialization.to_state_dict(agent.rf_bridge),
        'agent_rng': np.asarray(jax.device_get(agent.rng)),
        'bridge_rng': np.asarray(jax.device_get(agent.bridge_rng)),
        'collect_rng': np.asarray(jax.device_get(rng)),
        'replay': replay.state_dict(),
        'counters': dict(counters),
        'digests': digests,
        'numpy_random_state': np.random.get_state(),
        'python_random_state': random.getstate(),
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
    agent = agent.replace(
        critic=flax.serialization.from_state_dict(agent.critic, payload['critic']),
        actor=flax.serialization.from_state_dict(agent.actor, payload['actor']),
        det_bridge=flax.serialization.from_state_dict(
            agent.det_bridge, payload['det_bridge']
        ),
        rf_bridge=flax.serialization.from_state_dict(
            agent.rf_bridge, payload['rf_bridge']
        ),
        rng=jax.numpy.asarray(payload['agent_rng']),
        bridge_rng=jax.numpy.asarray(payload['bridge_rng']),
    )
    replay.load_state_dict(payload['replay'])
    np.random.set_state(payload['numpy_random_state'])
    random.setstate(payload['python_random_state'])

    # Re-derive the digests from the restored objects rather than trusting the
    # recorded ones; this is what proves the branches started identical.
    digests = _snapshot_digests(agent, replay)
    expected = payload.get('digests', {})
    mismatched = [key for key, value in expected.items() if digests[key] != value]
    if mismatched:
        raise RuntimeError(
            f'Snapshot {path} did not restore faithfully; {mismatched} differ '
            'from the digests recorded when it was written.'
        )
    return agent, jax.numpy.asarray(payload['collect_rng']), payload['counters'], digests


def main(_):
    config = FLAGS.agent
    variant = str(config.variant)
    if variant not in VARIANT_SETTINGS:
        raise ValueError(f'Unknown variant {variant!r}.')
    if FLAGS.total_env_steps < 1:
        raise ValueError('total_env_steps must be at least 1.')
    bridge_mode = str(VARIANT_SETTINGS[variant]['bridge_mode'])

    if FLAGS.train_bridges == 'both':
        train_det, train_rf = True, True
    elif FLAGS.train_bridges == 'none':
        train_det, train_rf = False, False
    else:
        train_det = bridge_mode == 'deterministic'
        train_rf = bridge_mode == 'rectified_flow'

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
        holdout_every=int(config.holdout_every),
    )

    run_dir = Path(FLAGS.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = jax.random.PRNGKey(FLAGS.seed + 10_000)
    counters = {
        'env_step': 0,
        'update_count': 0,
        'det_bridge_updates': 0,
        'rf_bridge_updates': 0,
        'episode_index': 0,
    }
    if FLAGS.resume_snapshot:
        agent, rng, restored, digests = _load_snapshot(
            Path(FLAGS.resume_snapshot), agent, replay
        )
        counters.update(restored)
        print(
            f'[main_online] branched from {FLAGS.resume_snapshot} at '
            f'{counters["env_step"]} env steps, {len(replay)} replay '
            f'transitions, critic {digests["critic"][:12]} '
            f'actor {digests["actor"][:12]} replay {digests["replay"][:12]}',
            flush=True,
        )

    with (run_dir / 'flags.json').open('w', encoding='utf-8') as file:
        json.dump(
            {
                'flags': get_flag_dict(),
                'agent': config.to_dict(),
                'bridge_mode': bridge_mode,
                'train_det_bridge': train_det,
                'train_rf_bridge': train_rf,
            },
            file,
            indent=2,
            sort_keys=True,
            default=str,
        )
        file.write('\n')

    batch_size = int(config.batch_size)
    min_replay_size = int(config.min_replay_size)
    updates_per_env_step = float(config.updates_per_env_step)
    bridge_rate = float(config.bridge_updates_per_env_step)
    alpha = float(config.bridge_alpha)

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

    resumed = bool(FLAGS.resume_snapshot)
    train_logger = CsvLogger(str(run_dir / 'train.csv'), resume=resumed)
    eval_logger = CsvLogger(str(run_dir / 'eval.csv'), resume=resumed)

    interrupted: list[str] = []

    def _request_stop(signal_number, _frame):
        if interrupted:
            raise KeyboardInterrupt('Second interrupt; aborting.')
        interrupted.append(signal.Signals(signal_number).name)
        print(
            f'\n[main_online] {interrupted[0]} received; stopping cleanly.', flush=True
        )

    for signal_number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signal_number, _request_stop)

    training_env_steps = 0
    recent_success: list[float] = []
    recent_distance: list[float] = []
    pending_updates = 0.0
    pending_bridge_updates = 0.0
    started = time.time()
    last_info: dict[str, float] = {}
    progress = (
        tqdm.tqdm(
            total=FLAGS.total_env_steps,
            initial=counters['env_step'],
            dynamic_ncols=True,
        )
        if FLAGS.use_tqdm
        else None
    )

    def run_evaluation(point: int) -> None:
        nonlocal rng
        rng, eval_rng, diag_rng = jax.random.split(rng, 3)
        results = evaluate_online(
            agent, env, manifest=manifest, rng=eval_rng, bridge_mode=bridge_mode
        )
        row = {
            f'evaluation/{key}': value
            for key, value in results.items()
            if not isinstance(value, list)
        }
        diagnostics: dict[str, float] = {}
        try:
            holdout = replay.sample_bridge(
                int(FLAGS.diagnostic_batch), alpha=alpha, holdout=True
            )
        except RuntimeError:
            holdout = None
        if holdout is not None:
            diagnostics = _host_metrics(agent.bridge_diagnostics(holdout, diag_rng))
            row.update(diagnostics)
        row.update(
            {
                'train/env_steps': float(counters['env_step']),
                'train/updates': float(counters['update_count']),
                'train/det_bridge_updates': float(counters['det_bridge_updates']),
                'train/rf_bridge_updates': float(counters['rf_bridge_updates']),
            }
        )
        eval_logger.log(row, step=point)
        with (run_dir / f'eval_{point}.json').open('w', encoding='utf-8') as file:
            json.dump(
                {
                    'variant': variant,
                    'bridge_mode': bridge_mode,
                    'seed': int(FLAGS.seed),
                    'eval_point': int(point),
                    **{f'counters/{k}': int(v) for k, v in counters.items()},
                    **{f'evaluation/{k}': v for k, v in results.items()},
                    **diagnostics,
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

    # A branch resumes at a point the warmup already evaluated, so only the
    # points strictly beyond the restored step remain.
    if counters['env_step'] in eval_points:
        eval_points = tuple(p for p in eval_points if p > counters['env_step'])

    while counters['env_step'] < FLAGS.total_env_steps and not interrupted:
        # -- collect one episode against the task's single fixed goal -----
        # Collection seeds are disjoint from the evaluation manifest's, so
        # training never visits the exact episodes it is scored on.
        collect_seed = (
            500_000_000 + int(FLAGS.seed) * 10_000_000 + counters['episode_index']
        )
        entry = {
            'task_id': int(config.task_id),
            'env_seed': collect_seed,
            'action_space_seed': collect_seed + 5,
        }
        random_actions = not replay.ready(min_replay_size)
        observations, actions, episode_metrics, rng = collect_episode(
            agent,
            env,
            entry,
            rng,
            random_actions=random_actions,
            bridge_mode=bridge_mode,
        )
        replay.add_episode(observations, actions)
        counters['episode_index'] += 1
        previous_step = counters['env_step']
        counters['env_step'] += len(actions)
        recent_success.append(episode_metrics['success'])
        recent_distance.append(episode_metrics['final_distance'])

        # -- train at the fixed update-to-data ratio ----------------------
        if replay.ready(min_replay_size):
            training_env_steps += len(actions)
            pending_updates += updates_per_env_step * len(actions)
            while pending_updates >= 1.0:
                agent, last_info = agent.update(replay.sample(batch_size))
                counters['update_count'] += 1
                pending_updates -= 1.0

            if train_det or train_rf:
                pending_bridge_updates += bridge_rate * len(actions)
                while pending_bridge_updates >= 1.0:
                    # Each bridge gets its own batch, so the deterministic and
                    # flow objectives see the same supervision distribution
                    # but never share a draw that could couple them.
                    if train_det:
                        agent, det_info = agent.update_det_bridge(
                            replay.sample_bridge(batch_size, alpha=alpha)
                        )
                        counters['det_bridge_updates'] += 1
                        last_info = {**last_info, **det_info}
                    if train_rf:
                        agent, rf_info = agent.update_rf_bridge(
                            replay.sample_bridge(batch_size, alpha=alpha)
                        )
                        counters['rf_bridge_updates'] += 1
                        last_info = {**last_info, **rf_info}
                    pending_bridge_updates -= 1.0

        if progress is not None:
            progress.update(len(actions))

        # -- log ----------------------------------------------------------
        if last_info and counters['env_step'] // FLAGS.log_interval != (
            previous_step // FLAGS.log_interval
        ):
            metrics = _host_metrics(last_info)
            metrics.update(
                {
                    'train/env_steps': float(counters['env_step']),
                    'train/updates': float(counters['update_count']),
                    'train/det_bridge_updates': float(counters['det_bridge_updates']),
                    'train/rf_bridge_updates': float(counters['rf_bridge_updates']),
                    # Logged, not assumed: this is the number that must match
                    # across variants for the comparison to be fair.
                    'train/updates_per_env_step': float(counters['update_count'])
                    / max(training_env_steps, 1),
                    'train/replay_transitions': float(len(replay)),
                    'train/replay_episodes': float(replay.num_episodes),
                    'train/behaviour_success_100': float(
                        np.mean(recent_success[-100:])
                    ),
                    'train/behaviour_final_distance_100': float(
                        np.mean(recent_distance[-100:])
                    ),
                    'train/successful_trajectory_fraction': float(
                        np.mean(recent_success)
                    ),
                    'train/elapsed_minutes': (time.time() - started) / 60.0,
                }
            )
            train_logger.log(metrics, step=counters['env_step'])

        # -- evaluate and snapshot at the configured interaction counts ---
        for point in eval_points:
            if previous_step < point <= counters['env_step']:
                run_evaluation(point)
        for point in snapshot_points:
            if previous_step < point <= counters['env_step']:
                digests = _write_snapshot(
                    run_dir / f'snapshot_{point}.pkl', agent, replay, rng, counters
                )
                print(
                    f'[main_online] snapshot at {counters["env_step"]} env steps: '
                    f'critic {digests["critic"][:12]} actor {digests["actor"][:12]} '
                    f'replay {digests["replay"][:12]}',
                    flush=True,
                )

    if progress is not None:
        progress.close()
    _write_snapshot(
        run_dir / f'final_{counters["env_step"]}.pkl', agent, replay, rng, counters
    )
    train_logger.close()
    eval_logger.close()
    print(
        f'[main_online] stopped at {counters["env_step"]} env steps, '
        f'{counters["update_count"]} RL updates, '
        f'{counters["det_bridge_updates"]} det-bridge, '
        f'{counters["rf_bridge_updates"]} rf-bridge -> {run_dir}',
        flush=True,
    )


def run():
    """Run the command-line online training entry point."""

    app.run(main)


if __name__ == '__main__':
    run()
