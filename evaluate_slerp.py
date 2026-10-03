"""Sweep the deterministic SLERP latent bridge against ``direct_goal``.

The SLERP bridge replaces Module B's generated waypoint with a fixed point on
the geodesic between ``psi(s_t)`` and ``psi(g)``, recomputed from the actual
state at every environment step::

    z_way = SLERP(psi(s_t), psi(g), alpha)
    a     = pi(s_t, z_way)

``alpha=1`` returns ``psi(g)`` exactly, so it must reproduce ``direct_goal``.
This script therefore runs ``direct_goal`` and the whole alpha sweep **in one
process, on one paired episode manifest**, which makes that equality an exact
check rather than one blurred by cross-process XLA kernel selection.

If no ``alpha < 1`` beats ``alpha = 1``, then moving the actor's conditioning
vector along the geodesic does not help, and a learned bridge that produces
points near that geodesic cannot help either.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from absl import app, flags
from ml_collections import config_flags

from agents.latentbridger import LatentBridgerAgent, restore_latent_params
from envs.env_utils import make_env_and_datasets
from utils.flax_utils import resolve_checkpoint
from utils.latent_datasets import LatentBridgerDataset
from utils.latent_evaluation import (
    DEFAULT_TASK_IDS,
    episode_manifest,
    evaluate_latent,
    slerp,
)

FLAGS = flags.FLAGS
_DEFAULT_CONFIG = str(
    Path(__file__).resolve().parent / 'configs' / 'latent' / 'cube_single.py'
)

flags.DEFINE_string('checkpoint_dir', '', 'Checkpoint directory or exact .pkl file.')
flags.DEFINE_integer('checkpoint_step', 0, 'Checkpoint step; inferred from a .pkl path.')
flags.DEFINE_string('dataset_dir', '', 'Optional OGBench dataset directory.')
flags.DEFINE_string(
    'alphas',
    '0.1,0.2,0.4,0.6,0.8,1.0',
    'Comma-separated SLERP interpolation fractions.',
)
flags.DEFINE_integer('episodes', 20, 'Episodes per task.')
flags.DEFINE_integer('seed', 0, 'Evaluation seed; also seeds the manifest.')
flags.DEFINE_string(
    'manifest_path',
    '',
    'Optional paired episode manifest JSON; generated from --seed if absent.',
)
flags.DEFINE_string('output_path', '', 'Optional JSON result path.')

config_flags.DEFINE_config_file(
    'agent',
    _DEFAULT_CONFIG,
    'LatentBridger config; append :<variant> to select a variant.',
    lock_config=False,
)


def _conditioning_equality(agent, dataset, num_rows: int = 256) -> dict[str, float]:
    """Check numerically that ``SLERP(.., alpha=1)`` is ``psi(g)``."""

    batch = dataset.sample(int(num_rows))
    states = np.asarray(agent.goal_latents(batch['observations']), dtype=np.float32)
    goals = np.asarray(agent.goal_latents(batch['actor_goals']), dtype=np.float32)
    deviations = [
        float(np.max(np.abs(slerp(state, goal, 1.0) - goal)))
        for state, goal in zip(states, goals)
    ]
    return {
        'alpha1_max_abs_deviation_from_psi_goal': float(np.max(deviations)),
        'alpha1_rows_checked': float(len(deviations)),
    }


def main(_):
    if not FLAGS.checkpoint_dir:
        raise ValueError('checkpoint_dir is required.')
    alphas = [float(value) for value in FLAGS.alphas.split(',') if value.strip()]
    if not alphas:
        raise ValueError('--alphas must list at least one value.')
    if any(not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise ValueError(f'All alphas must lie in [0, 1]; got {alphas}.')

    _, checkpoint_step = resolve_checkpoint(FLAGS.checkpoint_dir, FLAGS.checkpoint_step)
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

    if FLAGS.manifest_path:
        with Path(FLAGS.manifest_path).open('r', encoding='utf-8') as file:
            manifest = json.load(file)['episodes']
    else:
        manifest = episode_manifest(DEFAULT_TASK_IDS, FLAGS.episodes, FLAGS.seed)

    # Everything below runs in this one process against this one manifest, so
    # every number is paired episode-for-episode.
    runs: dict[str, dict] = {}
    baseline = evaluate_latent(
        agent, env, mode='direct_goal', seed=FLAGS.seed, manifest=manifest
    )
    runs['direct_goal'] = baseline
    for alpha in alphas:
        runs[f'alpha_{alpha:g}'] = evaluate_latent(
            agent,
            env,
            mode='slerp_bridge',
            seed=FLAGS.seed,
            alpha=alpha,
            manifest=manifest,
        )

    baseline_outcomes = np.asarray(baseline['episodes'], dtype=np.int64)
    comparisons: dict[str, dict[str, float]] = {}
    for name, metrics in runs.items():
        if name == 'direct_goal':
            continue
        outcomes = np.asarray(metrics['episodes'], dtype=np.int64)
        # Paired McNemar-style counts: episodes only this run solved, and
        # episodes only the baseline solved.
        comparisons[name] = {
            'success_delta': float(
                metrics['overall_success'] - baseline['overall_success']
            ),
            'wins_vs_direct_goal': float(np.sum((outcomes == 1) & (baseline_outcomes == 0))),
            'losses_vs_direct_goal': float(np.sum((outcomes == 0) & (baseline_outcomes == 1))),
            'identical_episodes': float(np.mean(outcomes == baseline_outcomes)),
        }

    alpha_one = runs.get('alpha_1')
    sanity = _conditioning_equality(agent, dataset)
    if alpha_one is not None:
        sanity['alpha1_matches_direct_goal_exactly'] = float(
            alpha_one['episodes'] == baseline['episodes']
        )
        sanity['alpha1_success'] = float(alpha_one['overall_success'])
        sanity['direct_goal_success'] = float(baseline['overall_success'])

    improving = {
        name: values['success_delta']
        for name, values in comparisons.items()
        if name != 'alpha_1' and values['success_delta'] > 0.0
    }
    result = {
        'checkpoint_step': checkpoint_step,
        'env_name': str(config.env_name),
        'variant': str(config.variant),
        'seed': int(FLAGS.seed),
        'num_episodes': len(manifest),
        'manifest_path': str(FLAGS.manifest_path),
        'alphas': alphas,
        'runs': {name: {k: v for k, v in m.items()} for name, m in runs.items()},
        'paired_comparisons': comparisons,
        'sanity': sanity,
        'best_alpha': (
            max(improving, key=improving.get).removeprefix('alpha_')
            if improving
            else ''
        ),
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
    """Run the command-line SLERP sweep entry point."""

    app.run(main)


if __name__ == '__main__':
    run()
