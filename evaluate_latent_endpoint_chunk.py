"""Evaluate one unified latent endpoint chunk checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

from absl import app, flags
from ml_collections import config_flags

from agents.latent_endpoint_chunk import (
    LatentEndpointChunkAgent,
    restore_latent_endpoint_params,
)
from envs.env_utils import make_env_and_datasets
from utils.chunk_diagnostics import run_chunk_diagnostics
from utils.chunk_relabeling import LatentEndpointChunkDataset
from utils.latent_endpoint_chunk_evaluation import (
    INFERENCE_MODES,
    evaluate_latent_endpoint_chunk,
)

FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint', '', 'Checkpoint directory or exact file.')
flags.DEFINE_integer('checkpoint_step', 0, 'Checkpoint step.')
flags.DEFINE_string('manifest', '', 'Paired evaluation manifest JSON.')
flags.DEFINE_string('output', '', 'Evaluation JSON path.')
flags.DEFINE_string('diagnostics_output', '', 'Optional offline diagnostics JSON path.')
flags.DEFINE_boolean(
    'diagnostics_only', False, 'Skip environment rollouts and only write diagnostics.'
)
flags.DEFINE_integer('diagnostic_batches', 16, 'Offline diagnostic batches.')
flags.DEFINE_integer('diagnostic_batch_size', 256, 'Offline diagnostic batch size.')
flags.DEFINE_string('dataset_dir', '', 'Optional OGBench dataset directory.')
flags.DEFINE_integer('episodes', 50, 'Episodes per predefined task.')
flags.DEFINE_integer('seed', 0, 'Evaluation seed.')
flags.DEFINE_integer('execute_h', 0, 'Execution prefix; zero uses config.')
flags.DEFINE_enum('inference_mode', 'direct', list(INFERENCE_MODES), 'Inference mode.')
flags.DEFINE_integer('support_bank_size', 4096, 'Dataset chunks for distance diagnostics.')
config_flags.DEFINE_config_file(
    'agent',
    str(Path(__file__).parent / 'configs/latent_endpoint_chunk/cube_single.py'),
    'Latent endpoint chunk config.',
    lock_config=False,
)


def _write(path_string: str, payload) -> None:
    output = json.dumps(payload, indent=2, sort_keys=True) + '\n'
    if path_string:
        path = Path(path_string)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output, encoding='utf-8')
    print(output, end='')


def main(_):
    if not FLAGS.checkpoint:
        raise ValueError('--checkpoint is required.')
    env, train_data, _ = make_env_and_datasets(
        str(FLAGS.agent.env_name), dataset_dir=FLAGS.dataset_dir or None
    )
    dataset = LatentEndpointChunkDataset(train_data, FLAGS.agent)
    example = dataset.sample(2)
    stage = f'policy_{FLAGS.agent.policy_type}'
    agent = LatentEndpointChunkAgent.create(
        FLAGS.seed,
        example['observations'],
        example['action_chunks'],
        example['endpoint_states'],
        example['goals'],
        FLAGS.agent,
        stage=stage,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
    )
    agent = restore_latent_endpoint_params(
        agent, FLAGS.checkpoint, FLAGS.checkpoint_step
    )
    manifest = None
    if FLAGS.manifest:
        manifest = json.loads(Path(FLAGS.manifest).read_text(encoding='utf-8'))
        if len(manifest) != 250:
            raise ValueError(
                f'The paired full-evaluation manifest must contain 250 episodes, got {len(manifest)}.'
            )
    if not FLAGS.diagnostics_only:
        result = evaluate_latent_endpoint_chunk(
            agent,
            env,
            manifest=manifest,
            episodes_per_task=FLAGS.episodes,
            seed=FLAGS.seed,
            execute_h=FLAGS.execute_h or None,
            inference_mode=FLAGS.inference_mode,
            support_chunks=dataset.support_chunks(FLAGS.support_bank_size, FLAGS.seed),
        )
        _write(FLAGS.output, result)
    elif FLAGS.output:
        raise ValueError('--output cannot be used with --diagnostics_only.')
    if FLAGS.diagnostics_output:
        diagnostics = run_chunk_diagnostics(
            agent,
            dataset,
            batches=FLAGS.diagnostic_batches,
            batch_size=FLAGS.diagnostic_batch_size,
        )
        path = Path(FLAGS.diagnostics_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
    elif FLAGS.diagnostics_only:
        raise ValueError('--diagnostics_only requires --diagnostics_output.')


if __name__ == '__main__':
    app.run(main)
