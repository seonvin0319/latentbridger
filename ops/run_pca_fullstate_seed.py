"""Thin wrapper around the verified full-state PCA/PCA trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))

from agents.fullstate_role_pathbridger import (  # noqa: E402
    FrozenEexecEncoder,
    FullStateRolePathBridgerAgent,
)
from configs.pbf.puzzle_3x3 import get_config  # noqa: E402
from eexec_lib import EPISODE_LENGTH, normalize  # noqa: E402
from evaluate_eexec_fullstate_fixed_resets import make_env  # noqa: E402
from run_eexec_fullstate_cross import train_arm  # noqa: E402
from utils.datasets import Dataset, PathBridgerDataset  # noqa: E402

SMOKE_SEED = 909_090


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def load_handoff(directory: Path) -> dict:
    handoff = json.loads((directory / 'HANDOFF.json').read_text())
    contract = json.loads((directory / 'config/execution_contract.json').read_text())
    mismatches = []
    for item in handoff['files']:
        path = directory / item['path']
        digest = sha256_file(path)
        if digest != item['sha256']:
            mismatches.append(item['path'])
    if mismatches:
        raise RuntimeError(f'handoff hash mismatch: {mismatches}')
    if (contract['source'], contract['goal'], contract['latent_dim'], contract['candidate_count']) != ('pca', 'pca', 16, 32):
        raise RuntimeError(f'unexpected execution contract {contract}')
    return handoff


def load_arrays(directory: Path):
    mean = np.load(directory / 'norm/obs_mean.npy')
    std = np.load(directory / 'norm/obs_std.npy')
    components = np.load(directory / 'pca/pca_components_16.npy')
    matrix = np.load(directory / 'pca/pca_matrix.npy')
    train = np.load(directory / 'splits/train_episodes.npy')
    with np.load(directory / 'data/puzzle-3x3-play-v0.npz') as data:
        observations = np.asarray(data['observations'], dtype=np.float32)
        actions = np.asarray(data['actions'], dtype=np.float32)
        terminals = np.asarray(data['terminals'])
    return mean, std, components, matrix, train, observations, actions, terminals


def train_arrays(observations, actions, terminals, train, mean, std):
    pieces_obs, pieces_act, pieces_term = [], [], []
    for episode in train:
        start = int(episode) * EPISODE_LENGTH
        stop = start + EPISODE_LENGTH
        pieces_obs.append(normalize(observations[start:stop], mean, std))
        pieces_act.append(actions[start:stop])
        pieces_term.append(terminals[start:stop])
    return (
        np.concatenate(pieces_obs).astype(np.float32),
        np.concatenate(pieces_act).astype(np.float32),
        np.concatenate(pieces_term),
    )


def unused_encoder(mean: np.ndarray, init_seed: int):
    """A fresh encoder pytree. PCA/PCA never applies it."""

    encoder = FrozenEexecEncoder()
    return encoder.init(
        jax.random.PRNGKey(int(init_seed)),
        jnp.zeros((2, int(mean.shape[0])), dtype=jnp.float32),
    )['params']


def same_tree(left, right) -> bool:
    leaves_left = jax.tree_util.tree_leaves(left)
    leaves_right = jax.tree_util.tree_leaves(right)
    return all(
        np.array_equal(np.asarray(first), np.asarray(second))
        for first, second in zip(leaves_left, leaves_right, strict=True)
    )


def check_projection(observations, mean, std, components, matrix) -> None:
    row = np.asarray(observations[0], dtype=np.float64)
    normalized = (row - mean) / std
    through_matrix = normalized @ matrix
    direct = (row - mean) @ components
    if not np.allclose(through_matrix, direct, atol=1e-5):
        raise RuntimeError('reloaded PCA projection does not match the centered-component rule')


def check_modules(mean, std, matrix, observations, actions, terminals, train) -> None:
    obs, act, term = train_arrays(observations, actions, terminals, train[:2], mean, std)
    dataset = PathBridgerDataset(
        Dataset.create(observations=obs, actions=act, terminals=term),
        get_config(),
    )
    np.random.seed(SMOKE_SEED)
    example = dataset.sample(4)
    config = get_config().to_dict()
    if int(config['eval_num_candidates']) != 32:
        raise RuntimeError('candidate_count is not 32')
    encoder = unused_encoder(mean, SMOKE_SEED)
    agent = FullStateRolePathBridgerAgent.create(
        SMOKE_SEED,
        example['observations'],
        example['actions'],
        config,
        source_kind='pca',
        goal_kind='pca',
        eexec_encoder_params=encoder,
        pca_matrix=jnp.asarray(matrix),
    )
    params = agent.network.params
    shapes = {
        'bridge_first': tuple(params['modules_bridge']['MLP_0']['Dense_0']['kernel'].shape),
        'bridge_output': tuple(params['modules_bridge']['MLP_0']['Dense_3']['kernel'].shape),
        'idm_first': tuple(params['modules_idm']['MLP_0']['Dense_0']['kernel'].shape),
        'endpoint_output': tuple(params['modules_endpoint']['MLP_0']['Dense_3']['kernel'].shape),
    }
    expected = {
        'bridge_first': (111, 512),
        'bridge_output': (512, 55),
        'idm_first': (110, 512),
        'endpoint_output': (512, 55),
    }
    if shapes != expected:
        raise RuntimeError(f'full-state tensor path mismatch {shapes}')
    probe = jnp.asarray(example['observations'][:1])
    moved = probe.at[:, 54].add(1.0)
    idm = agent.network.select('idm')
    if float(np.max(np.abs(np.asarray(idm(probe, probe) - idm(probe, moved))))) <= 1e-7:
        raise RuntimeError('IDM ignored full-state coordinate 54')
    prefix = np.asarray(agent._construct_bridge_prefix(probe, moved))
    if prefix.shape[-1] != 55 or not np.any(np.abs(prefix[:, 1:, 54] - prefix[:, :1, 54]) > 1e-7):
        raise RuntimeError('bridge is not using the full observation')
    pca_before = np.asarray(agent.pca_matrix).copy()
    encoder_before = agent.eexec_encoder_params
    updated, _info = agent.update(example)
    if not np.array_equal(pca_before, np.asarray(updated.pca_matrix)):
        raise RuntimeError('PCA matrix changed during a downstream update')
    if not same_tree(encoder_before, updated.eexec_encoder_params):
        raise RuntimeError('unused encoder parameters changed during a downstream update')
    del agent, updated


def check_resets(manifest: list[dict]) -> None:
    env, action_space = make_env()
    try:
        for row in manifest[:2]:
            env_seed = int(row['env_seed'])
            snapshots = []
            for _ in range(2):
                np.random.seed(env_seed)
                action_space.seed(env_seed)
                observation, info = env.reset(
                    seed=env_seed,
                    options={'task_id': int(row['task_id']), 'render_goal': False},
                )
                snapshots.append((
                    np.asarray(observation, dtype=np.float32).tobytes(),
                    np.asarray(info['goal'], dtype=np.float32).tobytes(),
                ))
            if snapshots[0] != snapshots[1]:
                raise RuntimeError(f'reset/goal mismatch for env_seed {env_seed}')
    finally:
        env.close()


def smoke(directory: Path) -> None:
    load_handoff(directory)
    mean, std, components, matrix, train, observations, actions, terminals = load_arrays(directory)
    check_projection(observations, mean, std, components, matrix)
    check_modules(mean, std, matrix, observations, actions, terminals, train)
    manifest = json.loads((directory / 'eval/eval_resets_final.json').read_text())
    check_resets(manifest)
    print(json.dumps({
        'status': 'smoke_passed',
        'weights_written': False,
        'smoke_seed_reused_for_training': False,
    }), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--max-updates', type=int, required=True)
    parser.add_argument('--checkpoint-steps', type=int, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    if args.smoke_only:
        smoke(args.handoff)
        return
    if args.seed == SMOKE_SEED:
        raise RuntimeError('training seed collides with the discarded smoke seed')
    load_handoff(args.handoff)
    mean, std, _components, matrix, train, observations, actions, terminals = load_arrays(args.handoff)
    obs, act, term = train_arrays(observations, actions, terminals, train, mean, std)
    dev = json.loads((args.handoff / 'eval/eval_resets_dev.json').read_text())
    final = json.loads((args.handoff / 'eval/eval_resets_final.json').read_text())
    train_arm(
        'pca_pca',
        seed=args.seed,
        max_updates=args.max_updates,
        checkpoint_steps=tuple(args.checkpoint_steps),
        output_dir=args.output,
        observations=obs,
        actions=act,
        terminals=term,
        mean=mean,
        std=std,
        pca_matrix=matrix,
        eexec_encoder_params=unused_encoder(mean, args.seed + 17),
        dev_manifest=dev,
        final_manifest=final,
        require_protocol=False,
    )


if __name__ == '__main__':
    main()
