"""Export the immutable PCA/PCA full-state seed-1 handoff. No training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/choi/latentbridger-goalspace')
PILOT = ROOT / 'exp/eexec_pilot/puzzle_3x3'
FULL = ROOT / 'exp/eexec_fullstate_cross/puzzle_3x3'
DATASET = Path('/home/choi/.ogbench/data/puzzle-3x3-play-v0.npz')
HANDOFF = ROOT / 'exp/handoff/pca_fullstate_seed1_20261013'
REPO = Path('/home/choi/latentbridger-pca-handoff-wt')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + '\n')


def copy_file(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def git(*args: str) -> str:
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()


def export_inputs() -> None:
    if (HANDOFF / 'READY').is_file():
        raise RuntimeError(f'{HANDOFF / "READY"} already exists; refusing to overwrite')
    HANDOFF.mkdir(parents=True, exist_ok=True)
    copy_file(DATASET, HANDOFF / 'data/puzzle-3x3-play-v0.npz')
    for name in ('train_episodes.npy', 'val_episodes.npy', 'test_episodes.npy'):
        copy_file(PILOT / name, HANDOFF / 'splits' / name)
    train = np.load(HANDOFF / 'splits/train_episodes.npy')
    val = np.load(HANDOFF / 'splits/val_episodes.npy')
    test = np.load(HANDOFF / 'splits/test_episodes.npy')
    if (train.size, val.size, test.size) != (800, 100, 100):
        raise RuntimeError(f'unexpected split sizes {(train.size, val.size, test.size)}')
    parts = [set(train.tolist()), set(val.tolist()), set(test.tolist())]
    if not parts[0].isdisjoint(parts[1]) or not parts[0].isdisjoint(parts[2]) or not parts[1].isdisjoint(parts[2]):
        raise RuntimeError('episode splits overlap')
    if set().union(*parts) != set(range(1000)):
        raise RuntimeError('episode splits do not cover 1000 episodes')
    write_json(HANDOFF / 'splits/metadata.json', {
        'dataset': 'puzzle-3x3-play-v0',
        'seed': 20261011,
        'generator': 'numpy.random.Generator(PCG64(20261011)).permutation(1000)',
        'order': 'train = sorted(permutation[:800]); val = sorted(permutation[800:900]); test = sorted(permutation[900:])',
        'episode_length': 1001,
        'terminal_offset': 1000,
        'train_episodes': 800,
        'val_episodes': 100,
        'test_episodes': 100,
        'used_for_pca_and_training': 'train only',
    })
    mean = np.load(PILOT / 'obs_mean.npy')
    std = np.load(PILOT / 'obs_std.npy')
    components = np.load(PILOT / 'pca_components.npy')[:, :16]
    matrix = np.asarray(std, dtype=np.float64)[:, None] * np.asarray(components, dtype=np.float64)
    if matrix.shape != (55, 16):
        raise RuntimeError(f'PCA matrix shape {matrix.shape}')
    (HANDOFF / 'norm').mkdir(parents=True, exist_ok=True)
    (HANDOFF / 'pca').mkdir(parents=True, exist_ok=True)
    np.save(HANDOFF / 'norm/obs_mean.npy', mean)
    np.save(HANDOFF / 'norm/obs_std.npy', std)
    np.save(HANDOFF / 'pca/pca_components_16.npy', components)
    np.save(HANDOFF / 'pca/pca_matrix.npy', matrix.astype(np.float32))
    write_json(HANDOFF / 'pca/projection_rule.json', {
        'feature_order': 'ogbench compact observation columns 0..54, unpermuted',
        'feature_indices': list(range(55)),
        'normalization': '(observation - obs_mean) / obs_std',
        'normalization_fit': 'train episodes only, population standard deviation, zero std replaced by 1',
        'components': 'pca/pca_components_16.npy columns are the leading 16 eigenvectors of the centered train observations',
        'frozen_parameter': 'pca/pca_matrix.npy = obs_std[:, None] * pca_components_16',
        'agent_input': 'normalized_observation @ pca_matrix',
        'equivalent_rule': '(observation - obs_mean) @ pca_components_16',
        'latent_dim': 16,
        'refit': False,
    })
    copy_file(FULL / 'downstream/pca_pca/config.json', HANDOFF / 'config/historical_seed0_config.json')
    write_json(HANDOFF / 'config/execution_contract.json', {
        'trainer': 'scripts/run_eexec_fullstate_cross.py:train_arm',
        'wrapper': 'ops/run_pca_fullstate_seed.py',
        'arm': 'pca_pca',
        'source': 'pca',
        'goal': 'pca',
        'latent_dim': 16,
        'candidate_count': 32,
        'execution_representation': 'normalized full 55-D observation',
        'bridge_and_idm': 'actual full observation, not the 16-D role features',
        'historical_seed': 0,
        'requested_new_seed': 1,
        'max_updates': 400000,
        'checkpoint_steps': [100000, 200000, 400000],
        'pca_refit': False,
        'eexec_checkpoint_included': False,
        'downstream_checkpoint_included': False,
        'batch_size': 1024,
        'optimizer': 'Adam 3e-4',
        'env': 'puzzle-3x3-play-v0',
    })
    copy_file(FULL / 'eval_resets_final.json', HANDOFF / 'eval/eval_resets_final.json')
    copy_file(FULL / 'eval_resets_dev.json', HANDOFF / 'eval/eval_resets_dev.json')
    write_json(HANDOFF / 'eval/protocol.json', {
        'h': 5,
        'h_meaning': 'number of actions taken from the fixed 5-step chunk before replanning',
        'episodes_per_task': 50,
        'tasks': [1, 2, 3, 4, 5],
        'final_manifest': 'eval/eval_resets_final.json',
        'dev_manifest': 'eval/eval_resets_dev.json',
        'env_seed': 'the stored env_seed, not a recomputed bare task_id*10000+episode formula',
        'numpy_reset_seed': 'np.random.seed(env_seed) immediately before reset',
        'action_space_seed': 'persistent Box(-1, 1) action space seeded with env_seed before reset, covering goal-stabilization samples',
        'planner_key': 'PRNGKey(h * 1000003 + env_seed), restarted every episode',
        'candidate_count': 32,
        'temperature': 1.0,
    })
    packages = {}
    for name in ('jax', 'flax', 'optax', 'mujoco', 'ogbench', 'gymnasium', 'numpy'):
        packages[name] = importlib.metadata.version(name)
    write_json(HANDOFF / 'env/versions.json', {
        'python': platform.python_version(),
        'packages': packages,
    })


def write_handoff_index() -> None:
    remote = git('remote', 'get-url', 'origin')
    if '@' in remote and '://' in remote:
        scheme, rest = remote.split('://', 1)
        remote = scheme + '://' + rest.split('@', 1)[1]
    files = []
    for path in sorted(HANDOFF.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(HANDOFF).as_posix()
        if relative in {'HANDOFF.json', 'SHA256SUMS', 'READY'}:
            continue
        files.append({'path': relative, 'sha256': sha256_file(path)})
    write_json(HANDOFF / 'HANDOFF.json', {
        'name': 'pca_fullstate_seed1_20261013',
        'code_commit': git('rev-parse', 'HEAD'),
        'branch': 'exp/pca-fullstate-handoff-20261013',
        'remote': remote,
        'execution_contract': 'config/execution_contract.json',
        'files': files,
        'excluded': [
            'downstream checkpoints',
            'Eexec pretrain checkpoint',
        ],
    })


def create_bundle() -> None:
    if (HANDOFF / 'READY').is_file():
        raise RuntimeError('READY exists')
    bundle = HANDOFF / 'code.bundle'
    if bundle.exists():
        bundle.unlink()
    subprocess.check_call([
        'git', '-C', str(REPO), 'bundle', 'create', str(bundle),
        'exp/pca-fullstate-handoff-20261013',
    ])


def seal() -> None:
    if (HANDOFF / 'READY').is_file():
        raise RuntimeError('READY exists')
    write_handoff_index()
    lines = []
    for path in sorted(HANDOFF.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(HANDOFF).as_posix()
        if relative in {'SHA256SUMS', 'READY'}:
            continue
        lines.append(f'{sha256_file(path)}  {relative}')
    text = '\n'.join(lines) + '\n'
    (HANDOFF / 'SHA256SUMS').write_text(text)
    subprocess.check_call(['sha256sum', '-c', 'SHA256SUMS'], cwd=HANDOFF)
    (HANDOFF / 'READY').write_text(
        'status: READY\n'
        f'commit: {git("rev-parse", "HEAD")}\n'
        'sha256sums: verified\n'
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop-before-ready', action='store_true')
    parser.add_argument('--seal-only', action='store_true')
    args = parser.parse_args()
    if not args.seal_only:
        export_inputs()
        create_bundle()
    if args.stop_before_ready:
        print(json.dumps({'status': 'exported_not_sealed', 'handoff': str(HANDOFF)}))
        return
    seal()
    print(json.dumps({'status': 'READY', 'handoff': str(HANDOFF)}))


if __name__ == '__main__':
    sys.exit(main())
