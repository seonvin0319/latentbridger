"""Prepare the isolated four-arm full-state representation cross."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eexec_fullstate_lib import (  # noqa: E402
    ARMS,
    BATCH_SIZE,
    DEV_BASE,
    DEV_EPISODES,
    DOWNSTREAM_SEED,
    DOWNSTREAM_STEPS,
    FINAL_BASE,
    FINAL_EPISODES,
    OUT,
    PILOT_OUT,
    ROOT,
    TASKS,
    reset_manifest,
    write_json,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def snapshot() -> str:
    directory = OUT / 'code_snapshot'
    directory.mkdir(parents=True, exist_ok=True)
    diff = subprocess.check_output(['git', '-C', str(ROOT), 'diff', 'HEAD'], text=True)
    status = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True)
    (directory / 'git.diff').write_text(diff)
    (directory / 'git_status.txt').write_text(status)
    hashes = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith(('exp/', '.venv/', '.git/')) or '__pycache__' in relative:
            continue
        if path.suffix not in {'.py', '.md', '.json', '.sh'}:
            continue
        hashes.append(f'{sha256_file(path)}  {relative}')
    (directory / 'source_hashes.txt').write_text('\n'.join(hashes) + '\n')
    digest = hashlib.sha256(
        (directory / 'git.diff').read_bytes()
        + b'\n'
        + (directory / 'source_hashes.txt').read_bytes()
    ).hexdigest()
    (directory / 'SNAPSHOT_SHA256').write_text(digest + '\n')
    return digest


def legacy_table() -> dict:
    table = {}
    for arm in ('pca_frozen', 'eexec_frozen'):
        source = PILOT_OUT / 'downstream' / arm / 'evaluation_400000_h5_final.json'
        result = json.loads(source.read_text())
        table[arm] = {
            'execution_representation': 'latent_16d',
            'overall_success': result['overall_success'],
            'success_count': result['success_count'],
            'episodes': len(TASKS) * int(result['episodes_per_task']),
            'source_path': str(source),
            'sha256': sha256_file(source),
            'interpretation': 'preserved historical result; not a full-state cross arm',
        }
    return table


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = reset_manifest(DEV_BASE, DEV_EPISODES, 'fullstate_cross_dev')
    final = reset_manifest(FINAL_BASE, FINAL_EPISODES, 'fullstate_cross_final')
    write_json(OUT / 'eval_resets_dev.json', dev)
    write_json(OUT / 'eval_resets_final.json', final)
    old_dev = json.loads((PILOT_OUT / 'eval_resets_dev.json').read_text())
    old_final = json.loads((PILOT_OUT / 'eval_resets_final.json').read_text())
    new_seeds = {row['env_seed'] for row in dev + final}
    old_seeds = {row['env_seed'] for row in old_dev + old_final}
    if not new_seeds.isdisjoint(old_seeds):
        raise RuntimeError('new evaluation reset list overlaps the latent pilot')
    if len(new_seeds) != len(dev) + len(final):
        raise RuntimeError('new dev/final evaluation lists overlap')

    legacy = legacy_table()
    write_json(OUT / 'legacy_latent_results.json', legacy)
    eexec_checkpoint = PILOT_OUT / 'pretrain/eexec/params_133333.pkl'
    pca_components = PILOT_OUT / 'pca_components.npy'
    protocol = {
        'name': 'eexec_fullstate_role_cross',
        'question': 'With full-state execution, is Eexec useful in the source or goal role?',
        'arms': {
            arm: {
                'source_representation': source,
                'goal_representation': goal,
                'bridge_idm_representation': 'normalized full 55-D observation',
                'updates': DOWNSTREAM_STEPS,
                'fresh_downstream_initialization': True,
            }
            for arm, (source, goal) in ARMS.items()
        },
        'frozen_inputs': {
            'eexec_checkpoint': str(eexec_checkpoint),
            'eexec_checkpoint_sha256': sha256_file(eexec_checkpoint),
            'pca_components': str(pca_components),
            'pca_components_sha256': sha256_file(pca_components),
            'additional_pretrain_updates': 0,
            'inside_downstream_optimizer': False,
        },
        'execution': {
            'observation_space': 'train-normalized full 55-D observation',
            'endpoint_candidates': 'full 55-D observation',
            'endpoint_displacements': 'full 55-D observation displacement',
            'bridge': 'full 55-D observation',
            'idm': 'two full 55-D observations to 5-D action',
            'value_source_width': 16,
            'value_goal_width': 16,
        },
        'matched': {
            'downstream_seed': DOWNSTREAM_SEED,
            'batch_size': BATCH_SIZE,
            'batch_stream': 'NumPy seed reset identically at the start of every arm',
            'optimizer': 'PathBridger Adam 3e-4',
            'sampling': 'same PathBridgerDataset recipe and train split',
            'evaluation_resets': 'identical stored lists for all four arms',
        },
        'evaluation': {
            'dev': {'steps': [100000, 200000], 'h': 5, 'episodes_per_task': DEV_EPISODES, 'role': 'anomaly check only'},
            'final': {'step': 400000, 'h': 5, 'episodes_per_task': FINAL_EPISODES, 'role': 'primary judgment'},
            'tasks': list(TASKS),
            'dev_base': DEV_BASE,
            'final_base': FINAL_BASE,
            'new_resets_disjoint_from_latent_pilot': True,
        },
        'budget': {
            'arms': len(ARMS),
            'updates_per_arm': DOWNSTREAM_STEPS,
            'total_updates': len(ARMS) * DOWNSTREAM_STEPS,
            'automatic_1m_extension': False,
        },
        'legacy_latent_results': {
            'path': str(OUT / 'legacy_latent_results.json'),
            'kept_as_separate_table': True,
            'not_claimed_to_explain_failure': True,
        },
        'identity': {
            'git_head': subprocess.check_output(
                ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True,
            ).strip(),
            'worktree': 'dirty; source hashes and diff stored in code_snapshot',
        },
    }
    protocol['identity']['code_snapshot_sha256'] = snapshot()
    write_json(OUT / 'protocol.json', protocol)
    report = {
        'status': 'ready_for_smoke',
        'arms': list(ARMS),
        'total_updates': len(ARMS) * DOWNSTREAM_STEPS,
        'new_reset_count': len(new_seeds),
        'new_resets_disjoint_from_latent_pilot': True,
        'legacy_latent_results': legacy,
        'code_snapshot_sha256': protocol['identity']['code_snapshot_sha256'],
    }
    write_json(OUT / 'prep_report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
