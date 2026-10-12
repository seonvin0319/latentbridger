"""Behavioral smoke checks for the full-state four-arm comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, '/home/choi/latentbridger-goalspace')

from agents.fullstate_role_pathbridger import FullStateRolePathBridgerAgent  # noqa: E402
from configs.pbf.puzzle_3x3 import get_config  # noqa: E402
from eexec_fullstate_lib import (  # noqa: E402
    ARMS,
    OUT,
    load_frozen_representations,
    load_full_train,
    write_json,
)
from utils.datasets import Dataset, PathBridgerDataset  # noqa: E402


def changed(before, after) -> bool:
    before_leaves = jax.tree_util.tree_leaves(before)
    after_leaves = jax.tree_util.tree_leaves(after)
    return any(
        not np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(before_leaves, after_leaves, strict=True)
    )


def main() -> None:
    if not (OUT / 'protocol.json').is_file():
        raise RuntimeError('run prepare_eexec_fullstate_cross.py before smoke')

    observations, actions, terminals = load_full_train()
    dataset = PathBridgerDataset(
        Dataset.create(
            observations=observations,
            actions=actions,
            terminals=terminals,
        ),
        get_config(),
    )
    np.random.seed(0)
    example = dataset.sample(8)
    encoder_params, pca_matrix = load_frozen_representations()
    config = get_config().to_dict()

    agents = {}
    role_shapes = {}
    for arm, (source_kind, goal_kind) in ARMS.items():
        agent = FullStateRolePathBridgerAgent.create(
            0,
            example['observations'],
            example['actions'],
            config,
            source_kind=source_kind,
            goal_kind=goal_kind,
            eexec_encoder_params=encoder_params,
            pca_matrix=pca_matrix,
        )
        source = np.asarray(agent.encode_source(example['observations']))
        goal = np.asarray(agent.encode_goal(example['observations']))
        if source.shape != (8, 16) or goal.shape != (8, 16):
            raise AssertionError(f'{arm}: invalid role shapes {source.shape}/{goal.shape}')
        if agent.source_kind != source_kind or agent.goal_kind != goal_kind:
            raise AssertionError(f'{arm}: role mapping changed during creation')
        agents[arm] = agent
        role_shapes[arm] = {'source': list(source.shape), 'goal': list(goal.shape)}

    agent = agents['pca_eexec']
    params = agent.network.params
    shapes = {
        'bridge_first_kernel': list(
            params['modules_bridge']['MLP_0']['Dense_0']['kernel'].shape
        ),
        'bridge_output_kernel': list(
            params['modules_bridge']['MLP_0']['Dense_3']['kernel'].shape
        ),
        'idm_first_kernel': list(
            params['modules_idm']['MLP_0']['Dense_0']['kernel'].shape
        ),
        'endpoint_output_kernel': list(
            params['modules_endpoint']['MLP_0']['Dense_3']['kernel'].shape
        ),
    }
    expected_shapes = {
        'bridge_first_kernel': [111, 512],
        'bridge_output_kernel': [512, 55],
        'idm_first_kernel': [110, 512],
        'endpoint_output_kernel': [512, 55],
    }
    if shapes != expected_shapes:
        raise AssertionError(f'full-state module shape mismatch: {shapes}')

    probe = jnp.asarray(example['observations'][:1])
    goal = jnp.asarray(example['endpoint_goals'][:1])
    endpoint = probe.at[:, 54].add(1.0)
    prefix = np.asarray(agent._construct_bridge_prefix(probe, endpoint))
    if prefix.shape != (1, 6, 55):
        raise AssertionError(f'invalid bridge prefix shape {prefix.shape}')
    if not np.any(np.abs(prefix[:, 1:, 54] - prefix[:, :1, 54]) > 1e-7):
        raise AssertionError('bridge ignored a perturbation in full-state coordinate 54')

    idm = agent.network.select('idm')
    idm_base = np.asarray(idm(probe, probe))
    idm_changed = np.asarray(idm(probe, endpoint))
    idm_delta = float(np.max(np.abs(idm_base - idm_changed)))
    if not idm_delta > 1e-7:
        raise AssertionError('IDM ignored a perturbation in full-state coordinate 54')

    candidates = np.asarray(agent._sample_endpoint_candidates(
        probe,
        goal,
        jax.random.PRNGKey(91),
        num_candidates=3,
        temperature=1.0,
    ))
    if candidates.shape != (1, 3, 55):
        raise AssertionError(f'invalid endpoint candidate shape {candidates.shape}')
    tail_delta = float(np.max(np.abs(candidates[:, :, 16:] - np.asarray(probe)[:, None, 16:])))
    if not tail_delta > 1e-7:
        raise AssertionError('endpoint candidates do not populate full-state coordinates')

    action_chunks = np.asarray(agent.sample_action_chunks(
        probe,
        goal,
        seed=jax.random.PRNGKey(92),
        num_candidates=3,
        temperature=1.0,
    ))
    if action_chunks.shape != (1, 5, actions.shape[-1]):
        raise AssertionError(f'invalid action chunk shape {action_chunks.shape}')

    frozen_encoder_before = jax.tree_util.tree_map(np.asarray, agent.eexec_encoder_params)
    frozen_pca_before = np.asarray(agent.pca_matrix).copy()
    bridge_before = params['modules_bridge']
    idm_before = params['modules_idm']
    updated, info = agent.update(example)
    jax.block_until_ready(info['loss/total'])
    if changed(frozen_encoder_before, updated.eexec_encoder_params):
        raise AssertionError('frozen Eexec encoder changed during downstream update')
    if not np.array_equal(frozen_pca_before, np.asarray(updated.pca_matrix)):
        raise AssertionError('frozen PCA map changed during downstream update')
    if not changed(bridge_before, updated.network.params['modules_bridge']):
        raise AssertionError('bridge did not update')
    if not changed(idm_before, updated.network.params['modules_idm']):
        raise AssertionError('IDM did not update')
    scalar_info = {key: float(np.asarray(value)) for key, value in info.items()}
    if not all(np.isfinite(value) for value in scalar_info.values()):
        raise AssertionError('non-finite loss in one-step update')

    report = {
        'status': 'passed',
        'arms_created': list(ARMS),
        'role_shapes': role_shapes,
        'module_shapes': shapes,
        'behavioral_checks': {
            'bridge_uses_full_coordinate_54': True,
            'idm_uses_full_coordinate_54': True,
            'idm_coordinate_54_max_action_delta': idm_delta,
            'endpoint_candidates_are_full_55d': True,
            'endpoint_tail_max_displacement': tail_delta,
            'action_chunk_shape': list(action_chunks.shape),
            'frozen_eexec_unchanged_after_update': True,
            'frozen_pca_unchanged_after_update': True,
            'bridge_and_idm_updated': True,
            'one_update_losses_finite': True,
        },
        'one_update_loss': scalar_info['loss/total'],
    }
    write_json(OUT / 'fullstate_smoke.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
