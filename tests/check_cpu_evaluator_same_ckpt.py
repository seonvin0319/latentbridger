"""Compare the agreed CPU evaluator path on one shared checkpoint.

Deployment vs unmodified legacy 3999f37: candidates, ratio scores, selected
endpoint, bridge trajectory, and IDM actions. Uses protocol
3999-idm-strict-reset-v2 seeding (episode key = episode index within task).
"""
from __future__ import annotations

import ast
import json
import pickle
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Force CPU before JAX import.
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench

from check_core_equivalence import oracle, exact
from agents import PathBridgerAgent
from configs.pbf.puzzle_3x3 import get_config
from utils.evaluation import EvalDynamics, RecordedEnv, stable_action_space
from utils.reference_evaluation import _idm_action_chunk

LEGACY = Path('/home/choi/pb_reimpl_workspace/PathBridger_legacy')
RUN = Path(
    '/home/choi/Pathbridger_flow/runs/_other/puzzle-3x3-play-v0/'
    '20260718_160615_seed0_puzzle-3x3-play-v0'
)
OUT = ROOT / 'validation' / 'cpu_evaluator_same_ckpt.json'


def main():
    assert jax.default_backend() == 'cpu', jax.default_backend()
    # Evaluator function bodies must match legacy main.py.
    source = (LEGACY / 'main.py').read_text()
    original = {
        n.name: ast.get_source_segment(source, n)
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef)
    }
    deployed = (ROOT / 'utils/reference_evaluation.py').read_text()
    matched = []
    for node in ast.parse(deployed).body:
        if isinstance(node, ast.FunctionDef) and node.name in original:
            assert ast.get_source_segment(deployed, node) == original[node.name], node.name
            matched.append(node.name)

    result = {
        'backend': 'cpu',
        'protocol': '3999-idm-strict-reset-v2',
        'checkpoint_run': str(RUN),
        'cases': 125,
        'evaluator_functions_matched': matched,
        'exact_outputs': [],
        'first_mismatch': None,
        'passed': False,
    }

    with tempfile.TemporaryDirectory() as temp, stable_action_space():
        D, C, *_ = oracle(LEGACY, temp)
        env = ogbench.make_env_and_datasets('puzzle-3x3-play-v0', env_only=True)
        env.reset()
        wrapped = RecordedEnv(env)
        o, info = wrapped.reset(options={'task_id': 1})
        act = np.zeros((1, env.action_space.shape[0]), np.float32)
        cfg = get_config()
        a = PathBridgerAgent.create(1, o[None], act, cfg)
        d = D.create(1, o[None], dict(a.dynamics.config), ex_actions=act)
        c = C.create(1, o[None], None, np.tile(act, (1, 5)), dict(a.critic.config))
        for name in ['dynamics', 'critic']:
            payload = pickle.load((RUN / f'checkpoints/{name}/params_100000.pkl').open('rb'))['agent']
            a = a.replace(**{name: flax.serialization.from_state_dict(getattr(a, name), payload)})
            if name == 'dynamics':
                d = flax.serialization.from_state_dict(d, payload)
            else:
                c = flax.serialization.from_state_dict(c, payload)
        exact(flax.serialization.to_state_dict(a.dynamics), flax.serialization.to_state_dict(d), 'loaded dynamics')
        exact(flax.serialization.to_state_dict(a.critic), flax.serialization.to_state_dict(c), 'loaded critic')

        policy = EvalDynamics(a.dynamics, 32, 1.0)
        reference = EvalDynamics(d, 32, 1.0)
        candidates = jax.jit(
            lambda dyn, obs, goal, key: dyn.sample_subgoal_candidates(
                obs, goal, key, num_candidates=32, include_mean=False
            )[0]
        )
        scores = jax.jit(
            lambda crit, obs, z, goal: crit.score_transitive_subgoals(
                obs, z, goal, network_params=crit.network.params, score_mode='ratio'
            )
        )

        for i in range(125):
            if i:
                o, info = wrapped.reset(options={'task_id': i // 25 + 1})
            oo = jnp.asarray(o)
            g = jnp.asarray(info['goal'])
            key = jax.random.PRNGKey(i % 25)
            try:
                new_z = candidates(policy.d, oo[None], g[None], key)
                old_z = candidates(reference.d, oo[None], g[None], key)
                exact(new_z, old_z, 'candidates')
                new_score = scores(a.critic, oo[None], new_z, g[None])
                old_score = scores(c, oo[None], old_z, g[None])
                exact(new_score, old_score, 'scores')
                az = policy.infer_subgoal_for_eval(oo, g, a.critic, key)
                dz = reference.infer_subgoal_for_eval(oo, g, c, key)
                exact(az, dz, 'selected endpoint')
                exact(policy.plan(oo, az), reference.plan(oo, dz), 'bridge')
                exact(
                    _idm_action_chunk(policy, o, np.asarray(az), 5),
                    _idm_action_chunk(reference, o, np.asarray(dz), 5),
                    'actions',
                )
                batched_z = reference.infer_subgoal_for_eval(oo[None], g[None], c, key)
                batched_traj = reference.plan(oo[None], batched_z)['trajectory']
                exact(
                    a.sample_action_chunks(oo[None], g[None], key, num_candidates=32, temperature=1.0),
                    reference._idm_actions_from_trajectories(batched_traj, 5),
                    'public action API',
                )
            except AssertionError as exc:
                result['first_mismatch'] = {'episode': i, 'detail': str(exc)}
                raise
            result['exact_outputs'].append(
                {
                    'episode': i,
                    'task_id': int(i // 25 + 1),
                    'observation_sha256': wrapped.records[-1]['observation_sha256'],
                    'goal_sha256': wrapped.records[-1]['goal_sha256'],
                    'candidates': True,
                    'scores': True,
                    'selected_endpoint': True,
                    'bridge': True,
                    'raw_actions': True,
                }
            )
            if i == 0 or (i + 1) % 25 == 0:
                print(f'CPU_EXACT {i+1}/125', flush=True)
        env.close()

    result['public_batched_action_api_exact'] = True
    result['passed'] = True
    # Keep output compact: drop per-episode hashes from the summary file.
    summary = {
        k: v for k, v in result.items() if k != 'exact_outputs'
    }
    summary['exact_episode_count'] = len(result['exact_outputs'])
    summary['episode_hash_dump'] = str(ROOT / 'validation' / 'cpu_evaluator_episode_hashes.json')
    (ROOT / 'validation' / 'cpu_evaluator_episode_hashes.json').write_text(
        json.dumps(result['exact_outputs'], indent=2) + '\n'
    )
    OUT.write_text(json.dumps(summary, indent=2) + '\n')
    print('CPU_EVALUATOR_ALL_EXACT', OUT, flush=True)


if __name__ == '__main__':
    main()
