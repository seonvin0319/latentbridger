"""125 fixed reset states, native candidate RNG, and the common CPU evaluator."""
import argparse
import ast
import json
from pathlib import Path
import pickle
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flax.serialization
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from check_core_equivalence import oracle, exact
from agents import PathBridgerAgent
from configs.pbf.puzzle_3x3 import get_config
from utils.evaluation import EvalDynamics, RecordedEnv, stable_action_space, evaluate
from utils.reference_evaluation import _idm_action_chunk
ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference-root', type=Path, required=True)
    p.add_argument('--reference-run', type=Path, required=True)
    p.add_argument('--reference-evaluation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    arg = p.parse_args()
    assert jax.default_backend() == 'cpu'
    # Verify extracted evaluator bodies against independent original main.py.
    source = (arg.reference_root / 'main.py').read_text()
    original = {n.name: ast.get_source_segment(source, n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    deployed = (ROOT/'utils/reference_evaluation.py').read_text()
    for node in ast.parse(deployed).body:
        if isinstance(node, ast.FunctionDef):
            assert ast.get_source_segment(deployed, node) == original[node.name]
    result = dict(backend='cpu', cases=125, policy_batch_size=1, exact_outputs=[], evaluator_source_exact=True)
    records = json.loads(arg.reference_evaluation.read_text())['episodes']
    with tempfile.TemporaryDirectory() as temp, stable_action_space():
        D, C, *_ = oracle(arg.reference_root, temp)
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
            payload = pickle.load((arg.reference_run/f'checkpoints/{name}/params_100000.pkl').open('rb'))['agent']
            a = a.replace(**{name: flax.serialization.from_state_dict(getattr(a, name), payload)})
            if name == 'dynamics':
                d = flax.serialization.from_state_dict(d, payload)
            else:
                c = flax.serialization.from_state_dict(c, payload)
        policy = EvalDynamics(a.dynamics, 32, 1.)
        reference = EvalDynamics(d, 32, 1.)
        candidates = jax.jit(lambda d, o, g, k: d.sample_subgoal_candidates(o, g, k, num_candidates=32, include_mean=False)[0])
        scores = jax.jit(lambda c, o, z, g: c.score_transitive_subgoals(o, z, g, network_params=c.network.params, score_mode='ratio'))
        for i in range(125):
            if i:
                o, info = wrapped.reset(options={'task_id': i//25+1})
            for name in ['observation_sha256', 'goal_sha256']:
                assert wrapped.records[-1][name] == records[i][name], (i, name)
            oo, g, key = jnp.asarray(o), jnp.asarray(info['goal']), jax.random.PRNGKey(i % 25)
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
            exact(_idm_action_chunk(policy, o, np.asarray(az), 5), _idm_action_chunk(reference, o, np.asarray(dz), 5), 'actions')
            # Public batched action API uses these same staged original calls.
            batched_z = reference.infer_subgoal_for_eval(oo[None], g[None], c, key)
            batched_traj = reference.plan(oo[None], batched_z)['trajectory']
            exact(a.sample_action_chunks(oo[None], g[None], key, num_candidates=32, temperature=1.),
                  reference._idm_actions_from_trajectories(batched_traj, 5), 'public action API')
            result['exact_outputs'].append(dict(episode=i, candidates=True, scores=True, selected_endpoint=True, bridge=True, raw_actions=True))
        env.close()
    result['public_batched_action_api_exact'] = True
    result['reset_observation_and_full_goal_hashes_exact'] = True
    result['passed'] = True
    arg.output.write_text(json.dumps(result, indent=2)+'\n')
    print('CPU_125_ALL_EXACT', flush=True)


if __name__ == '__main__':
    main()
