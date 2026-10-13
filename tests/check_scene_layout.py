"""Check scene phi against OGBench's own oracle and the prior deployment."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
import ogbench
from utils.evaluation import stable_action_space
from utils.goal_representation import goal_representation as prior_phi
from pathbridger_core.utils.goal_representation import goal_representation as core_phi
from check_core_equivalence import exact


def phi(x):
    return core_phi(jnp.asarray(x, dtype=jnp.float32), 'phi', (), env_name='scene-play-v0')


def main():
    assert jax.default_backend() == 'cpu'
    checked_states = checked_goals = 0
    actions = np.random.RandomState(481)
    with stable_action_space():
        env = ogbench.make_env_and_datasets('scene-play-v0', env_only=True)
        oracle_env = ogbench.make_env_and_datasets('scene-play-v0', env_only=True, use_oracle_rep=True)
        try:
            native = env.unwrapped
            assert (native._num_cubes, native._num_buttons, native._num_button_states) == (1, 2, 2)
            for task in range(1, 6):
                for episode in range(2):
                    index = (task-1)*2+episode
                    for e in [env, oracle_env]:
                        e.action_space.seed(930000+index)
                    o, info = env.reset(seed=730000+index, options={'task_id': task})
                    oo, oi = oracle_env.reset(seed=730000+index, options={'task_id': task})
                    exact(o, oo, 'paired environment observation')
                    exact(phi(info['goal']), np.asarray(oi['goal'], np.float32), 'actual oracle goal')
                    checked_goals += 1
                    for step in range(21):
                        exact(phi(o), np.asarray(native.compute_oracle_observation(), np.float32), 'actual oracle state')
                        checked_states += 1
                        if step < 20:
                            act = actions.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)
                            o, *_ = env.step(act)
            # Existing deployment conversion is also checked on offline observations.
            raw = ogbench.load_dataset(str(Path.home()/'.ogbench/data/scene-play-v0-val.npz'), compact_dataset=True)
            sample = raw['observations'][np.random.RandomState(1301).randint(len(raw['observations']), size=1024)]
            exact(phi(sample), prior_phi(jnp.asarray(sample), 'phi', (), env_name='scene-play-v0'), 'prior deployment dataset phi')
        finally:
            env.close()
            oracle_env.close()
    result = dict(passed=True, backend='cpu', layout=[1,2,2], observations_dim=40, phi_dim=7,
                  actual_oracle_states_exact=checked_states, actual_oracle_goals_exact=checked_goals,
                  previous_deployment_dataset_rows_exact=1024,
                  note='Independent oracle values come from installed OGBench SceneEnv, not the patched parser.')
    out = Path(__file__).resolve().parents[1]/'validation/scene_layout.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
