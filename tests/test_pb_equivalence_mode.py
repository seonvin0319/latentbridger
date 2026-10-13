"""PB mode must keep full-state value semantics of official PathBridger."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from agents.pathbridger import PathBridgerAgent, get_config
from pathbridger_core.utils.goal_representation import goal_representation


def _agent(mode: str = 'pb'):
    c = get_config()
    c.env_name = 'puzzle-3x3-play-v0'
    c.pathbridger_mode = mode
    c.endpoint_value_scale = 10.0
    c.value_distance_weight_power = 0.5
    c.eval_num_candidates = 4
    c.eval_temperature = 1.0
    c.dynamics_p = (0.0, 0.0, 1.0, 0.0)
    c.critic_p = (0.0, 1.0, 0.0, 0.0)
    obs = jnp.zeros((2, 55), dtype=jnp.float32)
    act = jnp.zeros((2, 5), dtype=jnp.float32)
    return PathBridgerAgent.create(0, obs, act, c)


def test_full_goal_representation_is_identity():
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 55))
    y = goal_representation(x, 'full', env_name='puzzle-3x3-play-v0')
    np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_pb_value_param_input_is_concat_full_states():
    agent = _agent('pb')
    w = agent.critic.network.params['modules_value']['MLP_0']['Dense_0']['kernel']
    assert int(w.shape[0]) == 110


def test_pb_scoring_sensitive_to_robot_channels():
    agent = _agent('pb')
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 55))
    g = jax.random.normal(jax.random.PRNGKey(2), (2, 55))
    u = jax.random.normal(jax.random.PRNGKey(3), (2, 4, 55))
    s0 = agent.critic.score_transitive_subgoals(x, u, g, score_mode='product')
    x2 = x.at[:, :19].add(25.0)
    s1 = agent.critic.score_transitive_subgoals(x2, u, g, score_mode='product')
    assert float(jnp.max(jnp.abs(s0 - s1))) > 1e-6


if __name__ == '__main__':
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print('PASS', name)
            except Exception as e:
                failed += 1
                print('FAIL', name, e)
    raise SystemExit(failed)
