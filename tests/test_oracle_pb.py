"""Oracle PB correctness + PB-mode non-regression on the latest PathBridger."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from agents.pathbridger import PathBridgerAgent, get_config
from pathbridger_core.utils.goal_representation import goal_representation


def _cfg(mode: str, env_name: str = 'puzzle-3x3-play-v0'):
    c = get_config()
    c.env_name = env_name
    c.pathbridger_mode = mode
    c.endpoint_value_scale = 10.0
    c.value_distance_weight_power = 0.5
    c.eval_num_candidates = 4
    c.eval_temperature = 1.0
    c.dynamics_p = (0.0, 0.0, 1.0, 0.0)
    c.critic_p = (0.0, 1.0, 0.0, 0.0)
    return c


def _agent(mode: str, obs_dim: int = 55, act_dim: int = 5, seed: int = 0):
    obs = jnp.zeros((2, obs_dim), dtype=jnp.float32)
    act = jnp.zeros((2, act_dim), dtype=jnp.float32)
    return PathBridgerAgent.create(seed, obs, act, _cfg(mode))


def test_pb_mode_defaults_and_full_state_value_sensitivity():
    agent = _agent('pb')
    assert agent.config['pathbridger_mode'] == 'pb'
    assert agent.critic.config['pathbridger_mode'] == 'pb'
    assert agent.critic.config['value_state_representation'] == 'full'
    assert agent.dynamics.config['subgoal_value_state_representation'] == 'full'
    x = jax.random.normal(jax.random.PRNGKey(0), (4, 55))
    g = jax.random.normal(jax.random.PRNGKey(1), (4, 55))
    v0 = agent.critic.network.select('value')(x, g)
    # Non-phi robot channels must affect PB value.
    changed = x.at[:, :19].add(10.0)
    v1 = agent.critic.network.select('value')(changed, g)
    assert float(jnp.max(jnp.abs(v0 - v1))) > 1e-5


def test_oracle_pb_value_is_invariant_to_non_phi_state_channels():
    agent = _agent('oracle_pb')
    assert agent.config['pathbridger_mode'] == 'oracle_pb'
    assert agent.critic.config['value_state_representation'] == 'phi'
    assert agent.critic.config['value_goal_representation'] == 'phi'
    assert agent.dynamics.config['subgoal_value_state_representation'] == 'phi'
    assert agent.dynamics.config['subgoal_value_goal_representation'] == 'phi'
    # Proposer still conditions on phi(g) in full-state endpoint space.
    assert agent.dynamics.config['subgoal_goal_representation'] == 'phi'

    x = jax.random.normal(jax.random.PRNGKey(2), (4, 55))
    g = jax.random.normal(jax.random.PRNGKey(3), (4, 55))
    v0 = agent.critic.network.select('value')(x, g)
    # Puzzle phi is button configuration; head dims 0..18 are robot-ish.
    changed = x.at[:, :19].add(1000.0)
    g_changed = g.at[:, :19].add(1000.0)
    v1 = agent.critic.network.select('value')(changed, g_changed)
    np.testing.assert_allclose(np.asarray(v0), np.asarray(v1), rtol=1e-5, atol=1e-5)

    # Same phi(s) => same value for fixed goal.
    phi = lambda z: goal_representation(z, 'phi', env_name='puzzle-3x3-play-v0')
    twin = x.at[:, :19].add(3.0)
    np.testing.assert_array_equal(np.asarray(phi(x)), np.asarray(phi(twin)))
    np.testing.assert_allclose(
        np.asarray(agent.critic.network.select('value')(x, g)),
        np.asarray(agent.critic.network.select('value')(twin, g)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_oracle_pb_value_changes_when_phi_channels_change():
    agent = _agent('oracle_pb')
    x = jax.random.normal(jax.random.PRNGKey(4), (3, 55))
    g = jax.random.normal(jax.random.PRNGKey(5), (3, 55))
    env = 'puzzle-3x3-play-v0'
    v0 = agent.critic.network.select('value')(x, g)
    # Perturb only phi channels (button block starts after manip head).
    x2 = x.at[:, 19:].add(jax.random.normal(jax.random.PRNGKey(6), (3, 55 - 19)))
    assert not np.allclose(
        np.asarray(goal_representation(x, 'phi', env_name=env)),
        np.asarray(goal_representation(x2, 'phi', env_name=env)),
    )
    v1 = agent.critic.network.select('value')(x2, g)
    assert float(jnp.max(jnp.abs(v0 - v1))) > 1e-6


def test_proposer_outputs_full_state_endpoints():
    agent = _agent('oracle_pb')
    x = jnp.zeros((2, 55), dtype=jnp.float32)
    g = jnp.zeros((2, 55), dtype=jnp.float32)
    seed = jax.random.PRNGKey(7)
    # infer_subgoal_for_eval path via sample_action_chunks → endpoints are full dim.
    chunks = agent.sample_action_chunks(x, g, seed, num_candidates=4, temperature=1.0)
    assert chunks.shape[-1] == 5  # action dim
    # Dynamics subgoal sample should be full state_dim.
    from utils.evaluation import EvalDynamics

    policy = EvalDynamics(agent.dynamics, 4, 1.0)
    z = policy.infer_subgoal_for_eval(x, g, agent.critic, seed)
    assert z.shape == (2, 55)


def test_invalid_mode_raises():
    try:
        _agent('latent')
    except ValueError as e:
        assert 'pathbridger_mode' in str(e)
        return
    raise AssertionError('expected ValueError for invalid pathbridger_mode')


def test_oracle_pb_scoring_uses_goal_space_product():
    """Candidate scores must be invariant to non-phi channels under oracle_pb."""
    agent = _agent('oracle_pb')
    x = jax.random.normal(jax.random.PRNGKey(8), (2, 55))
    g = jax.random.normal(jax.random.PRNGKey(9), (2, 55))
    u = jax.random.normal(jax.random.PRNGKey(10), (2, 4, 55))
    s0 = agent.critic.score_transitive_subgoals(x, u, g, score_mode='product')
    x2 = x.at[:, :19].add(50.0)
    u2 = u.at[:, :, :19].add(50.0)
    g2 = g.at[:, :19].add(50.0)
    s1 = agent.critic.score_transitive_subgoals(x2, u2, g2, score_mode='product')
    np.testing.assert_allclose(np.asarray(s0), np.asarray(s1), rtol=1e-5, atol=1e-5)


def test_pb_vs_oracle_pb_value_input_dims_differ():
    """PB value MLP sees 2*obs_dim; Oracle PB sees 2*phi_dim."""
    pb = _agent('pb')
    gs = _agent('oracle_pb')
    # Peek Dense kernel shapes on the first MLP layer.
    pb_w = pb.critic.network.params['modules_value']['MLP_0']['Dense_0']['kernel']
    gs_w = gs.critic.network.params['modules_value']['MLP_0']['Dense_0']['kernel']
    assert int(pb_w.shape[0]) == 110  # 55+55
    assert int(gs_w.shape[0]) < 110
    assert int(gs_w.shape[0]) % 2 == 0


if __name__ == '__main__':
    failed = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith('test_') or not callable(fn):
            continue
        try:
            fn()
            print('PASS', name)
        except Exception as e:
            failed += 1
            print('FAIL', name, e)
    raise SystemExit(failed)
