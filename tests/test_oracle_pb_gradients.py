"""Oracle PB gradient flow + no-phi on bridge/IDM/action-critic paths."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from agents.pathbridger import PathBridgerAgent, get_config
from pathbridger_core.utils.goal_representation import goal_representation


def _cfg(mode: str):
    c = get_config()
    c.env_name = 'puzzle-3x3-play-v0'
    c.pathbridger_mode = mode
    c.endpoint_value_scale = 10.0
    c.value_distance_weight_power = 0.5
    c.eval_num_candidates = 4
    c.eval_temperature = 1.0
    c.dynamics_p = (0.0, 0.0, 1.0, 0.0)
    c.critic_p = (0.0, 1.0, 0.0, 0.0)
    return c


def _agent(mode: str):
    obs = jnp.zeros((2, 55), dtype=jnp.float32)
    act = jnp.zeros((2, 5), dtype=jnp.float32)
    return PathBridgerAgent.create(0, obs, act, _cfg(mode))


def _trl_batch(n=4, obs_dim=55, act_dim=5, h=5):
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(n, obs_dim)).astype(np.float32)
    goals = rng.normal(size=(n, obs_dim)).astype(np.float32)
    return {
        'observations': obs,
        'value_goals': goals,
        'action_chunk_actions': rng.normal(size=(n, h, act_dim)).astype(np.float32),
        'action_chunk_next_observations': rng.normal(size=(n, obs_dim)).astype(np.float32),
        'value_offsets': rng.integers(1, 20, size=(n,)).astype(np.float32),
        'value_base_goals': goals,
        'value_base_offsets': rng.integers(1, 10, size=(n,)).astype(np.float32),
        'trans_v_split_observations': rng.normal(size=(n, obs_dim)).astype(np.float32),
        'trans_v_left_goals': goals,
        'trans_v_right_goals': goals,
        'trans_v_valid_mask': np.ones((n,), dtype=np.float32),
    }


def test_oracle_pb_value_grads_nonzero_and_phi_sensitive():
    agent = _agent('oracle_pb')
    batch = jax.tree_util.tree_map(jnp.asarray, _trl_batch())

    def loss_from_value_params(value_params):
        params = agent.critic.network.params
        params = params.copy() if hasattr(params, 'copy') else dict(params)
        # FrozenDict -> dict
        params = {k: params[k] for k in params}
        params['modules_value'] = value_params
        total, _info = agent.critic.total_loss(batch, params, rng=jax.random.PRNGKey(0))
        return total

    g = jax.grad(loss_from_value_params)(agent.critic.network.params['modules_value'])
    g_norm = float(sum(jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(g)))
    assert g_norm > 0.0, 'value grads should be nonzero under Oracle PB TRL loss'

    # First-layer kernel rows correspond to concat(phi(s), phi(g)).
    # Perturbing only non-phi obs channels must not change the loss (and thus grads
    # w.r.t. a fixed param snapshot of the forward at those inputs — checked via value).
    x = batch['observations']
    g_goal = batch['value_goals']
    v0 = agent.critic.network.select('value')(x, g_goal)
    x2 = x.at[:, :19].add(100.0)
    v1 = agent.critic.network.select('value')(x2, g_goal)
    np.testing.assert_allclose(np.asarray(v0), np.asarray(v1), atol=1e-5)


def test_proposer_weighting_uses_stopgrad_value_but_weights_subgoal():
    """Value params must be stop-grad in dynamics weighting; subgoal net still trains."""
    agent = _agent('oracle_pb')
    # Minimal: apply _subgoal_values and ensure stop_gradient path (no grad to value via apply).
    s = jax.random.normal(jax.random.PRNGKey(1), (8, 55))
    g = jax.random.normal(jax.random.PRNGKey(2), (8, 55))
    vp = agent.critic.network.params['modules_value']

    def weighted_mean(value_params):
        vals = agent.dynamics._subgoal_values(s, g, value_params)
        return jnp.mean(vals)

    # With stop_gradient inside _subgoal_values, grad w.r.t. value_params is 0.
    grads = jax.grad(weighted_mean)(vp)
    g_norm = float(sum(jnp.sum(jnp.abs(x)) for x in jax.tree_util.tree_leaves(grads)))
    assert g_norm == 0.0, f'expected stop-grad on value in weighting, got {g_norm}'

    # Values themselves must still depend on phi channels (weighting signal exists).
    v = agent.dynamics._subgoal_values(s, g, vp)
    s2 = s.at[:, 19:].add(1.0)
    v2 = agent.dynamics._subgoal_values(s2, g, vp)
    assert float(jnp.max(jnp.abs(v - v2))) > 1e-6


def test_action_critic_not_phi_projected():
    agent = _agent('oracle_pb')
    # BinaryChunkCritic keeps full observations; only goals may be represented.
    # With action_goal_rep=full, changing robot channels of observations must change Q.
    x = jax.random.normal(jax.random.PRNGKey(3), (4, 55))
    g = jax.random.normal(jax.random.PRNGKey(4), (4, 55))
    a = jax.random.normal(jax.random.PRNGKey(5), (4, 25))  # h*act
    q0 = agent.critic.network.select('action_critic')(x, g, a)
    x2 = x.at[:, :19].add(50.0)
    q1 = agent.critic.network.select('action_critic')(x2, g, a)
    assert float(jnp.max(jnp.abs(q0 - q1))) > 1e-5


def test_bridge_and_idm_consume_full_state_dims():
    agent = _agent('oracle_pb')
    # Subgoal proposer output dim == full obs dim
    obs = jnp.zeros((2, 55), dtype=jnp.float32)
    goals = jnp.zeros((2, 55), dtype=jnp.float32)
    from utils.evaluation import EvalDynamics

    pol = EvalDynamics(agent.dynamics, 4, 1.0)
    z = pol.infer_subgoal_for_eval(obs, goals, agent.critic, jax.random.PRNGKey(0))
    assert z.shape == (2, 55)
    plan = pol.plan(obs, z)
    traj = plan['trajectory']
    assert traj.shape[-1] == 55  # full-state bridge path
    acts = pol._idm_actions_from_trajectories(traj, 5)
    assert acts.shape[-1] == 5


def test_pb_value_grad_sensitive_to_robot_channels():
    agent = _agent('pb')
    x = jax.random.normal(jax.random.PRNGKey(6), (4, 55))
    g = jax.random.normal(jax.random.PRNGKey(7), (4, 55))
    v0 = agent.critic.network.select('value')(x, g)
    v1 = agent.critic.network.select('value')(x.at[:, :19].add(10.0), g)
    assert float(jnp.max(jnp.abs(v0 - v1))) > 1e-5


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
