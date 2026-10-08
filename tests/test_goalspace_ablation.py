"""Causal intervention contracts, checked against original PB objectives."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.contrastive_transitive_distance_pathbridger import ContrastiveTransitiveDistanceAgent, GoalSpaceScalarValue
from agents.pathbridger import PathBridgerAgent, ScalarTransitiveValue
from main_ctd_pathbridger import config_for
from test_ctd_pathbridger import _batch
from utils.goal_representation import goal_representation

VARIANTS = ('gs_trl_weighted', 'gsdtrl_uniform', 'gsdtrl_no_transitive_weighted', 'gs_symmetric_weighted')


def agent(variant):
    return ContrastiveTransitiveDistanceAgent.create(
        0, jnp.zeros((2, 8)), jnp.zeros((2, 2)),
        config_for('antmaze_medium', variant).to_dict(),
    )


def equal_tree(left, right):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for x, y in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        np.testing.assert_array_equal(x, y)


def test_scalar_uses_exact_pb_mlp_on_phi_pairs():
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 55))
    y = jax.random.normal(jax.random.PRNGKey(2), (4, 55))
    phi = lambda z: goal_representation(z, 'phi', env_name='puzzle-3x3-play-v0')
    gs = GoalSpaceScalarValue(env_name='puzzle-3x3-play-v0')
    pb = ScalarTransitiveValue()
    key = jax.random.PRNGKey(3)
    params = gs.init(key, x, y)
    equal_tree(params, pb.init(key, phi(x), phi(y)))
    np.testing.assert_array_equal(gs.apply(params, x, y), pb.apply(params, phi(x), phi(y)))
    changed = x.at[:, :19].add(1000.)
    np.testing.assert_array_equal(gs.apply(params, x, y), gs.apply(params, changed, y.at[:, :19].add(1000.)))


def test_scalar_has_original_trl_product_ranking_and_pb_weights():
    a = agent('gs_trl_weighted'); b = _batch()
    equal_tree(a.value_loss(b, a.network.params), PathBridgerAgent.value_loss(a, b, a.network.params))
    x, g, z = [jnp.asarray(b[k]) for k in ('observations', 'value_goals', 'endpoint_targets')]
    equal_tree(a._endpoint_weights(x, g, z), PathBridgerAgent._endpoint_weights(a, x, g, z))
    np.testing.assert_array_equal(
        a.sample_action_chunks(x, g, seed=jax.random.PRNGKey(5), num_candidates=4),
        PathBridgerAgent.sample_action_chunks(a, x, g, seed=jax.random.PRNGKey(5), num_candidates=4),
    )


def test_uniform_weights_are_exact_ones_and_proposer_updates():
    a = agent('gsdtrl_uniform'); b = _batch()
    w, _ = a._endpoint_weights(b['observations'], b['value_goals'], b['endpoint_targets'])
    np.testing.assert_array_equal(w, np.ones(8))
    # Including at 1M: there is no step-dependent proposer freeze.
    a = a.replace(network=a.network.replace(step=1_000_000))
    updated, _ = a.update(b)
    assert any(np.any(np.asarray(x) != np.asarray(y)) for x, y in zip(
        jax.tree_util.tree_leaves(a.network.params['modules_endpoint']),
        jax.tree_util.tree_leaves(updated.network.params['modules_endpoint']),
    ))


@pytest.mark.parametrize('power', [0.0, 0.5])
def test_no_transitive_removes_exact_gradient_and_preserves_self_base(power):
    a = agent('gsdtrl_no_transitive_weighted'); b = _batch()
    full = agent('gsdtrl_weighted')
    a = a.replace(config=a.config.copy({'value_distance_weight_power': power}))
    full = full.replace(config=full.config.copy({'value_distance_weight_power': power}))
    loss, info = a.value_loss(b, a.network.params)
    _, reference = full.value_loss(b, full.network.params)
    np.testing.assert_allclose(info['value/self_loss'], reference['value/self_loss'], rtol=1e-6)
    np.testing.assert_allclose(info['value/base_loss'], reference['value/base_loss'], rtol=1e-6)
    np.testing.assert_allclose(loss, reference['value/self_loss'] + reference['value/base_loss'], rtol=1e-6)
    assert info['value/self_loss'] > 0 and info['value/base_loss'] > 0
    poisoned = dict(b)
    for key in ('value_goals', 'value_offsets', 'transitive_subgoals', 'transitive_offsets', 'transitive_valids'):
        poisoned[key] = jnp.full_like(b[key], jnp.nan)
    equal_tree(a.value_loss(b, a.network.params), a.value_loss(poisoned, a.network.params))
    grads = jax.grad(lambda p: a.value_loss(b, p)[0])(a.network.params)
    equal_tree(grads, jax.grad(lambda p: a.value_loss(poisoned, p)[0])(a.network.params))
    expected = jax.grad(lambda p: sum(full.value_loss(b, p)[1][k] for k in ('value/self_loss', 'value/base_loss')))(a.network.params)
    for x, y in zip(jax.tree_util.tree_leaves(grads), jax.tree_util.tree_leaves(expected)):
        np.testing.assert_allclose(x, y, rtol=2e-5, atol=1e-7)
    tg = jax.grad(lambda p: a.value_loss(b, p)[1]['value/transitive_loss'])(a.network.params)
    assert all(np.count_nonzero(x) == 0 for x in jax.tree_util.tree_leaves(tg))
    assert any(np.count_nonzero(x) for x in jax.tree_util.tree_leaves(grads['modules_value']))


def test_symmetric_distance_has_no_potential_and_exact_identities():
    a = agent('gs_symmetric_weighted')
    x = jax.random.normal(jax.random.PRNGKey(0), (8, 8))
    y = jax.random.normal(jax.random.PRNGKey(1), (8, 8))
    d = lambda x,y: a._metric_distance(x,y,name='value')
    np.testing.assert_array_equal(d(x,y), d(y,x))
    np.testing.assert_array_equal(d(x,x), np.zeros(8))
    assert 'p' not in a.network.params['modules_value']
    assert all(np.isfinite(x).all() for x in jax.tree_util.tree_leaves(jax.grad(lambda z: d(z,z).sum())(x)))


@pytest.mark.parametrize('variant', VARIANTS)
def test_full_state_proposer_bridge_idm_are_unchanged_and_phi_only(variant):
    a = agent(variant); ref = agent('gsdtrl_weighted'); b = _batch()
    for module in ('modules_endpoint', 'modules_bridge', 'modules_idm'):
        equal_tree(a.network.params[module], ref.network.params[module])
    equal_tree(a.bridge_loss(b,a.network.params), ref.bridge_loss(b,ref.network.params))
    equal_tree(a.idm_loss(b,a.network.params), ref.idm_loss(b,ref.network.params))
    x = jnp.asarray(b['observations']); g = jnp.asarray(b['value_goals'])
    np.testing.assert_array_equal(
        a.network.select('value')(x,g),
        a.network.select('value')(x.at[:,2:].add(1000),g.at[:,2:].add(1000)),
    )
    assert a.config['metric_representation'] == 'phi'
    assert all(a.config[k] == 0 for k in ('lambda_nce','lambda_pathnce','lambda_bridge_geo'))
    assert a.config['endpoint_distribution'] == 'flow'
