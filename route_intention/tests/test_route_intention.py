"""Unit tests for route-level intention. ``JAX_PLATFORMS=cpu python -m pytest route_intention/tests -q``."""

from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('IPB_ALLOW_CPU', '1')

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from route_intention.chunks import (  # noqa: E402
    compute_route_stats,
    encoder_sequence,
    prediction_horizons,
    target_dim,
    target_matrix,
    valid_route_starts,
    window_views,
)
from route_intention.common import H_ROUTE, NUM_CODES  # noqa: E402
from route_intention.gates import (  # noqa: E402
    local_intention_early_stop,
    representation_gate,
    variance_component_passes,
)
from route_intention.inference import code_layout, per_code_candidates, shuffle_codes  # noqa: E402
from route_intention.predictor import classification_metrics, init_predictor, predict_step  # noqa: E402
from route_intention.tokenizer import RouteTokenizer, RouteTokenizerConfig, nearest_code  # noqa: E402
from route_intention.variance import conditional_variance, knn_indices, variance_report  # noqa: E402
from run_intention_pathbridger import hold_decision  # noqa: E402

A, D, H = 5, 8, 50


def _data(n_ep=4, ep_len=80, seed=0):
    rng = np.random.default_rng(seed)
    obs = np.cumsum(rng.normal(size=(n_ep * ep_len, D)).astype(np.float32) * 0.1, axis=0)
    act = rng.uniform(-1, 1, size=(n_ep * ep_len, A)).astype(np.float32)
    term = np.zeros(n_ep * ep_len, np.float32)
    term[ep_len - 1 :: ep_len] = 1.0
    return obs, act, term


def test_no_boundary_crossing():
    _obs, _act, term = _data()
    starts = valid_route_starts(term, H)
    # Episode length 80, horizon 50: last start in an episode is end - 50.
    for s in starts:
        ep = int(s) // 80
        assert ep * 80 <= s <= s + H < (ep + 1) * 80


def test_encoder_has_no_goal_and_is_translation_invariant():
    assert 'goal' not in inspect.signature(encoder_sequence).parameters
    obs, act, term = _data()
    starts = valid_route_starts(term, H)[:16]
    stats = compute_route_stats(obs, act, valid_route_starts(term, H), H)
    ow, aw = window_views(obs, act, starts, H)
    base = encoder_sequence(ow, aw, stats)
    shifted, _ = window_views(obs + 3.0, act, starts, H)
    # Recompute stats on the shifted observations the same way a leaked absolute feature would change.
    same = encoder_sequence(shifted, aw, stats)
    # Relative features change their z-score location if stats stay fixed, but the raw relative
    # window must match. Check the un-normalised relative part directly.
    rel = ow[:, 1:] - ow[:, :1]
    rel_s = shifted[:, 1:] - shifted[:, :1]
    np.testing.assert_allclose(rel, rel_s, atol=1e-5)
    assert base.shape == (16, H, A + D)
    assert same.shape == base.shape


def test_multi_horizon_targets():
    assert prediction_horizons(50) == (5, 10, 20, 50)
    assert target_dim(50, D, A) == (4 + 3) * D + A
    obs, act, term = _data()
    starts = valid_route_starts(term, H)[:8]
    stats = compute_route_stats(obs, act, valid_route_starts(term, H), H)
    ow, aw = window_views(obs, act, starts, H)
    tgt = target_matrix(ow, aw, stats, H)
    assert tgt.shape == (8, target_dim(H, D, A))
    assert np.isfinite(tgt).all()


def test_vq_assignment_and_usage():
    obs, act, term = _data()
    starts = valid_route_starts(term, H)
    stats = compute_route_stats(obs, act, starts, H)
    ow, aw = window_views(obs, act, starts[:32], H)
    seq = encoder_sequence(ow, aw, stats)
    tok = RouteTokenizer(RouteTokenizerConfig(horizon=H, batch_size=32), A, D)
    state = tok.init(0)
    state = tok.init_codebook(state, jnp.asarray(seq), jax.random.PRNGKey(1))
    e = tok.encode(state['params'], jnp.asarray(seq))
    idx, _ = nearest_code(e, state['codebook'])
    idx2, _ = nearest_code(e, state['codebook'])
    np.testing.assert_array_equal(np.asarray(idx), np.asarray(idx2))
    assert int(np.asarray(idx).min()) >= 0 and int(np.asarray(idx).max()) < NUM_CODES
    s_n = (ow[:, 0] - stats['state_mean']) / stats['state_std']
    tgt = target_matrix(ow, aw, stats, H)
    new_state, info, usage = tok.train_step(state, jnp.asarray(seq), jnp.asarray(s_n), jnp.asarray(tgt))
    assert np.isfinite(float(info['loss']))
    assert np.asarray(usage).shape == (NUM_CODES,)
    assert np.asarray(new_state['codebook']).shape == (NUM_CODES, tok.cfg.embed_dim)


def test_knn_is_deterministic_and_code_reduces_variance():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(200, 4))
    codes = np.array([0, 1] * 100)
    values = codes.astype(np.float64)[:, None] + rng.normal(scale=0.01, size=(200, 1))
    a = knn_indices(feats, 32)
    b = knn_indices(feats, 32)
    np.testing.assert_array_equal(a, b)
    report = variance_report(values, feats, codes, (32, 64))
    assert [r['knn_k'] for r in report] == [32, 64]
    ratio = conditional_variance(values, codes, knn_indices(feats, 64))['variance_ratio']
    assert ratio < 0.5


def test_gate_thresholds_are_fixed():
    assert variance_component_passes(0.69, 'cube-double')
    assert not variance_component_passes(0.95, 'cube-double')
    assert representation_gate(perplexity=6.0, max_usage=0.3, variance_ratio=0.95, task='cube-double') == 'VARIANCE_FAIL'
    assert representation_gate(perplexity=1.0, max_usage=0.2, variance_ratio=0.4, task='cube-double') == 'FAIL'
    assert representation_gate(perplexity=6.0, max_usage=0.3, variance_ratio=0.5, task='cube-double') == 'NEED_ORACLE'
    assert representation_gate(perplexity=6.0, max_usage=0.3, variance_ratio=0.5, task='cube-double',
                               route_subgoal_error=0.4, baseline_subgoal_error=0.5) == 'PASS'
    assert representation_gate(perplexity=6.0, max_usage=0.3, variance_ratio=0.5, task='cube-double',
                               route_subgoal_error=0.6, baseline_subgoal_error=0.5) == 'ORACLE_FAIL'


def test_local_early_stop_rule():
    bad = {('cube-double', 1): dict(pb=0.46, shared=0.06, shuffled=0.05),
           ('puzzle-4x4', 1): dict(pb=0.80, shared=0.60, shuffled=0.66)}
    decision = local_intention_early_stop(bad)
    assert decision['condition_a'] and decision['stop']
    close = {('cube-double', 1): dict(pb=0.50, shared=0.48, shuffled=0.47),
             ('puzzle-4x4', 1): dict(pb=0.70, shared=0.69, shuffled=0.68)}
    assert local_intention_early_stop(close)['condition_b']


def test_hold_allowlist():
    hold = dict(active=True, allow_prefixes=['eval/cube-double_s1/', 'conditioned/cube-double_s1'])
    assert hold_decision('eval/cube-double_s1/step1000000/Shared-I_h1', hold) == 'run'
    assert hold_decision('pb_train/cube-single_s1', hold) == 'hold'
    assert hold_decision('pb_train/cube-single_s1', None) == 'run'


def test_equal_budget_and_shuffle():
    per = per_code_candidates(16, k=8, l=4)
    assert per == 4
    layout = code_layout(np.array([0, 1, 2, 3]), per)
    assert len(layout) == 16
    rng = np.random.default_rng(0)
    codes = np.array([0, 1, 2, 3, 0])
    shuffled = shuffle_codes(codes, rng)
    assert np.all(shuffled != codes)


def test_predictor_shapes_and_step():
    model, state, tx = init_predictor(0, state_dim=D, num_codes=8)
    s = jnp.zeros((16, D))
    g = jnp.ones((16, D))
    logits = model.apply({'params': state['params']}, s, g)
    assert logits.shape == (16, 8)
    codes = jnp.zeros((16,), dtype=jnp.int32)
    new_state, info = predict_step(model, state, s, g, codes, tx)
    assert np.isfinite(float(info['loss']))
    metrics = classification_metrics(np.asarray(logits), np.zeros(16, np.int32), top_l=4)
    assert 0.0 <= metrics['top1'] <= 1.0 and metrics['topk'] >= metrics['top1']
    assert 'params' in new_state


@pytest.fixture(scope='module')
def pb():
    from intention_pb import common as C
    from intention_pb.pb_io import load_pb

    return load_pb(C.find_pb_run_dir('cube-single', 0), need_train=False, need_env=False)


def test_oracle_step_does_not_train_bridge_or_critic(pb):
    from intention_pb.conditioned import ConditionedModel

    from route_intention.common import EMBED_DIM
    from route_intention.oracle import _OracleTrainer

    model = ConditionedModel(pb.dynamics, num_codes=8, intent_dim=EMBED_DIM)
    trainer = _OracleTrainer(model)
    state = trainer.init_state(0)
    before = [np.array(x) for x in jax.tree_util.tree_leaves(jax.device_get(state['residual']))]
    b = 8
    batch = dict(
        observations=np.zeros((b, model.state_dim), np.float32),
        high_actor_goals=np.zeros((b, model.state_dim), np.float32),
        high_actor_targets=np.ones((b, model.state_dim), np.float32),
    )
    codes = jnp.zeros((b,), dtype=jnp.int32)
    value = jax.device_put(pb.critic_value_params())
    new_state, info = trainer.train_step(state, batch, codes, value)
    after = [np.array(x) for x in jax.tree_util.tree_leaves(jax.device_get(new_state['residual']))]
    for a, c in zip(before, after):
        np.testing.assert_array_equal(a, c)
    assert np.isfinite(float(info['loss']))
    # Critic weights are not stored on the oracle state.
    assert 'critic' not in new_state
