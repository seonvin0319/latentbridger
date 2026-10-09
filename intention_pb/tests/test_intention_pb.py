"""Unit tests for the intention experiment. Run: ``JAX_PLATFORMS=cpu python -m pytest intention_pb/tests -q``."""

from __future__ import annotations

import inspect
import json
import os
import time

import numpy as np
import pytest

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import intention_pb  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from intention_pb import common as C  # noqa: E402
from intention_pb.aggregate import mean_std, paired_deltas, rule_based_verdict  # noqa: E402
from intention_pb.conditioned import ConditionedModel  # noqa: E402
from intention_pb.policy import make_planner, shuffled_code  # noqa: E402
from intention_pb.tokenizer import Tokenizer, TokenizerConfig, chunk_features, compute_stats, gather_chunks, load_tokenizer, train_tokenizer  # noqa: E402

A, D, H = 5, 28, 5


def _synthetic(n_ep=6, ep_len=60, seed=0):
    rng = np.random.default_rng(seed)
    obs = np.cumsum(rng.normal(size=(n_ep * ep_len, D)).astype(np.float32) * 0.1, axis=0)
    act = rng.uniform(-1, 1, size=(n_ep * ep_len, A)).astype(np.float32)
    term = np.zeros(n_ep * ep_len, np.float32)
    term[ep_len - 1 :: ep_len] = 1
    return dict(observations=obs, actions=act, terminals=term)


@pytest.fixture(scope='module')
def tok_setup():
    data = _synthetic()
    from intention_pb.pb_io import chunk_valid_starts

    starts = chunk_valid_starts(data['terminals'], H)
    stats = {k: jnp.asarray(v) for k, v in compute_stats(data['observations'], data['actions'], starts).items()}
    tok = Tokenizer(TokenizerConfig(batch_size=64), A, D)
    state = tok.init(0)
    act, obs = gather_chunks(data['observations'], data['actions'], starts[:64], H)
    feats, _, _, _ = chunk_features(act, obs, stats)
    state = tok.init_codebook(state, feats, jax.random.PRNGKey(1))
    return tok, state, stats, data, starts


@pytest.fixture(scope='module')
def pb():
    from intention_pb.pb_io import load_pb

    return load_pb(C.find_pb_run_dir('cube-single', 0), need_train=False, need_env=False)


# ---------------------------------------------------------------- tokenizer
def test_vq_shapes(tok_setup):
    tok, state, stats, data, starts = tok_setup
    act, obs = gather_chunks(data['observations'], data['actions'], starts[:32], H)
    feats, a_n, d_n, s_n = chunk_features(act, obs, stats)
    assert feats.shape == (32, H * (A + D)) and a_n.shape == (32, H, A) and d_n.shape == (32, H, D) and s_n.shape == (32, D)
    e = tok.encode(state['params'], feats)
    assert e.shape == (32, C.INTENT_DIM)
    assert state['codebook'].shape == (C.NUM_CODES, C.INTENT_DIM)
    idx = tok.assign(state['params'], state['codebook'], act, obs, stats)
    assert idx.shape == (32,) and int(idx.min()) >= 0 and int(idx.max()) < C.NUM_CODES
    new_state, info, usage = tok.train_step(state, act, obs, stats)
    assert np.isfinite(float(info['loss'])) and usage.shape == (C.NUM_CODES,)


def test_deterministic_assignment(tok_setup):
    tok, state, stats, data, starts = tok_setup
    act, obs = gather_chunks(data['observations'], data['actions'], starts[:50], H)
    a1 = np.asarray(tok.assign(state['params'], state['codebook'], act, obs, stats))
    a2 = np.asarray(jax.jit(lambda p, cb, a, o: tok.assign(p, cb, a, o, stats))(state['params'], state['codebook'], act, obs))
    np.testing.assert_array_equal(a1, a2)


def test_no_absolute_goal_leakage(tok_setup):
    tok, state, stats, data, starts = tok_setup
    params = inspect.signature(tok.assign).parameters
    assert not any('goal' in name for name in params)
    act, obs = gather_chunks(data['observations'], data['actions'], starts[:50], H)
    feats, _, _, _ = chunk_features(act, obs, stats)
    feats_shift, _, _, _ = chunk_features(act, obs + 123.0, stats)  # absolute position / endpoint changes, deltas do not
    np.testing.assert_allclose(np.asarray(feats), np.asarray(feats_shift), atol=1e-3)
    a1 = tok.assign(state['params'], state['codebook'], act, obs, stats)
    a2 = tok.assign(state['params'], state['codebook'], act, obs + 123.0, stats)
    np.testing.assert_array_equal(np.asarray(a1), np.asarray(a2))


def test_tokenizer_resume(tmp_path):
    data = _synthetic(seed=3)
    cfg = TokenizerConfig(batch_size=32)
    train_tokenizer(out_dir=tmp_path, train=data, val=data, seed=0, total_steps=4, save_steps=(2, 4), log_every=2, cfg=cfg)
    _, s4, _, _ = load_tokenizer(tmp_path / 'checkpoints' / 'tokenizer_4.pkl')
    assert int(s4['step']) == 4
    train_tokenizer(out_dir=tmp_path, train=data, val=data, seed=0, total_steps=6, save_steps=(2, 4, 6), log_every=2, cfg=cfg)
    _, s6, _, _ = load_tokenizer(tmp_path / 'checkpoints' / 'tokenizer_6.pkl')
    assert int(s6['step']) == 6  # resumed from 4, not restarted
    lines = [json.loads(x) for x in (tmp_path / 'train_log.jsonl').read_text().splitlines()]
    assert [r['step'] for r in lines] == [2, 4, 6]


# ---------------------------------------------------------------- conditioned model
def _obs_goal(pb, n=6):
    obs = np.asarray(pb.val['observations'][:n], np.float32)
    goal = np.asarray(pb.val['observations'][100 : 100 + n], np.float32)
    return obs, goal


def test_conditioned_forward_shapes(pb):
    m = ConditionedModel(pb.dynamics, num_codes=C.NUM_CODES)
    st = m.init(0)
    obs, goal = _obs_goal(pb)
    cands, codes = m.sample_candidates(st['params']['subgoal'], obs, goal, jax.random.PRNGKey(0), temperature=1.0, per_code=2)
    assert cands.shape == (len(obs), 16, obs.shape[1])
    tr = m.bridge(st['params']['residual'], obs, cands[:, 0], jnp.zeros((len(obs),), jnp.int32))
    assert tr.shape == (len(obs), m.N + 1, obs.shape[1])
    np.testing.assert_allclose(np.asarray(tr[:, 0]), obs, atol=1e-5)
    np.testing.assert_allclose(np.asarray(tr[:, -1]), np.asarray(cands[:, 0]), atol=1e-4)
    with pytest.raises(ValueError):
        m.bridge(st['params']['residual'], obs, cands[:, 0], None)


def test_intention_disabled_reproduces_pb(pb):
    m0 = ConditionedModel(pb.dynamics, num_codes=0)
    p = m0.params_from_pb()
    obs, goal = _obs_goal(pb)
    noise = np.random.default_rng(0).normal(size=obs.shape).astype(np.float32)
    raw_pb = pb.dynamics._subgoal_flow_sample_raw(jnp.asarray(obs), jnp.asarray(goal), jnp.asarray(noise))
    raw_m = m0.flow_sample_raw(p['subgoal'], obs, goal, noise, None)
    np.testing.assert_allclose(np.asarray(raw_m), np.asarray(raw_pb), atol=1e-5)
    z = obs + 0.05
    tr_pb = pb.dynamics.plan(jnp.asarray(obs), jnp.asarray(z))['trajectory']
    tr_m = m0.bridge(p['residual'], obs, z, None)
    np.testing.assert_allclose(np.asarray(tr_m), np.asarray(tr_pb), atol=1e-5)
    # Same shapes / API as PB candidate sampling.
    c_pb, _ = pb.dynamics.sample_subgoal_candidates(jnp.asarray(obs), jnp.asarray(goal), jax.random.PRNGKey(0), num_candidates=4, include_mean=False)
    c_m = m0.sample_with_codes(p['subgoal'], obs, goal, None, jax.random.PRNGKey(0), temperature=1.0, num=4)
    assert c_pb.shape == c_m.shape
    np.testing.assert_allclose(np.asarray(c_m), np.asarray(c_pb), atol=1e-5)


def test_candidate_budget_and_enumeration(pb):
    m = ConditionedModel(pb.dynamics, num_codes=C.NUM_CODES)
    st = m.init(0)
    obs, goal = _obs_goal(pb, 1)
    np.testing.assert_array_equal(np.asarray(m.candidate_codes(2)), np.repeat(np.arange(8), 2))
    outs = {}
    for meth in ('PB', 'I-SG', 'Shared-I', 'Shuffled-I'):
        pl = make_planner(meth, pb, temperature=0.5, num_candidates=C.NUM_CANDIDATES, cond_model=m, cond_params=st['params'])
        outs[meth] = jax.device_get(pl(obs[0], goal[0], jax.random.PRNGKey(3)))
        assert outs[meth]['scores'].shape == (C.NUM_CANDIDATES,)
        assert outs[meth]['actions'].shape == (pb.idm_horizon, 5)
    assert sorted(set(outs['Shared-I']['cand_codes'].tolist())) == list(range(8))
    assert np.all(outs['PB']['cand_codes'] == -1)
    with pytest.raises(ValueError):
        make_planner('I-SG', pb, temperature=0.5, num_candidates=12, cond_model=m, cond_params=st['params'])


def test_shared_and_shuffled_share_z(pb):
    m = ConditionedModel(pb.dynamics, num_codes=C.NUM_CODES)
    st = m.init(1)
    obs, goal = _obs_goal(pb, 3)
    pls = {k: make_planner(k, pb, temperature=0.5, cond_model=m, cond_params=st['params']) for k in ('I-SG', 'Shared-I', 'Shuffled-I')}
    for i in range(3):
        o = {k: jax.device_get(pl(obs[i], goal[i], jax.random.PRNGKey(i))) for k, pl in pls.items()}
        np.testing.assert_array_equal(o['Shared-I']['subgoal'], o['Shuffled-I']['subgoal'])
        np.testing.assert_array_equal(o['Shared-I']['subgoal'], o['I-SG']['subgoal'])
        assert o['Shared-I']['code'] == o['Shuffled-I']['code'] == o['I-SG']['code']
        assert o['Shared-I']['exec_code'] == o['Shared-I']['code']
        assert o['Shuffled-I']['exec_code'] != o['Shuffled-I']['code']
        assert o['I-SG']['exec_code'] == -1


def test_shuffled_never_matches():
    for K in (2, 8):
        c = jnp.arange(K)
        cs = shuffled_code(c, K)
        assert bool(jnp.all(cs != c)) and bool(jnp.all((cs >= 0) & (cs < K)))


def test_frozen_components_receive_no_updates(pb):
    from utils.datasets import Dataset, PathHGCDataset

    m = ConditionedModel(pb.dynamics, num_codes=C.NUM_CODES)
    st = m.init(0)
    ds = PathHGCDataset(Dataset.create(**pb.val), pb.dynamics_config)
    np.random.seed(0)
    b = ds.sample(16)
    batch = {k: np.asarray(b[k], np.float32) for k in ('observations', 'high_actor_goals', 'high_actor_targets', 'trajectory_segment')}
    codes = np.arange(16, dtype=np.int32) % 8
    dyn_before = jax.tree_util.tree_map(np.asarray, pb.dynamics.network.params)
    critic_before = jax.tree_util.tree_map(np.asarray, pb.critic.network.params)
    new, info = m.train_step(st, batch, codes, pb.critic_value_params())
    assert set(new['params']) == {'subgoal', 'residual'}  # no tokenizer / IDM / critic parameters are trainable
    assert np.isfinite(float(info['loss']))
    changed = jax.tree_util.tree_leaves(jax.tree_util.tree_map(lambda a, b: bool(np.any(np.asarray(a) != np.asarray(b))), st['params'], new['params']))
    assert any(changed)
    for before, after in ((dyn_before, pb.dynamics.network.params), (critic_before, pb.critic.network.params)):
        jax.tree_util.tree_map(lambda a, b: np.testing.assert_array_equal(a, np.asarray(b)), before, after)
    # Teacher codes are integer inputs: there is no gradient path into the tokenizer.
    assert np.asarray(codes).dtype.kind == 'i'


# ---------------------------------------------------------------- launcher resume / skip
def test_launcher_resume_skip(tmp_path, monkeypatch):
    import run_intention_pathbridger as R

    monkeypatch.setattr(C, 'EXP_ROOT', tmp_path)
    marker_done = tmp_path / 'a.done'
    marker_done.write_text('x')
    out_b = tmp_path / 'b.done'
    jobs = [
        R._J(name='x/a', kind='cpu', args=['true'], priority=(0,), done_fn=marker_done.exists, deps_fn=lambda: 'ready'),
        R._J(name='x/b', kind='cpu', args=['bash', '-c', f'touch {out_b}'], priority=(1,), done_fn=out_b.exists, deps_fn=lambda: 'ready'),
        R._J(name='x/c', kind='cpu', args=['false'], priority=(2,), done_fn=lambda: False, deps_fn=lambda: 'ready'),
        R._J(name='x/d', kind='cpu', args=['true'], priority=(3,), done_fn=lambda: False, deps_fn=lambda: 'skip:collapsed'),
        R._J(name='x/e', kind='cpu', args=['true'], priority=(4,), done_fn=lambda: False, deps_fn=lambda: 'wait'),
    ]
    L = R.Launcher(jobs, max_gpu_jobs=0, max_cpu_threads=2, rerun_failed=False, poll=0.05)
    snap = L.run(aggregate_every=1e9)
    assert snap['done'] == 2 and snap['failed'] == 1 and snap['skipped'] == 1 and snap['wait'] == 1
    assert L.status(jobs[0]) == {}  # already-complete job was never launched
    st_c = L.status(jobs[2])
    assert st_c['state'] == 'failed' and st_c['exit_code'] == 1 and st_c['attempts'] == 1
    # Second launcher: failed job is not retried without --rerun-failed ...
    L2 = R.Launcher(jobs, max_gpu_jobs=0, max_cpu_threads=2, rerun_failed=False, poll=0.05)
    L2.run(aggregate_every=1e9)
    assert L2.status(jobs[2])['attempts'] == 1
    # ... and is retried (only it) with --rerun-failed.
    L3 = R.Launcher(jobs, max_gpu_jobs=0, max_cpu_threads=2, rerun_failed=True, poll=0.05)
    L3.run(aggregate_every=1e9)
    assert L3.status(jobs[2])['attempts'] == 2
    assert L3.status(jobs[1])['attempts'] == 1


# ---------------------------------------------------------------- aggregation math
def test_aggregation_math():
    m, s = mean_std([0.2, 0.4, 0.6])
    assert m == pytest.approx(0.4) and s == pytest.approx(0.2)
    m1, s1 = mean_std([0.5])
    assert m1 == 0.5 and np.isnan(s1)
    scores = {('t', 0, 'PB', 1): 0.5, ('t', 0, 'I-SG', 1): 0.6, ('t', 0, 'Shared-I', 1): 0.8, ('t', 0, 'Shuffled-I', 1): 0.3}
    d = {r['pair']: r['delta'] for r in paired_deltas(scores, ['t'], [0], [1])}
    assert d == pytest.approx({'I-SG - PB': 0.1, 'Shared-I - PB': 0.3, 'Shared-I - I-SG': 0.2, 'Shared-I - Shuffled-I': 0.5})
    dm = {('t', 1, k): v for k, v in d.items()}
    assert rule_based_verdict(dm, True)[0] == 'A'
    dm_neg = {('t', 1, k): -0.2 for k in d}
    assert rule_based_verdict(dm_neg, False)[0] == 'D'
    dm_zero = {('t', 1, k): 0.0 for k in d}
    assert rule_based_verdict(dm_zero, False)[0] == 'C'
