"""Offline diagnostics on the held-out OGBench ``-val`` split (no environment interaction)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from intention_pb.common import INTENT_HORIZON, NUM_CANDIDATES, atomic_write_json, refuse_overwrite, task_info

KNN_K = 64


def _sample_val(pb, num: int, seed: int) -> dict:
    from utils.datasets import Dataset, PathHGCDataset

    ds = PathHGCDataset(Dataset.create(**pb.val), pb.dynamics_config)
    rs = np.random.get_state()
    np.random.seed(seed)
    b = ds.sample(num)
    np.random.set_state(rs)
    idx = np.asarray(b['trajectory_indices'])
    starts = np.asarray(b['trajectory_start_indices'])
    h = INTENT_HORIZON
    unpadded = np.all(np.diff(idx[:, : h + 1], axis=1) == 1, axis=1)
    acts = np.asarray(pb.val['actions'], np.float32)[starts[:, None] + np.arange(h)[None, :]]
    return dict(
        obs=np.asarray(b['observations'], np.float32),
        goal=np.asarray(b['high_actor_goals'], np.float32),
        target=np.asarray(b['high_actor_targets'], np.float32),
        seg=np.asarray(b['trajectory_segment'], np.float32),
        starts=starts,
        unpadded=unpadded,
        actions=acts,
    )


def _batched(fn, n: int, bs: int, *arrays):
    outs = []
    for i in range(0, n, bs):
        outs.append(jax.device_get(fn(*[a[i : i + bs] for a in arrays])))
    if isinstance(outs[0], tuple):
        return tuple(np.concatenate([o[j] for o in outs]) for j in range(len(outs[0])))
    return np.concatenate(outs)


def _knn(features: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    out = np.empty((len(queries), k), dtype=np.int64)
    for i in range(0, len(queries), 256):
        q = features[queries[i : i + 256]]
        d2 = (q ** 2).sum(1)[:, None] - 2 * q @ features.T + (features ** 2).sum(1)[None, :]
        out[i : i + 256] = np.argsort(d2, axis=1)[:, :k]
    return out


def _pooled_within(values: np.ndarray, groups: np.ndarray) -> float:
    num, den = 0.0, 0
    for gval in np.unique(groups):
        m = groups == gval
        if m.sum() >= 2:
            num += values[m].var(axis=0, ddof=1).mean() * (m.sum() - 1)
            den += m.sum() - 1
    return num / den if den > 0 else float('nan')


def conditional_variance(values: np.ndarray, codes: np.ndarray, neigh: np.ndarray, rng: np.random.Generator) -> dict:
    """Mean neighbourhood variance of ``values`` without / with code conditioning (+ permutation baseline)."""
    tot, cond, perm = [], [], []
    for nb in neigh:
        v = values[nb]
        c = codes[nb]
        tot.append(v.var(axis=0, ddof=1).mean())
        cond.append(_pooled_within(v, c))
        perm.append(_pooled_within(v, rng.permutation(c)))
    tot, cond, perm = map(lambda x: float(np.nanmean(x)), (tot, cond, perm))
    return dict(var_uncond=tot, var_cond=cond, var_cond_perm=perm, ratio_cond=cond / tot, ratio_perm=perm / tot)


def run_diagnostics(*, task: str, seed: int, out_dir: Path, num_samples: int = 4096, num_queries: int = 1000,
                    num_diversity: int = 512, step: int = 1_000_000) -> dict:
    from intention_pb.common import cond_ckpt, find_pb_run_dir, tokenizer_ckpt, TOKENIZER_STEPS
    from intention_pb.conditioned import load_conditioned, teacher_codes_for
    from intention_pb.pb_io import load_pb
    from intention_pb.policy import pb_with_temperature, shuffled_code
    from intention_pb.tokenizer import evaluate_tokenizer, load_tokenizer
    from intention_pb.pb_io import chunk_valid_starts

    out_dir = Path(out_dir)
    refuse_overwrite(out_dir / 'diagnostics.json')
    t0 = time.time()
    run_dir = find_pb_run_dir(task, seed)
    pb = load_pb(run_dir, need_train=False, need_env=False)
    model, state, meta = load_conditioned(cond_ckpt(task, seed, step), pb.dynamics)
    if Path(meta['pb_run_dir']).resolve() != Path(run_dir).resolve():
        raise ValueError('Conditioned checkpoint / PB run mismatch.')
    tok_path = Path(meta['tokenizer'])
    if tok_path.resolve().parent != tokenizer_ckpt(task, seed, TOKENIZER_STEPS).resolve().parent or not tok_path.is_file():
        raise ValueError(f'Unexpected tokenizer {tok_path}')
    tok, tok_state, tok_stats, _ = load_tokenizer(tok_path)
    K = model.num_codes
    params = state['params']
    temp_eval = float(task_info(task)['temperature'])
    per_code = NUM_CANDIDATES // K
    dyn = pb.dynamics
    critic = pb.critic
    cparams = critic.network.params
    idm_h = pb.idm_horizon

    # --- A. tokenizer on held-out data
    val_obs = np.asarray(pb.val['observations'], np.float32)
    val_act = np.asarray(pb.val['actions'], np.float32)
    tok_val = evaluate_tokenizer(tok, tok_state['params'], tok_state['codebook'], tok_stats, val_obs, val_act,
                                 chunk_valid_starts(pb.val['terminals'], tok.cfg.horizon))

    S = _sample_val(pb, num_samples, seed=12345 + seed)
    M = len(S['obs'])
    c_true = teacher_codes_for(tok, tok_state, tok_stats, val_obs, val_act, S['starts'])
    c_shuf = np.asarray(shuffled_code(c_true, K))
    rel = S['target'] - S['obs']
    z_std = np.maximum(rel.std(0), 1e-3)
    step_rel = S['seg'][:, 1 : INTENT_HORIZON + 1] - S['obs'][:, None]
    path_std = np.maximum(step_rel.reshape(-1, step_rel.shape[-1]).std(0), 1e-3)

    def zerr(cands, target):
        d = (cands - target[:, None, :]) / z_std
        return np.sqrt((d ** 2).mean(-1))  # [B, N] normalised RMS error

    # --- B. subgoal prediction
    res: dict[str, Any] = dict(task=task, seed=int(seed), step=int(step), num_samples=M, eval_temperature=temp_eval,
                               tokenizer_val=tok_val, teacher_code_usage=(np.bincount(c_true, minlength=K) / M).tolist())
    keys = jax.random.split(jax.random.PRNGKey(777 + seed), 4)
    sub_rows = {}
    for tname, temp in (('eval', temp_eval), ('t1', 1.0)):
        dyn_t = pb_with_temperature(dyn, temp)
        pb_fn = jax.jit(lambda o, g: dyn_t.sample_subgoal_candidates(o, g, keys[0], num_candidates=NUM_CANDIDATES, include_mean=False)[0])
        pb_c = _batched(pb_fn, M, 512, S['obs'], S['goal'])
        enum_fn = jax.jit(lambda o, g: model.sample_candidates(params['subgoal'], o, g, keys[1], temperature=temp, per_code=per_code)[0])
        en_c = _batched(enum_fn, M, 512, S['obs'], S['goal'])
        teach_fn = jax.jit(lambda o, g, c: model.sample_with_codes(params['subgoal'], o, g, c, keys[2], temperature=temp, num=NUM_CANDIDATES))
        te_c = _batched(teach_fn, M, 512, S['obs'], S['goal'], c_true)
        score_fn = jax.jit(lambda o, z, g: critic.score_transitive_subgoals(o, z, g, network_params=cparams))
        pb_s = _batched(score_fn, M, 512, S['obs'], pb_c, S['goal'])
        en_s = _batched(score_fn, M, 512, S['obs'], en_c, S['goal'])
        e_pb, e_en, e_te = zerr(pb_c, S['target']), zerr(en_c, S['target']), zerr(te_c, S['target'])
        ar = np.arange(M)
        sel_pb, sel_en = pb_s.argmax(1), en_s.argmax(1)
        codes_layout = np.repeat(np.arange(K), per_code)
        per_code_err = np.stack([e_en[:, codes_layout == k].mean(1) for k in range(K)], 1)  # [M, K]
        match = per_code_err[ar, c_true].mean()
        off = (per_code_err.sum(1) - per_code_err[ar, c_true]) / (K - 1)
        sub_rows[tname] = dict(
            temperature=temp,
            pb_mean_err=float(e_pb.mean()), pb_selected_err=float(e_pb[ar, sel_pb].mean()), pb_best_of_n_err=float(e_pb.min(1).mean()),
            isg_teacher_mean_err=float(e_te.mean()), isg_teacher_best_of_n_err=float(e_te.min(1).mean()),
            isg_enum_mean_err=float(e_en.mean()), isg_enum_selected_err=float(e_en[ar, sel_en].mean()),
            isg_enum_best_of_n_err=float(e_en.min(1).mean()),
            per_code_err_matching=float(match), per_code_err_nonmatching=float(off.mean()),
            per_code_err_by_true_code=[float(per_code_err[c_true == k, k].mean()) if np.any(c_true == k) else float('nan') for k in range(K)],
            selected_code_matches_teacher=float(np.mean(codes_layout[sel_en] == c_true)),
            selected_code_histogram=(np.bincount(codes_layout[sel_en], minlength=K) / M).tolist(),
        )
    res['subgoal'] = sub_rows

    # --- C. bridge with true z: true c vs shuffled c vs PB bridge
    mask = S['unpadded']
    h = INTENT_HORIZON
    plan_pb = jax.jit(lambda o, z: dyn.plan(o, z)['trajectory'])
    br = jax.jit(lambda o, z, c: model.bridge(params['residual'], o, z, c))
    idm = jax.jit(lambda tr: dyn._idm_actions_from_trajectories(tr, idm_h))
    true_prefix = S['seg'][:, 1 : h + 1]
    bridge_rows = {}
    trajs = {'pb_bridge': _batched(plan_pb, M, 512, S['obs'], S['target']),
             'true_c': _batched(br, M, 512, S['obs'], S['target'], c_true),
             'shuffled_c': _batched(br, M, 512, S['obs'], S['target'], c_shuf)}
    for name, tr in trajs.items():
        pre = tr[:, 1 : h + 1]
        d = pre - true_prefix
        acts = _batched(idm, M, 512, tr)
        bridge_rows[name] = dict(
            path_mse=float((d[mask] ** 2).mean()),
            path_nmse=float(((d[mask] / path_std) ** 2).mean()),
            first_step_mse=float((d[mask][:, 0] ** 2).mean()),
            full_prefix_l2=float(np.linalg.norm(d[mask], axis=-1).mean()),
            idm_action_mse=float(((acts[mask] - S['actions'][mask]) ** 2).mean()),
            first_action_mse=float(((acts[mask][:, 0] - S['actions'][mask][:, 0]) ** 2).mean()),
        )
    res['bridge'] = bridge_rows
    res['bridge_num_unpadded'] = int(mask.sum())
    res['bridge_gap_shuffled_minus_true'] = {k: bridge_rows['shuffled_c'][k] - bridge_rows['true_c'][k] for k in bridge_rows['true_c']}

    # --- D. conditional multimodality (kNN neighbourhoods in standardised [s, g])
    rng = np.random.default_rng(99 + seed)
    feats = np.concatenate([S['obs'], S['goal']], 1)
    feats = (feats - feats.mean(0)) / np.maximum(feats.std(0), 1e-3)
    queries = rng.choice(M, size=min(num_queries, M), replace=False)
    neigh = _knn(feats, queries, KNN_K)
    act_std = np.maximum(S['actions'].reshape(-1, S['actions'].shape[-1]).std(0), 1e-3)
    var_rows = dict(
        knn_k=KNN_K, num_queries=int(len(queries)),
        subgoal_displacement=conditional_variance(rel / z_std, c_true, neigh, rng),
        first_action=conditional_variance(S['actions'][:, 0] / act_std, c_true, neigh, rng),
        action_chunk=conditional_variance((S['actions'] / act_std).reshape(M, -1), c_true, neigh, rng),
        prefix_path=conditional_variance((step_rel / path_std).reshape(M, -1), c_true, neigh, rng),
    )
    res['conditional_variance'] = var_rows

    # --- E. between-code vs within-code subgoal diversity
    Dn = min(num_diversity, M)
    div = {}
    for tname, temp in (('eval', temp_eval), ('t1', 1.0)):
        fn = jax.jit(lambda o, g: model.sample_candidates(params['subgoal'], o, g, keys[3], temperature=temp, per_code=2)[0])
        cands = _batched(fn, Dn, 256, S['obs'][:Dn], S['goal'][:Dn]).reshape(Dn, K, 2, -1) / z_std
        within = np.linalg.norm(cands[:, :, 0] - cands[:, :, 1], axis=-1).mean()
        means = cands.mean(2)
        pd = np.linalg.norm(means[:, :, None] - means[:, None, :], axis=-1)
        between = pd.sum((1, 2)) / (K * (K - 1))
        div[tname] = dict(temperature=temp, within_code=float(within), between_code=float(between.mean()))
    res['subgoal_diversity'] = div

    # --- F. bridge controllability for fixed (s, z)
    o = S['obs'][:Dn]
    z = S['target'][:Dn]
    all_tr = np.stack([_batched(br, Dn, 256, o, z, np.full((Dn,), k, np.int32)) for k in range(K)], 1)  # [Dn, K, N+1, D]
    pre = all_tr[:, :, 1 : h + 1] / path_std
    pdiff = np.linalg.norm((pre[:, :, None] - pre[:, None, :]).reshape(Dn, K, K, -1), axis=-1).sum((1, 2)) / (K * (K - 1))
    motion = np.linalg.norm((all_tr[:, :, 1 : h + 1] - o[:, None, None, :]) / path_std, axis=-1).mean()
    acts = np.stack([_batched(idm, Dn, 256, all_tr[:, k]) for k in range(K)], 1)
    adiff = np.linalg.norm((acts[:, :, None] - acts[:, None, :]).reshape(Dn, K, K, -1), axis=-1).sum((1, 2)) / (K * (K - 1))
    res['bridge_controllability'] = dict(pairwise_prefix_dist=float(pdiff.mean()), prefix_motion_norm=float(motion),
                                         relative=float(pdiff.mean() / max(motion, 1e-8)), pairwise_idm_action_dist=float(adiff.mean()))
    res['runtime_s'] = time.time() - t0
    res['conditioned_ckpt'] = str(cond_ckpt(task, seed, step))
    atomic_write_json(out_dir / 'diagnostics.json', res)
    return res
