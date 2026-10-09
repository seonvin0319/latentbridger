"""Intention-conditioned subgoal flow ``p(z | s, g, c)`` and deterministic bridge ``B(Y | s, z, c)``.

Both networks wrap the *exact* PB module definitions stored in the PB agent (``subgoal_net`` =
``SubgoalFlowNet``, ``path_residual_net`` = ``PathResidualNet``) and add a learned intention embedding to
their first input (the normalised current state / bridge anchor), i.e. the MLP input becomes
``[s, e_c, phi(g), x_u, t]`` and ``[s_t, e_c, z_K, t]`` respectively. With ``num_codes=0`` the wrapper is
the identity and PB parameters (nested under ``base``) reproduce PB exactly. Frame / normalisation
helpers are PB's own methods, so semantics match the PB code version that produced the checkpoints.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from functools import partial
from pathlib import Path
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from intention_pb.common import INTENT_DIM, INTENT_HORIZON, NUM_CODES, atomic_write_json, atomic_write_pickle, load_pickle, trim_jsonl_to_step

COND_VERSION = 'intention_conditioned_pb_v1'


class IntentionConditioned(nn.Module):
    """Concatenate ``Embed(codes)`` to the first positional input of a PB module."""

    base: nn.Module
    num_codes: int = 0
    intent_dim: int = INTENT_DIM

    @nn.compact
    def __call__(self, first, *rest, codes=None):
        if self.num_codes > 0:
            if codes is None:
                raise ValueError('Conditioned module with num_codes>0 requires codes.')
            emb = nn.Embed(self.num_codes, self.intent_dim, name='intention_embed')(jnp.asarray(codes, jnp.int32))
            first = jnp.concatenate([first, emb], axis=-1)
        elif codes is not None:
            raise ValueError('codes given to an unconditioned (num_codes=0) module.')
        return self.base(first, *rest)


class ConditionedModel:
    """Conditioned subgoal + bridge built from a PB ``DynamicsAgent`` (PB agent is read-only)."""

    def __init__(self, pb_dynamics, *, num_codes: int = NUM_CODES, intent_dim: int = INTENT_DIM):
        cfg = pb_dynamics.config
        self.pb = pb_dynamics
        self.num_codes = int(num_codes)
        self.N = int(cfg['dynamics_N'])
        if str(cfg.get('subgoal_distribution')) != 'flow':
            raise ValueError('Conditioned model requires a flow-subgoal PB baseline.')
        if str(cfg.get('planner_type', 'forward_bridge_residual')) != 'forward_bridge_residual':
            raise ValueError('Conditioned model requires the forward_bridge_residual planner.')
        if not bool(cfg.get('forward_bridge_use_path_loss', True)):
            raise ValueError('Unsupported PB path-loss configuration.')
        if float(cfg.get('subgoal_value_alpha', 0.0)) != 0.0 or float(cfg.get('subgoal_flow_velocity_reg', 0.0)) != 0.0:
            raise ValueError('subgoal_value_alpha / subgoal_flow_velocity_reg != 0 are not mirrored.')
        self.flow_steps = int(cfg.get('subgoal_flow_steps', 8))
        self.flow_t_min = float(cfg.get('subgoal_flow_t_min', 1e-4))
        self.flow_noise_scale = float(cfg.get('subgoal_flow_noise_scale', 1.0))
        self.path_H = int(cfg.get('forward_bridge_path_loss_horizon', 0) or 0)
        if not (0 < self.path_H < self.N):
            raise ValueError(f'Expected prefix path-loss horizon in (0, N); got {self.path_H} (N={self.N}).')
        self.path_normalized = bool(cfg.get('path_loss_normalized', True))
        self.path_w = float(cfg.get('path_loss_weight', 1.0))
        self.sub_w = float(cfg.get('subgoal_loss_weight', 1.0))
        a, b, _ = pb_dynamics.forward_bridge_coefficients(self.N)
        self.coef_a = np.asarray(a, dtype=np.float32)
        self.coef_b = np.asarray(b, dtype=np.float32)
        modules = pb_dynamics.network.model_def.modules
        self.subgoal_def = IntentionConditioned(base=modules['subgoal_net'].clone(), num_codes=self.num_codes, intent_dim=int(intent_dim))
        self.residual_def = IntentionConditioned(base=modules['path_residual_net'].clone(), num_codes=self.num_codes, intent_dim=int(intent_dim))
        res = pb_dynamics.network.params['modules_path_residual_net']['MLP_0']
        last = sorted((k for k in res if k.startswith('Dense_')), key=lambda k: int(k.split('_')[1]))[-1]
        self.state_dim = int(res[last]['kernel'].shape[-1])
        self.tx = optax.adam(learning_rate=float(cfg['lr']))

    # ------------------------------------------------------------------ init / params
    def _codes(self, codes):
        return None if self.num_codes == 0 else jnp.asarray(codes, dtype=jnp.int32)

    def init(self, seed: int) -> dict:
        k1, k2 = jax.random.split(jax.random.PRNGKey(int(seed)))
        B, D = 4, self.state_dim
        x = jnp.zeros((B, D), jnp.float32)
        codes = None if self.num_codes == 0 else jnp.zeros((B,), jnp.int32)
        params = dict(
            subgoal=self.subgoal_def.init(k1, x, x, x, jnp.zeros((B, 1)), codes=codes)['params'],
            residual=self.residual_def.init(k2, x, x, jnp.zeros((B, self.N + 1)), codes=codes)['params'],
        )
        return dict(params=params, opt_state=self.tx.init(params), rng=jax.random.PRNGKey(int(seed) + 1),
                    step=jnp.asarray(0, jnp.int32))

    def params_from_pb(self) -> dict:
        """PB parameters in this model's layout (only valid for ``num_codes == 0``)."""
        if self.num_codes != 0:
            raise ValueError('params_from_pb only applies to the unconditioned (num_codes=0) model.')
        p = self.pb.network.params
        return dict(subgoal={'base': p['modules_subgoal_net']}, residual={'base': p['modules_path_residual_net']})

    # ------------------------------------------------------------------ subgoal
    def flow_sample_raw(self, sub_params, observations, goals, noise, codes):
        steps = self.flow_steps
        dt = jnp.asarray(1.0 / steps, dtype=jnp.float32)
        obs_n = self.pb._normalize_abs_state(observations)
        goal_n = self.pb._normalize_abs_state(goals)
        c = self._codes(codes)

        def body(k, x):
            u = jnp.full((x.shape[0], 1), k.astype(jnp.float32) / jnp.asarray(steps, dtype=jnp.float32))
            v = self.subgoal_def.apply({'params': sub_params}, obs_n, goal_n, x, u, codes=c)
            return x + dt * v

        return jax.lax.fori_loop(0, steps, body, jnp.asarray(noise, jnp.float32))

    def candidate_codes(self, per_code: int) -> jnp.ndarray:
        """Code layout of the equal-budget candidate set: ``[0]*M + [1]*M + ... + [K-1]*M``."""
        return jnp.repeat(jnp.arange(self.num_codes, dtype=jnp.int32), int(per_code))

    def _sample(self, sub_params, obs, goals, flat_codes, rng, n, temperature):
        B, D = obs.shape
        noise = jax.random.normal(rng, (B, n, D), dtype=jnp.float32) * (self.flow_noise_scale * float(temperature))
        flat_obs = jnp.repeat(obs[:, None, :], n, axis=1).reshape(B * n, D)
        flat_g = jnp.repeat(goals[:, None, :], n, axis=1).reshape(B * n, D)
        raw = self.flow_sample_raw(sub_params, flat_obs, flat_g, noise.reshape(B * n, D), flat_codes)
        return self.pb._subgoal_candidates_abs_from_raw(obs, raw.reshape(B, n, D))

    def sample_candidates(self, sub_params, observations, goals, rng, *, temperature: float, per_code: int):
        """Enumerate all K codes with ``per_code`` flow samples each. Returns ``(cands [B,N,D], codes [N])``."""
        obs = jnp.asarray(observations, jnp.float32)
        g = jnp.asarray(goals, jnp.float32)
        codes = self.candidate_codes(per_code)
        n = int(codes.shape[0])
        return self._sample(sub_params, obs, g, jnp.tile(codes, obs.shape[0]), rng, n, temperature), codes

    def sample_with_codes(self, sub_params, observations, goals, codes, rng, *, temperature: float, num: int):
        """``num`` samples per row for given per-row codes. Returns ``[B, num, D]`` absolute subgoals."""
        obs = jnp.asarray(observations, jnp.float32)
        g = jnp.asarray(goals, jnp.float32)
        flat = None if self.num_codes == 0 else jnp.repeat(jnp.asarray(codes, jnp.int32), num)
        return self._sample(sub_params, obs, g, flat, rng, num, temperature)

    # ------------------------------------------------------------------ bridge
    def _path_at_indices_n(self, res_params, z0, zK, indices, anchor, codes):
        """Normalised planner-frame path at ``indices`` (PB ``_forward_bridge_path_at_indices`` semantics)."""
        N = self.N
        idx = jnp.asarray(indices, dtype=jnp.int32)
        idx_f = idx.astype(jnp.float32)
        z0_n = self.pb._normalize_planner_state(z0)
        zK_n = self.pb._normalize_planner_state(zK)
        a = jnp.asarray(self.coef_a)[idx]
        b = jnp.asarray(self.coef_b)[idx]
        mu = a[None, :, None] * z0_n[:, None, :] + b[None, :, None] * zK_n[:, None, :]
        t_norm = jnp.broadcast_to(idx_f[None, :] / float(N), (z0.shape[0], idx.shape[0]))
        anchor_n = self.pb._normalize_abs_state(anchor)
        residual = self.residual_def.apply({'params': res_params}, anchor_n, zK_n, t_norm, codes=self._codes(codes))
        w = idx_f * (float(N) - idx_f) / float(N * N)
        path_n = mu + w[None, :, None] * residual
        path_n = jnp.where((idx == 0)[None, :, None], z0_n[:, None, :], path_n)
        path_n = jnp.where((idx == N)[None, :, None], zK_n[:, None, :], path_n)
        return path_n

    def bridge(self, res_params, observations, subgoals, codes):
        """Full deterministic bridge ``[B, N+1, D]`` in absolute coordinates (PB ``plan`` semantics)."""
        obs = jnp.asarray(observations, jnp.float32)
        origin, z0, zK, anchor = self.pb._shift_to_displacement_frame(obs, jnp.asarray(subgoals, jnp.float32))
        path_n = self._path_at_indices_n(res_params, z0, zK, jnp.arange(self.N + 1), anchor, codes)
        return self.pb._denormalize_planner_state(path_n) + origin[:, None, :]

    # ------------------------------------------------------------------ training
    def loss(self, params, batch, codes, rng, value_params):
        pb = self.pb
        s = jnp.asarray(batch['observations'], jnp.float32)
        g = jnp.asarray(batch['high_actor_goals'], jnp.float32)
        target_abs = jnp.asarray(batch['high_actor_targets'], jnp.float32)
        seg_abs = jnp.asarray(batch['trajectory_segment'], jnp.float32)
        origin = pb._displacement_origin(s)
        anchor = pb._bridge_anchor(s)
        seg = seg_abs - origin[:, None, :]
        N, H = self.N, self.path_H
        indices = jnp.concatenate([jnp.arange(0, H + 1, dtype=jnp.int32), jnp.asarray([N], jnp.int32)])
        path_n = self._path_at_indices_n(params['residual'], seg[:, 0], seg[:, -1], indices, anchor, codes)
        path_pred = pb._denormalize_planner_state(path_n)
        seg_path = seg[:, indices, :]
        if self.path_normalized:
            pred_l, tgt_l = pb._normalize_planner_state(path_pred), pb._normalize_planner_state(seg_path)
        else:
            pred_l, tgt_l = path_pred, seg_path
        # PB (21a4042): interior L1 over indices[1:] (prefix 1..H plus the clamped endpoint) + next-step L1.
        interior_per = jnp.mean(jnp.abs(pred_l[:, 1:, :] - tgt_l[:, 1:, :]).sum(-1), axis=1)
        next_per = jnp.abs(pred_l[:, 1, :] - tgt_l[:, 1, :]).sum(-1)
        path_per = interior_per + next_per
        loss_path = jnp.mean(path_per)

        target = pb._subgoal_target_for_loss(s, target_abs)
        eps_rng, u_rng = jax.random.split(rng)
        eps = jax.random.normal(eps_rng, target.shape, dtype=target.dtype)
        u = jax.random.uniform(u_rng, (target.shape[0], 1), minval=self.flow_t_min, maxval=1.0 - self.flow_t_min,
                               dtype=target.dtype)
        x_u = (1.0 - u) * eps + u * target
        v_pred = self.subgoal_def.apply({'params': params['subgoal']}, pb._normalize_abs_state(s), pb._normalize_abs_state(g),
                                        x_u, u, codes=self._codes(codes))
        fm = jnp.mean((v_pred - (target - eps)) ** 2, axis=-1)
        values = pb._subgoal_values(jnp.concatenate([s, target_abs], 0), jnp.concatenate([g, g], 0), value_params)
        obs_v, tgt_v = jnp.split(values, 2, axis=0)
        gap = tgt_v - obs_v
        weight = jax.lax.stop_gradient(pb._subgoal_mse_weight_from_gap(gap))
        sub_per = weight * fm
        loss_sub = jnp.mean(sub_per)
        loss = self.path_w * loss_path + self.sub_w * loss_sub

        info = dict(loss=loss, loss_path=loss_path, loss_subgoal=loss_sub, subgoal_fm_raw=jnp.mean(fm),
                    subgoal_weight_mean=jnp.mean(weight), value_gap_mean=jnp.mean(gap),
                    first_step_mse=jnp.mean((path_pred[:, 1] - seg_path[:, 1]) ** 2))
        if self.num_codes > 0:
            oh = jax.nn.one_hot(codes, self.num_codes, dtype=jnp.float32)
            cnt = oh.sum(0)
            info['code_count'] = cnt
            info['per_code_subgoal'] = (oh.T @ sub_per) / jnp.maximum(cnt, 1.0)
            info['per_code_path'] = (oh.T @ path_per) / jnp.maximum(cnt, 1.0)
        return loss, info

    @partial(jax.jit, static_argnums=0)
    def train_step(self, state, batch, codes, value_params):
        rng, sub = jax.random.split(state['rng'])
        grads, info = jax.grad(self.loss, has_aux=True)(state['params'], batch, codes, sub, value_params)
        updates, opt_state = self.tx.update(grads, state['opt_state'], state['params'])
        params = optax.apply_updates(state['params'], updates)
        info = dict(info, grad_norm=optax.global_norm(grads), grad_norm_subgoal=optax.global_norm(grads['subgoal']),
                    grad_norm_residual=optax.global_norm(grads['residual']))
        return dict(params=params, opt_state=opt_state, rng=rng, step=state['step'] + 1), info


# ---------------------------------------------------------------------- checkpoint I/O
def save_conditioned(path: Path, model: ConditionedModel, state: dict, meta: dict) -> None:
    payload = dict(version=COND_VERSION, num_codes=model.num_codes,
                   state=jax.device_get(flax.serialization.to_state_dict(state)), meta=meta)
    atomic_write_pickle(path, payload)


def load_conditioned(path: Path, pb_dynamics) -> tuple[ConditionedModel, dict, dict]:
    payload = load_pickle(path)
    if payload.get('version') != COND_VERSION:
        raise ValueError(f'Unexpected conditioned checkpoint version in {path}: {payload.get("version")}')
    model = ConditionedModel(pb_dynamics, num_codes=int(payload['num_codes']))
    state = flax.serialization.from_state_dict(model.init(0), payload['state'])
    return model, state, payload.get('meta', {})


# ---------------------------------------------------------------------- teacher codes
def teacher_codes_for(tok, tok_state, tok_stats, observations, actions, idxs, *, batch: int = 16384) -> np.ndarray:
    from intention_pb.tokenizer import gather_chunks

    assign = jax.jit(lambda p, cb, a, o: tok.assign(p, cb, a, o, tok_stats))
    out = []
    h = tok.cfg.horizon
    for i in range(0, len(idxs), batch):
        act, obs = gather_chunks(observations, actions, idxs[i : i + batch], h)
        out.append(np.asarray(assign(tok_state['params'], tok_state['codebook'], act, obs)))
    return np.concatenate(out).astype(np.int32) if out else np.zeros((0,), np.int32)


BATCH_KEYS = ('observations', 'high_actor_goals', 'high_actor_targets', 'trajectory_segment')


def train_conditioned(*, out_dir: Path, pb, tok_path: Path, seed: int, total_steps: int, save_steps: tuple[int, ...],
                      log_every: int = 10_000, sample_every: int = 1000) -> None:
    """Train conditioned subgoal + bridge with frozen tokenizer / PB critic (resumable)."""
    from main import _intersect_valid_starts, _make_critic_dataset, _sample_shared_idxs
    from utils.datasets import Dataset, PathHGCDataset, lookup_final_indices

    from intention_pb.tokenizer import load_tokenizer

    out_dir = Path(out_dir)
    ck_dir = out_dir / 'checkpoints'
    ck_dir.mkdir(parents=True, exist_ok=True)
    tok, tok_state, tok_stats, _ = load_tokenizer(tok_path)
    if tok.cfg.horizon != INTENT_HORIZON or tok.cfg.horizon != pb.idm_horizon:
        raise ValueError(f'Tokenizer horizon {tok.cfg.horizon} != PB chunk horizon {pb.idm_horizon}')
    dyn_cfg = pb.dynamics_config
    train_ds = PathHGCDataset(Dataset.create(**pb.train), dyn_cfg)
    critic_ds = _make_critic_dataset(pb.train, pb.critic_config)
    common = _intersect_valid_starts(train_ds, critic_ds)
    del critic_ds
    finals = lookup_final_indices(train_ds.terminal_locs, common)
    if np.any(common + INTENT_HORIZON > finals):
        raise ValueError('Some training starts cannot host an intention chunk.')
    obs_all = np.asarray(pb.train['observations'], np.float32)
    act_all = np.asarray(pb.train['actions'], np.float32)
    codes_all = np.full((len(obs_all),), -1, dtype=np.int32)
    t0 = time.time()
    codes_all[common] = teacher_codes_for(tok, tok_state, tok_stats, obs_all, act_all, common)
    usage = np.bincount(codes_all[common], minlength=tok.cfg.num_codes) / len(common)
    print(f'[cond] teacher codes for {len(common)} starts in {time.time() - t0:.1f}s usage={np.round(usage, 3).tolist()}', flush=True)

    model = ConditionedModel(pb.dynamics, num_codes=tok.cfg.num_codes)
    state = model.init(seed)
    start_step = 0
    done = sorted(s for s in save_steps if (ck_dir / f'conditioned_{s}.pkl').is_file())
    if done:
        _, state, _ = load_conditioned(ck_dir / f'conditioned_{done[-1]}.pkl', pb.dynamics)
        start_step = int(done[-1])
        print(f'[cond] resumed from step {start_step}', flush=True)
    np.random.seed(int(seed) * 1_000_003 + start_step)
    value_params = jax.device_put(pb.critic_value_params())
    bs = int(dyn_cfg['batch_size'])
    meta = dict(pb_run_dir=str(pb.run_dir), pb_step=int(pb.step), tokenizer=str(tok_path), seed=int(seed),
                num_codes=int(tok.cfg.num_codes), env_name=pb.env_name, teacher_usage=usage.tolist())
    atomic_write_json(out_dir / 'meta.json', meta)

    q: queue.Queue = queue.Queue(maxsize=8)
    stop = threading.Event()

    def producer():
        while not stop.is_set():
            idxs = _sample_shared_idxs(common, bs)
            b = train_ds.sample(bs, idxs=idxs)
            item = ({k: np.asarray(b[k], np.float32) for k in BATCH_KEYS}, codes_all[idxs])
            while not stop.is_set():
                try:
                    q.put(item, timeout=1.0)
                    break
                except queue.Full:
                    continue

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    log_path = out_dir / 'train_log.jsonl'
    trim_jsonl_to_step(log_path, start_step)
    acc: list[dict] = []
    t_start = time.time()
    try:
        for step in range(start_step + 1, total_steps + 1):
            batch, codes = q.get()
            if np.any(codes < 0):
                raise RuntimeError('Sampled start without a teacher code.')
            state, info = model.train_step(state, batch, codes, value_params)
            if step % sample_every == 0 or step % log_every == 0 or step == total_steps:
                acc.append(jax.device_get(info))
            if step % log_every == 0 or step == total_steps:
                rec: dict[str, Any] = dict(step=step, elapsed_s=time.time() - t_start)
                for k in acc[0]:
                    vals = np.stack([np.asarray(a[k]) for a in acc])
                    if k == 'code_count':
                        tot = vals.sum(0)
                        rec['train_code_usage'] = (tot / tot.sum()).tolist()
                    else:
                        v = vals.mean(0)
                        rec[k] = v.tolist() if np.ndim(v) else float(v)
                acc = []
                if not np.isfinite(rec['loss']):
                    raise FloatingPointError(f'NaN loss at step {step}')
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(rec) + '\n')
                print(f"[cond] step={step} loss={rec['loss']:.4f} path={rec['loss_path']:.4f} "
                      f"sub={rec['loss_subgoal']:.4f} gn={rec['grad_norm']:.3f} {rec['elapsed_s']:.0f}s", flush=True)
            if step in save_steps:
                save_conditioned(ck_dir / f'conditioned_{step}.pkl', model, state, dict(meta, step=step))
                print(f'[cond] saved step {step}', flush=True)
    finally:
        stop.set()
