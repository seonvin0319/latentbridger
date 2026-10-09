"""Temporal VQ route tokenizer.

The encoder reads a length-H sequence of normalised actions and states relative to ``s_t``.
It does not receive the goal, reward, success, or the absolute state. A small temporal
convolution mixes the window; the codebook is the same EMA VQ used by the local tokenizer.
The decoder, which does see ``s_t``, predicts z-scored future path summaries so the code
has to explain route structure rather than only reconstruct the raw window.
"""

from __future__ import annotations

import dataclasses
import json
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

from intention_pb import ensure_pb_code_path

ensure_pb_code_path()

from utils.networks import MLP

from intention_pb.common import COLLAPSE_MAX_USAGE, COLLAPSE_MIN_PERPLEXITY

from route_intention.chunks import (
    compute_route_stats,
    encoder_sequence,
    prediction_horizons,
    state_t_normalized,
    target_dim,
    target_matrix,
    valid_route_starts,
    window_views,
)
from route_intention.common import (
    EMBED_DIM,
    H_ROUTE,
    NUM_CODES,
    atomic_write_json,
    atomic_write_pickle,
    load_pickle,
    trim_jsonl_to_step,
)

VERSION = 'route_tokenizer_v1'


def nearest_code(e, codebook):
    d2 = jnp.sum((e[:, None, :] - codebook[None, :, :]) ** 2, axis=-1)
    return jnp.argmin(d2, axis=-1), d2


def perplexity_np(usage: np.ndarray) -> float:
    u = np.asarray(usage, dtype=np.float64)
    u = u[u > 0]
    if len(u) == 0:
        return 0.0
    return float(np.exp(-np.sum(u * np.log(u))))


def _perplexity(usage):
    p = jnp.clip(usage, 1e-12, 1.0)
    return jnp.exp(-jnp.sum(jnp.where(usage > 0, usage * jnp.log(p), 0.0)))


@dataclasses.dataclass(frozen=True)
class RouteTokenizerConfig:
    num_codes: int = NUM_CODES
    embed_dim: int = EMBED_DIM
    horizon: int = H_ROUTE
    conv_dim: int = 64
    hidden_dims: tuple[int, ...] = (256, 256)
    lr: float = 3e-4
    batch_size: int = 1024
    lambda_multi: float = 1.0
    lambda_path: float = 1.0
    lambda_act: float = 1.0
    lambda_commit: float = 0.25
    lambda_balance: float = 0.01
    ema_decay: float = 0.99
    ema_eps: float = 1e-5


class TemporalEncoder(nn.Module):
    """Two same-padded convolutions, then mean-pool. Fixed and small on purpose."""

    conv_dim: int
    embed_dim: int

    @nn.compact
    def __call__(self, seq):
        h = nn.Conv(self.conv_dim, kernel_size=(5,), padding='SAME')(seq)
        h = nn.gelu(nn.LayerNorm()(h))
        h = nn.Conv(self.conv_dim, kernel_size=(5,), padding='SAME')(h)
        h = nn.gelu(nn.LayerNorm()(h))
        pooled = h.mean(axis=1)
        return nn.Dense(self.embed_dim)(pooled)


class RouteDecoder(nn.Module):
    hidden_dims: tuple[int, ...]
    out_dim: int

    @nn.compact
    def __call__(self, state_n, q):
        x = jnp.concatenate([state_n, q], axis=-1)
        return MLP(hidden_dims=(*self.hidden_dims, self.out_dim), activate_final=False, layer_norm=True)(x)


class RouteTokenizer:
    def __init__(self, cfg: RouteTokenizerConfig, action_dim: int, state_dim: int):
        self.cfg = cfg
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.step_dim = self.action_dim + self.state_dim
        self.out_dim = target_dim(cfg.horizon, self.state_dim, self.action_dim)
        self.n_endpoint = len(prediction_horizons(cfg.horizon))
        self.encoder = TemporalEncoder(int(cfg.conv_dim), int(cfg.embed_dim))
        self.decoder = RouteDecoder(tuple(cfg.hidden_dims), self.out_dim)
        self.tx = optax.adam(cfg.lr)

    def init(self, seed: int) -> dict:
        k_enc, k_dec = jax.random.split(jax.random.PRNGKey(int(seed)))
        seq = jnp.zeros((2, self.cfg.horizon, self.step_dim), jnp.float32)
        s = jnp.zeros((2, self.state_dim), jnp.float32)
        q = jnp.zeros((2, self.cfg.embed_dim), jnp.float32)
        params = dict(
            encoder=self.encoder.init(k_enc, seq)['params'],
            decoder=self.decoder.init(k_dec, s, q)['params'],
        )
        k = self.cfg.num_codes
        e = self.cfg.embed_dim
        return dict(
            params=params,
            opt_state=self.tx.init(params),
            codebook=jnp.zeros((k, e), jnp.float32),
            ema_count=jnp.ones((k,), jnp.float32),
            ema_sum=jnp.zeros((k, e), jnp.float32),
            step=jnp.asarray(0, jnp.int32),
        )

    def encode(self, params, seq):
        return self.encoder.apply({'params': params['encoder']}, seq)

    def init_codebook(self, state, seq, rng):
        e = self.encode(state['params'], seq)
        if e.shape[0] < self.cfg.num_codes:
            raise ValueError(f'Need at least {self.cfg.num_codes} rows to initialise the codebook.')
        pick = jax.random.choice(rng, e.shape[0], (self.cfg.num_codes,), replace=False)
        cb = e[pick]
        return dict(state, codebook=cb, ema_sum=cb, ema_count=jnp.ones_like(state['ema_count']))

    def loss(self, params, codebook, seq, s_n, target):
        cfg = self.cfg
        e = self.encode(params, seq)
        idx, d2 = nearest_code(e, codebook)
        q = codebook[idx]
        q_st = e + jax.lax.stop_gradient(q - e)
        pred = self.decoder.apply({'params': params['decoder']}, s_n, q_st)
        # Targets are already z-scored, so the three groups have comparable MSE scale.
        d = self.state_dim
        n_end = self.n_endpoint
        n_way = (target.shape[-1] - self.action_dim) // d - n_end
        end = n_end * d
        way = end + n_way * d
        l_multi = jnp.mean((pred[:, :end] - target[:, :end]) ** 2)
        l_path = jnp.mean((pred[:, end:way] - target[:, end:way]) ** 2)
        l_act = jnp.mean((pred[:, way:] - target[:, way:]) ** 2)
        l_commit = jnp.mean((e - jax.lax.stop_gradient(q)) ** 2)
        probs = jax.nn.softmax(-d2, axis=-1)
        pbar = jnp.mean(probs, axis=0)
        l_balance = jnp.sum(pbar * (jnp.log(pbar + 1e-10) - jnp.log(1.0 / cfg.num_codes)))
        loss = (
            cfg.lambda_multi * l_multi
            + cfg.lambda_path * l_path
            + cfg.lambda_act * l_act
            + cfg.lambda_commit * l_commit
            + cfg.lambda_balance * l_balance
        )
        info = dict(loss=loss, l_multi=l_multi, l_path=l_path, l_act=l_act, l_commit=l_commit, l_balance=l_balance)
        return loss, (info, idx, e)

    @partial(jax.jit, static_argnums=0)
    def train_step(self, state, seq, s_n, target):
        cfg = self.cfg
        grads, (info, idx, e) = jax.grad(self.loss, has_aux=True)(state['params'], state['codebook'], seq, s_n, target)
        updates, opt_state = self.tx.update(grads, state['opt_state'], state['params'])
        params = optax.apply_updates(state['params'], updates)
        onehot = jax.nn.one_hot(idx, cfg.num_codes, dtype=jnp.float32)
        e = jax.lax.stop_gradient(e)
        count = cfg.ema_decay * state['ema_count'] + (1.0 - cfg.ema_decay) * onehot.sum(0)
        esum = cfg.ema_decay * state['ema_sum'] + (1.0 - cfg.ema_decay) * onehot.T @ e
        n = jnp.sum(count)
        count_s = (count + cfg.ema_eps) / (n + cfg.num_codes * cfg.ema_eps) * n
        codebook = esum / count_s[:, None]
        usage = onehot.mean(0)
        info = dict(info, grad_norm=optax.global_norm(grads), batch_perplexity=_perplexity(usage), batch_max_usage=usage.max())
        new_state = dict(params=params, opt_state=opt_state, codebook=codebook, ema_count=count, ema_sum=esum, step=state['step'] + 1)
        return new_state, info, usage

    @partial(jax.jit, static_argnums=0)
    def eval_batch(self, params, codebook, seq, s_n, target):
        e = self.encode(params, seq)
        idx, _ = nearest_code(e, codebook)
        pred = self.decoder.apply({'params': params['decoder']}, s_n, codebook[idx])
        err = jnp.mean((pred - target) ** 2, axis=-1)
        return idx, err, e


def _as_jnp_stats(stats: dict) -> dict:
    return {k: jnp.asarray(v) for k, v in stats.items()}


def _batch_arrays(obs, act, starts, stats_np, horizon):
    obs_win, act_win = window_views(obs, act, starts, horizon)
    seq = encoder_sequence(obs_win, act_win, stats_np)
    s_n = state_t_normalized(obs_win, stats_np)
    tgt = target_matrix(obs_win, act_win, stats_np, horizon)
    return seq, s_n, tgt


def assign_codes(tok: RouteTokenizer, params, codebook, observations, actions, starts, stats) -> np.ndarray:
    horizon = tok.cfg.horizon
    stats_np = {k: np.asarray(v) for k, v in stats.items()}
    out = []
    batch = 4096
    for i in range(0, len(starts), batch):
        seq, _, _ = _batch_arrays(observations, actions, starts[i : i + batch], stats_np, horizon)
        e = tok.encode(params, jnp.asarray(seq))
        idx, _ = nearest_code(e, codebook)
        out.append(np.asarray(idx))
    return np.concatenate(out).astype(np.int32)


def evaluate_route_tokenizer(tok, params, codebook, stats, observations, actions, starts) -> dict:
    stats_np = {k: np.asarray(v) for k, v in stats.items()}
    codes, errs, embs = [], [], []
    horizon = tok.cfg.horizon
    for i in range(0, len(starts), 4096):
        sl = starts[i : i + 4096]
        seq, s_n, tgt = _batch_arrays(observations, actions, sl, stats_np, horizon)
        idx, err, e = tok.eval_batch(params, codebook, jnp.asarray(seq), jnp.asarray(s_n), jnp.asarray(tgt))
        codes.append(np.asarray(idx))
        errs.append(np.asarray(err))
        embs.append(np.asarray(e))
    codes = np.concatenate(codes)
    errs = np.concatenate(errs)
    embs = np.concatenate(embs)
    usage = np.bincount(codes, minlength=tok.cfg.num_codes).astype(np.float64)
    usage = usage / max(len(codes), 1)
    ppl = perplexity_np(usage)
    # Coarse trajectory summary for separability: relative endpoint of each window.
    obs_win, act_win = window_views(observations, actions, starts, horizon)
    endpoint = obs_win[:, -1, :] - obs_win[:, 0, :]
    mean_action = act_win.mean(1)
    per_code = []
    for k in range(tok.cfg.num_codes):
        m = codes == k
        row: dict[str, Any] = dict(code=k, frequency=float(usage[k]), count=int(m.sum()))
        if m.any():
            row['mean_endpoint'] = endpoint[m].mean(0).tolist()
            row['mean_displacement_norm'] = float(np.linalg.norm(endpoint[m], axis=-1).mean())
            row['mean_action'] = mean_action[m].mean(0).tolist()
            row['embedding_centroid'] = embs[m].mean(0).tolist()
            row['recon_mse'] = float(errs[m].mean())
        per_code.append(row)
    from route_intention.variance import between_within

    sep = between_within(endpoint, codes)
    return dict(
        num_samples=int(len(codes)),
        usage=[float(u) for u in usage],
        perplexity=ppl,
        max_usage=float(usage.max()) if len(usage) else 1.0,
        recon_mse=float(errs.mean()) if len(errs) else float('nan'),
        collapsed=bool(usage.max() > COLLAPSE_MAX_USAGE or ppl < COLLAPSE_MIN_PERPLEXITY),
        between_endpoint=sep['between'],
        within_endpoint=sep['within'],
        per_code=per_code,
    )


def save_route_tokenizer(path: Path, tok: RouteTokenizer, state: dict, stats: dict, extra: dict | None = None) -> None:
    payload = dict(
        version=VERSION,
        config=dataclasses.asdict(tok.cfg),
        action_dim=tok.action_dim,
        state_dim=tok.state_dim,
        state=jax.device_get(flax.serialization.to_state_dict(state)),
        stats={k: np.asarray(v) for k, v in stats.items()},
        extra=extra or {},
    )
    atomic_write_pickle(path, payload)


def load_route_tokenizer(path: Path):
    payload = load_pickle(path)
    if payload.get('version') != VERSION:
        raise ValueError(f'Unexpected route tokenizer version in {path}: {payload.get("version")}')
    cfg_d = dict(payload['config'])
    cfg_d['hidden_dims'] = tuple(cfg_d['hidden_dims'])
    tok = RouteTokenizer(RouteTokenizerConfig(**cfg_d), payload['action_dim'], payload['state_dim'])
    state = flax.serialization.from_state_dict(tok.init(0), payload['state'])
    return tok, state, payload['stats'], payload.get('extra', {})


def train_route_tokenizer(*, out_dir: Path, train: dict, val: dict, seed: int, total_steps: int,
                          save_steps: tuple[int, ...], log_every: int = 5000,
                          cfg: RouteTokenizerConfig | None = None) -> dict:
    cfg = cfg or RouteTokenizerConfig()
    out_dir = Path(out_dir)
    ck_dir = out_dir / 'checkpoints'
    ck_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'metrics').mkdir(exist_ok=True)
    obs_tr = np.asarray(train['observations'], np.float32)
    act_tr = np.asarray(train['actions'], np.float32)
    obs_va = np.asarray(val['observations'], np.float32)
    act_va = np.asarray(val['actions'], np.float32)
    starts_tr = valid_route_starts(train['terminals'], cfg.horizon)
    starts_va = valid_route_starts(val['terminals'], cfg.horizon)
    if len(starts_tr) < cfg.batch_size:
        raise RuntimeError(f'Only {len(starts_tr)} valid route starts; need at least a batch.')
    tok = RouteTokenizer(cfg, act_tr.shape[-1], obs_tr.shape[-1])
    stats_np = compute_route_stats(obs_tr, act_tr, starts_tr, cfg.horizon)
    state = tok.init(seed)
    start_step = 0
    done = sorted(s for s in save_steps if (ck_dir / f'tokenizer_{s}.pkl').is_file())
    rng = np.random.default_rng(seed)
    if done:
        _, state, _, _ = load_route_tokenizer(ck_dir / f'tokenizer_{done[-1]}.pkl')
        start_step = int(done[-1])
        rng = np.random.default_rng([seed, start_step])
        print(f'[route-tok] resumed from step {start_step}', flush=True)
    else:
        idx = starts_tr[rng.integers(len(starts_tr), size=max(cfg.batch_size, cfg.num_codes))]
        seq, _, _ = _batch_arrays(obs_tr, act_tr, idx, stats_np, cfg.horizon)
        state = tok.init_codebook(state, jnp.asarray(seq), jax.random.PRNGKey(seed + 17))
    log_path = out_dir / 'train_log.jsonl'
    trim_jsonl_to_step(log_path, start_step)
    t0 = time.time()
    for step in range(start_step + 1, total_steps + 1):
        idx = starts_tr[rng.integers(len(starts_tr), size=cfg.batch_size)]
        seq, s_n, tgt = _batch_arrays(obs_tr, act_tr, idx, stats_np, cfg.horizon)
        state, info, _usage = tok.train_step(state, jnp.asarray(seq), jnp.asarray(s_n), jnp.asarray(tgt))
        if step % log_every == 0 or step == total_steps:
            rec = {k: float(v) for k, v in jax.device_get(info).items()}
            rec.update(step=step, elapsed_s=time.time() - t0)
            if not np.isfinite(rec['loss']):
                raise FloatingPointError(f'NaN route tokenizer loss at step {step}')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"[route-tok] step={step} loss={rec['loss']:.4f} multi={rec['l_multi']:.4f} "
                  f"path={rec['l_path']:.4f} act={rec['l_act']:.4f} ppl={rec['batch_perplexity']:.2f} "
                  f"max={rec['batch_max_usage']:.3f}", flush=True)
        if step in save_steps:
            metrics = evaluate_route_tokenizer(tok, state['params'], state['codebook'], stats_np, obs_va, act_va, starts_va)
            metrics['step'] = step
            atomic_write_json(out_dir / 'metrics' / f'step_{step}.json', metrics)
            save_route_tokenizer(ck_dir / f'tokenizer_{step}.pkl', tok, state, stats_np, extra=dict(step=step, seed=int(seed)))
            print(f"[route-tok] saved step {step}: val ppl={metrics['perplexity']:.3f} "
                  f"max={metrics['max_usage']:.3f} collapsed={metrics['collapsed']}", flush=True)
    with open(out_dir / 'metrics' / f'step_{total_steps}.json', encoding='utf-8') as f:
        return json.load(f)
