"""Discrete VQ intention tokenizer over local behaviour chunks (actions + relative state changes).

The encoder only ever sees ``a_{t:t+h}`` and normalised ``Delta s_{t:t+h}``; it never receives the goal,
the absolute state, the absolute chunk endpoint, reward or success. The auxiliary decoder sees ``s_t``.
"""

from __future__ import annotations

import dataclasses
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

from intention_pb.common import (
    trim_jsonl_to_step,
    COLLAPSE_MAX_USAGE,
    COLLAPSE_MIN_PERPLEXITY,
    INTENT_DIM,
    INTENT_HORIZON,
    NUM_CODES,
    atomic_write_json,
    atomic_write_pickle,
    load_pickle,
)

STD_FLOOR = 1e-3


@dataclasses.dataclass(frozen=True)
class TokenizerConfig:
    num_codes: int = NUM_CODES
    intent_dim: int = INTENT_DIM
    horizon: int = INTENT_HORIZON
    hidden_dims: tuple[int, ...] = (512, 512)
    lr: float = 3e-4
    batch_size: int = 1024
    lambda_action: float = 1.0
    lambda_delta: float = 1.0
    lambda_commit: float = 0.25
    lambda_balance: float = 0.01
    ema_decay: float = 0.99
    ema_eps: float = 1e-5


class IntentionEncoder(nn.Module):
    hidden_dims: tuple[int, ...]
    intent_dim: int

    @nn.compact
    def __call__(self, feats):
        return MLP(hidden_dims=(*self.hidden_dims, self.intent_dim), activate_final=False, layer_norm=True)(feats)


class BehaviorDecoder(nn.Module):
    hidden_dims: tuple[int, ...]
    out_dim: int

    @nn.compact
    def __call__(self, state_n, q):
        x = jnp.concatenate([state_n, q], axis=-1)
        return MLP(hidden_dims=(*self.hidden_dims, self.out_dim), activate_final=False, layer_norm=True)(x)


def compute_stats(observations: np.ndarray, actions: np.ndarray, starts: np.ndarray) -> dict[str, np.ndarray]:
    """Per-dimension normalisation statistics from valid one-step transitions ``t -> t+1``."""
    obs = np.asarray(observations, dtype=np.float32)
    act = np.asarray(actions, dtype=np.float32)
    t = np.asarray(starts, dtype=np.int64)
    delta = obs[t + 1] - obs[t]
    return dict(
        action_mean=act[t].mean(0),
        action_std=np.maximum(act[t].std(0), STD_FLOOR),
        delta_mean=delta.mean(0),
        delta_std=np.maximum(delta.std(0), STD_FLOOR),
        state_mean=obs[t].mean(0),
        state_std=np.maximum(obs[t].std(0), STD_FLOOR),
    )


def chunk_features(action_chunk, obs_chunk, stats):
    """Return ``(encoder_features, action_targets_n, delta_targets_n, state_t_n)``.

    ``action_chunk``: (B, h, A) = ``a_t..a_{t+h-1}``; ``obs_chunk``: (B, h+1, D) = ``s_t..s_{t+h}``.
    Encoder features contain only actions and relative state changes.
    """
    a = jnp.asarray(action_chunk, dtype=jnp.float32)
    o = jnp.asarray(obs_chunk, dtype=jnp.float32)
    a_n = (a - stats['action_mean']) / stats['action_std']
    delta = o[:, 1:, :] - o[:, :-1, :]
    d_n = (delta - stats['delta_mean']) / stats['delta_std']
    feats = jnp.concatenate([a_n.reshape(a.shape[0], -1), d_n.reshape(d_n.shape[0], -1)], axis=-1)
    s_n = (o[:, 0, :] - stats['state_mean']) / stats['state_std']
    return feats, a_n, d_n, s_n


def nearest_code(e, codebook):
    d2 = jnp.sum((e[:, None, :] - codebook[None, :, :]) ** 2, axis=-1)
    return jnp.argmin(d2, axis=-1), d2


class Tokenizer:
    """Holds module definitions; parameters are passed explicitly."""

    def __init__(self, cfg: TokenizerConfig, action_dim: int, state_dim: int):
        self.cfg = cfg
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.encoder = IntentionEncoder(tuple(cfg.hidden_dims), int(cfg.intent_dim))
        self.decoder = BehaviorDecoder(tuple(cfg.hidden_dims), cfg.horizon * (self.action_dim + self.state_dim))
        self.tx = optax.adam(cfg.lr)

    @property
    def feat_dim(self) -> int:
        return self.cfg.horizon * (self.action_dim + self.state_dim)

    def init(self, seed: int) -> dict:
        k_enc, k_dec = jax.random.split(jax.random.PRNGKey(int(seed)))
        feats = jnp.zeros((2, self.feat_dim), jnp.float32)
        params = dict(
            encoder=self.encoder.init(k_enc, feats)['params'],
            decoder=self.decoder.init(k_dec, jnp.zeros((2, self.state_dim)), jnp.zeros((2, self.cfg.intent_dim)))['params'],
        )
        K, E = self.cfg.num_codes, self.cfg.intent_dim
        return dict(
            params=params,
            opt_state=self.tx.init(params),
            codebook=jnp.zeros((K, E), jnp.float32),
            ema_count=jnp.ones((K,), jnp.float32),
            ema_sum=jnp.zeros((K, E), jnp.float32),
            step=jnp.asarray(0, jnp.int32),
        )

    def encode(self, params, feats):
        return self.encoder.apply({'params': params['encoder']}, feats)

    def assign(self, params, codebook, action_chunk, obs_chunk, stats):
        """Deterministic code assignment ``c = argmin_j ||E(x) - e_j||``."""
        feats, _, _, _ = chunk_features(action_chunk, obs_chunk, stats)
        idx, _ = nearest_code(self.encode(params, feats), codebook)
        return idx

    def init_codebook(self, state, feats, rng):
        e = self.encode(state['params'], feats)
        pick = jax.random.choice(rng, e.shape[0], (self.cfg.num_codes,), replace=False)
        cb = e[pick]
        return dict(state, codebook=cb, ema_sum=cb, ema_count=jnp.ones_like(state['ema_count']))

    def loss(self, params, codebook, feats, a_n, d_n, s_n):
        cfg = self.cfg
        B = feats.shape[0]
        e = self.encode(params, feats)
        idx, d2 = nearest_code(e, codebook)
        q = codebook[idx]
        q_st = e + jax.lax.stop_gradient(q - e)
        pred = self.decoder.apply({'params': params['decoder']}, s_n, q_st)
        h, A = cfg.horizon, self.action_dim
        pred_a = pred[:, : h * A].reshape(B, h, A)
        pred_d = pred[:, h * A :].reshape(B, h, -1)
        l_action = jnp.mean((pred_a - a_n) ** 2)
        l_delta = jnp.mean((pred_d - d_n) ** 2)
        l_commit = jnp.mean((e - jax.lax.stop_gradient(q)) ** 2)
        probs = jax.nn.softmax(-d2, axis=-1)
        pbar = jnp.mean(probs, axis=0)
        l_balance = jnp.sum(pbar * (jnp.log(pbar + 1e-10) - jnp.log(1.0 / cfg.num_codes)))
        loss = (
            cfg.lambda_action * l_action
            + cfg.lambda_delta * l_delta
            + cfg.lambda_commit * l_commit
            + cfg.lambda_balance * l_balance
        )
        info = dict(loss=loss, l_action=l_action, l_delta=l_delta, l_commit=l_commit, l_balance=l_balance)
        return loss, (info, idx, e)

    @partial(jax.jit, static_argnums=0)
    def train_step(self, state, action_chunk, obs_chunk, stats):
        cfg = self.cfg
        feats, a_n, d_n, s_n = chunk_features(action_chunk, obs_chunk, stats)
        grads, (info, idx, e) = jax.grad(self.loss, has_aux=True)(state['params'], state['codebook'], feats, a_n, d_n, s_n)
        updates, opt_state = self.tx.update(grads, state['opt_state'], state['params'])
        params = optax.apply_updates(state['params'], updates)
        onehot = jax.nn.one_hot(idx, cfg.num_codes, dtype=jnp.float32)
        e = jax.lax.stop_gradient(e)
        count = cfg.ema_decay * state['ema_count'] + (1 - cfg.ema_decay) * onehot.sum(0)
        esum = cfg.ema_decay * state['ema_sum'] + (1 - cfg.ema_decay) * onehot.T @ e
        n = jnp.sum(count)
        count_s = (count + cfg.ema_eps) / (n + cfg.num_codes * cfg.ema_eps) * n
        codebook = esum / count_s[:, None]
        usage = onehot.mean(0)
        info = dict(info, grad_norm=optax.global_norm(grads), batch_perplexity=_perplexity(usage), batch_max_usage=usage.max())
        new_state = dict(params=params, opt_state=opt_state, codebook=codebook, ema_count=count, ema_sum=esum, step=state['step'] + 1)
        return new_state, info, usage

    @partial(jax.jit, static_argnums=0)
    def eval_batch(self, params, codebook, action_chunk, obs_chunk, stats):
        cfg = self.cfg
        feats, a_n, d_n, s_n = chunk_features(action_chunk, obs_chunk, stats)
        e = self.encode(params, feats)
        idx, _ = nearest_code(e, codebook)
        pred = self.decoder.apply({'params': params['decoder']}, s_n, codebook[idx])
        B, h, A = feats.shape[0], cfg.horizon, self.action_dim
        err_a = jnp.mean((pred[:, : h * A].reshape(B, h, A) - a_n) ** 2, axis=(1, 2))
        err_d = jnp.mean((pred[:, h * A :].reshape(B, h, -1) - d_n) ** 2, axis=(1, 2))
        return idx, err_a, err_d


def _perplexity(usage):
    p = jnp.clip(usage, 1e-12, 1.0)
    return jnp.exp(-jnp.sum(jnp.where(usage > 0, usage * jnp.log(p), 0.0)))


def perplexity_np(usage: np.ndarray) -> float:
    u = np.asarray(usage, dtype=np.float64)
    u = u[u > 0]
    return float(np.exp(-np.sum(u * np.log(u))))


def gather_chunks(observations, actions, idxs, horizon):
    offs = np.arange(horizon + 1)
    obs = observations[idxs[:, None] + offs[None, :]]
    act = actions[idxs[:, None] + offs[None, :-1]]
    return act, obs


def evaluate_tokenizer(tok: Tokenizer, params, codebook, stats, observations, actions, starts, *, batch: int = 8192) -> dict:
    K = tok.cfg.num_codes
    h = tok.cfg.horizon
    codes, ea, ed = [], [], []
    for i in range(0, len(starts), batch):
        idx = starts[i : i + batch]
        act, obs = gather_chunks(observations, actions, idx, h)
        c, a_err, d_err = tok.eval_batch(params, codebook, act, obs, stats)
        codes.append(np.asarray(c))
        ea.append(np.asarray(a_err))
        ed.append(np.asarray(d_err))
    codes = np.concatenate(codes)
    ea = np.concatenate(ea)
    ed = np.concatenate(ed)
    usage = np.bincount(codes, minlength=K).astype(np.float64) / len(codes)
    per_code = []
    for k in range(K):
        m = codes == k
        row: dict[str, Any] = dict(code=k, frequency=float(usage[k]), count=int(m.sum()))
        if m.any():
            sel = starts[m]
            act, obs = gather_chunks(observations, actions, sel[:20000], h)
            row['mean_action'] = act.mean(axis=(0, 1)).tolist()
            step_disp = obs[:, 1:] - obs[:, :-1]
            row['mean_step_displacement'] = step_disp.mean(axis=(0, 1)).tolist()
            row['mean_step_displacement_norm'] = float(np.linalg.norm(step_disp, axis=-1).mean())
            row['mean_chunk_displacement_norm'] = float(np.linalg.norm(obs[:, -1] - obs[:, 0], axis=-1).mean())
            row['action_recon_mse'] = float(ea[m].mean())
            row['delta_recon_mse'] = float(ed[m].mean())
        per_code.append(row)
    ppl = perplexity_np(usage)
    return dict(
        num_samples=int(len(codes)),
        usage=usage.tolist(),
        perplexity=ppl,
        max_usage=float(usage.max()),
        action_recon_mse=float(ea.mean()),
        delta_recon_mse=float(ed.mean()),
        collapsed=bool(usage.max() > COLLAPSE_MAX_USAGE or ppl < COLLAPSE_MIN_PERPLEXITY),
        per_code=per_code,
    )


def save_tokenizer(path: Path, tok: Tokenizer, state: dict, stats: dict, extra: dict | None = None) -> None:
    payload = dict(
        version='intention_tokenizer_v1',
        config=dataclasses.asdict(tok.cfg),
        action_dim=tok.action_dim,
        state_dim=tok.state_dim,
        state=jax.device_get(flax.serialization.to_state_dict(state)),
        stats={k: np.asarray(v) for k, v in stats.items()},
        extra=extra or {},
    )
    atomic_write_pickle(path, payload)


def load_tokenizer(path: Path) -> tuple[Tokenizer, dict, dict, dict]:
    payload = load_pickle(path)
    if payload.get('version') != 'intention_tokenizer_v1':
        raise ValueError(f'Unexpected tokenizer checkpoint version in {path}: {payload.get("version")}')
    cfg_d = dict(payload['config'])
    cfg_d['hidden_dims'] = tuple(cfg_d['hidden_dims'])
    tok = Tokenizer(TokenizerConfig(**cfg_d), payload['action_dim'], payload['state_dim'])
    template = tok.init(0)
    state = flax.serialization.from_state_dict(template, payload['state'])
    stats = {k: jnp.asarray(v) for k, v in payload['stats'].items()}
    return tok, state, stats, payload.get('extra', {})


def train_tokenizer(
    *,
    out_dir: Path,
    train: dict,
    val: dict,
    seed: int,
    total_steps: int,
    save_steps: tuple[int, ...],
    log_every: int = 5000,
    cfg: TokenizerConfig | None = None,
) -> dict:
    """Train the tokenizer (resumable from the latest saved checkpoint)."""
    from intention_pb.pb_io import chunk_valid_starts

    cfg = cfg or TokenizerConfig()
    out_dir = Path(out_dir)
    ck_dir = out_dir / 'checkpoints'
    ck_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'metrics').mkdir(exist_ok=True)
    obs_tr = np.asarray(train['observations'], dtype=np.float32)
    act_tr = np.asarray(train['actions'], dtype=np.float32)
    starts_tr = chunk_valid_starts(train['terminals'], cfg.horizon)
    obs_va = np.asarray(val['observations'], dtype=np.float32)
    act_va = np.asarray(val['actions'], dtype=np.float32)
    starts_va = chunk_valid_starts(val['terminals'], cfg.horizon)

    tok = Tokenizer(cfg, act_tr.shape[-1], obs_tr.shape[-1])
    stats_np = compute_stats(obs_tr, act_tr, starts_tr)
    stats = {k: jnp.asarray(v) for k, v in stats_np.items()}

    state = tok.init(seed)
    start_step = 0
    done = sorted(s for s in save_steps if (ck_dir / f'tokenizer_{s}.pkl').is_file())
    rng = np.random.default_rng(seed)
    if done:
        _, state, _, extra = load_tokenizer(ck_dir / f'tokenizer_{done[-1]}.pkl')
        start_step = int(done[-1])
        rng = np.random.default_rng([seed, start_step])
        print(f'[tokenizer] resumed from step {start_step}', flush=True)
    else:
        idx = starts_tr[rng.integers(len(starts_tr), size=cfg.batch_size)]
        act, obs = gather_chunks(obs_tr, act_tr, idx, cfg.horizon)
        feats, _, _, _ = chunk_features(act, obs, stats)
        state = tok.init_codebook(state, feats, jax.random.PRNGKey(seed + 17))

    log_path = out_dir / 'train_log.jsonl'
    trim_jsonl_to_step(log_path, start_step)
    t0 = time.time()
    usage_acc = np.zeros(cfg.num_codes)
    for step in range(start_step + 1, total_steps + 1):
        idx = starts_tr[rng.integers(len(starts_tr), size=cfg.batch_size)]
        act, obs = gather_chunks(obs_tr, act_tr, idx, cfg.horizon)
        state, info, usage = tok.train_step(state, act, obs, stats)
        if step % log_every == 0 or step == total_steps:
            usage_acc = np.asarray(usage)
            rec = {k: float(v) for k, v in jax.device_get(info).items()}
            rec.update(step=step, elapsed_s=time.time() - t0, batch_usage=usage_acc.tolist())
            if not np.isfinite(rec['loss']):
                raise FloatingPointError(f'NaN tokenizer loss at step {step}')
            with open(log_path, 'a', encoding='utf-8') as f:
                import json

                f.write(json.dumps(rec) + '\n')
            print(f"[tokenizer] step={step} loss={rec['loss']:.4f} la={rec['l_action']:.4f} "
                  f"ld={rec['l_delta']:.4f} ppl={rec['batch_perplexity']:.2f} max={rec['batch_max_usage']:.3f}", flush=True)
        if step in save_steps:
            metrics = evaluate_tokenizer(tok, state['params'], state['codebook'], stats, obs_va, act_va, starts_va)
            metrics['train_batch'] = evaluate_tokenizer(
                tok, state['params'], state['codebook'], stats, obs_tr, act_tr,
                starts_tr[np.random.default_rng(seed + step).integers(len(starts_tr), size=100_000)],
            )
            metrics['step'] = step
            atomic_write_json(out_dir / 'metrics' / f'step_{step}.json', metrics)
            save_tokenizer(ck_dir / f'tokenizer_{step}.pkl', tok, state, stats_np, extra=dict(step=step, seed=seed))
            print(f'[tokenizer] saved step {step}: val ppl={metrics["perplexity"]:.3f} max={metrics["max_usage"]:.3f} '
                  f'collapsed={metrics["collapsed"]}', flush=True)
    final = load_tokenizer(ck_dir / f'tokenizer_{total_steps}.pkl')
    del final
    import json as _json

    with open(out_dir / 'metrics' / f'step_{total_steps}.json') as f:
        return _json.load(f)
