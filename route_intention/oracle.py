"""Oracle-code subgoal model ``p(z | s, g, c_route)``.

The bridge is not trained here. PB's critic is an input to the value weight and is not
in the parameter tree, so it receives no gradient. Only the subgoal flow is updated.
"""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from intention_pb.common import find_pb_run_dir
from intention_pb.conditioned import ConditionedModel
from intention_pb.pb_io import load_pb

from route_intention.chunks import valid_route_starts
from route_intention.common import (
    CANDIDATE_BUDGET,
    EMBED_DIM,
    H_ROUTE,
    NUM_CODES,
    ORACLE_SAVE_STEPS,
    ORACLE_STEPS,
    TOKENIZER_STEPS,
    atomic_write_json,
    atomic_write_pickle,
    load_pickle,
    oracle_dir,
    tokenizer_dir,
    trim_jsonl_to_step,
)
from route_intention.tokenizer import assign_codes, load_route_tokenizer

ORACLE_VERSION = 'route_oracle_subgoal_v1'


def subgoal_only_loss(model: ConditionedModel, sub_params, batch, codes, rng, value_params):
    pb = model.pb
    s = jnp.asarray(batch['observations'], jnp.float32)
    g = jnp.asarray(batch['high_actor_goals'], jnp.float32)
    target_abs = jnp.asarray(batch['high_actor_targets'], jnp.float32)
    target = pb._subgoal_target_for_loss(s, target_abs)
    eps_rng, u_rng = jax.random.split(rng)
    eps = jax.random.normal(eps_rng, target.shape, dtype=target.dtype)
    u = jax.random.uniform(u_rng, (target.shape[0], 1), minval=model.flow_t_min, maxval=1.0 - model.flow_t_min, dtype=target.dtype)
    x_u = (1.0 - u) * eps + u * target
    v_pred = model.subgoal_def.apply(
        {'params': sub_params}, pb._normalize_abs_state(s), pb._normalize_abs_state(g), x_u, u, codes=model._codes(codes),
    )
    fm = jnp.mean((v_pred - (target - eps)) ** 2, axis=-1)
    values = pb._subgoal_values(jnp.concatenate([s, target_abs], 0), jnp.concatenate([g, g], 0), value_params)
    obs_v, tgt_v = jnp.split(values, 2, axis=0)
    weight = jax.lax.stop_gradient(pb._subgoal_mse_weight_from_gap(tgt_v - obs_v))
    loss = jnp.mean(weight * fm)
    return loss, dict(loss=loss, fm=jnp.mean(fm), weight=jnp.mean(weight))


class _OracleTrainer:
    def __init__(self, model: ConditionedModel):
        self.model = model
        self.tx = optax.adam(float(model.pb.config['lr']))

    def init_state(self, seed: int) -> dict:
        full = self.model.init(seed)
        sub = full['params']['subgoal']
        return dict(subgoal=sub, opt_state=self.tx.init(sub), rng=full['rng'], step=jnp.asarray(0, jnp.int32),
                    residual=full['params']['residual'])

    @partial(jax.jit, static_argnums=0)
    def train_step(self, state, batch, codes, value_params):
        rng, sub = jax.random.split(state['rng'])

        def loss_fn(sub_params):
            return subgoal_only_loss(self.model, sub_params, batch, codes, sub, value_params)

        grads, info = jax.grad(loss_fn, has_aux=True)(state['subgoal'])
        updates, opt_state = self.tx.update(grads, state['opt_state'], state['subgoal'])
        new_sub = optax.apply_updates(state['subgoal'], updates)
        info = dict(info, grad_norm=optax.global_norm(grads))
        return dict(subgoal=new_sub, opt_state=opt_state, rng=rng, step=state['step'] + 1, residual=state['residual']), info


def _route_codes(task: str, seed: int, observations, actions, terminals, starts) -> np.ndarray:
    ckpt = tokenizer_dir(task, seed) / 'checkpoints' / f'tokenizer_{TOKENIZER_STEPS}.pkl'
    tok, state, stats, _ = load_route_tokenizer(ckpt)
    if int(tok.cfg.horizon) != H_ROUTE:
        raise ValueError(f'Expected H_route={H_ROUTE}, checkpoint has {tok.cfg.horizon}')
    ok = np.zeros(len(terminals), dtype=bool)
    ok[valid_route_starts(terminals, H_ROUTE)] = True
    if not np.all(ok[np.asarray(starts)]):
        raise ValueError('A training start does not fit inside one H_route window.')
    return assign_codes(tok, state['params'], state['codebook'], observations, actions, np.asarray(starts), stats)


def save_oracle(path: Path, trainer: _OracleTrainer, state: dict, meta: dict) -> None:
    payload = dict(
        version=ORACLE_VERSION,
        num_codes=trainer.model.num_codes,
        intent_dim=trainer.model.subgoal_def.intent_dim,
        state=jax.device_get(flax.serialization.to_state_dict(state)),
        meta=meta,
    )
    atomic_write_pickle(path, payload)


def load_oracle(path: Path, pb_dynamics):
    payload = load_pickle(path)
    if payload.get('version') != ORACLE_VERSION:
        raise ValueError(f'Unexpected oracle checkpoint in {path}')
    model = ConditionedModel(pb_dynamics, num_codes=int(payload['num_codes']), intent_dim=int(payload['intent_dim']))
    trainer = _OracleTrainer(model)
    state = flax.serialization.from_state_dict(trainer.init_state(0), payload['state'])
    return trainer, state, payload.get('meta', {})


def train_oracle_subgoal(*, task: str, seed: int, total_steps: int = ORACLE_STEPS,
                         save_steps: tuple[int, ...] = ORACLE_SAVE_STEPS, log_every: int = 10_000) -> None:
    from main import _intersect_valid_starts, _make_critic_dataset, _sample_shared_idxs
    from utils.datasets import Dataset, PathHGCDataset

    out_dir = oracle_dir(task, seed)
    ck_dir = out_dir / 'checkpoints'
    ck_dir.mkdir(parents=True, exist_ok=True)
    pb = load_pb(find_pb_run_dir(task, seed), need_train=True, need_env=False)
    train_ds = PathHGCDataset(Dataset.create(**pb.train), pb.dynamics_config)
    critic_ds = _make_critic_dataset(pb.train, pb.critic_config)
    common = _intersect_valid_starts(train_ds, critic_ds)
    del critic_ds
    ok = np.zeros(len(pb.train['terminals']), dtype=bool)
    ok[valid_route_starts(pb.train['terminals'], H_ROUTE)] = True
    common = common[ok[common]]
    if len(common) < int(pb.dynamics_config['batch_size']):
        raise RuntimeError(f'Too few route-valid PB starts for {task} seed{seed}: {len(common)}')
    obs = np.asarray(pb.train['observations'], np.float32)
    act = np.asarray(pb.train['actions'], np.float32)
    t0 = time.time()
    codes_at = _route_codes(task, seed, obs, act, pb.train['terminals'], common)
    print(f'[route-oracle] codes for {len(common)} starts in {time.time() - t0:.1f}s', flush=True)
    code_of = np.full(len(obs), -1, np.int32)
    code_of[common] = codes_at
    model = ConditionedModel(pb.dynamics, num_codes=NUM_CODES, intent_dim=EMBED_DIM)
    trainer = _OracleTrainer(model)
    state = trainer.init_state(seed)
    start_step = 0
    done = sorted(s for s in save_steps if (ck_dir / f'oracle_{s}.pkl').is_file())
    if done:
        _, state, _ = load_oracle(ck_dir / f'oracle_{done[-1]}.pkl', pb.dynamics)
        start_step = int(done[-1])
        print(f'[route-oracle] resumed from step {start_step}', flush=True)
    value_params = jax.device_put(pb.critic_value_params())
    residual0 = jax.device_get(state['residual'])
    bs = int(pb.dynamics_config['batch_size'])
    meta = dict(task=task, seed=int(seed), pb_run_dir=str(pb.run_dir), horizon=H_ROUTE, num_codes=NUM_CODES, embed_dim=EMBED_DIM)
    atomic_write_json(out_dir / 'meta.json', meta)
    np.random.seed(int(seed) * 1_000_003 + start_step)
    log_path = out_dir / 'train_log.jsonl'
    trim_jsonl_to_step(log_path, start_step)
    t_start = time.time()
    for step in range(start_step + 1, int(total_steps) + 1):
        idxs = _sample_shared_idxs(common, bs)
        batch_np = train_ds.sample(bs, idxs=idxs)
        batch = {k: np.asarray(batch_np[k], np.float32) for k in ('observations', 'high_actor_goals', 'high_actor_targets')}
        codes = code_of[idxs]
        if np.any(codes < 0):
            raise RuntimeError('Sampled a start without a route code.')
        state, info = trainer.train_step(state, batch, jnp.asarray(codes), value_params)
        if step % log_every == 0 or step == total_steps:
            rec = {k: float(v) for k, v in jax.device_get(info).items()}
            rec.update(step=step, elapsed_s=time.time() - t_start)
            if not np.isfinite(rec['loss']):
                raise FloatingPointError(f'NaN oracle loss at step {step}')
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec) + '\n')
            print(f"[route-oracle] {task} s{seed} step={step} loss={rec['loss']:.4f} {rec['elapsed_s']:.0f}s", flush=True)
        if step in save_steps:
            save_oracle(ck_dir / f'oracle_{step}.pkl', trainer, state, dict(meta, step=step))
            print(f'[route-oracle] saved step {step}', flush=True)
    after = jax.tree_util.tree_leaves(jax.device_get(state['residual']))
    before = jax.tree_util.tree_leaves(residual0)
    for a, b in zip(after, before):
        if not np.array_equal(np.asarray(a), np.asarray(b)):
            raise RuntimeError('Bridge residual changed during oracle subgoal training.')


def normalized_best_and_mean(pred: np.ndarray, target: np.ndarray, scale: np.ndarray) -> tuple[float, float]:
    """``pred`` is ``[B, N, D]`` absolute subgoals. Returns ``(mean error, best-of-N error)``."""
    scale = np.maximum(np.asarray(scale, np.float32), 1e-3)
    err = np.linalg.norm((pred - target[:, None, :]) / scale, axis=-1)
    return float(err.mean()), float(err.min(axis=1).mean())


def evaluate_oracle_subgoal(*, task: str, seed: int, step: int = ORACLE_STEPS, num: int = 1024,
                            budget: int = CANDIDATE_BUDGET) -> dict:
    """Compare frozen PB subgoals with oracle route-conditioned subgoals on the val split."""
    pb = load_pb(find_pb_run_dir(task, seed), need_train=False, need_env=False)
    trainer, state, _meta = load_oracle(oracle_dir(task, seed) / 'checkpoints' / f'oracle_{int(step)}.pkl', pb.dynamics)
    from utils.datasets import Dataset, PathHGCDataset

    ds = PathHGCDataset(Dataset.create(**pb.val), pb.dynamics_config)
    np.random.seed(50_000 + int(seed))
    raw = ds.sample(int(num) * 2)
    starts = np.asarray(raw['trajectory_start_indices'], np.int64)
    ok = np.zeros(len(pb.val['terminals']), dtype=bool)
    ok[valid_route_starts(pb.val['terminals'], H_ROUTE)] = True
    keep = ok[starts]
    obs = np.asarray(raw['observations'], np.float32)[keep][:num]
    goal = np.asarray(raw['high_actor_goals'], np.float32)[keep][:num]
    target = np.asarray(raw['high_actor_targets'], np.float32)[keep][:num]
    starts = starts[keep][:num]
    obs_all = np.asarray(pb.val['observations'], np.float32)
    act_all = np.asarray(pb.val['actions'], np.float32)
    codes = _route_codes(task, seed, obs_all, act_all, pb.val['terminals'], starts)
    scale = np.maximum(np.std(target - obs, axis=0), 1e-3)
    rng = jax.random.PRNGKey(7 + int(seed))
    pb_cands, _mu = pb.dynamics.sample_subgoal_candidates(
        jnp.asarray(obs), jnp.asarray(goal), rng, num_candidates=int(budget), include_mean=False,
    )
    route_cands = trainer.model.sample_with_codes(
        state['subgoal'], jnp.asarray(obs), jnp.asarray(goal), jnp.asarray(codes), rng,
        temperature=1.0, num=int(budget),
    )
    pb_mean, pb_best = normalized_best_and_mean(np.asarray(pb_cands), target, scale)
    rt_mean, rt_best = normalized_best_and_mean(np.asarray(route_cands), target, scale)
    per_code = {}
    err = np.linalg.norm((np.asarray(route_cands) - target[:, None, :]) / scale, axis=-1).min(1)
    for c in range(NUM_CODES):
        m = codes == c
        if m.any():
            per_code[str(c)] = float(err[m].mean())
    result = dict(
        task=task, seed=int(seed), step=int(step), num=int(len(obs)), budget=int(budget),
        pb_mean_error=pb_mean, pb_best_error=pb_best,
        route_mean_error=rt_mean, route_best_error=rt_best,
        per_code_best_error=per_code,
    )
    atomic_write_json(oracle_dir(task, seed) / 'eval.json', result)
    print(f"[route-oracle-eval] {task} s{seed} pb_best={pb_best:.4f} route_best={rt_best:.4f}", flush=True)
    return result
