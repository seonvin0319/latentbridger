"""Inference-time route predictor ``p(c | s, g)``.

Pseudo-labels are the frozen route tokenizer's codes. The network does not see the
future trajectory. It is only trained for tasks that pass the representation gate.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from intention_pb import ensure_pb_code_path

ensure_pb_code_path()

from utils.networks import MLP

from route_intention.common import NUM_CODES


class RoutePredictor(nn.Module):
    num_codes: int = NUM_CODES
    hidden: tuple[int, ...] = (256, 256)

    @nn.compact
    def __call__(self, state_n, goal_n):
        return MLP(hidden_dims=(*self.hidden, self.num_codes), activate_final=False, layer_norm=True)(
            jnp.concatenate([state_n, goal_n], axis=-1)
        )


def init_predictor(seed: int, state_dim: int, num_codes: int = NUM_CODES) -> tuple[RoutePredictor, dict, optax.GradientTransformation]:
    model = RoutePredictor(num_codes=num_codes)
    params = model.init(jax.random.PRNGKey(int(seed)), jnp.zeros((2, state_dim)), jnp.zeros((2, state_dim)))['params']
    tx = optax.adam(3e-4)
    return model, dict(params=params, opt_state=tx.init(params)), tx


def predict_step(model: RoutePredictor, state, state_n, goal_n, codes, tx):
    """One SGD step. ``tx`` is closed over so JAX does not try to trace the optimizer."""

    @jax.jit
    def _step(params, opt_state, s, g, y):
        def loss_fn(p):
            logits = model.apply({'params': p}, s, g)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, new_opt = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        return new_params, new_opt, dict(loss=loss, accuracy=acc)

    new_params, new_opt, info = _step(state['params'], state['opt_state'], state_n, goal_n, codes)
    return dict(params=new_params, opt_state=new_opt), info


def classification_metrics(logits: np.ndarray, codes: np.ndarray, top_l: int) -> dict[str, float]:
    logits = np.asarray(logits)
    codes = np.asarray(codes)
    order = np.argsort(-logits, axis=-1)
    top1 = order[:, 0] == codes
    topk = (order[:, : int(top_l)] == codes[:, None]).any(axis=1)
    probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs /= probs.sum(axis=-1, keepdims=True)
    nll = float(-np.log(probs[np.arange(len(codes)), codes] + 1e-12).mean())
    ent = float(-(probs * np.log(probs + 1e-12)).sum(-1).mean())
    # Class-balanced top-1: unweighted mean of per-code accuracies.
    bal = []
    for c in range(logits.shape[-1]):
        m = codes == c
        if m.any():
            bal.append(float(top1[m].mean()))
    return dict(top1=float(top1.mean()), topk=float(topk.mean()), nll=nll, entropy=ent,
                balanced_top1=float(np.mean(bal)) if bal else float('nan'))
