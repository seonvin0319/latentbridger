"""Oracle-free multi-horizon full-observation NCE pretraining."""

from __future__ import annotations

from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from learned_goalspace.pretrain import (
    LATENT_DIM,
    LEARNING_RATE,
    QUERY_REPR_DIM,
    TEMPERATURE,
    GoalEncoder,
    PretrainModuleDict,
    _HiddenMLP,
    augment,
)
from utils.flax_utils import TrainState, nonpytree_field

VARIANT = 'fullobs_multihorizon_nce'
ARCHITECTURE = 'shared_q512x3_ln_gelu_64_proj16x3_short_medium_long_E512x3_ln_gelu_16'
BANDS = ('short', 'medium', 'long')


class MultiHorizonQueryEncoder(nn.Module):
    """Shared query trunk with three distinct projection heads."""

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> dict[str, jnp.ndarray]:
        hidden = _HiddenMLP(name='trunk')(observations)
        representation = nn.Dense(QUERY_REPR_DIM, name='representation')(hidden)
        outputs = {}
        for band in BANDS:
            projection = nn.Dense(LATENT_DIM, name=f'q_{band}')(representation)
            outputs[band] = projection / jnp.maximum(
                jnp.linalg.norm(projection, axis=-1, keepdims=True),
                1e-8,
            )
        return outputs


def _masked_infonce(
    query_embeddings: jnp.ndarray,
    goal_embeddings: jnp.ndarray,
    mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Row-wise InfoNCE using only in-band valid goals as positives/negatives."""

    valid = mask > 0.5
    n_valid = valid.sum()
    safe = n_valid >= 2
    logits = query_embeddings @ goal_embeddings.T / TEMPERATURE
    # Drop invalid band goals from the softmax denominator (and as positives).
    logits = jnp.where(valid[None, :], logits, jnp.asarray(-1e9, logits.dtype))
    labels = jnp.arange(logits.shape[0])
    row_losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    ranks = 1 + jnp.sum(
        (logits > jnp.diag(logits)[:, None]) & valid[None, :],
        axis=1,
    )
    weights = valid.astype(row_losses.dtype)
    denom = jnp.maximum(weights.sum(), 1.0)
    zero = jnp.asarray(0.0, row_losses.dtype)
    loss = jnp.where(safe, (row_losses * weights).sum() / denom, zero)
    recall = jnp.where(
        safe,
        ((ranks == 1).astype(row_losses.dtype) * weights).sum() / denom,
        zero,
    )
    return loss, recall, n_valid.astype(row_losses.dtype)


class MultiHorizonNCEPretrainer(flax.struct.PyTreeNode):
    rng: Any
    network: TrainState
    feature_std: Any
    config: Any = nonpytree_field()

    def encode_goal(self, observations: jnp.ndarray) -> jnp.ndarray:
        """Return the unnormalized 16-D E latent exported downstream."""

        return self.network(
            observations,
            method='encode_goal',
            normalize=False,
        )

    def loss(
        self,
        batch: dict[str, jnp.ndarray],
        params: Any,
        rng: jax.Array,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        keys = jax.random.split(rng, 8)
        queries = augment(batch['queries'], self.feature_std, keys[0], keys[1])
        goals = {
            'short': augment(batch['goals_short'], self.feature_std, keys[2], keys[3]),
            'medium': augment(batch['goals_medium'], self.feature_std, keys[4], keys[5]),
            'long': augment(batch['goals_long'], self.feature_std, keys[6], keys[7]),
        }
        query_heads = self.network.select('query_encoder')(queries, params=params)
        info: dict[str, jnp.ndarray] = {}
        losses = []
        for band in BANDS:
            goal_embeddings = self.network.select('goal_encoder')(
                goals[band],
                normalize=True,
                params=params,
            )
            band_loss, recall, n_valid = _masked_infonce(
                query_heads[band],
                goal_embeddings,
                batch[f'{band}_mask'],
            )
            losses.append(band_loss)
            info[f'loss_{band}'] = band_loss
            info[f'recall_at_1_{band}'] = recall
            info[f'n_valid_{band}'] = n_valid
        loss = losses[0] + losses[1] + losses[2]
        info['loss'] = loss
        return loss, info

    @jax.jit
    def update(self, batch: dict[str, jnp.ndarray]) -> tuple['MultiHorizonNCEPretrainer', dict[str, jnp.ndarray]]:
        next_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(params):
            return self.loss(batch, params, loss_rng)

        network, info = self.network.apply_loss_fn(loss_fn)
        return self.replace(rng=next_rng, network=network), info

    @classmethod
    def create(
        cls,
        seed: int,
        observations: jnp.ndarray,
        feature_std: jnp.ndarray,
        *,
        env_name: str,
    ) -> 'MultiHorizonNCEPretrainer':
        observations = jnp.asarray(observations, dtype=jnp.float32)
        feature_std = jnp.asarray(feature_std, dtype=jnp.float32)
        if observations.ndim != 2 or feature_std.shape != observations.shape[-1:]:
            raise ValueError('Invalid observation example or per-dimension std.')
        model = PretrainModuleDict(
            {
                'query_encoder': MultiHorizonQueryEncoder(),
                'goal_encoder': GoalEncoder(),
            }
        )
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        params = model.init(
            init_rng,
            query_encoder=(observations,),
            goal_encoder=(observations,),
        )['params']
        network = TrainState.create(model, params, tx=optax.adam(LEARNING_RATE))
        config = flax.core.FrozenDict(
            {
                'variant': VARIANT,
                'env_name': str(env_name),
                'obs_dim': int(observations.shape[-1]),
                'latent_dim': LATENT_DIM,
                'architecture': ARCHITECTURE,
                'temperature': TEMPERATURE,
            }
        )
        return cls(rng=rng, network=network, feature_std=feature_std, config=config)


def goal_encoder_params(agent: MultiHorizonNCEPretrainer) -> Any:
    return agent.network.params['modules_goal_encoder']


__all__ = [
    'ARCHITECTURE',
    'BANDS',
    'MultiHorizonNCEPretrainer',
    'MultiHorizonQueryEncoder',
    'VARIANT',
    'goal_encoder_params',
]
