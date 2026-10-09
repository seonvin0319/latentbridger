"""Oracle-free offline full-observation FutureNCE pretraining."""

from __future__ import annotations

from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field

HIDDEN_DIMS = (512, 512, 512)
QUERY_REPR_DIM = 64
LATENT_DIM = 16
TEMPERATURE = 0.1
NOISE_SCALE = 0.01
FEATURE_DROPOUT = 0.05
LEARNING_RATE = 3e-4
VARIANT = 'fullobs_future_nce'
ARCHITECTURE = 'separate_q512x3_ln_gelu_64_proj16_E512x3_ln_gelu_16'


class _HiddenMLP(nn.Module):
    widths: Sequence[int] = HIDDEN_DIMS

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        value = inputs
        for width in self.widths:
            value = nn.Dense(width)(value)
            value = nn.gelu(value)
            value = nn.LayerNorm()(value)
        return value


class QueryEncoder(nn.Module):
    """Independent query subtree: trunk -> 64 representation -> 16 projection."""

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        *,
        return_preprojection: bool = False,
    ) -> jnp.ndarray | tuple[jnp.ndarray, jnp.ndarray]:
        hidden = _HiddenMLP(name='trunk')(observations)
        representation = nn.Dense(QUERY_REPR_DIM, name='representation')(hidden)
        projection = nn.Dense(LATENT_DIM, name='projection')(representation)
        normalized = projection / jnp.maximum(jnp.linalg.norm(projection, axis=-1, keepdims=True), 1e-8)
        if return_preprojection:
            return normalized, representation
        return normalized


class GoalEncoder(nn.Module):
    """Independent downstream-exported full-observation encoder."""

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        *,
        normalize: bool = False,
    ) -> jnp.ndarray:
        hidden = _HiddenMLP(name='trunk')(observations)
        latent = nn.Dense(LATENT_DIM, name='latent')(hidden)
        if normalize:
            return latent / jnp.maximum(jnp.linalg.norm(latent, axis=-1, keepdims=True), 1e-8)
        return latent


class PretrainModuleDict(ModuleDict):
    def encode_goal(self, observations, *, normalize: bool = False):
        return self.modules['goal_encoder'](observations, normalize=normalize)


def augment(
    values: jnp.ndarray,
    std: jnp.ndarray,
    noise_rng: jax.Array,
    dropout_rng: jax.Array,
) -> jnp.ndarray:
    """Independent Gaussian and inverted feature-dropout augmentation."""

    noise = jax.random.normal(noise_rng, values.shape, dtype=values.dtype)
    noisy = values + jnp.asarray(NOISE_SCALE, values.dtype) * std * noise
    keep = jax.random.bernoulli(dropout_rng, p=1.0 - FEATURE_DROPOUT, shape=values.shape)
    return noisy * keep.astype(values.dtype) / (1.0 - FEATURE_DROPOUT)


class FutureNCEPretrainer(flax.struct.PyTreeNode):
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
        query_noise, query_drop, goal_noise, goal_drop = jax.random.split(rng, 4)
        queries = augment(batch['queries'], self.feature_std, query_noise, query_drop)
        goals = augment(batch['goals'], self.feature_std, goal_noise, goal_drop)
        query_embeddings = self.network.select('query_encoder')(queries, params=params)
        goal_embeddings = self.network.select('goal_encoder')(goals, normalize=True, params=params)
        logits = query_embeddings @ goal_embeddings.T / TEMPERATURE
        labels = jnp.arange(logits.shape[0])
        row_losses = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
        ranks = 1 + jnp.sum(logits > jnp.diag(logits)[:, None], axis=1)
        loss = row_losses.mean()
        return loss, {
            'loss': loss,
            'positive_score': jnp.diag(logits).mean(),
            'negative_score': ((logits.sum() - jnp.trace(logits)) / jnp.maximum(logits.size - logits.shape[0], 1)),
            'recall_at_1': (ranks == 1).mean(),
            'positive_rank': ranks.mean(),
        }

    @jax.jit
    def update(self, batch: dict[str, jnp.ndarray]) -> tuple['FutureNCEPretrainer', dict[str, jnp.ndarray]]:
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
    ) -> 'FutureNCEPretrainer':
        observations = jnp.asarray(observations, dtype=jnp.float32)
        feature_std = jnp.asarray(feature_std, dtype=jnp.float32)
        if observations.ndim != 2 or feature_std.shape != observations.shape[-1:]:
            raise ValueError('Invalid observation example or per-dimension std.')
        model = PretrainModuleDict(
            {
                'query_encoder': QueryEncoder(),
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


def goal_encoder_params(agent: FutureNCEPretrainer) -> Any:
    return agent.network.params['modules_goal_encoder']


__all__ = [
    'ARCHITECTURE',
    'FEATURE_DROPOUT',
    'FutureNCEPretrainer',
    'GoalEncoder',
    'HIDDEN_DIMS',
    'LATENT_DIM',
    'NOISE_SCALE',
    'QueryEncoder',
    'TEMPERATURE',
    'VARIANT',
    'augment',
    'goal_encoder_params',
]
