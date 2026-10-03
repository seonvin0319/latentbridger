"""LatentBridger: an experimental action-conditioned contrastive successor.

This agent is an ablation branch, **not** a change to the released PathBridger
method in :mod:`agents.pathbridger`.  It tests whether explicit state-space
subgoals, transitive-relabelling (TRL) ranking, and the inverse-dynamics model
can all be removed in favour of

* Module A, an action-conditioned temporal contrastive critic

      C(s, a, g) = phi_sa(s, a)^T psi(g) / tau,

  whose goal encoder ``psi`` doubles as the control interface, and
* Module B, a rectified flow that generates a short *latent* trajectory prefix

      z_1, ..., z_H ~ v_eta( . | psi(s_t), psi(g) ),   H = 5,

  which the latent-conditioned actor consumes one step at a time under
  receding-horizon replanning.

Every module lives in one unified checkpoint so the staged research protocol
(critic -> actor -> flow) can restore and freeze predecessors.  Freezing is
enforced by the optimizer itself: frozen modules are routed through
``optax.set_to_zero`` so their parameters are bit-identical across a stage.
"""

from __future__ import annotations

import pickle
import random
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, resolve_checkpoint
from utils.goal_representation import goal_representation
from utils.latent_datasets import (
    ACTOR_GOAL_SAMPLING_MODES,
    FLOW_TARGET_MODES,
    sparse_prefix_offsets,
)
from utils.networks import MLP

_MODULE_NAMES = ('phi_sa', 'phi_s', 'psi', 'actor', 'flow')

STAGES = ('critic', 'actor', 'flow', 'joint')
VARIANTS = (
    'gcbc',
    'state_cl',
    'sa_cl',
    'sa_cl_bc',
    'sa_cl_bc_actnce',
    'latent_rf',
    'latent_rf_actnce',
    # v1: latent temporal-scale mismatch.
    'actnce_local',
    'actnce_multihorizon',
    'latent_rf_sparse',
    'actnce_geometric',
)
CRITIC_TYPES = ('none', 'state', 'sa')
ACTOR_GOAL_INPUTS = ('raw', 'latent')
ACTOR_OBJECTIVES = ('bc', 'contrastive')
EVAL_MODES = ('direct_goal', 'latent_flow')

# v1 defaults.  The v0 pilot showed the consecutive five-step latent prefix
# spans less cosine distance than the flow's own reconstruction error, so the
# multi-horizon actor and the sparse bridge exist to widen that target.
V1_ACTOR_GOAL_OFFSETS = (1, 2, 4, 8, 16)

# Structural choices implied by each named variant.  They are written into the
# config so a run is fully reproducible from its serialized flags alone.
VARIANT_SETTINGS: dict[str, dict[str, Any]] = {
    'gcbc': dict(
        critic_type='none',
        actor_goal_input='raw',
        actor_objective='bc',
        actor_bc_coef=10.0,
        action_nce_coef=0.0,
        use_flow=False,
        eval_mode='direct_goal',
    ),
    'state_cl': dict(
        critic_type='state',
        actor_goal_input='latent',
        actor_objective='bc',
        actor_bc_coef=10.0,
        action_nce_coef=0.0,
        use_flow=False,
        eval_mode='direct_goal',
    ),
    'sa_cl': dict(
        critic_type='sa',
        actor_goal_input='latent',
        actor_objective='contrastive',
        actor_bc_coef=0.0,
        action_nce_coef=0.0,
        use_flow=False,
        eval_mode='direct_goal',
    ),
    'sa_cl_bc': dict(
        critic_type='sa',
        actor_goal_input='latent',
        actor_objective='contrastive',
        actor_bc_coef=10.0,
        action_nce_coef=0.0,
        use_flow=False,
        eval_mode='direct_goal',
    ),
    'sa_cl_bc_actnce': dict(
        critic_type='sa',
        actor_goal_input='latent',
        actor_objective='contrastive',
        actor_bc_coef=10.0,
        action_nce_coef=1.0,
        use_flow=False,
        eval_mode='direct_goal',
    ),
    'latent_rf': dict(
        critic_type='sa',
        actor_goal_input='latent',
        actor_objective='contrastive',
        actor_bc_coef=10.0,
        action_nce_coef=0.0,
        use_flow=True,
        eval_mode='latent_flow',
    ),
    # Module B on top of the action-sensitive critic.  Kept as its own variant
    # so the original latent_rf result stays intact and comparable.
    'latent_rf_actnce': dict(
        critic_type='sa',
        actor_goal_input='latent',
        actor_objective='contrastive',
        actor_bc_coef=10.0,
        action_nce_coef=1.0,
        use_flow=True,
        eval_mode='latent_flow',
    ),
    # ------------------------------------------------------------------
    # v1: does the latent bridge fail because its target is too fine-grained?
    # All three keep the v1 default critic (SA-InfoNCE + action-NCE) and BC.
    # ------------------------------------------------------------------
    'actnce_local': dict(
        critic_type='sa',
        actor_goal_input='latent',
        actor_objective='contrastive',
        actor_bc_coef=10.0,
        action_nce_coef=1.0,
        actor_goal_offsets=(1,),
        use_flow=False,
        eval_mode='direct_goal',
    ),
    'actnce_multihorizon': dict(
        critic_type='sa',
        actor_goal_input='latent',
        actor_objective='contrastive',
        actor_bc_coef=10.0,
        action_nce_coef=1.0,
        actor_goal_offsets=V1_ACTOR_GOAL_OFFSETS,
        use_flow=False,
        eval_mode='direct_goal',
    ),
    'latent_rf_sparse': dict(
        critic_type='sa',
        actor_goal_input='latent',
        actor_objective='contrastive',
        actor_bc_coef=10.0,
        action_nce_coef=1.0,
        actor_goal_offsets=V1_ACTOR_GOAL_OFFSETS,
        use_flow=True,
        flow_target_mode='sparse',
        eval_mode='latent_flow',
    ),
    # Five discrete rungs beat a single one; a geometric horizon replaces the
    # rungs with a smooth distribution matched to the critic's own discount.
    'actnce_geometric': dict(
        critic_type='sa',
        actor_goal_input='latent',
        actor_objective='contrastive',
        actor_bc_coef=10.0,
        action_nce_coef=1.0,
        actor_goal_sampling='geometric',
        use_flow=False,
        eval_mode='direct_goal',
    ),
}

# Every released v0 variant predates these knobs; filling them in here keeps
# their sampling byte-identical while letting the variant table own them.
for _settings in VARIANT_SETTINGS.values():
    _settings.setdefault('actor_goal_sampling', 'offsets')
    _settings.setdefault('actor_goal_offsets', (1,))
    _settings.setdefault('flow_target_mode', 'consecutive')
    # A sparse bridge must be replanned every step: its k-th waypoint is h_k
    # steps ahead, not k steps ahead, so executing the prefix as a chunk would
    # hand the actor a target it is nowhere near reaching.
    _settings.setdefault(
        'replan_interval',
        1 if _settings['flow_target_mode'] == 'sparse' else 5,
    )
del _settings

_STAGE_TRAINABLE: dict[str, tuple[str, ...]] = {
    'critic': ('phi_sa', 'phi_s', 'psi'),
    'actor': ('actor',),
    'flow': ('flow',),
    'joint': _MODULE_NAMES,
}

_BATCH_KEYS_BY_STAGE: dict[str, tuple[str, ...]] = {
    'critic': ('observations', 'actions', 'contrastive_goals'),
    'actor': ('observations', 'actions', 'actor_goals'),
    'flow': ('observations', 'bridge_goals', 'bridge_targets'),
    'joint': (
        'observations',
        'actions',
        'contrastive_goals',
        'actor_goals',
        'bridge_goals',
        'bridge_targets',
    ),
}

_EPS = 1e-8


def _l2_normalize(x: jnp.ndarray) -> jnp.ndarray:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), _EPS)


class StateActionEncoder(nn.Module):
    """``phi_sa(s, a) -> R^d``: the action-conditioned anchor encoder."""

    repr_dim: int
    hidden_dims: Sequence[int]
    layer_norm: bool = True

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        return MLP(
            (*self.hidden_dims, self.repr_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(inputs)


class StateEncoder(nn.Module):
    """``phi_s(s) -> R^d``: the state-only anchor encoder for the CL ablation."""

    repr_dim: int
    hidden_dims: Sequence[int]
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        return MLP(
            (*self.hidden_dims, self.repr_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(observations)


class GoalEncoder(nn.Module):
    """``psi(g) -> R^d``: the shared goal encoder and control interface.

    The goal representation defaults to ``full``, matching
    :class:`agents.pathbridger.ScalarTransitiveValue`.  The reduced ``phi``
    endpoint representation is reachable only by explicit configuration.
    """

    repr_dim: int
    hidden_dims: Sequence[int]
    env_name: str
    goal_representation_mode: str = 'full'
    phi_goal_obs_indices: tuple[int, ...] = ()
    layer_norm: bool = True

    @nn.compact
    def __call__(self, goals: jnp.ndarray) -> jnp.ndarray:
        goal_inputs = goal_representation(
            goals,
            self.goal_representation_mode,
            self.phi_goal_obs_indices,
            env_name=self.env_name,
        )
        return MLP(
            (*self.hidden_dims, self.repr_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(goal_inputs)


class LatentConditionedActor(nn.Module):
    """``pi(s, z_goal) -> a`` squashed into the environment action bounds."""

    action_dim: int
    hidden_dims: Sequence[int]
    action_low: tuple[float, ...]
    action_high: tuple[float, ...]
    layer_norm: bool = True

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        goal_inputs: jnp.ndarray,
    ) -> jnp.ndarray:
        inputs = jnp.concatenate([observations, goal_inputs], axis=-1)
        pre_activations = MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(inputs)
        low = jnp.asarray(self.action_low, dtype=pre_activations.dtype)
        high = jnp.asarray(self.action_high, dtype=pre_activations.dtype)
        # tanh keeps the actor strictly inside the real action box, so it can
        # never game the contrastive critic with out-of-support actions.
        return low + 0.5 * (high - low) * (jnp.tanh(pre_activations) + 1.0)


class LatentPrefixFlow(nn.Module):
    """Rectified-flow velocity field over the joint ``[H, d]`` latent prefix."""

    repr_dim: int
    action_horizon: int
    hidden_dims: Sequence[int]
    layer_norm: bool = True

    @nn.compact
    def __call__(
        self,
        noisy_prefix: jnp.ndarray,
        times: jnp.ndarray,
        state_latents: jnp.ndarray,
        goal_latents: jnp.ndarray,
    ) -> jnp.ndarray:
        batch_size = noisy_prefix.shape[0]
        flat_prefix = noisy_prefix.reshape(batch_size, -1)
        times = jnp.asarray(times, dtype=flat_prefix.dtype)
        if times.ndim == 1:
            times = times[:, None]
        inputs = jnp.concatenate(
            [flat_prefix, state_latents, goal_latents, times],
            axis=-1,
        )
        velocities = MLP(
            (*self.hidden_dims, self.action_horizon * self.repr_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(inputs)
        return velocities.reshape(batch_size, self.action_horizon, self.repr_dim)


def _as_float_tuple(value: Any, *, size: int, name: str) -> tuple[float, ...]:
    array = np.broadcast_to(np.asarray(value, dtype=np.float64), (size,))
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must be finite, got {value!r}.')
    return tuple(float(entry) for entry in array)


def _as_comparable(value: Any) -> Any:
    """Normalize a setting so a ConfigDict list compares equal to a tuple."""

    if isinstance(value, (list, tuple)):
        return tuple(_as_comparable(entry) for entry in value)
    return value


def _freeze_config(config: dict[str, Any]) -> Any:
    """Make the config hashable so it can ride along as a static jit argument."""

    hashable: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, (list, tuple)):
            hashable[key] = tuple(value)
        elif isinstance(value, np.ndarray):
            hashable[key] = tuple(value.reshape(-1).tolist())
        else:
            hashable[key] = value
    return flax.core.FrozenDict(hashable)


class LatentBridgerAgent(flax.struct.PyTreeNode):
    """Staged LatentBridger training and evaluation state."""

    rng: Any
    network: TrainState
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------------
    def _maybe_normalize(self, embeddings: jnp.ndarray) -> jnp.ndarray:
        if bool(self.config['repr_norm']):
            return _l2_normalize(embeddings)
        return embeddings

    def encode_goal(
        self,
        goals: jnp.ndarray,
        *,
        params: Any | None = None,
    ) -> jnp.ndarray:
        """``psi(g)`` after the configured normalization.

        This is *the* latent interface: the actor consumes it, the flow
        generates it, and the critic scores against it.
        """

        return self._maybe_normalize(
            self.network.select('psi')(goals, params=params)
        )

    def encode_state_action(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        *,
        params: Any | None = None,
    ) -> jnp.ndarray:
        """``phi_sa(s, a)`` after the configured normalization."""

        return self._maybe_normalize(
            self.network.select('phi_sa')(observations, actions, params=params)
        )

    def encode_state(
        self,
        observations: jnp.ndarray,
        *,
        params: Any | None = None,
    ) -> jnp.ndarray:
        """``phi_s(s)`` after the configured normalization."""

        return self._maybe_normalize(
            self.network.select('phi_s')(observations, params=params)
        )

    def _temperature(self) -> jnp.ndarray:
        return jnp.asarray(
            self.config['contrastive_temperature'],
            dtype=jnp.float32,
        )

    def _paired_score(
        self,
        anchors: jnp.ndarray,
        goal_latents: jnp.ndarray,
    ) -> jnp.ndarray:
        return jnp.sum(anchors * goal_latents, axis=-1) / self._temperature()

    def _matrix_score(
        self,
        anchors: jnp.ndarray,
        goal_latents: jnp.ndarray,
    ) -> jnp.ndarray:
        return anchors @ goal_latents.T / self._temperature()

    def critic_score(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        goals: jnp.ndarray,
        *,
        params: Any | None = None,
    ) -> jnp.ndarray:
        """``C(s, a, g)`` for paired rows.

        ``params=None`` evaluates the critic at the agent's current parameters.
        Gradients therefore flow into ``actions`` but never into the critic
        weights, which is exactly the frozen-critic actor objective.
        """

        if str(self.config['critic_type']) == 'state':
            anchors = self.encode_state(observations, params=params)
        else:
            anchors = self.encode_state_action(observations, actions, params=params)
        return self._paired_score(anchors, self.encode_goal(goals, params=params))

    def actor_goal_inputs(
        self,
        goals: jnp.ndarray,
        *,
        params: Any | None = None,
    ) -> jnp.ndarray:
        """Whatever the actor conditions on: ``psi(g)`` or the raw goal."""

        if str(self.config['actor_goal_input']) == 'latent':
            return self.encode_goal(goals, params=params)
        return goal_representation(
            goals,
            str(self.config['goal_representation_mode']),
            tuple(self.config['phi_goal_obs_indices']),
            env_name=str(self.config['env_name']),
        )

    # ------------------------------------------------------------------
    # Module A: temporal contrastive critic
    # ------------------------------------------------------------------
    def contrastive_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """In-batch InfoNCE with trajectory-future positives on the diagonal."""

        observations = batch['observations']
        goals = batch['contrastive_goals']
        critic_type = str(self.config['critic_type'])

        raw_anchors = (
            self.network.select('phi_s')(observations, params=grad_params)
            if critic_type == 'state'
            else self.network.select('phi_sa')(
                observations,
                batch['actions'],
                params=grad_params,
            )
        )
        raw_goal_latents = self.network.select('psi')(goals, params=grad_params)
        anchors = self._maybe_normalize(raw_anchors)
        goal_latents = self._maybe_normalize(raw_goal_latents)

        logits = self._matrix_score(anchors, goal_latents)
        batch_size = logits.shape[0]
        labels = jnp.arange(batch_size)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

        logsumexp_coef = float(self.config['logsumexp_coef'])
        logsumexp_penalty = jnp.mean(
            jnp.square(jax.scipy.special.logsumexp(logits, axis=-1))
        )
        if logsumexp_coef != 0.0:
            loss = loss + logsumexp_coef * logsumexp_penalty

        diagonal = jnp.diag(logits)
        off_diagonal_mask = 1.0 - jnp.eye(batch_size)
        negative_logit = jnp.sum(logits * off_diagonal_mask) / jnp.maximum(
            jnp.sum(off_diagonal_mask),
            1.0,
        )
        info = {
            'contrastive/loss': loss,
            'contrastive/positive_logit': diagonal.mean(),
            'contrastive/negative_logit': negative_logit,
            'contrastive/logit_gap': diagonal.mean() - negative_logit,
            'contrastive/embedding_norm_phi': jnp.linalg.norm(
                raw_anchors,
                axis=-1,
            ).mean(),
            'contrastive/embedding_norm_psi': jnp.linalg.norm(
                raw_goal_latents,
                axis=-1,
            ).mean(),
            'contrastive/logsumexp_penalty': logsumexp_penalty,
            'contrastive/accuracy': jnp.mean(
                (jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)
            ),
        }
        return loss, info

    def action_contrastive_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Softmax contrast of the data action against K shuffled alternatives.

        Negatives are cyclic shifts of the batch's own actions, so the cost is
        ``K`` extra encoder rows rather than a ``B x B`` action grid.
        """

        observations = batch['observations']
        actions = batch['actions']
        goals = batch['contrastive_goals']
        batch_size = observations.shape[0]
        num_negatives = int(self.config['num_action_negatives'])
        num_negatives = max(1, min(num_negatives, batch_size - 1))

        goal_latents = self._maybe_normalize(
            self.network.select('psi')(goals, params=grad_params)
        )
        positive_anchors = self._maybe_normalize(
            self.network.select('phi_sa')(observations, actions, params=grad_params)
        )
        positive_scores = self._paired_score(positive_anchors, goal_latents)

        negative_actions = jnp.stack(
            [jnp.roll(actions, shift=shift + 1, axis=0) for shift in range(num_negatives)],
            axis=0,
        )
        tiled_observations = jnp.broadcast_to(
            observations[None],
            (num_negatives, *observations.shape),
        ).reshape(num_negatives * batch_size, -1)
        negative_anchors = self._maybe_normalize(
            self.network.select('phi_sa')(
                tiled_observations,
                negative_actions.reshape(num_negatives * batch_size, -1),
                params=grad_params,
            )
        ).reshape(num_negatives, batch_size, -1)
        negative_scores = self._paired_score(
            negative_anchors,
            jnp.broadcast_to(goal_latents[None], negative_anchors.shape),
        )

        logits = jnp.concatenate(
            [positive_scores[:, None], negative_scores.T],
            axis=-1,
        )
        labels = jnp.zeros(batch_size, dtype=jnp.int32)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        info = {
            'action_nce/loss': loss,
            'action_nce/positive_score': positive_scores.mean(),
            'action_nce/negative_score': negative_scores.mean(),
            'action_nce/gap': positive_scores.mean() - negative_scores.mean(),
            'action_nce/accuracy': jnp.mean(
                (jnp.argmax(logits, axis=-1) == labels).astype(jnp.float32)
            ),
            'action_nce/num_negatives': jnp.asarray(num_negatives, dtype=jnp.float32),
        }
        return loss, info

    def critic_total_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        loss, info = self.contrastive_loss(batch, grad_params)
        action_nce_coef = float(self.config['action_nce_coef'])
        if action_nce_coef != 0.0:
            action_loss, action_info = self.action_contrastive_loss(batch, grad_params)
            loss = loss + action_nce_coef * action_loss
            info = {**info, **action_info}
        info['critic/loss'] = loss
        return loss, info

    # ------------------------------------------------------------------
    # Latent-conditioned actor
    # ------------------------------------------------------------------
    def actor_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Frozen-critic score maximization plus optional BC regularization."""

        observations = batch['observations']
        data_actions = batch['actions']
        goals = batch['actor_goals']

        # The goal encoding is a fixed conditioning vector for the actor; the
        # critic parameters below are likewise fixed, but the score remains
        # differentiable with respect to the predicted action.
        goal_inputs = jax.lax.stop_gradient(self.actor_goal_inputs(goals))
        predicted_actions = self.network.select('actor')(
            observations,
            goal_inputs,
            params=grad_params,
        )

        squared_errors = jnp.sum(
            jnp.square(predicted_actions - data_actions),
            axis=-1,
        )
        bc_loss = squared_errors.mean()
        actor_bc_coef = float(self.config['actor_bc_coef'])

        if str(self.config['actor_objective']) == 'contrastive':
            critic_scores = self.critic_score(
                observations,
                predicted_actions,
                goals,
            )
            score_loss = -critic_scores.mean()
            loss = score_loss + actor_bc_coef * bc_loss
        else:
            critic_scores = jax.lax.stop_gradient(
                self.critic_score(observations, predicted_actions, goals)
            )
            score_loss = jnp.zeros((), dtype=bc_loss.dtype)
            loss = bc_loss

        info = {
            'actor/loss': loss,
            'actor/score_loss': score_loss,
            'actor/bc_loss': bc_loss,
            'actor/action_mse': jnp.mean(
                jnp.square(predicted_actions - data_actions)
            ),
            'actor/critic_score': critic_scores.mean(),
            'actor/action_abs_mean': jnp.mean(jnp.abs(predicted_actions)),
        }
        return loss, info

    # ------------------------------------------------------------------
    # Module B: rectified-flow latent bridge
    # ------------------------------------------------------------------
    def _encode_prefix(
        self,
        prefix_states: jnp.ndarray,
        *,
        params: Any | None = None,
    ) -> jnp.ndarray:
        batch_size, horizon, state_dim = prefix_states.shape
        latents = self.encode_goal(
            prefix_states.reshape(batch_size * horizon, state_dim),
            params=params,
        )
        return latents.reshape(batch_size, horizon, -1)

    def flow_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        rng: jax.Array,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Joint rectified-flow matching over the whole ``[B, H, d]`` prefix."""

        observations = batch['observations']
        goals = batch['bridge_goals']
        prefix_states = batch['bridge_targets']

        state_latents = self.encode_goal(observations)
        goal_latents = self.encode_goal(goals)
        prefix_latents = self._encode_prefix(prefix_states)
        if bool(self.config['flow_stop_psi_gradient']):
            state_latents = jax.lax.stop_gradient(state_latents)
            goal_latents = jax.lax.stop_gradient(goal_latents)
            prefix_latents = jax.lax.stop_gradient(prefix_latents)

        noise_rng, time_rng = jax.random.split(rng)
        noise = jax.random.normal(noise_rng, prefix_latents.shape, dtype=jnp.float32)
        times = jax.random.uniform(
            time_rng,
            (prefix_latents.shape[0], 1),
            dtype=jnp.float32,
        )
        broadcast_times = times[:, :, None]
        noisy_prefix = (1.0 - broadcast_times) * noise + broadcast_times * prefix_latents
        target_velocities = prefix_latents - noise

        predicted_velocities = self.network.select('flow')(
            noisy_prefix,
            times,
            state_latents,
            goal_latents,
            params=grad_params,
        )
        squared_errors = jnp.square(predicted_velocities - target_velocities)
        loss = squared_errors.mean()
        info = {
            'flow/loss': loss,
            'flow/velocity_mse': loss,
            'flow/target_norm': jnp.linalg.norm(
                target_velocities.reshape(target_velocities.shape[0], -1),
                axis=-1,
            ).mean(),
            'flow/time_mean': times.mean(),
            'flow/prefix_latent_norm': jnp.linalg.norm(
                prefix_latents,
                axis=-1,
            ).mean(),
        }
        return loss, info

    def _integrate_latent_prefix(
        self,
        state_latents: jnp.ndarray,
        goal_latents: jnp.ndarray,
        initial_noise: jnp.ndarray,
    ) -> jnp.ndarray:
        """Fixed-step Euler integration of the latent rectified flow."""

        flow_steps = int(self.config['flow_steps'])
        step_size = jnp.asarray(1.0 / flow_steps, dtype=jnp.float32)
        prefix = initial_noise
        for step in range(flow_steps):
            times = jnp.full(
                (prefix.shape[0], 1),
                step / flow_steps,
                dtype=jnp.float32,
            )
            velocities = self.network.select('flow')(
                prefix,
                times,
                state_latents,
                goal_latents,
            )
            prefix = prefix + step_size * velocities
        if bool(self.config['repr_norm']) and bool(self.config['flow_renormalize']):
            # Keep generated latents on the same manifold the actor was trained
            # on; without this the actor sees off-sphere conditioning vectors.
            prefix = _l2_normalize(prefix)
        return prefix

    # ------------------------------------------------------------------
    # Staged updates
    # ------------------------------------------------------------------
    def total_loss(
        self,
        batch: dict[str, jnp.ndarray],
        grad_params: Any,
        rng: jax.Array,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        """Dispatch the objective for the configured training stage."""

        stage = str(self.config['stage'])
        info: dict[str, jnp.ndarray] = {}
        loss = jnp.zeros((), dtype=jnp.float32)

        trains_critic = stage in ('critic', 'joint')
        trains_actor = stage in ('actor', 'joint')
        trains_flow = stage in ('flow', 'joint')

        if trains_critic and str(self.config['critic_type']) != 'none':
            critic_loss, critic_info = self.critic_total_loss(batch, grad_params)
            loss = loss + critic_loss
            info.update(critic_info)
        if trains_actor:
            actor_loss, actor_info = self.actor_loss(batch, grad_params)
            loss = loss + actor_loss
            info.update(actor_info)
        if trains_flow and bool(self.config['use_flow']):
            flow_loss, flow_info = self.flow_loss(batch, grad_params, rng)
            loss = loss + flow_loss
            info.update(flow_info)

        info['loss/total'] = loss
        return loss, info

    def update(
        self,
        batch: dict[str, jnp.ndarray],
    ) -> tuple['LatentBridgerAgent', dict[str, jnp.ndarray]]:
        """Apply one gradient update for the configured stage."""

        stage = str(self.config['stage'])
        required = _BATCH_KEYS_BY_STAGE[stage]
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f'LatentBridger {stage} batch is missing keys: {missing}')
        if stage in ('flow', 'joint') and bool(self.config['use_flow']):
            horizon = int(self.config['action_horizon'])
            if int(batch['bridge_targets'].shape[1]) != horizon:
                raise ValueError(
                    'bridge_targets must have shape [B, action_horizon, D]; '
                    f'expected length {horizon}, got '
                    f'{batch["bridge_targets"].shape[1]}.'
                )
        return self._update_impl(batch)

    @jax.jit
    def _update_impl(
        self,
        batch: dict[str, jnp.ndarray],
    ) -> tuple['LatentBridgerAgent', dict[str, jnp.ndarray]]:
        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, loss_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(rng=new_rng, network=new_network), info

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @jax.jit
    def goal_latents(self, goals: jnp.ndarray) -> jnp.ndarray:
        """``psi(g)`` for evaluation and diagnostics."""

        squeeze = goals.ndim == 1
        if squeeze:
            goals = goals[None, :]
        latents = self.encode_goal(goals)
        return latents[0] if squeeze else latents

    @jax.jit
    def actor_conditioning(self, goals: jnp.ndarray) -> jnp.ndarray:
        """The actor's goal-conditioning vector for raw environment goals."""

        squeeze = goals.ndim == 1
        if squeeze:
            goals = goals[None, :]
        conditioning = self.actor_goal_inputs(goals)
        return conditioning[0] if squeeze else conditioning

    @jax.jit
    def sample_actions(
        self,
        observations: jnp.ndarray,
        goal_inputs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Deterministic action for an already-encoded conditioning vector."""

        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None, :]
            goal_inputs = goal_inputs[None, :]
        actions = self.network.select('actor')(observations, goal_inputs)
        return actions[0] if squeeze else actions

    @jax.jit
    def sample_actions_from_goals(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
    ) -> jnp.ndarray:
        """``pi(s, psi(g))`` straight from a raw goal observation."""

        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None, :]
            goals = goals[None, :]
        actions = self.network.select('actor')(
            observations,
            self.actor_goal_inputs(goals),
        )
        return actions[0] if squeeze else actions

    @jax.jit
    def sample_latent_prefix(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        seed: jax.Array,
    ) -> jnp.ndarray:
        """Generate ``[B, H, d]`` latent prefixes with the rectified flow."""

        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None, :]
            goals = goals[None, :]
        state_latents = self.encode_goal(observations)
        goal_latents = self.encode_goal(goals)
        noise = jax.random.normal(
            seed,
            (
                observations.shape[0],
                int(self.config['action_horizon']),
                int(self.config['repr_dim']),
            ),
            dtype=jnp.float32,
        )
        noise = jnp.asarray(self.config['flow_noise_scale'], dtype=jnp.float32) * noise
        prefix = self._integrate_latent_prefix(state_latents, goal_latents, noise)
        return prefix[0] if squeeze else prefix

    @jax.jit
    def critic_score_matrix(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        goals: jnp.ndarray,
    ) -> jnp.ndarray:
        """``C(s_i, a_i, g_j)`` for every anchor/goal pair in a batch."""

        if str(self.config['critic_type']) == 'state':
            anchors = self.encode_state(observations)
        else:
            anchors = self.encode_state_action(observations, actions)
        return self._matrix_score(anchors, self.encode_goal(goals))

    @jax.jit
    def critic_scores(
        self,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        goals: jnp.ndarray,
    ) -> jnp.ndarray:
        """Paired ``C(s, a, g)``."""

        return self.critic_score(observations, actions, goals)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        config: Any,
        *,
        stage: str = 'joint',
        action_low: Any = None,
        action_high: Any = None,
    ) -> 'LatentBridgerAgent':
        """Initialize every module and the stage-specific optimizer."""

        config = dict(config.to_dict() if hasattr(config, 'to_dict') else config)

        variant = str(config.get('variant', 'sa_cl_bc'))
        if variant not in VARIANTS:
            raise ValueError(f'variant must be one of {VARIANTS}, got {variant!r}.')
        stage = str(stage).lower()
        if stage not in STAGES:
            raise ValueError(f'stage must be one of {STAGES}, got {stage!r}.')
        config['stage'] = stage

        # The variant owns the structural choices; a config may not silently
        # disagree with them.  Coefficients stay freely editable.
        settings = VARIANT_SETTINGS[variant]
        for key in (
            'critic_type',
            'actor_goal_input',
            'actor_objective',
            'use_flow',
            'actor_goal_sampling',
            'actor_goal_offsets',
            'flow_target_mode',
        ):
            expected = settings[key]
            if key in config and _as_comparable(config[key]) != _as_comparable(expected):
                raise ValueError(
                    f'variant={variant!r} fixes {key}={expected!r}, but the '
                    f'config requests {config[key]!r}. Choose a different '
                    'variant instead of overriding its structure.'
                )
            config[key] = expected
        config['actor_goal_offsets'] = tuple(
            int(offset) for offset in config['actor_goal_offsets']
        )
        for key in ('actor_bc_coef', 'action_nce_coef', 'eval_mode'):
            config.setdefault(key, settings[key])
        if str(config['critic_type']) not in CRITIC_TYPES:
            raise ValueError(
                f'critic_type must be one of {CRITIC_TYPES}, '
                f'got {config["critic_type"]!r}.'
            )
        if str(config['actor_goal_input']) not in ACTOR_GOAL_INPUTS:
            raise ValueError(
                f'actor_goal_input must be one of {ACTOR_GOAL_INPUTS}, '
                f'got {config["actor_goal_input"]!r}.'
            )
        if str(config['actor_objective']) not in ACTOR_OBJECTIVES:
            raise ValueError(
                f'actor_objective must be one of {ACTOR_OBJECTIVES}, '
                f'got {config["actor_objective"]!r}.'
            )
        if str(config['eval_mode']) not in EVAL_MODES:
            raise ValueError(
                f'eval_mode must be one of {EVAL_MODES}, '
                f'got {config["eval_mode"]!r}.'
            )
        if str(config['flow_target_mode']) not in FLOW_TARGET_MODES:
            raise ValueError(
                f'flow_target_mode must be one of {FLOW_TARGET_MODES}, '
                f'got {config["flow_target_mode"]!r}.'
            )
        if str(config['actor_goal_sampling']) not in ACTOR_GOAL_SAMPLING_MODES:
            raise ValueError(
                'actor_goal_sampling must be one of '
                f'{ACTOR_GOAL_SAMPLING_MODES}, '
                f'got {config["actor_goal_sampling"]!r}.'
            )
        # 0.0 means "follow the critic discount"; the sampler resolves it.
        actor_discount = float(config.setdefault('actor_discount', 0.0))
        if actor_discount and not 0.0 < actor_discount < 1.0:
            raise ValueError(
                f'actor_discount must be 0 or in (0, 1), got {actor_discount}.'
            )
        if stage == 'critic' and str(config['critic_type']) == 'none':
            raise ValueError(
                f'variant={variant!r} has no contrastive critic; '
                "its research protocol starts at stage='actor'."
            )
        if stage == 'flow' and not bool(config['use_flow']):
            raise ValueError(
                f'variant={variant!r} does not use the latent rectified flow.'
            )
        if (
            str(config['actor_goal_input']) == 'latent'
            and str(config['critic_type']) == 'none'
        ):
            raise ValueError(
                "actor_goal_input='latent' requires a trained goal encoder; "
                "set critic_type to 'state' or 'sa'."
            )
        if bool(config['use_flow']) and str(config['actor_goal_input']) != 'latent':
            raise ValueError(
                'The latent rectified flow can only drive a latent-conditioned '
                'actor.'
            )

        repr_dim = int(config['repr_dim'])
        if repr_dim < 1:
            raise ValueError(f'repr_dim must be positive, got {repr_dim}.')
        action_horizon = int(config['action_horizon'])
        if action_horizon < 1:
            raise ValueError(
                f'action_horizon must be positive, got {action_horizon}.'
            )
        if float(config['contrastive_temperature']) <= 0.0:
            raise ValueError('contrastive_temperature must be positive.')
        if int(config['flow_steps']) < 1:
            raise ValueError('flow_steps must be at least 1.')
        if float(config['flow_noise_scale']) < 0.0:
            raise ValueError('flow_noise_scale must be non-negative.')
        # Resolve the bridge's target offsets once, so the evaluator and the
        # diagnostics read the same waypoint schedule the sampler supervises.
        if str(config['flow_target_mode']) == 'sparse':
            config['flow_target_offsets'] = sparse_prefix_offsets(
                int(config['horizon']),
                action_horizon,
            )
        else:
            config['flow_target_offsets'] = tuple(range(1, action_horizon + 1))
        replan_interval = int(config.setdefault('replan_interval', action_horizon))
        if not 1 <= replan_interval <= action_horizon:
            raise ValueError(
                'replan_interval must lie in [1, action_horizon] = '
                f'[1, {action_horizon}], got {replan_interval}.'
            )
        if str(config['flow_target_mode']) == 'sparse' and replan_interval != 1:
            raise ValueError(
                "flow_target_mode='sparse' requires replan_interval=1: waypoint "
                f'k targets {config["flow_target_offsets"]}[k] steps ahead, not '
                f'k steps ahead, so a chunk of {replan_interval} would feed the '
                'actor goals it cannot reach. Got '
                f'replan_interval={replan_interval}.'
            )
        if int(config['num_action_negatives']) < 1:
            raise ValueError('num_action_negatives must be at least 1.')
        discount = float(config['discount'])
        if not 0.0 < discount < 1.0:
            raise ValueError(f'discount must be in (0, 1), got {discount}.')

        hidden_dims = tuple(int(width) for width in config['hidden_dims'])
        layer_norm = bool(config['layer_norm'])
        env_name = str(config['env_name'])
        goal_mode = str(config.get('goal_representation_mode', 'full'))
        phi_goal_obs_indices = tuple(
            int(index) for index in config.get('phi_goal_obs_indices', ())
        )
        config['hidden_dims'] = hidden_dims
        config['phi_goal_obs_indices'] = phi_goal_obs_indices
        config['goal_representation_mode'] = goal_mode
        config['variant'] = variant

        observations = jnp.asarray(ex_observations, dtype=jnp.float32)
        actions = jnp.asarray(ex_actions, dtype=jnp.float32)
        if observations.ndim != 2 or actions.ndim != 2:
            raise ValueError(
                'ex_observations and ex_actions must be batched rank-2 arrays.'
            )
        if observations.shape[0] != actions.shape[0]:
            raise ValueError(
                'ex_observations and ex_actions must have equal batch sizes.'
            )
        action_dim = int(actions.shape[-1])
        low = _as_float_tuple(
            -1.0 if action_low is None else action_low,
            size=action_dim,
            name='action_low',
        )
        high = _as_float_tuple(
            1.0 if action_high is None else action_high,
            size=action_dim,
            name='action_high',
        )
        if any(hi <= lo for lo, hi in zip(low, high)):
            raise ValueError(
                f'action_high must exceed action_low elementwise; got {low} / {high}.'
            )
        config['action_low'] = low
        config['action_high'] = high
        config['action_dim'] = action_dim
        config['state_dim'] = int(observations.shape[-1])

        psi_def = GoalEncoder(
            repr_dim=repr_dim,
            hidden_dims=hidden_dims,
            env_name=env_name,
            goal_representation_mode=goal_mode,
            phi_goal_obs_indices=phi_goal_obs_indices,
            layer_norm=layer_norm,
        )
        example_goal_latents = jnp.zeros(
            (observations.shape[0], repr_dim),
            dtype=jnp.float32,
        )
        example_actor_goal_inputs = (
            example_goal_latents
            if str(config['actor_goal_input']) == 'latent'
            else goal_representation(
                observations,
                goal_mode,
                phi_goal_obs_indices,
                env_name=env_name,
            )
        )
        network_info = {
            'phi_sa': (
                StateActionEncoder(
                    repr_dim=repr_dim,
                    hidden_dims=hidden_dims,
                    layer_norm=layer_norm,
                ),
                (observations, actions),
            ),
            'phi_s': (
                StateEncoder(
                    repr_dim=repr_dim,
                    hidden_dims=hidden_dims,
                    layer_norm=layer_norm,
                ),
                (observations,),
            ),
            'psi': (psi_def, (observations,)),
            'actor': (
                LatentConditionedActor(
                    action_dim=action_dim,
                    hidden_dims=hidden_dims,
                    action_low=low,
                    action_high=high,
                    layer_norm=layer_norm,
                ),
                (observations, example_actor_goal_inputs),
            ),
            'flow': (
                LatentPrefixFlow(
                    repr_dim=repr_dim,
                    action_horizon=action_horizon,
                    hidden_dims=hidden_dims,
                    layer_norm=layer_norm,
                ),
                (
                    jnp.zeros(
                        (observations.shape[0], action_horizon, repr_dim),
                        dtype=jnp.float32,
                    ),
                    jnp.zeros((observations.shape[0], 1), dtype=jnp.float32),
                    example_goal_latents,
                    example_goal_latents,
                ),
            ),
        }
        network_def = ModuleDict(
            {name: definition for name, (definition, _) in network_info.items()}
        )
        network_args = {
            name: arguments for name, (_, arguments) in network_info.items()
        }

        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        network_params = network_def.init(init_rng, **network_args)['params']

        tx = _stage_optimizer(
            network_params,
            stage=stage,
            learning_rate=float(config['learning_rate']),
        )
        network = TrainState.create(network_def, network_params, tx=tx)
        return cls(rng=rng, network=network, config=_freeze_config(config))


def trainable_modules(stage: str) -> tuple[str, ...]:
    """Module names updated by ``stage``; everything else is frozen."""

    stage = str(stage).lower()
    if stage not in _STAGE_TRAINABLE:
        raise ValueError(f'stage must be one of {STAGES}, got {stage!r}.')
    return _STAGE_TRAINABLE[stage]


def _stage_optimizer(
    params: Any,
    *,
    stage: str,
    learning_rate: float,
) -> optax.GradientTransformation:
    """Adam on the stage's modules, a hard zero transform on the rest.

    Routing frozen modules through ``optax.set_to_zero`` is stronger than
    zeroing their gradients: it is independent of any optimizer state carried
    in from an earlier stage, so a frozen module cannot drift.
    """

    trainable = trainable_modules(stage)
    allowed = {f'modules_{name}' for name in _MODULE_NAMES}

    def label_params(tree: Any) -> Any:
        tree = flax.core.unfreeze(tree) if isinstance(tree, flax.core.FrozenDict) else tree
        labels = {}
        for key, subtree in tree.items():
            if key not in allowed:
                raise ValueError(f'Unexpected LatentBridger module subtree {key!r}.')
            label = 'train' if key[len('modules_') :] in trainable else 'freeze'
            labels[key] = jax.tree_util.tree_map(lambda _, tag=label: tag, subtree)
        return labels

    return optax.multi_transform(
        {'train': optax.adam(learning_rate), 'freeze': optax.set_to_zero()},
        label_params(params),
    )


def restore_latent_params(
    agent: LatentBridgerAgent,
    restore_path: str,
    step: int = 0,
    *,
    restore_host_rng: bool = True,
) -> LatentBridgerAgent:
    """Restore module parameters only, leaving the stage optimizer fresh.

    Staged training deliberately rebuilds the optimizer for each stage, so the
    serialized ``opt_state`` of the previous stage is neither restorable nor
    wanted.  Parameters are shared by every stage and are restored in full.
    """

    checkpoint_path, _ = resolve_checkpoint(restore_path, step)
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')
    with Path(checkpoint_path).open('rb') as file:
        payload = pickle.load(file)
    if not isinstance(payload, dict) or 'agent' not in payload:
        raise ValueError(f'Invalid LatentBridger checkpoint: {checkpoint_path}')
    try:
        saved_params = payload['agent']['network']['params']
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f'Checkpoint {checkpoint_path} does not contain network parameters.'
        ) from exc

    params = flax.serialization.from_state_dict(agent.network.params, saved_params)
    network = agent.network.replace(params=params)
    if restore_host_rng:
        if 'numpy_random_state' in payload:
            np.random.set_state(payload['numpy_random_state'])
        if 'python_random_state' in payload:
            random.setstate(payload['python_random_state'])
    return agent.replace(network=network)


def get_config() -> ml_collections.ConfigDict:
    """Default LatentBridger settings (cube-single scale, ``sa_cl_bc``)."""

    return ml_collections.ConfigDict(
        dict(
            env_name='antmaze-medium-navigate-v0',
            variant='sa_cl_bc',
            horizon=25,
            discount=0.99,
            # Representation.
            repr_dim=64,
            hidden_dims=(512, 512, 512),
            layer_norm=True,
            repr_norm=True,
            contrastive_temperature=0.1,
            logsumexp_coef=0.0,
            goal_representation_mode='full',
            phi_goal_obs_indices=(),
            # Dataset supervision.
            future_sampling='geometric',
            bridge_goal_sampling='trajectory',
            actor_goal_sampling='offsets',
            actor_goal_max_offset=1,
            actor_goal_offsets=(1,),
            actor_discount=0.0,   # 0 tracks `discount`
            action_horizon=5,
            flow_target_mode='consecutive',
            # Objectives.
            critic_type='sa',
            actor_goal_input='latent',
            actor_objective='contrastive',
            actor_bc_coef=10.0,
            action_nce_coef=0.0,
            num_action_negatives=16,
            # Latent rectified flow.
            use_flow=False,
            flow_steps=8,
            flow_noise_scale=1.0,
            flow_renormalize=True,
            flow_stop_psi_gradient=True,
            replan_interval=5,
            # Optimization and evaluation.
            learning_rate=3e-4,
            eval_mode='direct_goal',
        )
    )


__all__ = [
    'ACTOR_GOAL_INPUTS',
    'ACTOR_OBJECTIVES',
    'CRITIC_TYPES',
    'EVAL_MODES',
    'GoalEncoder',
    'LatentBridgerAgent',
    'LatentConditionedActor',
    'LatentPrefixFlow',
    'STAGES',
    'StateActionEncoder',
    'StateEncoder',
    'V1_ACTOR_GOAL_OFFSETS',
    'VARIANTS',
    'VARIANT_SETTINGS',
    'get_config',
    'restore_latent_params',
    'trainable_modules',
]
