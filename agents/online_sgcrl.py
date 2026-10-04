"""Online SGCRL and two latent-interface variants.

This is a faithful single-process port of ``graliuce/sgcrl``'s
``contrastive_cpc`` setting.  The derivation from the original source, and the
list of what is and is not reproduced, is in
``docs/sgcrl_online_semantics.md``.  Nothing here touches the offline
PathBridger or LatentBridger agents.

Three variants share one critic, one actor architecture, one replay
distribution, and one update-to-data ratio.  They differ only in what the
actor is conditioned on::

    online_sgcrl               pi(a | s, g)          raw goal, the baseline
    online_sgcrl_latent        pi(a | s, psi(g))     the latent interface alone
    online_sgcrl_latent_bridge pi(a | s, z_way)      the latent interface plus a bridge

``z_way`` is the first waypoint of a rectified-flow prefix generated from
``psi(s)`` toward ``psi(g)``.  Because only the conditioning changes, the gap
between the second and third is the bridge's marginal contribution.

The critic is SGCRL's: unnormalized 64-dimensional inner products, a
one-directional InfoNCE over the batch's goals, and the ``0.01 * logsumexp^2``
penalty that keeps the logit scale bounded in the absence of normalization or
a learned temperature.  There is no behavioural cloning, no action-NCE, and no
target network.

The critic, the actor, and the bridge each own a separate optimizer, as in
SGCRL's ``q_optimizer`` / ``policy_optimizer`` split.  Sharing one optimizer
would let a critic update move the actor through Adam's momentum even with
zero actor gradients.
"""

from __future__ import annotations

import functools
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field

VARIANTS = (
    'online_sgcrl',
    'online_sgcrl_latent',
    'online_sgcrl_latent_bridge',
)

# `actor_goal_input` and `use_bridge` are the only structural knobs the
# variants disagree on; everything else is shared by construction.
VARIANT_SETTINGS: dict[str, dict[str, Any]] = {
    'online_sgcrl': dict(actor_goal_input='raw', use_bridge=False),
    'online_sgcrl_latent': dict(actor_goal_input='latent', use_bridge=False),
    'online_sgcrl_latent_bridge': dict(actor_goal_input='latent', use_bridge=True),
}

# Acme's NormalTanhDistribution default: scale = softplus(x) + min_scale, with
# a zero-initialized bias so the policy starts at softplus(0) + 1e-3 ~= 0.694.
# A log-std parameterization would start near 1.0 and, with no entropy bonus,
# can drive the scale to zero instead of stopping at this floor.
_ACTOR_MIN_SCALE = 1e-3
# TanhTransformedDistribution clips the event before the log-det correction.
_TANH_EVENT_CLIP = 0.99999997


def _encoder_init():
    """SGCRL's encoder init: ``VarianceScaling(1.0, 'fan_avg', 'uniform')``."""

    return nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform')


def _policy_init():
    """SGCRL's policy trunk init: ``VarianceScaling(1.0, 'fan_in', 'uniform')``."""

    return nn.initializers.variance_scaling(1.0, 'fan_in', 'uniform')


class StateActionEncoder(nn.Module):
    """``phi(s, a)``: MLP([256, 256, repr_dim]) with no final activation."""

    hidden_dims: tuple[int, ...]
    repr_dim: int
    repr_norm: bool = False

    @nn.compact
    def __call__(self, observations, actions):
        x = jnp.concatenate([observations, actions], axis=-1)
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width, kernel_init=_encoder_init())(x))
        x = nn.Dense(self.repr_dim, kernel_init=_encoder_init())(x)
        if self.repr_norm:
            x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x


class GoalEncoder(nn.Module):
    """``psi(g)``: MLP([256, 256, repr_dim]) with no final activation."""

    hidden_dims: tuple[int, ...]
    repr_dim: int
    repr_norm: bool = False

    @nn.compact
    def __call__(self, goals):
        x = goals
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width, kernel_init=_encoder_init())(x))
        x = nn.Dense(self.repr_dim, kernel_init=_encoder_init())(x)
        if self.repr_norm:
            x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        return x


class TanhGaussianActor(nn.Module):
    """SGCRL's ``NormalTanhDistribution`` head over an MLP trunk."""

    hidden_dims: tuple[int, ...]
    action_dim: int

    @nn.compact
    def __call__(self, observations, conditioning):
        x = jnp.concatenate([observations, conditioning], axis=-1)
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width, kernel_init=_policy_init())(x))
        mean = nn.Dense(self.action_dim, kernel_init=_policy_init())(x)
        scale = nn.Dense(self.action_dim, kernel_init=_policy_init())(x)
        return mean, jax.nn.softplus(scale) + _ACTOR_MIN_SCALE


class LatentBridgeFlow(nn.Module):
    """Rectified-flow velocity field over a latent waypoint prefix."""

    hidden_dims: tuple[int, ...]
    repr_dim: int
    num_waypoints: int

    @nn.compact
    def __call__(self, state_latents, goal_latents, prefixes, times):
        flat = prefixes.reshape(prefixes.shape[0], -1)
        x = jnp.concatenate([state_latents, goal_latents, flat, times], axis=-1)
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width, kernel_init=_encoder_init())(x))
        velocity = nn.Dense(
            self.num_waypoints * self.repr_dim, kernel_init=_encoder_init()
        )(x)
        return velocity.reshape(prefixes.shape)


def _sample_tanh_gaussian(mean, scale, rng):
    """Sample and score SGCRL's tanh-squashed Gaussian policy.

    The log-probability is only a diagnostic here: the launcher's
    ``entropy_coefficient = 0.0`` removes it from the actor's gradient.
    """

    noise = jax.random.normal(rng, mean.shape)
    pre_tanh = mean + scale * noise
    action = jnp.tanh(pre_tanh)
    log_prob = jax.scipy.stats.norm.logpdf(pre_tanh, mean, scale).sum(axis=-1)
    clipped = jnp.clip(action, -_TANH_EVENT_CLIP, _TANH_EVENT_CLIP)
    log_prob -= jnp.log1p(-jnp.square(clipped)).sum(axis=-1)
    return action, log_prob


def _freeze_config(config: dict[str, Any]) -> Any:
    return flax.core.FrozenDict(
        {
            key: tuple(value) if isinstance(value, (list, tuple)) else value
            for key, value in config.items()
        }
    )


class OnlineSGCRLAgent(flax.struct.PyTreeNode):
    """SGCRL's contrastive critic and actor, with a swappable goal interface."""

    rng: Any
    critic: Any
    actor: Any
    flow: Any
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------------
    def goal_latents(self, goals, params=None):
        return self.critic.select('psi')(goals, params=params)

    def critic_logits(self, observations, actions, goals, params=None):
        """``logits[i, j] = phi(s_i, a_i) . psi(g_j)``, unnormalized."""

        sa = self.critic.select('phi')(observations, actions, params=params)
        g = self.critic.select('psi')(goals, params=params)
        return jnp.einsum('ik,jk->ij', sa, g)

    def critic_scores(self, observations, actions, goals, params=None):
        """The paired score ``phi(s_i, a_i) . psi(g_i)``."""

        sa = self.critic.select('phi')(observations, actions, params=params)
        g = self.critic.select('psi')(goals, params=params)
        return jnp.sum(sa * g, axis=-1)

    def actor_conditioning(self, goals):
        """What the actor is conditioned on, ignoring the bridge."""

        if self.config['actor_goal_input'] == 'raw':
            return goals
        return self.goal_latents(goals)

    # ------------------------------------------------------------------
    # Latent bridge
    # ------------------------------------------------------------------
    def _integrate_prefix(self, state_latents, goal_latents, rng, params=None):
        """Euler-integrate the rectified flow from noise to a latent prefix."""

        batch_size = state_latents.shape[0]
        shape = (
            batch_size,
            int(self.config['num_waypoints']),
            int(self.config['repr_dim']),
        )
        prefix = float(self.config['flow_noise_scale']) * jax.random.normal(rng, shape)
        steps = int(self.config['flow_steps'])
        for step in range(steps):
            times = jnp.full((batch_size, 1), step / steps, dtype=prefix.dtype)
            velocity = self.flow.select('flow')(
                state_latents, goal_latents, prefix, times, params=params
            )
            prefix = prefix + velocity / steps
        return prefix

    def bridge_waypoint(self, observations, goals, rng):
        """The first latent waypoint on the way from ``s`` toward ``g``."""

        state_latents = self.goal_latents(observations)
        goal_latents = self.goal_latents(goals)
        return self._integrate_prefix(state_latents, goal_latents, rng)[:, 0]

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------
    def critic_loss(self, batch, grad_params):
        """One-directional InfoNCE with SGCRL's logsumexp penalty."""

        logits = self.critic_logits(
            batch['observations'], batch['actions'], batch['goals'], params=grad_params
        )
        batch_size = logits.shape[0]
        labels = jnp.eye(batch_size)
        cross_entropy = optax.softmax_cross_entropy(logits=logits, labels=labels)
        # Without representation normalization or a temperature this penalty
        # is the only thing bounding the logit scale.
        penalty = self.config['logsumexp_coef'] * jax.nn.logsumexp(logits, axis=1) ** 2
        loss = jnp.mean(cross_entropy + penalty)

        positives = jnp.diag(logits)
        negatives = (jnp.sum(logits) - jnp.sum(positives)) / (
            batch_size * (batch_size - 1)
        )
        ranks = jnp.sum(logits >= positives[:, None], axis=1)
        return loss, {
            'critic/loss': loss,
            'critic/recall_at_1': jnp.mean(ranks <= 1),
            'critic/recall_at_5': jnp.mean(ranks <= 5),
            'critic/positive_rank': jnp.mean(ranks),
            'critic/logits_pos': jnp.mean(positives),
            'critic/logits_neg': negatives,
            'critic/logit_gap': jnp.mean(positives) - negatives,
            'critic/logsumexp': jnp.mean(jax.nn.logsumexp(logits, axis=1) ** 2),
        }

    def _actor_batch(self, batch):
        """SGCRL's ``random_goals=0.5`` split: half hindsight, half shuffled."""

        states = batch['observations']
        goals = batch['goals']
        fraction = float(self.config['random_goals'])
        if fraction == 0.0:
            return states, goals
        if fraction == 1.0:
            return states, jnp.roll(goals, 1, axis=0)
        return (
            jnp.concatenate([states, states], axis=0),
            jnp.concatenate([goals, jnp.roll(goals, 1, axis=0)], axis=0),
        )

    def actor_loss(self, batch, grad_params, rng):
        """Maximize the critic score of the sampled action.  No BC term."""

        states, goals = self._actor_batch(batch)
        rng, bridge_rng, sample_rng = jax.random.split(rng, 3)
        if self.config['use_bridge']:
            conditioning = self.bridge_waypoint(states, goals, bridge_rng)
        else:
            conditioning = self.actor_conditioning(goals)
        # The conditioning comes from the critic and the bridge, neither of
        # which this loss trains; detaching it keeps the actor's gradient the
        # only thing this update can move.
        conditioning = jax.lax.stop_gradient(conditioning)

        mean, scale = self.actor.select('actor')(states, conditioning, params=grad_params)
        actions, log_prob = _sample_tanh_gaussian(mean, scale, sample_rng)
        # The critic is evaluated at its current, non-differentiated
        # parameters, so the gradient reaches the policy through the action
        # with no stop_gradient on the critic's output.
        scores = self.critic_scores(states, actions, goals)

        entropy = -log_prob
        loss = jnp.mean(-scores - float(self.config['entropy_coefficient']) * entropy)
        return loss, {
            'actor/loss': loss,
            'actor/critic_score': jnp.mean(scores),
            'actor/entropy': jnp.mean(entropy),
            'actor/action_saturation': jnp.mean(jnp.abs(actions) > 0.99),
            'actor/mean_scale': jnp.mean(scale),
        }

    def bridge_loss(self, batch, grad_params, rng):
        """Rectified-flow matching onto sparse latent waypoints.

        The targets are encoded with the *current* ``psi`` inside the loss.
        Online training moves the representation under the bridge, so a cached
        latent target would be stale within a few thousand updates and the
        flow would be chasing a representation that no longer exists.
        """

        state_latents = jax.lax.stop_gradient(self.goal_latents(batch['observations']))
        goal_latents = jax.lax.stop_gradient(self.goal_latents(batch['goals']))
        targets = batch['bridge_targets']
        batch_size, num_waypoints, _ = targets.shape
        target_latents = jax.lax.stop_gradient(
            self.goal_latents(targets.reshape(batch_size * num_waypoints, -1))
        ).reshape(batch_size, num_waypoints, -1)

        noise_rng, time_rng = jax.random.split(rng)
        noise = float(self.config['flow_noise_scale']) * jax.random.normal(
            noise_rng, target_latents.shape
        )
        times = jax.random.uniform(time_rng, (batch_size, 1))
        interpolated = (1.0 - times[:, :, None]) * noise + times[
            :, :, None
        ] * target_latents
        velocity_target = target_latents - noise

        velocity = self.flow.select('flow')(
            state_latents, goal_latents, interpolated, times, params=grad_params
        )
        # A waypoint past the episode's end carries no supervision.
        valid = batch['bridge_valid'][:, :, None]
        error = jnp.square(velocity - velocity_target) * valid
        loss = jnp.sum(error) / jnp.maximum(jnp.sum(valid) * targets.shape[-1], 1.0)
        return loss, {
            'bridge/loss': loss,
            'bridge/valid_fraction': jnp.mean(batch['bridge_valid']),
        }

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    @jax.jit
    def update(self, batch):
        """One critic update then one actor update, as SGCRL's learner does."""

        new_rng, actor_rng = jax.random.split(self.rng)

        critic, critic_info = self.critic.apply_loss_fn(
            loss_fn=lambda params: self.critic_loss(batch, params)
        )
        # SGCRL computes the actor loss against the pre-update critic
        # parameters, so the actor sees the same critic the critic loss did.
        actor, actor_info = self.actor.apply_loss_fn(
            loss_fn=lambda params: self.actor_loss(batch, params, actor_rng)
        )
        return (
            self.replace(rng=new_rng, critic=critic, actor=actor),
            {**critic_info, **actor_info},
        )

    @jax.jit
    def update_bridge(self, batch):
        """One bridge update, kept separate so it never buys extra RL steps."""

        new_rng, bridge_rng = jax.random.split(self.rng)
        flow, info = self.flow.apply_loss_fn(
            loss_fn=lambda params: self.bridge_loss(batch, params, bridge_rng)
        )
        return self.replace(rng=new_rng, flow=flow), info

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    @functools.partial(jax.jit, static_argnames=('deterministic', 'use_bridge'))
    def act(self, observations, goals, rng, deterministic=False, use_bridge=None):
        """Act for one step.  Collection samples; evaluation takes the mode.

        ``use_bridge`` overrides the variant default, which is what lets one
        warmed-up latent run branch into a bridge and a non-bridge behaviour
        from the same parameters.
        """

        if use_bridge is None:
            use_bridge = bool(self.config['use_bridge'])
        rng, bridge_rng = jax.random.split(rng)
        if use_bridge:
            conditioning = self.bridge_waypoint(observations, goals, bridge_rng)
        else:
            conditioning = self.actor_conditioning(goals)
        mean, scale = self.actor.select('actor')(observations, conditioning)
        if deterministic:
            return jnp.tanh(mean)
        action, _ = _sample_tanh_gaussian(mean, scale, rng)
        return action

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations,
        ex_actions,
        config: Any,
    ) -> 'OnlineSGCRLAgent':
        config = dict(config.to_dict() if hasattr(config, 'to_dict') else config)
        variant = str(config.get('variant', 'online_sgcrl'))
        if variant not in VARIANTS:
            raise ValueError(f'variant must be one of {VARIANTS}, got {variant!r}.')
        settings = VARIANT_SETTINGS[variant]
        for key, expected in settings.items():
            if key in config and config[key] != expected:
                raise ValueError(
                    f'variant={variant!r} fixes {key}={expected!r}, but the '
                    f'config requests {config[key]!r}. Choose a different '
                    'variant instead of overriding its structure.'
                )
            config[key] = expected
        if float(config['random_goals']) not in (0.0, 0.5, 1.0):
            raise ValueError(
                'SGCRL only defines random_goals in {0.0, 0.5, 1.0}, got '
                f'{config["random_goals"]}.'
            )
        if float(config['entropy_coefficient']) < 0.0:
            raise ValueError('entropy_coefficient cannot be negative.')

        hidden_dims = tuple(int(width) for width in config['hidden_dims'])
        repr_dim = int(config['repr_dim'])
        repr_norm = bool(config['repr_norm'])
        num_waypoints = int(config['num_waypoints'])
        observation_dim = int(ex_observations.shape[-1])
        action_dim = int(ex_actions.shape[-1])
        rows = int(ex_observations.shape[0])

        rng = jax.random.PRNGKey(seed)
        rng, critic_rng, actor_rng, flow_rng = jax.random.split(rng, 4)
        learning_rate = float(config['learning_rate'])

        def optimizer():
            # SGCRL uses Adam with eps=1e-7 for both the critic and the actor.
            return optax.adam(learning_rate=learning_rate, eps=1e-7)

        critic_def = ModuleDict(
            {
                'phi': StateActionEncoder(hidden_dims, repr_dim, repr_norm),
                'psi': GoalEncoder(hidden_dims, repr_dim, repr_norm),
            }
        )
        critic_params = critic_def.init(
            critic_rng,
            phi=(ex_observations, ex_actions),
            psi=(ex_observations,),
        )['params']
        critic = TrainState.create(critic_def, critic_params, tx=optimizer())

        conditioning_dim = (
            repr_dim if config['actor_goal_input'] == 'latent' else observation_dim
        )
        ex_conditioning = jnp.zeros((rows, conditioning_dim), dtype=jnp.float32)
        actor_def = ModuleDict({'actor': TanhGaussianActor(hidden_dims, action_dim)})
        actor_params = actor_def.init(
            actor_rng, actor=(ex_observations, ex_conditioning)
        )['params']
        actor = TrainState.create(actor_def, actor_params, tx=optimizer())

        ex_latents = jnp.zeros((rows, repr_dim), dtype=jnp.float32)
        ex_prefix = jnp.zeros((rows, num_waypoints, repr_dim), dtype=jnp.float32)
        ex_times = jnp.zeros((rows, 1), dtype=jnp.float32)
        flow_def = ModuleDict(
            {'flow': LatentBridgeFlow(hidden_dims, repr_dim, num_waypoints)}
        )
        flow_params = flow_def.init(
            flow_rng, flow=(ex_latents, ex_latents, ex_prefix, ex_times)
        )['params']
        flow = TrainState.create(flow_def, flow_params, tx=optimizer())

        return cls(
            rng=rng,
            critic=critic,
            actor=actor,
            flow=flow,
            config=_freeze_config(config),
        )


def get_config() -> ml_collections.ConfigDict:
    """SGCRL's launcher defaults, derived in docs/sgcrl_online_semantics.md."""

    return ml_collections.ConfigDict(
        dict(
            env_name='cube-single-play-v0',
            variant='online_sgcrl',
            task_id=1,
            # Critic and actor.
            repr_dim=64,
            repr_norm=False,
            hidden_dims=(256, 256),
            learning_rate=3e-4,
            discount=0.99,
            batch_size=256,
            random_goals=0.5,
            logsumexp_coef=0.01,
            # The launcher sets entropy_coefficient=0.0, which bypasses the
            # adaptive-alpha branch; the actor stays stochastic but gets no
            # entropy bonus.
            entropy_coefficient=0.0,
            # Replay.
            min_replay_size=10_000,
            max_replay_size=1_000_000,
            # Update-to-data ratio: one batch-256 update per environment step.
            updates_per_env_step=1,
            # Latent bridge (variant C only).  `actor_goal_input` and
            # `use_bridge` are deliberately absent: the variant is their only
            # source, so no config can quietly build a fourth hybrid.
            bridge_horizon=40,
            num_waypoints=5,
            flow_steps=8,
            flow_noise_scale=1.0,
            bridge_updates_per_env_step=1,
        )
    )


__all__ = [
    'GoalEncoder',
    'LatentBridgeFlow',
    'OnlineSGCRLAgent',
    'StateActionEncoder',
    'TanhGaussianActor',
    'VARIANTS',
    'VARIANT_SETTINGS',
    'get_config',
]
