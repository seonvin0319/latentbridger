"""Online SGCRL, with and without a raw-state intermediate waypoint bridge.

This is a faithful single-process port of ``graliuce/sgcrl``'s
``contrastive_cpc`` setting (tree ``2d1b59d``).  The derivation from the
original source is in ``docs/sgcrl_online_semantics.md``.  Nothing here
touches the offline PathBridger or LatentBridger agents.

The question is whether handing the actor a learned intermediate waypoint
instead of the final goal helps online single-goal RL.  Three variants::

    online_sgcrl            a ~ pi(. | s, g*)        the faithful baseline
    online_sgcrl_det_bridge a ~ pi(. | s, B(s, g*))  deterministic waypoint
    online_sgcrl_rf_bridge  a ~ pi(. | s, w ~ p(.))  rectified-flow waypoint

The critic, the actor network, the critic objective, the actor objective, the
replay, the hindsight sampling, and the update-to-data ratio are *identical*
across all three.  The only difference is what goes into the actor's goal slot
during behaviour collection and evaluation.  In particular the bridge does not
appear in the actor loss: the actor is always trained on raw hindsight goals,
so the three variants share one actor objective exactly as required.

Both bridges are supervised on states the replay trajectories actually
visited.  For a segment ``s_t ... s_j`` the target is ``s_i`` with
``i = t + floor(alpha * (j - t))``.  This is not an interpolation between the
endpoints; it is where the behaviour policy actually was partway through.

Both bridge modules are built in every variant so that a parameter snapshot
transfers between them unchanged, which is what makes the shared-warmup branch
comparison exact.  The baseline simply never trains or calls them.
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
    'online_sgcrl_det_bridge',
    'online_sgcrl_rf_bridge',
)

BRIDGE_MODES = ('none', 'deterministic', 'rectified_flow')

# `bridge_mode` is the only structural knob the variants disagree on.
VARIANT_SETTINGS: dict[str, dict[str, Any]] = {
    'online_sgcrl': dict(bridge_mode='none'),
    'online_sgcrl_det_bridge': dict(bridge_mode='deterministic'),
    'online_sgcrl_rf_bridge': dict(bridge_mode='rectified_flow'),
}

# distributional.NormalTanhDistribution, with make_networks' actor_min_std.
_ACTOR_MIN_SCALE = 1e-6
# The repo's "modified Tanh mean": loc = 10 * tanh(raw / 10).
_ACTOR_LOC_LIMIT = 10.0
# TanhTransformedDistribution clips the event to +-0.999 before log_prob.
_TANH_THRESHOLD = 0.999


def _encoder_init():
    """SGCRL's encoder init: ``VarianceScaling(1.0, 'fan_avg', 'uniform')``."""

    return nn.initializers.variance_scaling(1.0, 'fan_avg', 'uniform')


def _policy_init():
    """SGCRL's policy init: ``VarianceScaling(1.0, 'fan_in', 'uniform')``."""

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
    """SGCRL's ``NormalTanhDistribution`` over an MLP trunk.

    Two details from ``distributional.py`` that change behaviour: the mean is
    squashed to ``10 * tanh(raw / 10)`` before the outer tanh, and the scale is
    ``softplus(raw) + 1e-6`` rather than an exponentiated log-std.
    """

    hidden_dims: tuple[int, ...]
    action_dim: int

    @nn.compact
    def __call__(self, observations, goals):
        x = jnp.concatenate([observations, goals], axis=-1)
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width, kernel_init=_policy_init())(x))
        loc = nn.Dense(self.action_dim, kernel_init=_policy_init())(x)
        loc = _ACTOR_LOC_LIMIT * jnp.tanh(loc / _ACTOR_LOC_LIMIT)
        scale = nn.Dense(self.action_dim, kernel_init=_policy_init())(x)
        return loc, jax.nn.softplus(scale) + _ACTOR_MIN_SCALE


class DeterministicBridge(nn.Module):
    """``B(s, g) -> w_hat``: a raw-state waypoint regressor."""

    hidden_dims: tuple[int, ...]
    observation_dim: int

    @nn.compact
    def __call__(self, observations, goals):
        x = jnp.concatenate([observations, goals], axis=-1)
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width, kernel_init=_encoder_init())(x))
        return nn.Dense(self.observation_dim, kernel_init=_encoder_init())(x)


class RectifiedFlowBridge(nn.Module):
    """``v(x_tau, tau | s, g)``: velocity over the waypoint displacement.

    The flow models the displacement ``w - s`` rather than ``w`` itself, so
    the field does not have to relearn the identity map for every state.
    """

    hidden_dims: tuple[int, ...]
    observation_dim: int

    @nn.compact
    def __call__(self, observations, goals, displacements, times):
        x = jnp.concatenate([observations, goals, displacements, times], axis=-1)
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width, kernel_init=_encoder_init())(x))
        return nn.Dense(self.observation_dim, kernel_init=_encoder_init())(x)


def _sample_tanh_gaussian(loc, scale, rng):
    """Sample and score SGCRL's tanh-squashed Gaussian policy.

    The log-probability is a diagnostic only: the launcher's
    ``entropy_coefficient = 0.0`` multiplies it out of the actor's gradient.
    """

    noise = jax.random.normal(rng, loc.shape)
    pre_tanh = loc + scale * noise
    action = jnp.tanh(pre_tanh)
    log_prob = jax.scipy.stats.norm.logpdf(pre_tanh, loc, scale).sum(axis=-1)
    clipped = jnp.clip(action, -_TANH_THRESHOLD, _TANH_THRESHOLD)
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
    """SGCRL's contrastive critic and actor, plus two optional bridges."""

    rng: Any
    # The bridges draw from their own stream so that training one cannot
    # shift the randomness the critic and actor consume.  Without this, the
    # flow bridge's noise draws would desynchronize the reinforcement-learning
    # updates between variants that are supposed to differ only in behaviour.
    bridge_rng: Any
    critic: Any
    actor: Any
    det_bridge: Any
    rf_bridge: Any
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------
    # Critic
    # ------------------------------------------------------------------
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

    def critic_loss(self, batch, grad_params):
        """One-directional InfoNCE with SGCRL's logsumexp penalty."""

        logits = self.critic_logits(
            batch['observations'], batch['actions'], batch['goals'], params=grad_params
        )
        batch_size = logits.shape[0]
        labels = jnp.eye(batch_size)
        cross_entropy = optax.softmax_cross_entropy(logits=logits, labels=labels)
        # With repr_norm=False and no temperature this penalty is the only
        # thing bounding the logit scale.
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

    # ------------------------------------------------------------------
    # Actor
    # ------------------------------------------------------------------
    def actor_distribution(self, observations, goals, params=None):
        return self.actor.select('actor')(observations, goals, params=params)

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
        """Maximize the critic score of the sampled action.

        No behavioural cloning and no bridge: this objective is byte-identical
        across the three variants, which is what lets the comparison attribute
        any difference to the behaviour-time goal substitution alone.
        """

        states, goals = self._actor_batch(batch)
        loc, scale = self.actor_distribution(states, goals, params=grad_params)
        actions, log_prob = _sample_tanh_gaussian(loc, scale, rng)
        # The critic is evaluated at its current parameters; the gradient
        # reaches the policy through the action, with no stop_gradient on the
        # critic's output.
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

    # ------------------------------------------------------------------
    # Bridges
    # ------------------------------------------------------------------
    def det_waypoint(self, observations, goals, params=None):
        return self.det_bridge.select('bridge')(observations, goals, params=params)

    def det_bridge_loss(self, batch, grad_params):
        """Regress the observed intermediate state ``s_i``."""

        predicted = self.det_waypoint(
            batch['observations'], batch['goals'], params=grad_params
        )
        target = batch['waypoints']
        loss = jnp.mean(jnp.square(predicted - target))
        return loss, {
            'det_bridge/loss': loss,
            'det_bridge/waypoint_mse': loss,
            'det_bridge/predicted_displacement_norm': jnp.mean(
                jnp.linalg.norm(predicted - batch['observations'], axis=-1)
            ),
            'det_bridge/target_displacement_norm': jnp.mean(
                jnp.linalg.norm(target - batch['observations'], axis=-1)
            ),
        }

    def rf_bridge_loss(self, batch, grad_params, rng):
        """Rectified-flow matching on the waypoint displacement."""

        observations = batch['observations']
        goals = batch['goals']
        x_1 = batch['waypoints'] - observations
        noise_rng, time_rng = jax.random.split(rng)
        x_0 = jax.random.normal(noise_rng, x_1.shape)
        tau = jax.random.uniform(time_rng, (x_1.shape[0], 1))
        x_tau = (1.0 - tau) * x_0 + tau * x_1
        velocity_target = x_1 - x_0

        velocity = self.rf_bridge.select('bridge')(
            observations, goals, x_tau, tau, params=grad_params
        )
        loss = jnp.mean(jnp.square(velocity - velocity_target))
        return loss, {
            'rf_bridge/loss': loss,
            'rf_bridge/target_displacement_norm': jnp.mean(
                jnp.linalg.norm(x_1, axis=-1)
            ),
        }

    def rf_waypoint(self, observations, goals, rng, params=None):
        """Euler-integrate the flow from noise to a displacement, then add it."""

        displacement = jax.random.normal(rng, observations.shape)
        steps = int(self.config['flow_steps'])
        for step in range(steps):
            tau = jnp.full((observations.shape[0], 1), step / steps, dtype=jnp.float32)
            velocity = self.rf_bridge.select('bridge')(
                observations, goals, displacement, tau, params=params
            )
            displacement = displacement + velocity / steps
        return observations + displacement

    @functools.partial(jax.jit, static_argnames=('bridge_mode',))
    def waypoint(self, observations, goals, rng, bridge_mode):
        """The raw state the actor is pointed at, per the variant's mode."""

        if bridge_mode == 'none':
            return goals
        if bridge_mode == 'deterministic':
            return self.det_waypoint(observations, goals)
        if bridge_mode == 'rectified_flow':
            return self.rf_waypoint(observations, goals, rng)
        raise ValueError(f'bridge_mode must be one of {BRIDGE_MODES}.')

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------
    @jax.jit
    def update(self, batch):
        """One critic update then one actor update, as SGCRL's learner does.

        This is the only call that touches the critic or the actor, so the
        reinforcement-learning update count cannot depend on the variant.
        """

        new_rng, actor_rng = jax.random.split(self.rng)
        critic, critic_info = self.critic.apply_loss_fn(
            loss_fn=lambda params: self.critic_loss(batch, params)
        )
        # SGCRL computes both losses against the pre-update parameters.
        actor, actor_info = self.actor.apply_loss_fn(
            loss_fn=lambda params: self.actor_loss(batch, params, actor_rng)
        )
        return (
            self.replace(rng=new_rng, critic=critic, actor=actor),
            {**critic_info, **actor_info},
        )

    @jax.jit
    def update_det_bridge(self, batch):
        """One deterministic-bridge update.  Touches no other module."""

        bridge, info = self.det_bridge.apply_loss_fn(
            loss_fn=lambda params: self.det_bridge_loss(batch, params)
        )
        return self.replace(det_bridge=bridge), info

    @jax.jit
    def update_rf_bridge(self, batch):
        """One rectified-flow-bridge update.  Touches no other module."""

        new_rng, loss_rng = jax.random.split(self.bridge_rng)
        bridge, info = self.rf_bridge.apply_loss_fn(
            loss_fn=lambda params: self.rf_bridge_loss(batch, params, loss_rng)
        )
        return self.replace(bridge_rng=new_rng, rf_bridge=bridge), info

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------
    @functools.partial(jax.jit, static_argnames=('deterministic', 'bridge_mode'))
    def act(self, observations, goals, rng, deterministic=False, bridge_mode='none'):
        """Act for one step; the waypoint is recomputed from the current state.

        ``bridge_mode`` is an argument rather than a config read so that one
        set of parameters can drive all three behaviours from a shared
        snapshot at the branch point.
        """

        waypoint_rng, action_rng = jax.random.split(rng)
        goal_input = self.waypoint(observations, goals, waypoint_rng, bridge_mode)
        loc, scale = self.actor_distribution(observations, goal_input)
        if deterministic:
            return jnp.tanh(loc)
        action, _ = _sample_tanh_gaussian(loc, scale, action_rng)
        return action

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @jax.jit
    def bridge_diagnostics(self, batch, rng):
        """Score both bridges on a batch of held-out supervision tuples.

        The three critic scores separate the two ways a waypoint can fail: it
        can be unreachable (low score for the waypoint itself), or reachable
        but useless (high waypoint score, low final-goal score).
        """

        observations = batch['observations']
        goals = batch['goals']
        targets = batch['waypoints']
        true_displacement = targets - observations
        info: dict[str, Any] = {}

        def action_for(goal_input, key):
            loc, scale = self.actor_distribution(observations, goal_input)
            action, _ = _sample_tanh_gaussian(loc, scale, key)
            return action

        rng, direct_rng = jax.random.split(rng)
        direct_action = action_for(goals, direct_rng)
        info['diagnostics/final_goal_score_direct'] = jnp.mean(
            self.critic_scores(observations, direct_action, goals)
        )

        # -- deterministic bridge ---------------------------------------
        predicted = self.det_waypoint(observations, goals)
        rng, det_rng = jax.random.split(rng)
        det_action = action_for(predicted, det_rng)
        info.update(
            {
                'diagnostics/det_waypoint_mse': jnp.mean(
                    jnp.square(predicted - targets)
                ),
                'diagnostics/det_displacement_norm': jnp.mean(
                    jnp.linalg.norm(predicted - observations, axis=-1)
                ),
                'diagnostics/det_final_goal_score': jnp.mean(
                    self.critic_scores(observations, det_action, goals)
                ),
                'diagnostics/det_waypoint_score': jnp.mean(
                    self.critic_scores(observations, det_action, predicted)
                ),
            }
        )

        # -- rectified-flow bridge, single sample for the primary number --
        rng, flow_rng, rf_action_rng = jax.random.split(rng, 3)
        sampled = self.rf_waypoint(observations, goals, flow_rng)
        rf_action = action_for(sampled, rf_action_rng)
        info.update(
            {
                'diagnostics/rf_waypoint_mse': jnp.mean(jnp.square(sampled - targets)),
                'diagnostics/rf_displacement_norm': jnp.mean(
                    jnp.linalg.norm(sampled - observations, axis=-1)
                ),
                'diagnostics/rf_final_goal_score': jnp.mean(
                    self.critic_scores(observations, rf_action, goals)
                ),
                'diagnostics/rf_waypoint_score': jnp.mean(
                    self.critic_scores(observations, rf_action, sampled)
                ),
            }
        )

        # -- rectified-flow spread, diagnostic only -----------------------
        # Eight independent samples per (s, g).  This never feeds control:
        # behaviour always uses one sample, with no ranking or best-of-N.
        num_samples = int(self.config['diagnostic_flow_samples'])
        rng, *sample_rngs = jax.random.split(rng, num_samples + 1)
        samples = jnp.stack(
            [self.rf_waypoint(observations, goals, key) for key in sample_rngs]
        )
        mean_sample = jnp.mean(samples, axis=0)
        # Mean distance between two distinct samples for the same (s, g).
        # The diagonal contributes zero, so summing the whole matrix and
        # dividing by the N(N-1) off-diagonal pairs gives their mean.
        differences = samples[:, None] - samples[None, :]
        distances = jnp.linalg.norm(differences, axis=-1)
        off_diagonal = num_samples * (num_samples - 1)
        info.update(
            {
                'diagnostics/rf_pairwise_distance': jnp.mean(
                    jnp.sum(distances, axis=(0, 1)) / off_diagonal
                ),
                'diagnostics/rf_sample_std': jnp.mean(jnp.std(samples, axis=0)),
                'diagnostics/rf_conditional_variance': jnp.mean(
                    jnp.var(samples, axis=0)
                ),
                'diagnostics/rf_mean_sample_mse': jnp.mean(
                    jnp.square(mean_sample - targets)
                ),
            }
        )

        rng, *action_rngs = jax.random.split(rng, num_samples + 1)
        sample_actions = jnp.stack(
            [
                action_for(sample, key)
                for sample, key in zip(samples, action_rngs)
            ]
        )
        sample_scores = jnp.stack(
            [
                self.critic_scores(observations, action, goals)
                for action in sample_actions
            ]
        )
        info.update(
            {
                'diagnostics/rf_action_diversity': jnp.mean(
                    jnp.std(sample_actions, axis=0)
                ),
                'diagnostics/rf_score_std': jnp.mean(jnp.std(sample_scores, axis=0)),
                'diagnostics/rf_score_mean': jnp.mean(sample_scores),
                'diagnostics/true_displacement_norm': jnp.mean(
                    jnp.linalg.norm(true_displacement, axis=-1)
                ),
            }
        )
        return info

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
        if not 0.0 < float(config['bridge_alpha']) < 1.0:
            raise ValueError(
                f'bridge_alpha must lie in (0, 1), got {config["bridge_alpha"]}.'
            )

        hidden_dims = tuple(int(width) for width in config['hidden_dims'])
        repr_dim = int(config['repr_dim'])
        repr_norm = bool(config['repr_norm'])
        observation_dim = int(ex_observations.shape[-1])
        action_dim = int(ex_actions.shape[-1])
        rows = int(ex_observations.shape[0])

        rng = jax.random.PRNGKey(seed)
        rng, critic_rng, actor_rng, det_rng, rf_rng = jax.random.split(rng, 5)
        bridge_rng = jax.random.PRNGKey(seed + 777)

        def optimizer(learning_rate):
            # SGCRL uses a plain Adam per module; separate optimizers keep a
            # critic step from moving the actor through shared momentum.
            return optax.adam(learning_rate=float(learning_rate))

        critic_def = ModuleDict(
            {
                'phi': StateActionEncoder(hidden_dims, repr_dim, repr_norm),
                'psi': GoalEncoder(hidden_dims, repr_dim, repr_norm),
            }
        )
        critic_params = critic_def.init(
            critic_rng, phi=(ex_observations, ex_actions), psi=(ex_observations,)
        )['params']
        critic = TrainState.create(
            critic_def, critic_params, tx=optimizer(config['learning_rate'])
        )

        # The actor's goal slot always takes a raw observation-shaped vector,
        # whether that is the final goal or a bridge waypoint.
        actor_def = ModuleDict({'actor': TanhGaussianActor(hidden_dims, action_dim)})
        actor_params = actor_def.init(
            actor_rng, actor=(ex_observations, ex_observations)
        )['params']
        actor = TrainState.create(
            actor_def, actor_params, tx=optimizer(config['actor_learning_rate'])
        )

        bridge_dims = tuple(int(width) for width in config['bridge_hidden_dims'])
        det_def = ModuleDict(
            {'bridge': DeterministicBridge(bridge_dims, observation_dim)}
        )
        det_params = det_def.init(
            det_rng, bridge=(ex_observations, ex_observations)
        )['params']
        det_bridge = TrainState.create(
            det_def, det_params, tx=optimizer(config['bridge_learning_rate'])
        )

        ex_times = jnp.zeros((rows, 1), dtype=jnp.float32)
        rf_def = ModuleDict(
            {'bridge': RectifiedFlowBridge(bridge_dims, observation_dim)}
        )
        rf_params = rf_def.init(
            rf_rng, bridge=(ex_observations, ex_observations, ex_observations, ex_times)
        )['params']
        rf_bridge = TrainState.create(
            rf_def, rf_params, tx=optimizer(config['bridge_learning_rate'])
        )

        return cls(
            rng=rng,
            bridge_rng=bridge_rng,
            critic=critic,
            actor=actor,
            det_bridge=det_bridge,
            rf_bridge=rf_bridge,
            config=_freeze_config(config),
        )


def get_config() -> ml_collections.ConfigDict:
    """SGCRL's launcher defaults, derived in docs/sgcrl_online_semantics.md."""

    return ml_collections.ConfigDict(
        dict(
            env_name='cube-single-play-v0',
            variant='online_sgcrl',
            task_id=1,
            # Critic and actor, all from contrastive/config.py.
            repr_dim=64,
            repr_norm=False,
            hidden_dims=(256, 256),
            learning_rate=3e-4,
            actor_learning_rate=3e-4,
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
            holdout_every=20,
            # Update-to-data ratio: one batch-256 update per environment step.
            updates_per_env_step=1,
            # Bridges.  `bridge_mode` is absent on purpose: the variant is its
            # only source, so no config can build a fourth hybrid.
            bridge_alpha=0.5,
            bridge_hidden_dims=(256, 256),
            bridge_learning_rate=3e-4,
            bridge_updates_per_env_step=1,
            flow_steps=8,
            diagnostic_flow_samples=8,
        )
    )


__all__ = [
    'BRIDGE_MODES',
    'DeterministicBridge',
    'GoalEncoder',
    'OnlineSGCRLAgent',
    'RectifiedFlowBridge',
    'StateActionEncoder',
    'TanhGaussianActor',
    'VARIANTS',
    'VARIANT_SETTINGS',
    'get_config',
]
