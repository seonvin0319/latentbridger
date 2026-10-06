"""Goal-abstraction actors on an unchanged SGCRL critic.

The raw full-goal baseline and the waypoint bridges stay in
``agents/online_sgcrl.py``.  This module only adds three actors that consume
a transformed goal while the critic is still scored on the original full goal:

    sgcrl_psi_goal     pi(a | s, stop_gradient(psi(g)))
    sgcrl_state_goal   pi(a | s, LayerNorm(psi(g) + 0.5 tanh f(e(s), psi(g))))
    sgcrl_state_mask   pi(a | s, s + m (g - s)), m = sigmoid(M([s, g]))

The actor objective is SGCRL's: sample an action and ascend
``C(s, a, g_full) = phi(s, a) · psi(g_full)``.  No reward, return, or oracle
cube position enters that loss.  Actor gradients do not update psi.
"""

from __future__ import annotations

import functools
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from agents.online_sgcrl import (
    GoalEncoder,
    StateActionEncoder,
    TanhGaussianActor,
    _encoder_init,
    _freeze_config,
    _sample_tanh_gaussian,
)
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field

VARIANTS = (
    'sgcrl_psi_goal',
    'sgcrl_state_goal',
    'sgcrl_state_mask',
)

ABSTRACTIONS = ('psi_goal', 'state_goal', 'state_mask')

VARIANT_SETTINGS: dict[str, dict[str, Any]] = {
    'sgcrl_psi_goal': dict(abstraction='psi_goal'),
    'sgcrl_state_goal': dict(abstraction='state_goal'),
    'sgcrl_state_mask': dict(abstraction='state_mask'),
}


class StateEncoder(nn.Module):
    """``e_s(s)``: MLP(256, 256, repr_dim) with no final activation."""

    hidden_dims: tuple[int, ...]
    repr_dim: int

    @nn.compact
    def __call__(self, states):
        x = states
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width, kernel_init=_encoder_init())(x))
        return nn.Dense(self.repr_dim, kernel_init=_encoder_init())(x)


class GoalResidual(nn.Module):
    """``f_abs(z_s, z_g)``: MLP(256, 256, repr_dim) residual in latent space."""

    hidden_dims: tuple[int, ...]
    repr_dim: int

    @nn.compact
    def __call__(self, state_latent, goal_latent):
        x = jnp.concatenate([state_latent, goal_latent], axis=-1)
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width, kernel_init=_encoder_init())(x))
        return nn.Dense(self.repr_dim, kernel_init=_encoder_init())(x)


class LatentGoalNorm(nn.Module):
    """``LayerNorm(z_g + scale * tanh(delta))``."""

    scale: float

    @nn.compact
    def __call__(self, goal_latent, delta):
        mixed = goal_latent + self.scale * jnp.tanh(delta)
        return nn.LayerNorm()(mixed)


class GoalMask(nn.Module):
    """``sigmoid(M([s, g]))`` with values in ``(0, 1)``."""

    hidden_dims: tuple[int, ...]
    goal_dim: int

    @nn.compact
    def __call__(self, states, goals):
        x = jnp.concatenate([states, goals], axis=-1)
        for width in self.hidden_dims:
            x = nn.relu(nn.Dense(width, kernel_init=_encoder_init())(x))
        logits = nn.Dense(self.goal_dim, kernel_init=_encoder_init())(x)
        return jax.nn.sigmoid(logits)


def _pairwise_mean_distance(values: Any) -> Any:
    """Mean distance between two distinct rows.  The diagonal is excluded."""

    differences = values[:, None, :] - values[None, :, :]
    distances = jnp.linalg.norm(differences, axis=-1)
    count = values.shape[0]
    return jnp.sum(distances) / (count * (count - 1))


def linear_probe_oracle(features: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Held-out linear regression from a representation onto cube xyz.

    This is an analysis tool.  Training never calls it, and its outputs are
    not a reward, a target, or an input to the actor.
    """

    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    if features.ndim != 2 or targets.ndim != 2 or len(features) != len(targets):
        raise ValueError('features and targets must be aligned 2-D arrays.')
    if len(features) < 4:
        raise ValueError('A held-out probe needs at least four rows.')
    split = len(features) // 2
    train_x, test_x = features[:split], features[split:]
    train_y, test_y = targets[:split], targets[split:]
    design = np.concatenate([train_x, np.ones((len(train_x), 1))], axis=1)
    coefficient, _, _, _ = np.linalg.lstsq(design, train_y, rcond=None)
    prediction = np.concatenate([test_x, np.ones((len(test_x), 1))], axis=1) @ coefficient
    residual = test_y - prediction
    mse = float(np.mean(np.square(residual)))
    centered = test_y - np.mean(test_y, axis=0, keepdims=True)
    total = float(np.sum(np.square(centered)))
    r2 = float(1.0 - np.sum(np.square(residual)) / total) if total > 0.0 else 0.0
    return {'oracle_probe_mse': mse, 'oracle_probe_r2': r2}


class GoalAbstractionAgent(flax.struct.PyTreeNode):
    """SGCRL critic plus one goal-abstraction actor."""

    rng: Any
    critic: Any
    policy: Any
    config: Any = nonpytree_field()

    def critic_logits(self, observations, actions, goals, params=None):
        sa = self.critic.select('phi')(observations, actions, params=params)
        g = self.critic.select('psi')(goals, params=params)
        return jnp.einsum('ik,jk->ij', sa, g)

    def critic_scores(self, observations, actions, goals, params=None):
        sa = self.critic.select('phi')(observations, actions, params=params)
        g = self.critic.select('psi')(goals, params=params)
        return jnp.sum(sa * g, axis=-1)

    def critic_loss(self, batch, grad_params):
        """SGCRL's one-directional InfoNCE.  The goal is the raw full goal."""

        logits = self.critic_logits(
            batch['observations'], batch['actions'], batch['goals'], params=grad_params
        )
        batch_size = logits.shape[0]
        labels = jnp.eye(batch_size)
        cross_entropy = optax.softmax_cross_entropy(logits=logits, labels=labels)
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
        """SGCRL's ``random_goals`` split.  Half the rows use a shuffled goal."""

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

    def _psi(self, goals):
        """Goal encoder output with the actor gradient stopped."""

        return jax.lax.stop_gradient(self.critic.select('psi')(goals))

    def goal_features(self, states, goals, params=None):
        """The actor's goal input, plus the quantities diagnostics want.

        ``goal_input`` is what the policy concatenates with ``s``.  It is never
        what the critic scores.  The critic always receives the raw ``goals``
        argument, which callers must set to ``g_full``.  ``params`` is the
        policy tree being differentiated; diagnostics omit it and read the
        stored weights.
        """

        mode = str(self.config['abstraction'])
        psi = self._psi(goals)
        if mode == 'psi_goal':
            return {
                'goal_input': psi,
                'psi': psi,
                'residual': jnp.zeros_like(psi),
                'mask': jnp.zeros(states.shape, dtype=states.dtype),
            }
        if mode == 'state_goal':
            state_latent = self.policy.select('state_encoder')(states, params=params)
            delta = self.policy.select('abstraction')(state_latent, psi, params=params)
            goal_input = self.policy.select('residual')(psi, delta, params=params)
            return {
                'goal_input': goal_input,
                'psi': psi,
                'residual': float(self.config['abstraction_scale']) * jnp.tanh(delta),
                'mask': jnp.zeros(states.shape, dtype=states.dtype),
            }
        if mode == 'state_mask':
            mask = self.policy.select('mask')(states, goals, params=params)
            goal_input = states + mask * (goals - states)
            return {
                'goal_input': goal_input,
                'psi': psi,
                'residual': jnp.zeros_like(psi),
                'mask': mask,
            }
        raise ValueError(f'abstraction must be one of {ABSTRACTIONS}.')

    def actor_distribution(self, observations, goal_input, params=None):
        return self.policy.select('actor')(observations, goal_input, params=params)

    def actor_loss(self, batch, grad_params, rng):
        """Ascend ``C(s, a, g_full)``.  The abstraction is only the actor input."""

        states, goals = self._actor_batch(batch)
        features = self.goal_features(states, goals, params=grad_params)
        loc, scale = self.actor_distribution(
            states, features['goal_input'], params=grad_params
        )
        actions, log_prob = _sample_tanh_gaussian(loc, scale, rng)
        # ``goals`` here is g_full (or a shuffled g_full).  Not goal_input.
        scores = self.critic_scores(states, actions, goals)
        entropy = -log_prob
        loss = jnp.mean(-scores - float(self.config['entropy_coefficient']) * entropy)
        mask = features['mask']
        mask_mean = jnp.mean(mask)
        if str(self.config['abstraction']) == 'state_mask':
            loss = loss + float(self.config['mask_coef']) * mask_mean
        clipped = jnp.clip(mask, 1e-6, 1.0 - 1e-6)
        mask_entropy = jnp.mean(
            -(clipped * jnp.log(clipped) + (1.0 - clipped) * jnp.log(1.0 - clipped))
        )
        sensitivity_goal = jnp.repeat(goals[:1], states.shape[0], axis=0)
        sensitivity = _pairwise_mean_distance(
            self.goal_features(states, sensitivity_goal, params=grad_params)['goal_input']
        )
        return loss, {
            'actor/loss': loss,
            'actor/critic_score': jnp.mean(scores),
            'actor/entropy': jnp.mean(entropy),
            'actor/action_saturation': jnp.mean(jnp.abs(actions) > 0.99),
            'actor/mean_scale': jnp.mean(scale),
            'goal/psi_pairwise': _pairwise_mean_distance(features['psi']),
            'goal/state_sensitivity': sensitivity,
            'goal/residual_norm': jnp.mean(jnp.linalg.norm(features['residual'], axis=-1)),
            'mask/mean': mask_mean,
            'mask/entropy': mask_entropy,
        }

    @jax.jit
    def update(self, batch):
        """One critic step, then one actor step against the pre-update critic."""

        new_rng, actor_rng = jax.random.split(self.rng)
        critic, critic_info = self.critic.apply_loss_fn(
            loss_fn=lambda params: self.critic_loss(batch, params)
        )
        policy, actor_info = self.policy.apply_loss_fn(
            loss_fn=lambda params: self.actor_loss(batch, params, actor_rng)
        )
        return (
            self.replace(rng=new_rng, critic=critic, policy=policy),
            {**critic_info, **actor_info},
        )

    @jax.jit
    def update_actor_only(self, batch):
        """Actor step with the critic frozen.  Used only by the optional pretrain."""

        new_rng, actor_rng = jax.random.split(self.rng)
        policy, actor_info = self.policy.apply_loss_fn(
            loss_fn=lambda params: self.actor_loss(batch, params, actor_rng)
        )
        return self.replace(rng=new_rng, policy=policy), actor_info

    @functools.partial(jax.jit, static_argnames=('deterministic', 'bridge_mode'))
    def act(self, observations, goals, rng, deterministic=False, bridge_mode='none'):
        """Act toward the abstracted goal.  There is no waypoint bridge."""

        if bridge_mode != 'none':
            raise ValueError(
                'Goal-abstraction variants do not use a waypoint bridge, got '
                f'{bridge_mode!r}.'
            )
        features = self.goal_features(observations, goals)
        loc, scale = self.actor_distribution(observations, features['goal_input'])
        if deterministic:
            return jnp.tanh(loc)
        action, _ = _sample_tanh_gaussian(loc, scale, rng)
        return action

    @jax.jit
    def abstraction_diagnostics(self, batch):
        """Scalar checks of the goal interface.  This does not step the agent."""

        states = batch['observations']
        goals = batch['goals']
        features = self.goal_features(states, goals)
        rng = jax.random.PRNGKey(0)
        loc, scale = self.actor_distribution(states, features['goal_input'])
        actions, log_prob = _sample_tanh_gaussian(loc, scale, rng)
        scores = self.critic_scores(states, actions, goals)
        mask = features['mask']
        per_dim_mean = jnp.mean(mask, axis=0)
        per_dim_std = jnp.std(mask, axis=0)
        info = {
            'diagnostics/actor_critic_score': jnp.mean(scores),
            'diagnostics/action_entropy': jnp.mean(-log_prob),
            'diagnostics/action_saturation': jnp.mean(jnp.abs(actions) > 0.99),
            'diagnostics/mean_scale': jnp.mean(scale),
            'diagnostics/psi_pairwise': _pairwise_mean_distance(features['psi']),
            'diagnostics/residual_norm': jnp.mean(
                jnp.linalg.norm(features['residual'], axis=-1)
            ),
            'diagnostics/mask_mean': jnp.mean(mask),
            'diagnostics/mask_entropy': jnp.mean(
                -(
                    jnp.clip(mask, 1e-6, 1 - 1e-6) * jnp.log(jnp.clip(mask, 1e-6, 1 - 1e-6))
                    + (1 - jnp.clip(mask, 1e-6, 1 - 1e-6))
                    * jnp.log(1 - jnp.clip(mask, 1e-6, 1 - 1e-6))
                )
            ),
        }
        # Per-dimension mask statistics stay out of the scalar CSV path.
        info['diagnostics/mask_dim_mean_avg'] = jnp.mean(per_dim_mean)
        info['diagnostics/mask_dim_std_avg'] = jnp.mean(per_dim_std)
        sensitivity_goal = jnp.repeat(goals[:1], states.shape[0], axis=0)
        info['diagnostics/state_sensitivity'] = _pairwise_mean_distance(
            self.goal_features(states, sensitivity_goal)['goal_input']
        )
        return info

    def mask_profile(self, states, goals) -> dict[str, np.ndarray]:
        """Per-dimension mask mean and std.  Empty for the latent variants."""

        features = self.goal_features(states, goals)
        mask = np.asarray(jax.device_get(features['mask']))
        if str(self.config['abstraction']) != 'state_mask':
            return {'mask_mean': np.zeros((0,), dtype=np.float32), 'mask_std': np.zeros((0,), dtype=np.float32)}
        return {
            'mask_mean': mask.mean(axis=0).astype(np.float32),
            'mask_std': mask.std(axis=0).astype(np.float32),
        }

    @classmethod
    def create(cls, seed: int, ex_observations, ex_actions, config: Any):
        config = dict(config.to_dict() if hasattr(config, 'to_dict') else config)
        variant = str(config.get('variant', 'sgcrl_psi_goal'))
        if variant not in VARIANTS:
            raise ValueError(f'variant must be one of {VARIANTS}, got {variant!r}.')
        settings = VARIANT_SETTINGS[variant]
        for key, expected in settings.items():
            if key in config and config[key] != expected:
                raise ValueError(
                    f'variant={variant!r} fixes {key}={expected!r}, but the '
                    f'config requests {config[key]!r}.'
                )
            config[key] = expected
        if float(config['random_goals']) not in (0.0, 0.5, 1.0):
            raise ValueError(
                'SGCRL only defines random_goals in {0.0, 0.5, 1.0}, got '
                f'{config["random_goals"]}.'
            )

        hidden_dims = tuple(int(width) for width in config['hidden_dims'])
        abstraction_dims = tuple(int(width) for width in config['abstraction_hidden_dims'])
        repr_dim = int(config['repr_dim'])
        repr_norm = bool(config['repr_norm'])
        observation_dim = int(ex_observations.shape[-1])
        action_dim = int(ex_actions.shape[-1])
        rows = int(ex_observations.shape[0])
        mode = str(config['abstraction'])

        rng = jax.random.PRNGKey(seed)
        rng, critic_rng, policy_rng = jax.random.split(rng, 3)

        def optimizer(learning_rate):
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

        goal_dim = repr_dim if mode != 'state_mask' else observation_dim
        modules: dict[str, nn.Module] = {
            'actor': TanhGaussianActor(hidden_dims, action_dim),
        }
        ex_goal = jnp.zeros((rows, goal_dim), dtype=jnp.float32)
        init_kwargs: dict[str, Any] = {'actor': (ex_observations, ex_goal)}
        if mode == 'state_goal':
            modules['state_encoder'] = StateEncoder(abstraction_dims, repr_dim)
            modules['abstraction'] = GoalResidual(abstraction_dims, repr_dim)
            modules['residual'] = LatentGoalNorm(scale=float(config['abstraction_scale']))
            ex_latent = jnp.zeros((rows, repr_dim), dtype=jnp.float32)
            init_kwargs['state_encoder'] = (ex_observations,)
            init_kwargs['abstraction'] = (ex_latent, ex_latent)
            init_kwargs['residual'] = (ex_latent, ex_latent)
        elif mode == 'state_mask':
            modules['mask'] = GoalMask(abstraction_dims, observation_dim)
            init_kwargs['mask'] = (ex_observations, ex_observations)

        policy_def = ModuleDict(modules)
        policy_params = policy_def.init(policy_rng, **init_kwargs)['params']
        policy = TrainState.create(
            policy_def, policy_params, tx=optimizer(config['actor_learning_rate'])
        )
        return cls(
            rng=rng,
            critic=critic,
            policy=policy,
            config=_freeze_config(config),
        )


def export_agent(agent: GoalAbstractionAgent) -> dict[str, Any]:
    """Weights and RNG, without the replay.  The trainer stores this in a snapshot."""

    return {
        'critic': flax.serialization.to_state_dict(agent.critic),
        'policy': flax.serialization.to_state_dict(agent.policy),
        'rng': np.asarray(jax.device_get(agent.rng)),
    }


def import_agent(agent: GoalAbstractionAgent, payload: dict[str, Any]) -> GoalAbstractionAgent:
    """Restore ``export_agent`` into a freshly constructed agent of the same variant."""

    return agent.replace(
        critic=flax.serialization.from_state_dict(agent.critic, payload['critic']),
        policy=flax.serialization.from_state_dict(agent.policy, payload['policy']),
        rng=jnp.asarray(payload['rng']),
    )


def get_config(variant: str = 'sgcrl_psi_goal') -> ml_collections.ConfigDict:
    """SGCRL hyperparameters plus the abstraction knobs.  Bridges are absent."""

    if variant not in VARIANT_SETTINGS:
        raise ValueError(f'variant must be one of {tuple(VARIANT_SETTINGS)}, got {variant!r}.')
    return ml_collections.ConfigDict(
        dict(
            env_name='cube-single-play-v0',
            variant=variant,
            abstraction=VARIANT_SETTINGS[variant]['abstraction'],
            task_id=1,
            repr_dim=64,
            repr_norm=False,
            hidden_dims=(256, 256),
            learning_rate=3e-4,
            actor_learning_rate=3e-4,
            discount=0.99,
            batch_size=256,
            random_goals=0.5,
            logsumexp_coef=0.01,
            entropy_coefficient=0.0,
            min_replay_size=10_000,
            max_replay_size=1_000_000,
            holdout_every=20,
            updates_per_env_step=1,
            abstraction_hidden_dims=(256, 256),
            abstraction_scale=0.5,
            mask_coef=1e-4,
        )
    )
