"""Offline diagnostics that must pass before the latent flow is trusted.

The central risk of an action-conditioned contrastive critic is that the
in-batch InfoNCE objective can be solved perfectly while ``phi_sa`` ignores its
action argument: distinguishing *which future* a state leads to rarely requires
knowing *which action* was taken.  A critic like that still reports a healthy
contrastive loss and a healthy future-retrieval recall, yet it gives the actor
no usable gradient, because ``dC/da`` is approximately zero.

:func:`action_sensitivity` is therefore the gate for Module B, and
:func:`future_retrieval` alone is explicitly not sufficient.
"""

from __future__ import annotations

from typing import Any

import jax
import numpy as np


def _device_array(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=np.float32)


def _row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity per row, safe for unnormalized embeddings."""

    denominator = np.maximum(
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1),
        1e-8,
    )
    return np.sum(a * b, axis=-1) / denominator


def future_retrieval(
    agent: Any,
    dataset: Any,
    *,
    batch_size: int = 256,
    num_batches: int = 8,
    recall_ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Rank the true future goal against the other goals in the batch."""

    ranks: list[np.ndarray] = []
    gaps: list[float] = []
    for _ in range(int(num_batches)):
        batch = dataset.sample(int(batch_size))
        scores = _device_array(
            agent.critic_score_matrix(
                batch['observations'],
                batch['actions'],
                batch['contrastive_goals'],
            )
        )
        positive = np.diag(scores)
        # Rank 1 means the positive outscored every in-batch alternative.
        rank = 1 + np.sum(scores > positive[:, None], axis=1)
        ranks.append(rank)
        off_diagonal = scores[~np.eye(scores.shape[0], dtype=bool)]
        gaps.append(float(positive.mean() - off_diagonal.mean()))

    all_ranks = np.concatenate(ranks)
    metrics: dict[str, float] = {
        'retrieval/mean_positive_rank': float(all_ranks.mean()),
        'retrieval/median_positive_rank': float(np.median(all_ranks)),
        'retrieval/score_gap': float(np.mean(gaps)),
        'retrieval/batch_size': float(batch_size),
    }
    for k in recall_ks:
        metrics[f'retrieval/recall_at_{int(k)}'] = float(
            np.mean(all_ranks <= int(k))
        )
    return metrics


def action_sensitivity(
    agent: Any,
    dataset: Any,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
    batch_size: int = 256,
    num_batches: int = 8,
    num_hard_candidates: int = 16,
) -> dict[str, float]:
    """Check that ``C(s, a, g)`` actually depends on ``a``.

    For a fixed ``(s, g)`` the dataset action is compared against a shuffled
    dataset action, against a uniform action inside the real action box, and
    against the *best* of ``num_hard_candidates`` uniform actions.  A critic
    that ignores the action lands at probability 0.5 with a zero margin; the
    hard variant additionally catches a critic that is only weakly ordered,
    because beating one random action is much easier than beating sixteen.
    """

    action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    num_hard_candidates = int(num_hard_candidates)

    wins: dict[str, list[np.ndarray]] = {'shuffled': [], 'uniform': [], 'hard': []}
    margins: dict[str, list[np.ndarray]] = {'shuffled': [], 'uniform': [], 'hard': []}
    for _ in range(int(num_batches)):
        batch = dataset.sample(int(batch_size))
        observations = batch['observations']
        goals = batch['contrastive_goals']
        data_actions = batch['actions']

        permutation = np.random.permutation(len(data_actions))
        shuffled_actions = data_actions[permutation]
        uniform_actions = np.random.uniform(
            low=action_low,
            high=action_high,
            size=data_actions.shape,
        ).astype(np.float32)

        data_scores = _device_array(
            agent.critic_scores(observations, data_actions, goals)
        )
        candidate_scores = [
            _device_array(
                agent.critic_scores(
                    observations,
                    np.random.uniform(
                        low=action_low,
                        high=action_high,
                        size=data_actions.shape,
                    ).astype(np.float32),
                    goals,
                )
            )
            for _ in range(num_hard_candidates)
        ]
        hard_scores = np.max(np.stack(candidate_scores, axis=0), axis=0)

        for name, other in (
            ('shuffled', _device_array(
                agent.critic_scores(observations, shuffled_actions, goals)
            )),
            ('uniform', _device_array(
                agent.critic_scores(observations, uniform_actions, goals)
            )),
            ('hard', hard_scores),
        ):
            wins[name].append((data_scores > other).astype(np.float32))
            margins[name].append(data_scores - other)

    metrics: dict[str, float] = {
        'action_sensitivity/num_hard_candidates': float(num_hard_candidates),
    }
    for name in ('shuffled', 'uniform', 'hard'):
        metrics[f'action_sensitivity/p_data_gt_{name}'] = float(
            np.concatenate(wins[name]).mean()
        )
        metrics[f'action_sensitivity/margin_{name}'] = float(
            np.concatenate(margins[name]).mean()
        )
    return metrics


def actor_goal_sensitivity(
    agent: Any,
    dataset: Any,
    *,
    horizons: tuple[int, ...] = (1, 2, 4, 8, 16),
    batch_size: int = 256,
) -> dict[str, float]:
    """Does the actor's action change when the latent goal moves further away?

    A latent-conditioned actor can quietly collapse into a goal-agnostic
    policy: it reproduces the data action from ``s`` alone and ignores the
    conditioning vector.  Such a policy can never be steered by a generated
    waypoint, no matter how good the flow is.  For each horizon ``h`` this
    compares

        a_current = pi(s, psi(s))            (condition on standing still)
        a_h       = pi(s, psi(s_{t+h}))      (condition on the real future)

    and reports both the action displacement and whether the critic actually
    prefers ``a_h`` for reaching ``s_{t+h}``.
    """

    metrics: dict[str, float] = {}
    for horizon in horizons:
        try:
            pairs = dataset.sample_offset_pairs(int(batch_size), int(horizon))
        except ValueError:
            continue
        observations = pairs['observations']
        futures = pairs['future_observations']

        current_actions = _device_array(
            agent.sample_actions_from_goals(observations, observations)
        )
        horizon_actions = _device_array(
            agent.sample_actions_from_goals(observations, futures)
        )
        metrics[f'actor_goal_sensitivity/action_delta_{int(horizon)}'] = float(
            np.mean(np.linalg.norm(horizon_actions - current_actions, axis=-1))
        )

        if str(agent.config['critic_type']) == 'none':
            continue
        # Both actions are judged on reaching s_{t+h}; a steerable actor wins.
        horizon_scores = _device_array(
            agent.critic_scores(observations, horizon_actions, futures)
        )
        current_scores = _device_array(
            agent.critic_scores(observations, current_actions, futures)
        )
        metrics[f'actor_goal_sensitivity/critic_gain_{int(horizon)}'] = float(
            np.mean(horizon_scores - current_scores)
        )
    return metrics


def actor_diagnostics(
    agent: Any,
    dataset: Any,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
    batch_size: int = 256,
    num_batches: int = 8,
    saturation_tolerance: float = 1e-2,
) -> dict[str, float]:
    """Imitation error, bound saturation, and actor-vs-data critic scores."""

    action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    scale = np.maximum(action_high - action_low, 1e-6)

    mses: list[float] = []
    distances: list[float] = []
    saturations: list[float] = []
    actor_scores: list[float] = []
    data_scores: list[float] = []
    for _ in range(int(num_batches)):
        batch = dataset.sample(int(batch_size))
        observations = batch['observations']
        goals = batch['actor_goals']
        data_actions = batch['actions']

        predicted = _device_array(
            agent.sample_actions_from_goals(observations, goals)
        )
        mses.append(float(np.mean((predicted - data_actions) ** 2)))
        distances.append(
            float(np.mean(np.linalg.norm(predicted - data_actions, axis=-1)))
        )
        near_bound = np.minimum(
            np.abs(predicted - action_low),
            np.abs(predicted - action_high),
        )
        saturations.append(
            float(np.mean((near_bound / scale) < float(saturation_tolerance)))
        )
        actor_scores.append(
            float(
                _device_array(
                    agent.critic_scores(observations, predicted, goals)
                ).mean()
            )
        )
        data_scores.append(
            float(
                _device_array(
                    agent.critic_scores(observations, data_actions, goals)
                ).mean()
            )
        )

    return {
        'actor/bc_mse': float(np.mean(mses)),
        'actor/mean_distance_from_data_action': float(np.mean(distances)),
        'actor/saturation_fraction': float(np.mean(saturations)),
        'actor/critic_score_actor': float(np.mean(actor_scores)),
        'actor/critic_score_data': float(np.mean(data_scores)),
        'actor/critic_score_advantage': float(
            np.mean(actor_scores) - np.mean(data_scores)
        ),
    }


def latent_geometry(
    agent: Any,
    dataset: Any,
    *,
    deltas: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    batch_size: int = 256,
) -> dict[str, float]:
    """Similarity of ``psi(s_t)`` and ``psi(s_{t+Delta})`` versus random states."""

    metrics: dict[str, float] = {}
    for delta in deltas:
        try:
            pairs = dataset.sample_offset_pairs(int(batch_size), int(delta))
        except ValueError:
            # Episodes shorter than delta: report nothing rather than a
            # cross-episode number.
            continue
        anchors = _device_array(agent.goal_latents(pairs['observations']))
        futures = _device_array(agent.goal_latents(pairs['future_observations']))
        randoms = _device_array(agent.goal_latents(pairs['random_observations']))
        future_similarity = float(np.mean(_row_cosine(anchors, futures)))
        metrics[f'latent_geometry/future_similarity_delta_{int(delta)}'] = (
            future_similarity
        )
        metrics[f'latent_geometry/random_similarity_delta_{int(delta)}'] = float(
            np.mean(_row_cosine(anchors, randoms))
        )
        # D_h: how far the latent actually travels in h steps.  This is the
        # signal any latent bridge has to beat.
        metrics[f'latent_geometry/distance_delta_{int(delta)}'] = (
            1.0 - future_similarity
        )
    metrics['latent_geometry/normalized'] = float(bool(agent.config['repr_norm']))
    return metrics


def flow_reconstruction(
    agent: Any,
    dataset: Any,
    *,
    batch_size: int = 256,
    num_batches: int = 4,
    seed: int = 0,
) -> dict[str, float]:
    """Compare generated latent prefixes with the encoded ground-truth prefix.

    Latent targets are never decoded back into raw states: the flow is judged
    purely in the representation it actually has to produce.

    The decisive number here is the latent signal-to-noise ratio per waypoint

        D_h    = 1 - cos(psi(s_t), psi(s_{t+h}))      how far the latent moves
        E_h    = 1 - cos(zhat_h,   psi(s_{t+h}))      how wrong the flow is
        LSNR_h = D_h / E_h

    ``LSNR_h > 1`` means the generated waypoint is closer to the true future
    than simply standing still is, i.e. it carries usable control signal.
    Below 1 the actor is better off being handed ``psi(s_t)``.

    Rows whose waypoint was clipped at the conditioning goal are excluded from
    the per-horizon statistics, so ``h`` always means what it says.
    """

    rng = jax.random.PRNGKey(int(seed))
    offsets = tuple(int(offset) for offset in agent.config['flow_target_offsets'])

    prefix_mses: list[float] = []
    endpoint_hits: list[float] = []
    # Per-waypoint accumulators, each a list of per-row arrays.
    step_cosines: list[list[np.ndarray]] = [[] for _ in offsets]
    anchor_cosines: list[list[np.ndarray]] = [[] for _ in offsets]
    step_hits: list[list[float]] = [[] for _ in offsets]
    unclipped_fractions: list[list[float]] = [[] for _ in offsets]

    for _ in range(int(num_batches)):
        batch = dataset.sample(int(batch_size))
        rng, prefix_seed = jax.random.split(rng)
        generated = _device_array(
            agent.sample_latent_prefix(
                batch['observations'],
                batch['bridge_goals'],
                prefix_seed,
            )
        )
        targets = batch['bridge_targets']
        flat_targets = targets.reshape(-1, targets.shape[-1])
        target_latents = _device_array(agent.goal_latents(flat_targets)).reshape(
            generated.shape
        )
        anchor_latents = _device_array(agent.goal_latents(batch['observations']))
        realized = np.asarray(batch['bridge_target_offsets'])

        prefix_mses.append(float(np.mean((generated - target_latents) ** 2)))

        for index, offset in enumerate(offsets):
            # Clipped rows describe a nearer waypoint than ``offset`` claims.
            keep = realized[:, index] >= offset - 0.5
            unclipped_fractions[index].append(float(np.mean(keep)))
            if not np.any(keep):
                continue
            generated_h = generated[keep, index, :]
            target_h = target_latents[keep, index, :]
            step_cosines[index].append(_row_cosine(generated_h, target_h))
            anchor_cosines[index].append(
                _row_cosine(anchor_latents[keep], target_h)
            )
            similarity = generated_h @ target_h.T
            step_hits[index].append(
                float(
                    np.mean(
                        np.argmax(similarity, axis=1) == np.arange(len(similarity))
                    )
                )
            )

        generated_endpoint = generated[:, -1, :]
        target_endpoint = target_latents[:, -1, :]
        similarity = generated_endpoint @ target_endpoint.T
        endpoint_hits.append(
            float(np.mean(np.argmax(similarity, axis=1) == np.arange(len(similarity))))
        )

    metrics: dict[str, float] = {
        'flow/prefix_latent_mse': float(np.mean(prefix_mses)),
        'flow/endpoint_retrieval_consistency': float(np.mean(endpoint_hits)),
    }

    all_step_cosines: list[float] = []
    for index, offset in enumerate(offsets):
        metrics[f'flow/target_offset_step_{index + 1}'] = float(offset)
        metrics[f'flow/unclipped_fraction_h{offset}'] = float(
            np.mean(unclipped_fractions[index])
        )
        if not step_cosines[index]:
            continue
        step_cosine = float(np.mean(np.concatenate(step_cosines[index])))
        anchor_cosine = float(np.mean(np.concatenate(anchor_cosines[index])))
        reconstruction_error = 1.0 - step_cosine
        latent_motion = 1.0 - anchor_cosine
        all_step_cosines.append(step_cosine)

        metrics[f'flow/prefix_cosine_step_{index + 1}'] = step_cosine
        metrics[f'flow/E_h{offset}'] = reconstruction_error
        metrics[f'flow/D_h{offset}'] = latent_motion
        metrics[f'flow/lsnr_h{offset}'] = latent_motion / max(
            reconstruction_error,
            1e-8,
        )
        metrics[f'flow/retrieval_h{offset}'] = float(np.mean(step_hits[index]))

    if all_step_cosines:
        metrics['flow/mean_prefix_cosine'] = float(np.mean(all_step_cosines))
        metrics['flow/min_lsnr'] = min(
            metrics[f'flow/lsnr_h{offset}']
            for offset in offsets
            if f'flow/lsnr_h{offset}' in metrics
        )
    return metrics


def run_all_diagnostics(
    agent: Any,
    dataset: Any,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
    batch_size: int = 256,
    num_batches: int = 8,
    deltas: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    actor_goal_horizons: tuple[int, ...] = (1, 2, 4, 8, 16),
    seed: int = 0,
) -> dict[str, float]:
    """Run every diagnostic that applies to the agent's configured variant."""

    metrics: dict[str, float] = {}
    has_critic = str(agent.config['critic_type']) != 'none'
    if has_critic:
        metrics.update(
            future_retrieval(
                agent,
                dataset,
                batch_size=batch_size,
                num_batches=num_batches,
            )
        )
        metrics.update(
            latent_geometry(agent, dataset, deltas=deltas, batch_size=batch_size)
        )
    if str(agent.config['critic_type']) == 'sa':
        metrics.update(
            action_sensitivity(
                agent,
                dataset,
                action_low=action_low,
                action_high=action_high,
                batch_size=batch_size,
                num_batches=num_batches,
            )
        )
    metrics.update(
        actor_diagnostics(
            agent,
            dataset,
            action_low=action_low,
            action_high=action_high,
            batch_size=batch_size,
            num_batches=num_batches,
        )
    )
    metrics.update(
        actor_goal_sensitivity(
            agent,
            dataset,
            horizons=actor_goal_horizons,
            batch_size=batch_size,
        )
    )
    if bool(agent.config['use_flow']):
        metrics.update(
            flow_reconstruction(
                agent,
                dataset,
                batch_size=batch_size,
                num_batches=max(1, num_batches // 2),
                seed=seed,
            )
        )
    return metrics


__all__ = [
    'action_sensitivity',
    'actor_diagnostics',
    'actor_goal_sensitivity',
    'flow_reconstruction',
    'future_retrieval',
    'latent_geometry',
    'run_all_diagnostics',
]
