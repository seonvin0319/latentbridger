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
) -> dict[str, float]:
    """Check that ``C(s, a, g)`` actually depends on ``a``.

    For a fixed ``(s, g)`` the dataset action is compared against a shuffled
    dataset action and against a uniform action inside the real action box.  A
    critic that ignores the action lands at probability 0.5 with a zero margin.
    """

    action_low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(action_high, dtype=np.float32).reshape(-1)

    shuffled_wins: list[np.ndarray] = []
    uniform_wins: list[np.ndarray] = []
    shuffled_margins: list[np.ndarray] = []
    uniform_margins: list[np.ndarray] = []
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
        shuffled_scores = _device_array(
            agent.critic_scores(observations, shuffled_actions, goals)
        )
        uniform_scores = _device_array(
            agent.critic_scores(observations, uniform_actions, goals)
        )

        shuffled_wins.append((data_scores > shuffled_scores).astype(np.float32))
        uniform_wins.append((data_scores > uniform_scores).astype(np.float32))
        shuffled_margins.append(data_scores - shuffled_scores)
        uniform_margins.append(data_scores - uniform_scores)

    return {
        'action_sensitivity/p_data_gt_shuffled': float(
            np.concatenate(shuffled_wins).mean()
        ),
        'action_sensitivity/p_data_gt_uniform': float(
            np.concatenate(uniform_wins).mean()
        ),
        'action_sensitivity/margin_shuffled': float(
            np.concatenate(shuffled_margins).mean()
        ),
        'action_sensitivity/margin_uniform': float(
            np.concatenate(uniform_margins).mean()
        ),
    }


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
        metrics[f'latent_geometry/future_similarity_delta_{int(delta)}'] = float(
            np.mean(np.sum(anchors * futures, axis=-1))
        )
        metrics[f'latent_geometry/random_similarity_delta_{int(delta)}'] = float(
            np.mean(np.sum(anchors * randoms, axis=-1))
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
    """

    rng = jax.random.PRNGKey(int(seed))
    prefix_mses: list[float] = []
    step_cosines: list[np.ndarray] = []
    endpoint_hits: list[float] = []
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

        prefix_mses.append(float(np.mean((generated - target_latents) ** 2)))
        numerator = np.sum(generated * target_latents, axis=-1)
        denominator = np.maximum(
            np.linalg.norm(generated, axis=-1) * np.linalg.norm(target_latents, axis=-1),
            1e-8,
        )
        step_cosines.append(np.mean(numerator / denominator, axis=0))

        # Endpoint-conditioned retrieval consistency: does the generated final
        # latent still identify its own trajectory among the batch's endpoints?
        generated_endpoint = generated[:, -1, :]
        target_endpoint = target_latents[:, -1, :]
        similarity = generated_endpoint @ target_endpoint.T
        endpoint_hits.append(
            float(np.mean(np.argmax(similarity, axis=1) == np.arange(len(similarity))))
        )

    mean_step_cosine = np.mean(np.stack(step_cosines, axis=0), axis=0)
    metrics: dict[str, float] = {
        'flow/prefix_latent_mse': float(np.mean(prefix_mses)),
        'flow/endpoint_retrieval_consistency': float(np.mean(endpoint_hits)),
        'flow/mean_prefix_cosine': float(np.mean(mean_step_cosine)),
    }
    for index, cosine in enumerate(mean_step_cosine, start=1):
        metrics[f'flow/prefix_cosine_step_{index}'] = float(cosine)
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
    'flow_reconstruction',
    'future_retrieval',
    'latent_geometry',
    'run_all_diagnostics',
]
