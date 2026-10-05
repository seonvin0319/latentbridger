"""Offline diagnostics for the latent endpoint chunk architecture."""

from __future__ import annotations

from collections import defaultdict

import jax
import numpy as np


def _recall(logits: np.ndarray, at: int) -> float:
    positive = np.diag(logits)
    ranks = np.sum(logits >= positive[:, None], axis=1)
    return float(np.mean(ranks <= int(at)))


def run_chunk_diagnostics(agent, dataset, *, batches: int = 16, batch_size: int = 256):
    """Measure representation retrieval, action sensitivity, and policy support."""

    rows: dict[str, list[float]] = defaultdict(list)
    for index in range(int(batches)):
        batch = dataset.sample(int(batch_size))
        observations = batch['observations']
        chunks = batch['action_chunks']
        endpoints = batch['endpoint_states']
        goals = batch['goals']

        endpoint_logits = np.asarray(
            agent.endpoint_latents(observations, chunks)
            @ agent.state_latents(endpoints).T
        )
        sg_logits = np.asarray(
            agent.state_latents(observations) @ agent.goal_latents(goals).T
        )
        for name, logits in (('endpoint', endpoint_logits), ('state_goal', sg_logits)):
            rows[f'{name}/recall_at_1'].append(_recall(logits, 1))
            rows[f'{name}/recall_at_5'].append(_recall(logits, 5))

        rng = jax.random.PRNGKey(index + 91_337)
        negative = np.asarray(
            agent.sample_proposal_chunks(
                observations, goals, rng, num_candidates=1
            )[:, 0]
        )
        positive_score = np.asarray(agent.composed_scores(observations, chunks, goals))
        negative_score = np.asarray(agent.composed_scores(observations, negative, goals))
        rows['action/p_positive_gt_q'].append(
            float(np.mean(positive_score > negative_score))
        )

        predicted = np.asarray(agent.sample_action_chunks(observations, goals))
        rows['policy/chunk_bc_mse'].append(float(np.mean(np.square(predicted - chunks))))
        rows['policy/support_logprob'].append(
            float(np.mean(np.asarray(agent.proposal_log_prob(observations, goals, predicted))))
        )
        rows['policy/critic_score'].append(
            float(np.mean(np.asarray(agent.composed_scores(observations, predicted, goals))))
        )

        if str(agent.config['policy_type']) == 'awr':
            score = np.asarray(agent.composed_scores(observations, chunks, goals))
            value = np.asarray(agent.state_goal_values(observations, goals))
            weights = np.clip(
                np.exp((score - value) / float(agent.config['awr_beta'])),
                0.0,
                float(agent.config['awr_weight_max']),
            )
            rows['awr/weight_mean'].append(float(np.mean(weights)))
            rows['awr/weight_std'].append(float(np.std(weights)))
            rows['awr/weight_max'].append(float(np.max(weights)))
            rows['awr/weight_ess'].append(
                float(np.square(np.sum(weights)) / (np.sum(np.square(weights)) + 1e-8))
            )

    return {key: float(np.mean(values)) for key, values in sorted(rows.items())}


__all__ = ['run_chunk_diagnostics']
