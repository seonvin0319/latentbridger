"""Checkpoint diagnostics for the temporal quasimetric and its proposer."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from agents.contrastive_transitive_distance_pathbridger import pairwise_distance
from utils.goal_representation import goal_representation


def _effective_rank(embedding: np.ndarray) -> float:
    eig = np.maximum(np.linalg.eigvalsh(np.cov(embedding, rowvar=False)), 0.0)
    total = float(eig.sum())
    if total <= 0.0:
        return 0.0
    probability = eig / total
    return float(np.exp(-np.sum(probability * np.log(probability + 1e-12))))


def _correlation(left, right) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.std() < 1e-12 or right.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _nearest(query: np.ndarray, refs: np.ndarray, row_chunk=256, col_chunk=4096) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32)
    refs = np.asarray(refs, dtype=np.float32)
    best = np.full(len(query), np.inf, dtype=np.float32)
    q2 = np.sum(query * query, axis=1)
    for start in range(0, len(refs), col_chunk):
        block = refs[start:start + col_chunk]
        r2 = np.sum(block * block, axis=1)
        for row in range(0, len(query), row_chunk):
            q = query[row:row + row_chunk]
            dist = q2[row:row + row_chunk, None] + r2[None, :] - 2.0 * q @ block.T
            best[row:row + row_chunk] = np.minimum(best[row:row + row_chunk], dist.min(axis=1))
    return np.sqrt(np.maximum(best, 0.0))


def diagnostics(agent, batch, reference_states) -> dict[str, float]:
    """Distance, transitivity, and proposer-coverage numbers for one checkpoint."""

    observations = jnp.asarray(batch['observations'])
    goals = jnp.asarray(batch['value_goals'])
    base_goals = jnp.asarray(batch['base_goals'])
    base_offsets = np.asarray(batch['base_offsets'], dtype=np.float32)
    subgoals = jnp.asarray(batch['transitive_subgoals'])
    z_true = jnp.asarray(batch['z_true'])

    scalar = agent.config['variant'] == 'gs_trl_weighted'
    out = {}
    if not scalar:
        h_state, p_state = agent._metric_encode(observations, name='value')
        h_goal, p_goal = agent._metric_encode(goals, name='value')
        h_base, p_base = agent._metric_encode(base_goals, name='value')
        h_sub, p_sub = agent._metric_encode(subgoals, name='value')
        base_distance = np.asarray(agent._metric_distance(observations, base_goals, name='value'))
        same = np.asarray(agent._metric_distance(observations, observations, name='value'))
        forward = np.asarray(agent._metric_distance(observations, goals, name='value'))
        backward = np.asarray(agent._metric_distance(goals, observations, name='value'))
        via_subgoal = np.asarray(agent._metric_distance(observations, subgoals, name='value'))
        subgoal_to_goal = np.asarray(agent._metric_distance(subgoals, goals, name='value'))
        residual = via_subgoal + subgoal_to_goal - forward
        value_offsets = np.asarray(batch['value_offsets'], dtype=np.float32)
        discount = float(agent.config['discount'])
        value_d = np.power(discount, forward)

        h_np = np.asarray(h_state)
        p_np = np.asarray(p_state)
        out = {
            'distance/base_mae': float(np.mean(np.abs(base_distance - base_offsets))),
            'distance/base_correlation': _correlation(base_distance, base_offsets),
            'distance/self_mean': float(same.mean()),
            'distance/self_max': float(same.max()),
            'distance/direct_mean': float(forward.mean()),
            'distance/direct_median': float(np.median(forward)),
            'distance/direct_p90': float(np.quantile(forward, 0.90)),
            'distance/direct_p99': float(np.quantile(forward, 0.99)),
            'distance/temporal_mae': float(np.mean(np.abs(forward - value_offsets))),
            'distance/temporal_correlation': _correlation(forward, value_offsets),
            'distance/composed_mean': float((via_subgoal + subgoal_to_goal).mean()),
            'distance/path_residual_mean': float(residual.mean()),
            'distance/path_residual_std': float(residual.std()),
            'distance/path_residual_min': float(residual.min()),
            'distance/asymmetry_mean': float(np.mean(np.abs(forward - backward))),
            'value/vd_mean': float(value_d.mean()),
            'value/vd_median': float(np.median(value_d)),
            'value/vd_p10': float(np.quantile(value_d, 0.10)),
            'value/vd_p01': float(np.quantile(value_d, 0.01)),
            'representation/h_norm_mean': float(np.linalg.norm(h_np, axis=-1).mean()),
            'representation/p_norm_mean': float(np.linalg.norm(p_np, axis=-1).mean()),
            'representation/h_effective_rank': _effective_rank(h_np),
            'representation/p_effective_rank': _effective_rank(p_np),
        }

        count = min(64, len(h_np))
        left = h_np[:count]
        mid = np.roll(h_np[:count], 1, axis=0)
        right = np.roll(h_np[:count], 2, axis=0)
        p_left, p_mid, p_right = p_np[:count], np.roll(p_np[:count], 1, axis=0), np.roll(p_np[:count], 2, axis=0)
        d_direct = np.asarray(pairwise_distance(
            jnp.asarray(left), jnp.asarray(p_left), jnp.asarray(right), jnp.asarray(p_right),
        ))
        # Pairwise returns [B,B]; the matching triplet is the diagonal after the rolls above
        # were applied to the whole row, so use the explicit per-row distance instead.
        del d_direct
        direct = np.linalg.norm(left - right, axis=-1) + np.mean(
            np.maximum(p_right - p_left, 0.0), axis=-1)
        composed = (
            np.linalg.norm(left - mid, axis=-1) + np.mean(np.maximum(p_mid - p_left, 0.0), axis=-1)
            + np.linalg.norm(mid - right, axis=-1) + np.mean(np.maximum(p_right - p_mid, 0.0), axis=-1)
        )
        violation = direct - composed
        tolerance = 1e-4
        out['triangle/violation_fraction'] = float((violation > tolerance).mean())
        out['triangle/max_violation'] = float(violation.max())

    n = int(agent.config['eval_num_candidates'])
    temperature = float(agent.config['eval_temperature'])
    candidates = np.asarray(agent._sample_endpoint_candidates(
        observations,
        goals,
        jax.random.PRNGKey(73521),
        num_candidates=n,
        temperature=temperature,
    ))
    if n == 1:
        selected = candidates[:, 0, :]
        selected_index = np.zeros(len(candidates), dtype=np.int32)
        pairwise = np.zeros(len(candidates), dtype=np.float32)
    else:
        diff = candidates[:, :, None, :] - candidates[:, None, :, :]
        pairwise = np.linalg.norm(diff, axis=-1).sum(axis=(1, 2)) / (n * (n - 1))
        flat_s = jnp.broadcast_to(observations[:, None, :], candidates.shape).reshape(-1, candidates.shape[-1])
        flat_z = jnp.asarray(candidates.reshape(-1, candidates.shape[-1]))
        flat_g = jnp.broadcast_to(goals[:, None, :], (len(goals), n, goals.shape[-1])).reshape(-1, goals.shape[-1])
        if scalar:
            logits = agent.network.select('value')(
                jnp.concatenate([flat_s, flat_z]), jnp.concatenate([flat_z, flat_g]),
            )
            left, right = jnp.split(jax.nn.sigmoid(logits), 2)
            scores = np.asarray(left * right).reshape(len(goals), n)
            selected_index = scores.argmax(axis=1)
        else:
            scores = np.asarray(
                agent._metric_distance(flat_s, flat_z, name='value')
                + agent._metric_distance(flat_z, flat_g, name='value')
            ).reshape(len(goals), n)
            selected_index = scores.argmin(axis=1)
        selected = candidates[np.arange(len(candidates)), selected_index]
    truth = np.asarray(z_true)
    truth_d = np.linalg.norm(candidates - truth[:, None, :], axis=-1)
    goal_d = np.linalg.norm(
        np.asarray(goal_representation(jnp.asarray(candidates), 'phi', env_name=agent.config['env_name']))
        - np.asarray(goal_representation(jnp.asarray(truth), 'phi', env_name=agent.config['env_name']))[:, None, :],
        axis=-1,
    )
    nearest = _nearest(candidates.reshape(-1, candidates.shape[-1]), np.asarray(reference_states))
    selected_true = np.linalg.norm(selected - truth, axis=-1)
    selected_displacement = np.linalg.norm(selected - np.asarray(observations), axis=-1)
    out.update({
        'proposer/pairwise_distance': float(np.mean(pairwise)),
        'proposer/min_true_distance': float(truth_d.min(axis=1).mean()),
        'proposer/min_goal_distance': float(goal_d.min(axis=1).mean()),
        'proposer/nearest_reference': float(nearest.mean()),
        'proposer/selected_true_distance': float(selected_true.mean()),
        'proposer/selected_displacement': float(selected_displacement.mean()),
    })

    weights, gap = agent._endpoint_weights(observations, goals, z_true)
    weights = np.asarray(weights)
    gap = np.asarray(gap)
    denominator = max(float(np.square(weights).sum()), 1e-12)
    ess = float((weights.sum() ** 2) / denominator)
    out.update({
        'weight/mean': float(weights.mean()),
        'weight/std': float(weights.std()),
        'weight/max': float(weights.max()),
        'weight/cap_fraction': float((weights >= 5.0 - 1e-5).mean()),
        'weight/ess_fraction': ess / len(weights),
        'weight/delta_mean': float(gap.mean()),
        'weight/delta_std': float(gap.std()),
    })
    if not scalar and 'path_positive_states' in batch:
        _, path_info = agent.path_nce_loss(batch, agent.network.params)
        for key, value in path_info.items():
            out[key] = float(np.asarray(value))
    if 'endpoint_offsets' in batch and 'bridge_targets' in batch:
        endpoints = jnp.asarray(batch['endpoint_targets'])
        prefix = np.asarray(agent._construct_bridge_prefix(observations, endpoints))
        predicted = prefix[:, 1:6]
        truth = np.asarray(batch['bridge_targets'])
        step_l1 = np.abs(predicted - truth).sum(axis=-1)
        out['bridge/prefix_l1'] = float(step_l1.mean())
        out['bridge/first_step_l1'] = float(step_l1[:, 0].mean())
        for step in range(step_l1.shape[1]):
            out[f'bridge/prefix_l1_step{step + 1}'] = float(step_l1[:, step].mean())
        if not scalar:
            _, geo_info = agent.bridge_geometry_loss(batch, agent.network.params)
            for key, value in geo_info.items():
                out[key] = float(np.asarray(value))
        if 'future_actions' in batch:
            current = jnp.asarray(prefix[:, :-1].reshape(-1, prefix.shape[-1]))
            nxt = jnp.asarray(prefix[:, 1:].reshape(-1, prefix.shape[-1]))
            decoded = np.asarray(agent.network.select('idm')(current, nxt)).reshape(len(observations), 5, -1)
            out['bridge/idm_action_mse'] = float(np.mean((decoded - np.asarray(batch['future_actions'])) ** 2))
    if float(agent.config['lambda_nce']) > 0.0:
        _, nce_info = agent.nce_loss(batch, agent.network.params)
        for key, value in nce_info.items():
            if key.startswith('nce/'):
                out[key] = float(np.asarray(value))
    return out
