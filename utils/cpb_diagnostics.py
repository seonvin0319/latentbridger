"""Held-out calibrated CPB diagnostics; none feed into the training objective."""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import rankdata


def correlation(left, right):
    left, right = np.asarray(left).reshape(-1), np.asarray(right).reshape(-1)
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return 0., False
    return float(np.corrcoef(left, right)[0, 1]), True


def nearest_state_distances(selected, dataset_states):
    selected = np.asarray(selected)
    nearest = np.full(len(selected), np.inf)
    for start in range(0, len(dataset_states), 4096):
        states = np.asarray(dataset_states[start:start + 4096])
        distances = np.maximum((selected**2).sum(1)[:, None] + (states**2).sum(1)[None] - 2 * selected @ states.T, 0.)
        nearest = np.minimum(nearest, distances.min(axis=1))
    return np.sqrt(nearest)


def diagnostics(agent, batch, dataset_states, action_low=-1., action_high=1.):
    agent = agent.with_reference_cache()
    s, g = batch['observations'], batch['value_goals']
    out, emb = {}, []
    for name, states in [('phi', s), ('psi', g)]:
        x = np.asarray(agent.network.select(name)(states))
        emb.append(x)
        std = x.std(axis=0)
        norm = np.linalg.norm(x, axis=-1)
        eig = np.maximum(np.linalg.eigvalsh(np.cov(x, rowvar=False)), 0.)
        p = eig / max(eig.sum(), 1e-12)
        effective_rank = np.exp(-np.sum(p * np.log(p + 1e-12))) if eig.sum() > 0 else 0.
        out.update({f'{name}/std_mean': float(std.mean()), f'{name}/std_min': float(std.min()),
                    f'{name}/norm_mean': float(norm.mean()), f'{name}/norm_std': float(norm.std()),
                    f'{name}/effective_rank': float(effective_rank)})
        out.update({f'{name}/std_dim_{i}': float(v) for i, v in enumerate(std)})
        out.update({f'{name}/cov_eigenvalue_{i}': float(v) for i, v in enumerate(eig)})
    u, v = [x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8) for x in emb]
    out['embedding/positive_cosine'] = float(np.sum(u * v, axis=-1).mean())
    out['embedding/random_pair_cosine'] = float(np.sum(u * np.roll(v, 1, axis=0), axis=-1).mean())
    n = int(agent.config['eval_num_candidates'])
    candidates = agent._sample_endpoint_candidates(s, g, jax.random.PRNGKey(718),
                     num_candidates=n, temperature=float(agent.config['eval_temperature']))
    selected, best = agent.rank_candidates(candidates, g)
    flat_g = jnp.broadcast_to(g[:, None], (*candidates.shape[:2], g.shape[-1])).reshape(-1, g.shape[-1])
    flat_z = candidates.reshape(-1, s.shape[-1])
    raw = np.asarray(agent.raw_score(flat_z, flat_g)).reshape(len(s), n)
    scores = np.asarray(agent.calibrated_score(flat_z, flat_g)).reshape(len(s), n)
    targets = np.asarray(agent.target_calibrated_score(flat_z, flat_g)).reshape(len(s), n)
    out['candidate/selected_Cbar'] = float(scores[np.arange(len(s)), np.asarray(best)].mean())
    out['candidate/Cbar_mean'], out['candidate/Cbar_std'] = float(scores.mean()), float(scores.std())
    out['candidate/selected_rank'] = float((1 + (scores > scores[np.arange(len(s)), np.asarray(best), None]).sum(axis=1)).mean())
    out['candidate/top1_top2_margin'] = float(np.diff(np.sort(scores, axis=1)[:, -2:], axis=1).mean()) if n > 1 else 0.
    out['candidate/target_ranking_agreement'] = float((targets.argmax(axis=1) == np.asarray(best)).mean())
    correlations = [correlation(rankdata(a), rankdata(b)) for a, b in zip(raw, scores)]
    defined = [value for value, valid in correlations if valid]
    out['calibration/ranking_spearman'] = float(np.mean(defined)) if defined else 0.
    out['calibration/ranking_correlation_defined_fraction'] = len(defined) / len(s)
    out['calibration/raw_calibrated_top1_agreement'] = float((raw.argmax(axis=1) == scores.argmax(axis=1)).mean())
    z = batch['endpoint_targets']
    eg = batch['endpoint_goals']
    raw_delta = np.asarray(agent.raw_score(z, eg, target=True) - agent.raw_score(s, eg, target=True))
    cal_delta = np.asarray(agent.target_calibrated_score(z, eg) - agent.target_calibrated_score(s, eg))
    corr, defined = correlation(raw_delta, cal_delta)
    out['calibration/progress_pearson'] = corr
    out['calibration/progress_correlation_defined'] = float(defined)
    out['calibration/progress_sign_agreement'] = float((np.sign(raw_delta) == np.sign(cal_delta)).mean())
    for name, values in [('raw_delta', raw_delta), ('calibrated_delta', cal_delta)]:
        out[f'calibration/{name}_mean'] = float(values.mean())
        out[f'calibration/{name}_std'] = float(values.std())
        for q in (0, 25, 50, 75, 100):
            out[f'calibration/{name}_p{q}'] = float(np.percentile(values, q))
    logz = np.asarray(agent.log_partition(s))
    out['calibration/log_partition_mean'], out['calibration/log_partition_std'] = float(logz.mean()), float(logz.std())
    _, _, progress = agent._progress(s, eg, z)
    out.update({f'heldout/{k}': float(np.asarray(v)) for k, v in progress.items()})
    out['endpoint/target_displacement_norm'] = float(np.linalg.norm(z-s, axis=-1).mean())
    out['endpoint/sampled_displacement_norm'] = float(np.linalg.norm(np.asarray(candidates) - s[:, None], axis=-1).mean())
    out['endpoint/selected_displacement_norm'] = float(np.linalg.norm(np.asarray(selected) - s, axis=-1).mean())
    out['endpoint/selected_nearest_dataset_state_distance'] = float(nearest_state_distances(selected, dataset_states).mean())
    if 'puzzle' in agent.config['env_name']:
        # Nested subsets of the same candidate pool isolate candidate-count effects.
        for count in (1, 8, 32):
            if count <= n:
                chosen, _ = agent.rank_candidates(candidates[:, :count], g)
                out[f'endpoint/N{count}_nearest_dataset_distance'] = float(nearest_state_distances(chosen, dataset_states).mean())
    prefix = np.asarray(agent._construct_bridge_prefix(s, z))
    errors = np.abs(prefix[:, 1:] - batch['bridge_targets']).sum(axis=-1)
    out['bridge/first_step_state_error'] = float(errors[:, 0].mean())
    out['bridge/five_step_prefix_error'] = float(errors.mean())
    bridge = np.asarray(agent.construct_bridge(s, z))
    out['bridge/endpoint_pin_error'] = float(np.max(np.abs(bridge[:, -1] - z)))
    out['bridge/start_pin_error'] = float(np.max(np.abs(bridge[:, 0] - s)))
    actions = np.asarray(agent.sample_action_chunks(s, g))
    out['idm/action_saturation_fraction'] = float(np.mean((actions <= action_low) | (actions >= action_high)))
    return out
