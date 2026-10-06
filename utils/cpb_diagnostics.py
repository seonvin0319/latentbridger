"""Infrequent held-out diagnostics; never feed back into training."""
import jax
import jax.numpy as jnp
import numpy as np


def diagnostics(agent, batch, dataset_states):
    s, g = batch['observations'], batch['value_goals']
    out = {}
    emb = []
    for name, states in [('phi', s), ('psi', g)]:
        x = np.asarray(agent.network.select(name)(states))
        emb.append(x)
        std = x.std(axis=0)
        norm = np.linalg.norm(x, axis=-1)
        eig = np.maximum(np.linalg.eigvalsh(np.cov(x, rowvar=False)), 0.)
        p = eig / max(eig.sum(), 1e-12)
        out.update({f'{name}/std_mean': float(std.mean()), f'{name}/std_min': float(std.min()),
                    f'{name}/norm_mean': float(norm.mean()), f'{name}/norm_std': float(norm.std()),
                    f'{name}/effective_rank': float(np.exp(-np.sum(p * np.log(p + 1e-12))))})
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
    scores = np.asarray(agent.score(candidates.reshape(-1, s.shape[-1]), flat_g)).reshape(len(s), n)
    targets = np.asarray(agent.score(candidates.reshape(-1, s.shape[-1]), flat_g, target=True)).reshape(len(s), n)
    out['candidate/selected_C'] = float(scores[np.arange(len(s)), np.asarray(best)].mean())
    out['candidate/C_mean'], out['candidate/C_std'] = float(scores.mean()), float(scores.std())
    out['candidate/selected_rank'] = float((1 + (scores > scores[np.arange(len(s)), np.asarray(best), None]).sum(axis=1)).mean())
    out['candidate/top1_top2_margin'] = float(np.diff(np.sort(scores, axis=1)[:, -2:], axis=1).mean()) if n > 1 else 0.
    out['candidate/target_ranking_agreement'] = float((targets.argmax(axis=1) == np.asarray(best)).mean())
    out['endpoint/sampled_displacement_norm'] = float(np.linalg.norm(np.asarray(candidates) - s[:, None], axis=-1).mean())
    out['endpoint/selected_displacement_norm'] = float(np.linalg.norm(np.asarray(selected) - s, axis=-1).mean())
    # Exact nearest state over the entire training dataset, chunked for bounded memory.
    nearest = np.full(len(s), np.inf)
    selected_np = np.asarray(selected)
    for start in range(0, len(dataset_states), 4096):
        states = np.asarray(dataset_states[start:start + 4096])
        distances = np.maximum((selected_np**2).sum(1)[:, None] + (states**2).sum(1)[None] - 2 * selected_np @ states.T, 0.)
        nearest = np.minimum(nearest, distances.min(axis=1))
    out['endpoint/selected_nearest_dataset_state_distance'] = float(np.sqrt(nearest).mean())
    prefix = np.asarray(agent._construct_bridge_prefix(s, batch['endpoint_targets']))
    errors = np.abs(prefix[:, 1:] - batch['bridge_targets']).sum(axis=-1)
    out['bridge/first_step_state_error'] = float(errors[:, 0].mean())
    out['bridge/five_step_prefix_error'] = float(errors.mean())
    return out
