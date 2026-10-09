"""Deterministic kNN conditional-variance estimator.

Same construction as the local-intention diagnostic (neighbourhood includes the
query itself; variance is the mean over dimensions with ``ddof=1``). The
decision neighbourhood size is fixed at ``KNN_DEFAULT`` before looking at route
results. Other sizes are reported and then ignored by the gate.
"""

from __future__ import annotations

import numpy as np

from route_intention.common import KNN_DEFAULT


def knn_indices(features: np.ndarray, k: int, queries: np.ndarray | None = None) -> np.ndarray:
    """Return ``[Q, k]`` neighbour indices. ``queries`` index ``features`` (default: all rows)."""
    feats = np.asarray(features, dtype=np.float64)
    if queries is None:
        queries = np.arange(len(feats))
    queries = np.asarray(queries, dtype=np.int64)
    k = int(k)
    if k < 2 or k > len(feats):
        raise ValueError(f'k={k} is outside [2, {len(feats)}]')
    out = np.empty((len(queries), k), dtype=np.int64)
    norms = (feats ** 2).sum(1)
    for i in range(0, len(queries), 256):
        q_idx = queries[i : i + 256]
        q = feats[q_idx]
        d2 = (q ** 2).sum(1)[:, None] - 2.0 * q @ feats.T + norms[None, :]
        part = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
        # Stable order inside the neighbourhood so ties do not depend on argpartition.
        row = np.arange(len(q_idx))[:, None]
        order = np.argsort(d2[row, part], axis=1, kind='mergesort')
        out[i : i + len(q_idx)] = part[row, order]
    return out


def _pooled_within(values: np.ndarray, groups: np.ndarray) -> float:
    num = 0.0
    den = 0
    for gval in np.unique(groups):
        m = groups == gval
        if int(m.sum()) >= 2:
            num += float(values[m].var(axis=0, ddof=1).mean()) * (int(m.sum()) - 1)
            den += int(m.sum()) - 1
    return num / den if den else float('nan')


def conditional_variance(values: np.ndarray, codes: np.ndarray, neigh: np.ndarray) -> dict[str, float]:
    """Mean neighbourhood variance of ``values`` with and without grouping by ``codes``."""
    values = np.asarray(values, dtype=np.float64)
    codes = np.asarray(codes)
    if values.ndim == 1:
        values = values[:, None]
    tot, cond = [], []
    for nb in neigh:
        v = values[nb]
        tot.append(float(v.var(axis=0, ddof=1).mean()))
        cond.append(_pooled_within(v, codes[nb]))
    var_uncond = float(np.nanmean(tot))
    var_cond = float(np.nanmean(cond))
    ratio = var_cond / var_uncond if var_uncond > 0 else float('nan')
    return dict(var_uncond=var_uncond, var_cond=var_cond, variance_ratio=ratio, knn_k=int(neigh.shape[1]), num_queries=int(len(neigh)))


def variance_report(values: np.ndarray, features: np.ndarray, codes: np.ndarray, ks: tuple[int, ...] = (KNN_DEFAULT,)) -> list[dict]:
    rows = []
    for k in ks:
        neigh = knn_indices(features, int(k))
        rows.append(conditional_variance(values, codes, neigh))
    return rows


def between_within(embeddings: np.ndarray, codes: np.ndarray) -> dict[str, float]:
    """Mean pairwise L2 of per-code centroids vs mean pairwise L2 inside each code."""
    emb = np.asarray(embeddings, dtype=np.float64)
    codes = np.asarray(codes)
    within = []
    centroids = []
    for c in np.unique(codes):
        m = emb[codes == c]
        centroids.append(m.mean(0))
        if len(m) >= 2:
            d = np.linalg.norm(m[:, None, :] - m[None, :, :], axis=-1)
            within.append(float(d[np.triu_indices(len(m), k=1)].mean()))
    centroids = np.stack(centroids)
    if len(centroids) >= 2:
        bd = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
        between = float(bd[np.triu_indices(len(centroids), k=1)].mean())
    else:
        between = float('nan')
    return dict(between=between, within=float(np.mean(within)) if within else float('nan'))
