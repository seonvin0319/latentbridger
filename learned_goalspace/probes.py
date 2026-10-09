"""Post-hoc NumPy probes; this is the only learned-goalspace oracle boundary."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from learned_goalspace.checkpoints import load_goal_encoder
from learned_goalspace.pretrain import GoalEncoder, LATENT_DIM
from utils.goal_representation import (
    goal_representation,
    infer_phi_goal_obs_indices,
)

PROBE_SEED = 41719


def frozen_embeddings(params: Any, observations: np.ndarray) -> np.ndarray:
    """Materialize frozen host embeddings, severing every autodiff path."""

    values = GoalEncoder().apply(
        {'params': params},
        jnp.asarray(observations, dtype=jnp.float32),
        normalize=False,
    )
    return np.asarray(jax.device_get(values), dtype=np.float64)


def deterministic_split(size: int, seed: int = PROBE_SEED) -> tuple[np.ndarray, np.ndarray]:
    if size < 4:
        raise ValueError('At least four probe observations are required.')
    order = np.random.default_rng(seed).permutation(size)
    cut = max(1, min(size - 1, int(round(0.8 * size))))
    return order[:cut], order[cut:]


def _linear_fit(x: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> np.ndarray:
    design = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    gram = design.T @ design
    gram.flat[:: len(gram) + 1] += ridge
    return np.linalg.solve(gram, design.T @ y)


def _linear_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((len(x), 1))], axis=1) @ weights


def _regression(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean(np.square(prediction - y)))
    denominator = float(np.sum(np.square(y - y.mean(axis=0, keepdims=True))))
    r2 = 1.0 - float(np.sum(np.square(prediction - y))) / max(denominator, 1e-12)
    return r2, mse


def _effective_rank(embeddings: np.ndarray) -> float:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    probabilities = np.square(singular)
    probabilities /= max(float(probabilities.sum()), 1e-12)
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12)))
    return float(np.exp(entropy))


def representation_statistics(embeddings: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(embeddings, axis=1)
    std = embeddings.std(axis=0)
    result = {
        'per_dim_std_mean': float(std.mean()),
        'per_dim_std_min': float(std.min()),
        'effective_rank': _effective_rank(embeddings),
        'norm_mean': float(norms.mean()),
        'norm_std': float(norms.std()),
    }
    result.update({f'dim_{index}_std': float(value) for index, value in enumerate(std)})
    return result


def _representations(
    learned: np.ndarray,
    observations: np.ndarray,
    train: np.ndarray,
) -> dict[str, np.ndarray]:
    centered = observations - observations[train].mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered[train], full_matrices=False)
    components = vt[: min(LATENT_DIM, len(vt))]
    pca = centered @ components.T
    if pca.shape[1] < LATENT_DIM:
        pca = np.pad(pca, ((0, 0), (0, LATENT_DIM - pca.shape[1])))
    random_projection = np.random.default_rng(PROBE_SEED).normal(size=(observations.shape[1], LATENT_DIM)) / np.sqrt(
        observations.shape[1]
    )
    return {
        'learned_E': learned,
        'pca16': pca,
        'random16': observations @ random_projection,
    }


def _nearest_metrics(
    embeddings: np.ndarray,
    phi: np.ndarray,
    *,
    discrete: bool,
) -> dict[str, float]:
    limit = min(len(embeddings), 4096)
    chosen = np.random.default_rng(PROBE_SEED + 1).choice(len(embeddings), size=limit, replace=False)
    x, target = embeddings[chosen], phi[chosen]
    distance = np.sum(np.square(x[:, None] - x[None, :]), axis=-1)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argmin(distance, axis=1)
    if discrete:
        equal = np.asarray(target == target[nearest])
        return {
            'nearest_neighbor_purity': float(equal.mean()),
            'nearest_neighbor_exact_purity': float(equal.all(axis=1).mean()),
        }
    error = np.linalg.norm(target - target[nearest], axis=1)
    return {
        'nearest_neighbor_error_mean': float(error.mean()),
        'nearest_neighbor_error_median': float(np.median(error)),
    }


def run_checkpoint_probes(
    *,
    checkpoint: str | Path,
    step: int,
    env_name: str,
    observations: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations = np.asarray(observations, dtype=np.float32)
    params, _ = load_goal_encoder(
        checkpoint,
        env_name=env_name,
        obs_dim=observations.shape[-1],
        step=step,
        allow_nonproduction_step=True,
    )
    embeddings = frozen_embeddings(params, observations)
    train, test = deterministic_split(len(observations))
    phi = np.asarray(
        jax.device_get(
            goal_representation(
                jnp.asarray(observations),
                'phi',
                env_name=env_name,
            )
        ),
        dtype=np.float64,
    )
    indices = infer_phi_goal_obs_indices(env_name, observations.shape[-1])
    nuisance_indices = np.asarray(
        [index for index in range(observations.shape[-1]) if index not in indices],
        dtype=np.int64,
    )
    nuisance = observations[:, nuisance_indices]
    discrete = 'puzzle' in env_name.lower()

    representation_rows: list[dict[str, Any]] = []
    probe_rows: list[dict[str, Any]] = []
    for name, features in _representations(embeddings, observations, train).items():
        for metric, value in representation_statistics(features).items():
            representation_rows.append(dict(checkpoint=step, representation=name, metric=metric, value=value))

        if discrete:
            accuracies = []
            exact_predictions = []
            for dimension in range(phi.shape[1]):
                classes = np.unique(phi[train, dimension])
                one_hot = (phi[train, dimension, None] == classes[None]).astype(float)
                prediction = _linear_predict(
                    features[test],
                    _linear_fit(features[train], one_hot),
                )
                predicted = classes[np.argmax(prediction, axis=1)]
                accuracies.append(float(np.mean(predicted == phi[test, dimension])))
                exact_predictions.append(predicted == phi[test, dimension])
                probe_rows.append(
                    dict(
                        checkpoint=step,
                        representation=name,
                        metric=f'button_{dimension}_accuracy',
                        value=accuracies[-1],
                    )
                )
            aggregate = {
                'button_accuracy_mean': float(np.mean(accuracies)),
                'button_exact_state_accuracy': float(np.stack(exact_predictions, axis=1).all(axis=1).mean()),
            }
        else:
            prediction = _linear_predict(features[test], _linear_fit(features[train], phi[train]))
            r2, mse = _regression(phi[test], prediction)
            aggregate = {'cube_positions_r2': r2, 'cube_positions_mse': mse}
        aggregate.update(_nearest_metrics(features[test], phi[test], discrete=discrete))

        nuisance_prediction = _linear_predict(features[test], _linear_fit(features[train], nuisance[train]))
        nuisance_r2, nuisance_mse = _regression(nuisance[test], nuisance_prediction)
        aggregate.update({'nuisance_r2': nuisance_r2, 'nuisance_mse': nuisance_mse})
        probe_rows.extend(
            dict(checkpoint=step, representation=name, metric=metric, value=value)
            for metric, value in aggregate.items()
        )
    return representation_rows, probe_rows


def write_metric_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    fields = (
        'env',
        'variant',
        'seed',
        'checkpoint',
        'representation',
        'metric',
        'value',
    )
    with temporary.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


__all__ = [
    'PROBE_SEED',
    'deterministic_split',
    'frozen_embeddings',
    'representation_statistics',
    'run_checkpoint_probes',
    'write_metric_csv',
]
