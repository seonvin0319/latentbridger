"""Validated, deterministic learned-goalspace checkpoints."""

from __future__ import annotations

import os
import pickle
import random
import re
from pathlib import Path
from typing import Any

import flax
import numpy as np

from learned_goalspace import multihorizon as multihorizon_mod
from learned_goalspace import pretrain as pretrain_mod

_NAME = re.compile(r'^params_(\d+)\.pkl$')

VARIANT_ARCHITECTURE = {
    pretrain_mod.VARIANT: pretrain_mod.ARCHITECTURE,
    multihorizon_mod.VARIANT: multihorizon_mod.ARCHITECTURE,
}


def _path(path: str | os.PathLike[str], step: int | None = None) -> tuple[Path, int]:
    path = Path(path)
    if path.suffix == '.pkl':
        match = _NAME.fullmatch(path.name)
        inferred = int(match.group(1)) if match else None
        if step is None:
            if inferred is None:
                raise ValueError('Nonstandard checkpoint filename needs an explicit step.')
            step = inferred
        if inferred is not None and inferred != int(step):
            raise ValueError('Checkpoint filename and requested step disagree.')
        return path, int(step)
    if step is None or int(step) < 1:
        raise ValueError('Checkpoint directory requires a positive step.')
    return path / f'params_{int(step)}.pkl', int(step)


def save_pretrain_checkpoint(agent: Any, path: str | os.PathLike[str], step: int) -> str:
    checkpoint, step = _path(path, step)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(agent.config)
    metadata['step'] = step
    payload = {
        'format': 'learned_goalspace_pretrain_v1',
        'metadata': metadata,
        'agent': flax.serialization.to_state_dict(agent),
        'numpy_random_state': np.random.get_state(),
        'python_random_state': random.getstate(),
    }
    temporary = checkpoint.with_suffix('.pkl.tmp')
    with temporary.open('wb') as file:
        pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, checkpoint)
    return str(checkpoint)


def _validate_metadata(
    metadata: dict[str, Any],
    *,
    env_name: str,
    obs_dim: int,
    expected_step: int | None,
    allow_nonproduction_step: bool,
    expected_variant: str | None = None,
) -> None:
    variant = metadata.get('variant')
    if expected_variant is not None:
        if variant != expected_variant:
            raise ValueError(f'Pretrain checkpoint variant mismatch: expected {expected_variant!r}, got {variant!r}.')
    elif variant not in VARIANT_ARCHITECTURE:
        raise ValueError(f'Unsupported pretrain variant {variant!r}; known={sorted(VARIANT_ARCHITECTURE)}.')
    architecture = VARIANT_ARCHITECTURE[str(variant)]
    expected = {
        'variant': str(variant),
        'env_name': str(env_name),
        'obs_dim': int(obs_dim),
        'latent_dim': pretrain_mod.LATENT_DIM,
        'architecture': architecture,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'Pretrain checkpoint {key} mismatch: expected {value!r}, got {metadata.get(key)!r}.')
    step = int(metadata.get('step', -1))
    if expected_step is not None and step != int(expected_step):
        raise ValueError(f'Expected pretrain step {expected_step}, got {step}.')
    if not allow_nonproduction_step and step != 500_000:
        raise ValueError(f'Production downstream requires predetermined step 500000, got {step}.')


def restore_pretrainer(
    template: Any,
    path: str | os.PathLike[str],
    *,
    step: int | None = None,
) -> Any:
    checkpoint, resolved_step = _path(path, step)
    with checkpoint.open('rb') as file:
        payload = pickle.load(file)
    if payload.get('format') != 'learned_goalspace_pretrain_v1':
        raise ValueError(f'Invalid learned-goalspace checkpoint: {checkpoint}')
    expected_variant = str(template.config['variant'])
    _validate_metadata(
        payload['metadata'],
        env_name=template.config['env_name'],
        obs_dim=int(template.config['obs_dim']),
        expected_step=resolved_step,
        allow_nonproduction_step=True,
        expected_variant=expected_variant,
    )
    restored = flax.serialization.from_state_dict(template, payload['agent'])
    np.random.set_state(payload['numpy_random_state'])
    random.setstate(payload['python_random_state'])
    return restored


def load_goal_encoder(
    path: str | os.PathLike[str],
    *,
    env_name: str,
    obs_dim: int,
    step: int | None = None,
    allow_nonproduction_step: bool = False,
    expected_variant: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    checkpoint, resolved_step = _path(path, step)
    with checkpoint.open('rb') as file:
        payload = pickle.load(file)
    if payload.get('format') != 'learned_goalspace_pretrain_v1':
        raise ValueError(f'Invalid learned-goalspace checkpoint: {checkpoint}')
    metadata = dict(payload['metadata'])
    _validate_metadata(
        metadata,
        env_name=env_name,
        obs_dim=obs_dim,
        expected_step=resolved_step,
        allow_nonproduction_step=allow_nonproduction_step,
        expected_variant=expected_variant,
    )
    params = payload['agent']['network']['params']['modules_goal_encoder']
    # Match the container type produced by the installed Flax model.init.  A
    # nested FrozenDict inside an otherwise plain dict changes the JAX treedef.
    return flax.core.unfreeze(flax.core.freeze(params)), metadata


__all__ = [
    'VARIANT_ARCHITECTURE',
    'load_goal_encoder',
    'restore_pretrainer',
    'save_pretrain_checkpoint',
]
