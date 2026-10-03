#!/usr/bin/env python3
"""Shared planning logic for the long LatentBridger sweep.

Two things here are load-bearing for the sweep's integrity.

**Stage fingerprints.**  A checkpoint may only be resumed or shared when the
computation that produced it is the computation we are about to continue.  The
fingerprint pins the seven fields that decide that: the resolved agent config,
the objective signature, the representation dimension, the action-NCE setting,
the actor goal offsets, the flow target offsets, and the seed.  Any mismatch
demotes a ``[RESUME]`` to a ``[TRAIN]`` instead of silently continuing from an
incompatible checkpoint.

**Stage sharing.**  Variants whose critic (or actor) stage is literally the
same computation must train it once.  Separate processes do not reproduce each
other bitwise anyway, so a shared checkpoint is both cheaper and a stricter
ablation than two nominally identical runs.
"""

from __future__ import annotations

import functools
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.latentbridger import VARIANT_SETTINGS, get_config  # noqa: E402


@functools.lru_cache(maxsize=1)
def _defaults() -> dict:
    return dict(get_config())

# The fields a resume must agree on, in the order the spec lists them.
FINGERPRINT_FIELDS = (
    'config_hash',
    'objective_signature',
    'repr_dim',
    'action_nce',
    'actor_goal_offsets',
    'flow_target_offsets',
    'seed',
    # Not in the spec's list, but a stage trained at a different batch size is
    # a different computation, and batch size lives in the flags rather than
    # the config, so nothing else would catch it.
    'batch_size',
    # The frozen parent is part of the objective.  An actor trained against the
    # 100k critic is not a prefix of an actor trained against the 1M critic:
    # continuing the former would spend 900k updates optimizing against a
    # representation the pipeline has already moved past.  Root stages (the
    # critic, and the actor of a critic-free variant) record 0.
    'parent_step',
)

# The config entries that decide *what* a stage computes.  This is an explicit
# allowlist rather than "every key except a few" so that adding a new config
# field with a behaviour-preserving default does not invalidate every existing
# checkpoint: a field absent from an older run is filled from the current
# default, which for a behaviour-preserving field is exactly what that run did.
# `variant`, `eval_mode`, `replan_interval`, and `use_flow` are deliberately
# absent -- they name or evaluate a stage without changing its gradients.
_STRUCTURAL_KEYS = (
    'env_name',
    'horizon',
    'discount',
    'repr_dim',
    'hidden_dims',
    'layer_norm',
    'repr_norm',
    'contrastive_temperature',
    'logsumexp_coef',
    'goal_representation_mode',
    'phi_goal_obs_indices',
    'future_sampling',
    'bridge_goal_sampling',
    'actor_goal_sampling',
    'actor_goal_max_offset',
    'actor_goal_offsets',
    'actor_discount',
    'action_horizon',
    'flow_target_mode',
    'flow_target_offsets',
    'critic_type',
    'actor_goal_input',
    'actor_objective',
    'actor_bc_coef',
    'action_nce_coef',
    'num_action_negatives',
    'flow_steps',
    'flow_noise_scale',
    'flow_renormalize',
    'flow_stop_psi_gradient',
    'learning_rate',
)


def stage_plan(variant: str) -> tuple[str, ...]:
    """Stages the research protocol runs for ``variant``, in order."""

    settings = VARIANT_SETTINGS[variant]
    stages: list[str] = []
    if settings['critic_type'] != 'none':
        stages.append('critic')
    stages.append('actor')
    if settings['use_flow']:
        stages.append('flow')
    return tuple(stages)


def critic_signature(variant: str) -> str | None:
    """Name the critic computation, or ``None`` when there is no critic."""

    settings = VARIANT_SETTINGS[variant]
    if settings['critic_type'] == 'none':
        return None
    return f'critic_{settings["critic_type"]}_actnce{settings["action_nce_coef"]:g}'


def actor_signature(variant: str) -> str:
    """Name the actor computation.

    ``latent_rf_sparse`` is ``actnce_multihorizon`` plus Module B, and
    ``latent_rf_actnce`` is ``sa_cl_bc_actnce`` plus Module B.  In both pairs
    the actor stage is the same objective against the same frozen critic, so
    the flow must be built on the *same* controller.
    """

    settings = VARIANT_SETTINGS[variant]
    critic = critic_signature(variant) or 'nocritic'
    if settings.get('actor_goal_sampling', 'offsets') == 'geometric':
        horizon = f'geo{settings.get("actor_discount", 0.0):g}'
    else:
        offsets = '-'.join(
            str(int(offset)) for offset in settings['actor_goal_offsets']
        )
        horizon = f'h{offsets}'
    return (
        f'actor_{critic}_{settings["actor_goal_input"]}'
        f'_{settings["actor_objective"]}_bc{settings["actor_bc_coef"]:g}'
        f'_{horizon}'
    )


def flow_signature(variant: str) -> str | None:
    """Name the flow computation, or ``None`` when the variant has no flow."""

    settings = VARIANT_SETTINGS[variant]
    if not settings['use_flow']:
        return None
    return f'flow_{actor_signature(variant)}_{settings["flow_target_mode"]}'


def stage_signature(variant: str, stage: str) -> str | None:
    return {
        'critic': critic_signature,
        'actor': actor_signature,
        'flow': flow_signature,
    }[stage](variant)


def _normalize(value):
    if isinstance(value, (list, tuple)):
        return [_normalize(entry) for entry in value]
    if isinstance(value, dict):
        return {str(key): _normalize(entry) for key, entry in sorted(value.items())}
    if isinstance(value, float) and value.is_integer():
        return float(value)
    return value


def fill_derived(resolved_config: dict) -> dict:
    """Add config fields that the agent derives at construction time.

    Pilot runs predate ``flow_target_offsets`` being written into the resolved
    config, so it is recomputed here by the same rule the agent uses.  Without
    this, every pre-v1 checkpoint would look structurally different from an
    identically-configured run today and be needlessly retrained.
    """

    from utils.latent_datasets import sparse_prefix_offsets

    resolved = dict(resolved_config)
    if resolved.get('flow_target_offsets') is None:
        defaults = _defaults()
        action_horizon = int(resolved.get('action_horizon', defaults['action_horizon']))
        mode = str(resolved.get('flow_target_mode', defaults['flow_target_mode']))
        if mode == 'sparse':
            horizon = int(resolved.get('horizon', defaults['horizon']))
            resolved['flow_target_offsets'] = sparse_prefix_offsets(horizon, action_horizon)
        else:
            resolved['flow_target_offsets'] = tuple(range(1, action_horizon + 1))
    return resolved


def config_hash(resolved_config: dict) -> str:
    """Hash the structural part of a resolved agent config."""

    defaults = _defaults()
    resolved_config = fill_derived(resolved_config)
    structural = {
        key: _normalize(resolved_config.get(key, defaults.get(key)))
        for key in _STRUCTURAL_KEYS
    }
    payload = json.dumps(structural, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:16]


def stage_fingerprint(
    resolved_config: dict,
    variant: str,
    stage: str,
    seed: int,
    batch_size: int = 1024,
    parent_step: int = 0,
) -> dict:
    """The seven-field identity a resumable checkpoint must match."""

    settings = VARIANT_SETTINGS[variant]
    defaults = _defaults()
    resolved_config = fill_derived(resolved_config)

    def _entry(key):
        return resolved_config.get(key, defaults.get(key))

    return {
        'config_hash': config_hash(resolved_config),
        'objective_signature': stage_signature(variant, stage),
        'repr_dim': int(_entry('repr_dim')),
        'action_nce': float(settings['action_nce_coef']),
        'actor_goal_offsets': [
            int(offset) for offset in _entry('actor_goal_offsets') or ()
        ],
        'flow_target_offsets': [
            int(offset) for offset in _entry('flow_target_offsets') or ()
        ],
        'seed': int(seed),
        'batch_size': int(batch_size),
        'parent_step': int(parent_step),
        # Recorded for the log, not compared: the stage name is already in the
        # objective signature.
        'stage': stage,
    }


def fingerprint_matches(left: dict, right: dict) -> tuple[bool, list[str]]:
    """Compare two fingerprints, returning the fields that disagree."""

    mismatched = [
        field_name
        for field_name in FINGERPRINT_FIELDS
        if _normalize(left.get(field_name)) != _normalize(right.get(field_name))
    ]
    return not mismatched, mismatched


def read_fingerprint(stage_dir: Path) -> dict | None:
    """Load a stage fingerprint, reconstructing it from a legacy run if needed."""

    explicit = stage_dir / 'fingerprint.json'
    if explicit.is_file():
        with explicit.open('r', encoding='utf-8') as file:
            return json.load(file)

    # Pilot runs predate fingerprints but recorded the resolved config, which
    # is everything the fingerprint is derived from.
    legacy = stage_dir / 'flags.json'
    if not legacy.is_file():
        return None
    with legacy.open('r', encoding='utf-8') as file:
        payload = json.load(file)
    resolved = payload.get('resolved_agent')
    flags = payload.get('flags', {})
    if not resolved:
        return None
    variant = str(resolved.get('variant', ''))
    stage = str(flags.get('stage', ''))
    if variant not in VARIANT_SETTINGS or stage not in ('critic', 'actor', 'flow'):
        return None
    restore_step = int(flags.get('restore_step') or 0)
    if not restore_step:
        restore_step = _step_from_path(str(flags.get('restore_path') or ''))
    return stage_fingerprint(
        resolved,
        variant,
        stage,
        int(flags.get('seed', -1)),
        int(flags.get('batch_size', 1024)),
        restore_step,
    )


def _step_from_path(path: str) -> int:
    name = Path(path).name if path else ''
    try:
        return int(name.removeprefix('params_').removesuffix('.pkl'))
    except ValueError:
        return 0


def write_fingerprint(stage_dir: Path, fingerprint: dict) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    with (stage_dir / 'fingerprint.json').open('w', encoding='utf-8') as file:
        json.dump(fingerprint, file, indent=2, sort_keys=True)
        file.write('\n')


def latest_checkpoint(stage_dir: Path, max_step: int | None = None) -> tuple[Path | None, int]:
    """Highest ``params_<step>.pkl`` in ``stage_dir/checkpoints``."""

    checkpoint_dir = stage_dir / 'checkpoints'
    if not checkpoint_dir.is_dir():
        return None, 0
    best: tuple[Path | None, int] = (None, 0)
    for path in checkpoint_dir.glob('params_*.pkl'):
        try:
            step = int(path.stem.split('_')[1])
        except (IndexError, ValueError):
            continue
        if max_step is not None and step > max_step:
            continue
        if step > best[1]:
            best = (path, step)
    return best


def file_digest(path: Path) -> str:
    """SHA-256 of a checkpoint, used to prove two variants really shared one."""

    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()[:16]


@dataclass
class StageJob:
    """One trainable stage, possibly shared by several variants."""

    env_key: str
    config_path: str
    stage: str
    signature: str
    seed: int
    variant: str  # The variant whose config is used to run the stage.
    stage_dir: Path
    target_step: int
    fingerprint: dict
    parent: 'StageJob | None' = None
    action: str = 'TRAIN'
    start_step: int = 0
    source: str = ''
    shared_by: list[str] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, str, str, int]:
        return (self.env_key, self.stage, self.signature, self.seed)

    @property
    def final_checkpoint(self) -> Path:
        return self.stage_dir / 'checkpoints' / f'params_{self.target_step}.pkl'


__all__ = [
    'FINGERPRINT_FIELDS',
    'StageJob',
    'actor_signature',
    'config_hash',
    'critic_signature',
    'file_digest',
    'fill_derived',
    'fingerprint_matches',
    'flow_signature',
    'latest_checkpoint',
    'read_fingerprint',
    'stage_fingerprint',
    'stage_plan',
    'stage_signature',
    'write_fingerprint',
]
