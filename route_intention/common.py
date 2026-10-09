"""Constants, paths, and engineering gates for Route-Level Intention PathBridger.

Thresholds below are fixed before any route-intention result is computed.
They decide whether to spend a 1M control run. They are not paper claims.
"""

from __future__ import annotations

import os
from pathlib import Path

from intention_pb.common import (
    COLLAPSE_MAX_USAGE,
    COLLAPSE_MIN_PERPLEXITY,
    EXP_ROOT as LOCAL_EXP_ROOT,
    REPO,
    SEEDS,
    TASK_ORDER,
    atomic_write_json,
    atomic_write_pickle,
    load_pickle,
    read_json,
    refuse_overwrite,
    require_gpu,
    task_info,
    trim_jsonl_to_step,
)

EXP_ROOT = Path(os.environ.get('ROUTE_EXP_ROOT', REPO / 'exp' / 'route_intention')).resolve()

# Local intention chunk was 5 steps. PB planning horizons in the seed-0 flags are
# 40 (cube-*) and 25 (puzzle-4x4). H_route=25 would stop before the cube subgoal
# s_{t+40}, so the common default is the longer candidate, 50, which is still
# far inside the 1001-step episodes and does not cross episode boundaries.
H_ROUTE = 50
H_LOCAL_INTENTION = 5
NUM_CODES = 8
EMBED_DIM = 64
TOKENIZER_STEPS = 200_000
TOKENIZER_SAVE_STEPS = (50_000, 100_000, 200_000)
ORACLE_STEPS = 250_000
ORACLE_SAVE_STEPS = (50_000, 250_000)
PREDICTOR_STEPS = 150_000
CANDIDATE_BUDGET = 16
TOP_L = 4  # min(4, K) at K=8

KNN_DEFAULT = 64
KNN_SENSITIVITY = (32, 64, 128)

# Engineering gates (do not retune after control results).
VARIANCE_RATIO_GATE = 0.70
VARIANCE_CLEAR_CEILING = 0.85
VARIANCE_CLEAR_MARGIN = 0.10
BRIDGE_ERROR_RATIO_GATE = 0.80

# Seed-0 local-intention subgoal variance ratios (k=64), from the finished diagnostic.
LOCAL_SUBGOAL_VARIANCE_RATIO = {
    'cube-double': 0.954,
    'puzzle-4x4': 0.984,
    'cube-single': 0.910,
}

HARD_TASKS = ('cube-double', 'puzzle-4x4')


def tokenizer_dir(task: str, seed: int) -> Path:
    return EXP_ROOT / 'tokenizer' / f'{task}_seed{seed}'


def diagnostic_dir(task: str, seed: int) -> Path:
    return EXP_ROOT / 'diagnostics' / f'{task}_seed{seed}'


def oracle_dir(task: str, seed: int) -> Path:
    return EXP_ROOT / 'oracle_subgoal' / f'{task}_seed{seed}'


def predictor_dir(task: str, seed: int) -> Path:
    return EXP_ROOT / 'route_predictor' / f'{task}_seed{seed}'


def conditioned_dir(task: str, seed: int) -> Path:
    return EXP_ROOT / 'conditioned' / f'{task}_seed{seed}'


def aggregate_dir() -> Path:
    return EXP_ROOT / 'aggregate'


__all__ = [
    'COLLAPSE_MAX_USAGE',
    'COLLAPSE_MIN_PERPLEXITY',
    'EMBED_DIM',
    'EXP_ROOT',
    'H_ROUTE',
    'HARD_TASKS',
    'KNN_DEFAULT',
    'KNN_SENSITIVITY',
    'LOCAL_EXP_ROOT',
    'LOCAL_SUBGOAL_VARIANCE_RATIO',
    'NUM_CODES',
    'ORACLE_STEPS',
    'REPO',
    'SEEDS',
    'TASK_ORDER',
    'VARIANCE_RATIO_GATE',
    'atomic_write_json',
    'atomic_write_pickle',
    'diagnostic_dir',
    'load_pickle',
    'oracle_dir',
    'read_json',
    'refuse_overwrite',
    'require_gpu',
    'task_info',
    'tokenizer_dir',
    'trim_jsonl_to_step',
]
