"""Shared constants, paths and small I/O helpers for the intention experiment."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EXP_ROOT = Path(os.environ.get('IPB_EXP_ROOT', REPO / 'exp' / 'intention_pathbridger')).resolve()
SEED0_PB_ROOT = REPO / 'checkpoints' / '1m_env_best'

NUM_CODES = 8
INTENT_DIM = 32
INTENT_HORIZON = 5
NUM_CANDIDATES = 16
TOKENIZER_STEPS = 200_000
TOKENIZER_SAVE_STEPS = (50_000, 100_000, 200_000)
COND_STEPS = 1_000_000
COND_SAVE_STEPS = (100_000, 250_000, 500_000, 750_000, 1_000_000)
INTERMEDIATE_EVAL_STEPS = (250_000, 500_000, 750_000)
FINAL_STEP = 1_000_000
FINAL_EPISODES_PER_TASK = 20
INTERMEDIATE_EPISODES_PER_TASK = 4
EVAL_TASK_IDS = (1, 2, 3, 4, 5)
SEEDS = (0, 1, 2)
COLLAPSE_MAX_USAGE = 0.85
COLLAPSE_MIN_PERPLEXITY = 2.0

# Priority order (multimodal tasks first).
TASK_ORDER = ('cube-double', 'puzzle-4x4', 'cube-single')
TASKS: dict[str, dict[str, Any]] = {
    'cube-double': dict(env_name='cube-double-play-v0', seed0_label='cd_cube-double', temperature=0.0,
                        ref_num_candidates=1, ref_temperature=0.0),
    'puzzle-4x4': dict(env_name='puzzle-4x4-play-v0', seed0_label='p4_puzzle-4x4', temperature=0.5,
                       ref_num_candidates=32, ref_temperature=0.5),
    'cube-single': dict(env_name='cube-single-play-v0', seed0_label='cs_cube-single', temperature=0.0,
                        ref_num_candidates=1, ref_temperature=0.0),
}

METHODS = ('PB', 'I-SG', 'Shared-I', 'Shuffled-I')
REF_METHOD = 'PB-ref'
INTENTION_METHODS = ('I-SG', 'Shared-I', 'Shuffled-I')
H_EXECS = (1, 5)


def task_info(task: str) -> dict[str, Any]:
    if task not in TASKS:
        raise KeyError(f'Unknown task {task!r}; expected one of {sorted(TASKS)}')
    return TASKS[task]


def pb_train_root(task: str, seed: int) -> Path:
    return EXP_ROOT / 'pb_runs' / f'{task}_seed{seed}'


def find_pb_run_dir(task: str, seed: int) -> Path:
    """Return the PB run dir for task/seed. Raises if it does not exist (no fallback)."""
    if int(seed) == 0:
        run_dir = SEED0_PB_ROOT / task_info(task)['seed0_label']
        if not (run_dir / 'flags.json').is_file():
            raise FileNotFoundError(f'Seed-0 PB run missing: {run_dir}')
        return run_dir
    root = pb_train_root(task, seed)
    if not root.is_dir():
        raise FileNotFoundError(f'PB run root missing for {task} seed{seed}: {root}')
    cands = sorted(p for p in root.iterdir() if p.is_dir() and (p / 'flags.json').is_file())
    if len(cands) != 1:
        raise FileNotFoundError(f'Expected exactly one PB run under {root}, found {[c.name for c in cands]}')
    return cands[0]


def pb_checkpoint_paths(run_dir: Path, step: int = FINAL_STEP) -> dict[str, Path]:
    base = Path(run_dir) / 'checkpoints'
    return {name: base / name / f'params_{int(step)}.pkl' for name in ('dynamics', 'critic')}


def pb_run_complete(task: str, seed: int) -> bool:
    try:
        run_dir = find_pb_run_dir(task, seed)
    except FileNotFoundError:
        return False
    return all(p.is_file() and p.stat().st_size > 0 for p in pb_checkpoint_paths(run_dir).values())


def tokenizer_dir(task: str, seed: int) -> Path:
    return EXP_ROOT / 'tokenizer' / f'{task}_seed{seed}'


def cond_dir(task: str, seed: int) -> Path:
    return EXP_ROOT / 'conditioned' / f'{task}_seed{seed}'


def diag_dir(task: str, seed: int) -> Path:
    return EXP_ROOT / 'diagnostics' / f'{task}_seed{seed}'


def eval_dir(task: str, seed: int, step: int, method: str, h_exec: int) -> Path:
    return EXP_ROOT / 'eval' / f'{task}_seed{seed}' / f'step{int(step)}' / f'{method}_h{int(h_exec)}'


def tokenizer_ckpt(task: str, seed: int, step: int) -> Path:
    return tokenizer_dir(task, seed) / 'checkpoints' / f'tokenizer_{int(step)}.pkl'


def cond_ckpt(task: str, seed: int, step: int) -> Path:
    return cond_dir(task, seed) / 'checkpoints' / f'conditioned_{int(step)}.pkl'


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def trim_jsonl_to_step(path: Path, max_step: int) -> None:
    """Drop log records past the checkpoint a run resumes from (no duplicated steps)."""
    path = Path(path)
    if not path.is_file():
        return
    keep = [l for l in path.read_text().splitlines() if l.strip() and int(json.loads(l)['step']) <= max_step]
    atomic_write_bytes(path, ''.join(l + '\n' for l in keep).encode())


def require_gpu() -> None:
    """GPU training jobs must not silently fall back to CPU."""
    import jax

    if os.environ.get('IPB_ALLOW_CPU') != '1' and jax.default_backend() != 'gpu':
        raise RuntimeError(f'JAX backend is {jax.default_backend()!r}, expected gpu (set IPB_ALLOW_CPU=1 to override).')


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, (json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + '\n').encode())


def atomic_write_pickle(path: Path, obj: Any) -> None:
    atomic_write_bytes(path, pickle.dumps(obj))


def read_json(path: Path) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_pickle(path: Path) -> Any:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint missing: {path}')
    with open(path, 'rb') as f:
        return pickle.load(f)


def _json_default(o: Any):
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f'Not JSON serialisable: {type(o)}')


def refuse_overwrite(marker: Path) -> None:
    if Path(marker).exists():
        raise FileExistsError(f'Refusing to overwrite completed output: {marker}')
