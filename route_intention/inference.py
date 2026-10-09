"""Candidate-budget and code-shuffle helpers shared by the control variants.

No InFOM occupancy, generative Q, or contrastive ratio is used here.
"""

from __future__ import annotations

import numpy as np

from route_intention.common import CANDIDATE_BUDGET, NUM_CODES, TOP_L


def top_l(k: int = NUM_CODES, l: int | None = None) -> int:
    return min(int(l if l is not None else TOP_L), int(k))


def per_code_candidates(budget: int = CANDIDATE_BUDGET, k: int = NUM_CODES, l: int | None = None) -> int:
    """How many subgoals each selected route code may propose.

    ``L * per_code == budget``. There is no extra candidate budget versus PB.
    """
    use = top_l(k, l)
    if int(budget) % use != 0:
        raise ValueError(f'budget {budget} is not divisible by L={use}')
    return int(budget) // use


def code_layout(selected: np.ndarray, per_code: int) -> np.ndarray:
    """Repeat each selected code ``per_code`` times. ``selected`` is ``[L]``."""
    selected = np.asarray(selected, dtype=np.int32)
    return np.repeat(selected, int(per_code))


def shuffle_codes(codes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A code that is never equal to the correct code (derangement when K>1)."""
    codes = np.asarray(codes, dtype=np.int32)
    k = int(codes.max()) + 1 if len(codes) else NUM_CODES
    if k < 2:
        raise ValueError('Shuffling a route code requires at least 2 codes.')
    out = codes.copy()
    # Shift by a non-zero offset. This is a derangement and does not depend on matching z.
    shift = int(rng.integers(1, k))
    return (out + shift) % k
