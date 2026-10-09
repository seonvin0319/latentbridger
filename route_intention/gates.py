"""Engineering gates. Thresholds live in ``common.py`` and are not fit to outcomes."""

from __future__ import annotations

from route_intention.common import (
    BRIDGE_ERROR_RATIO_GATE,
    COLLAPSE_MAX_USAGE,
    COLLAPSE_MIN_PERPLEXITY,
    HARD_TASKS,
    LOCAL_SUBGOAL_VARIANCE_RATIO,
    VARIANCE_CLEAR_CEILING,
    VARIANCE_CLEAR_MARGIN,
    VARIANCE_RATIO_GATE,
)


def tokenizer_collapsed(perplexity: float, max_usage: float) -> bool:
    return float(max_usage) > COLLAPSE_MAX_USAGE or float(perplexity) < COLLAPSE_MIN_PERPLEXITY


def variance_component_passes(ratio: float, task: str) -> bool:
    """True when route intention cuts subgoal variance enough to justify a cheap oracle model.

    Primary rule: ``ratio < 0.70``. Alternate rule, still fixed in advance: the ratio is
    below 0.85 and at least 0.10 below that task's seed-0 local-intention ratio.
    """
    ratio = float(ratio)
    if ratio < VARIANCE_RATIO_GATE:
        return True
    local = LOCAL_SUBGOAL_VARIANCE_RATIO.get(task)
    if local is None:
        return False
    return ratio < VARIANCE_CLEAR_CEILING and ratio <= float(local) - VARIANCE_CLEAR_MARGIN


def oracle_component_passes(route_error: float, baseline_error: float) -> bool:
    """Held-out subgoal error must be strictly lower with the oracle route code."""
    return float(route_error) < float(baseline_error)


def bridge_component_passes(correct_error: float, shuffled_error: float) -> bool:
    if float(shuffled_error) <= 0:
        return False
    return float(correct_error) / float(shuffled_error) < BRIDGE_ERROR_RATIO_GATE


def representation_gate(*, perplexity: float, max_usage: float, variance_ratio: float, task: str,
                       route_subgoal_error: float | None = None, baseline_subgoal_error: float | None = None) -> str:
    """``PASS`` only when tokenizer, variance, and oracle-subgoal checks all pass.

    ``VARIANCE_FAIL`` means do not train the oracle model.
    ``ORACLE_FAIL`` means variance was interesting but the oracle predictor did not help.
    """
    if tokenizer_collapsed(perplexity, max_usage):
        return 'FAIL'
    if not variance_component_passes(variance_ratio, task):
        return 'VARIANCE_FAIL'
    if route_subgoal_error is None or baseline_subgoal_error is None:
        return 'NEED_ORACLE'
    if not oracle_component_passes(route_subgoal_error, baseline_subgoal_error):
        return 'ORACLE_FAIL'
    return 'PASS'


def local_intention_early_stop(cells: dict[tuple[str, int], dict[str, float]]) -> dict[str, bool]:
    """Early-stop rule for the unfinished local-intention sweep.

    ``cells[(task, h)]`` maps ``pb`` / ``shared`` / ``shuffled`` to success rates in ``[0, 1]``.
    Condition A: for some ``h``, both hard tasks have Shared-I at least 5 points below PB.
    Condition B: for some ``h``, both hard tasks have |Shared-I - Shuffled-I| < 5 points.
    """
    def _has(h: int) -> bool:
        return all((t, h) in cells for t in HARD_TASKS)

    cond_a = False
    cond_b = False
    for h in (1, 5):
        if not _has(h):
            continue
        cond_a = cond_a or all(cells[(t, h)]['shared'] <= cells[(t, h)]['pb'] - 0.05 for t in HARD_TASKS)
        cond_b = cond_b or all(abs(cells[(t, h)]['shared'] - cells[(t, h)]['shuffled']) < 0.05 for t in HARD_TASKS)
    return dict(condition_a=cond_a, condition_b=cond_b, stop=cond_a or cond_b)
