#!/usr/bin/env python3
"""Aggregate a LatentBridger experiment tree into one comparison table.

Reads every ``results/*.json`` written by
``scripts/run_latentbridger_suite.py``, groups by (environment, variant),
aggregates across seeds, and writes ``summary.csv`` and ``summary.md`` next to
the experiment root while printing the table to stdout.

Example::

    python scripts/summarize_latentbridger.py --root exp/latentbridger_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

# Column label -> JSON key in the diagnostics / evaluation payloads.
_DIAGNOSTIC_COLUMNS: tuple[tuple[str, str], ...] = (
    ('recall@1', 'diagnostics/retrieval/recall_at_1'),
    ('recall@5', 'diagnostics/retrieval/recall_at_5'),
    ('act_sens_shuf', 'diagnostics/action_sensitivity/p_data_gt_shuffled'),
    ('act_sens_unif', 'diagnostics/action_sensitivity/p_data_gt_uniform'),
    ('act_sens_hard', 'diagnostics/action_sensitivity/p_data_gt_hard'),
    ('actor_bc_mse', 'diagnostics/actor/bc_mse'),
    ('flow_prefix_mse', 'diagnostics/flow/prefix_latent_mse'),
    ('flow_min_lsnr', 'diagnostics/flow/min_lsnr'),
)

# Per-horizon families, whose horizons depend on the variant's offsets and so
# cannot be listed ahead of time.  Label prefix -> diagnostics key prefix.
_DIAGNOSTIC_FAMILIES: tuple[tuple[str, str], ...] = (
    ('goalsens_a', 'diagnostics/actor_goal_sensitivity/action_delta_'),
    ('D_h', 'diagnostics/flow/D_h'),
    ('E_h', 'diagnostics/flow/E_h'),
    ('lsnr_h', 'diagnostics/flow/lsnr_h'),
)
_DIRECT_GOAL_FILE = 'eval_direct_goal.json'


def _family_columns(diagnostics: dict[str, Any]) -> list[tuple[str, str]]:
    """Expand the per-horizon diagnostic families present in one payload."""

    columns: list[tuple[str, str]] = []
    for label_prefix, key_prefix in _DIAGNOSTIC_FAMILIES:
        for key in diagnostics:
            if not key.startswith(key_prefix):
                continue
            suffix = key[len(key_prefix):]
            if not suffix.isdigit():
                continue
            columns.append((f'{label_prefix}{suffix}', key))
    return columns


def _sorted_family_columns(columns: set[tuple[str, str]]) -> list[tuple[str, str]]:
    """Order family columns by family first, then numerically by horizon."""

    order = {prefix: index for index, (prefix, _) in enumerate(_DIAGNOSTIC_FAMILIES)}

    def sort_key(column: tuple[str, str]) -> tuple[int, int]:
        label, _ = column
        for prefix, index in order.items():
            if label.startswith(prefix):
                return index, int(label[len(prefix):])
        return len(order), 0

    return sorted(columns, key=sort_key)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Experiment root directory.')
    parser.add_argument(
        '--output_dir',
        default='',
        help='Where to write summary.csv/summary.md; defaults to --root.',
    )
    parser.add_argument(
        '--precision',
        type=int,
        default=3,
        help='Decimal places used in the printed and Markdown tables.',
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open('r', encoding='utf-8') as file:
        return json.load(file)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def _success_column(path: Path, payload: dict[str, Any]) -> str:
    """Name the success column an evaluation file belongs to.

    Latent-flow files are split by replan interval, so a suite that evaluated
    several intervals gets one column each instead of silently averaging them.
    """

    if path.name == _DIRECT_GOAL_FILE:
        return 'direct_goal'
    interval = payload.get('replan_interval')
    if interval is None:
        interval = payload.get('evaluation/replan_interval')
    if interval is None:
        return 'latent_flow'
    return f'latent_flow_r{int(interval)}'


def collect(root: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Group every per-seed result directory by (environment, variant)."""

    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seeds: dict[tuple[str, str], set[int]] = defaultdict(set)
    success_columns: set[str] = set()
    family_columns: set[tuple[str, str]] = set()

    for results_dir in sorted(root.rglob('results')):
        if not results_dir.is_dir():
            continue
        evaluation_paths = sorted(results_dir.glob('eval_*.json'))
        diagnostics = _read_json(results_dir / 'diagnostics.json')
        evaluations = [
            (path, payload)
            for path in evaluation_paths
            if (payload := _read_json(path)) is not None
        ]
        identity = diagnostics or (evaluations[0][1] if evaluations else None)
        if identity is None:
            continue
        key = (
            str(identity.get('env_name', 'unknown')),
            str(identity.get('variant', 'unknown')),
        )
        seeds[key].add(int(identity.get('seed', -1)))

        for path, payload in evaluations:
            value = payload.get('evaluation/overall_success')
            if value is None:
                continue
            column = _success_column(path, payload)
            success_columns.add(column)
            grouped[key][column].append(float(value))

        if diagnostics is not None:
            present = _family_columns(diagnostics)
            family_columns.update(present)
            for column, json_key in (*_DIAGNOSTIC_COLUMNS, *present):
                if json_key in diagnostics:
                    grouped[key][column].append(float(diagnostics[json_key]))

    ordered_success = ['direct_goal'] + sorted(success_columns - {'direct_goal'})
    ordered_families = _sorted_family_columns(family_columns)
    rows: list[dict[str, Any]] = []
    for (env_name, variant), metrics in sorted(grouped.items()):
        row: dict[str, Any] = {
            'env_name': env_name,
            'variant': variant,
            'num_seeds': len(seeds[(env_name, variant)]),
        }
        for column in ordered_success:
            mean, std = _mean_std(metrics.get(column, []))
            row[f'{column}_success_mean'] = mean
            row[f'{column}_success_std'] = std
        for column, _ in (*_DIAGNOSTIC_COLUMNS, *ordered_families):
            mean, std = _mean_std(metrics.get(column, []))
            row[f'{column}_mean'] = mean
            row[f'{column}_std'] = std
        rows.append(row)
    return rows, ordered_success, [label for label, _ in ordered_families]


def _format(value: Any, precision: int) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return '-'
        return f'{value:.{precision}f}'
    return str(value)


def _headers(success_columns: list[str], family_columns: list[str]) -> list[str]:
    return [
        'env_name',
        'variant',
        'num_seeds',
        *[
            f'{column}_success_{statistic}'
            for column in success_columns
            for statistic in ('mean', 'std')
        ],
        *[f'{column}_mean' for column, _ in _DIAGNOSTIC_COLUMNS],
        *[f'{column}_mean' for column in family_columns],
    ]


def _render_table(
    rows: list[dict[str, Any]],
    success_columns: list[str],
    family_columns: list[str],
    precision: int,
) -> list[str]:
    headers = _headers(success_columns, family_columns)
    table = [headers]
    table.extend(
        [_format(row.get(header), precision) for header in headers] for row in rows
    )
    widths = [max(len(row[index]) for row in table) for index in range(len(headers))]
    return [
        '  '.join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip()
        for row in table
    ]


def write_outputs(
    rows: list[dict[str, Any]],
    success_columns: list[str],
    family_columns: list[str],
    output_dir: Path,
    precision: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ['env_name', 'variant', 'num_seeds']

    csv_path = output_dir / 'summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    headers = _headers(success_columns, family_columns)
    lines = [
        '# LatentBridger summary',
        '',
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
    ]
    lines.extend(
        '| ' + ' | '.join(_format(row.get(header), precision) for header in headers) + ' |'
        for row in rows
    )
    lines.append('')
    (output_dir / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote {csv_path} and {output_dir / "summary.md"}')


def main() -> int:
    args = _parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f'Experiment root not found: {root}')

    rows, success_columns, family_columns = collect(root)
    if not rows:
        raise SystemExit(f'No LatentBridger result files found under {root}.')

    print('\n'.join(_render_table(rows, success_columns, family_columns, args.precision)))
    write_outputs(
        rows,
        success_columns,
        family_columns,
        Path(args.output_dir or root),
        args.precision,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
