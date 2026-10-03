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
    ('actor_bc_mse', 'diagnostics/actor/bc_mse'),
    ('flow_prefix_mse', 'diagnostics/flow/prefix_latent_mse'),
)
_SUCCESS_COLUMNS: tuple[tuple[str, str], ...] = (
    ('direct_goal', 'eval_direct_goal.json'),
    ('latent_flow', 'eval_latent_flow.json'),
)


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


def collect(root: Path) -> list[dict[str, Any]]:
    """Group every per-seed result directory by (environment, variant)."""

    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seeds: dict[tuple[str, str], set[int]] = defaultdict(set)

    for results_dir in sorted(root.rglob('results')):
        if not results_dir.is_dir():
            continue
        payloads = {
            name: _read_json(results_dir / name)
            for name in ('diagnostics.json', *(file for _, file in _SUCCESS_COLUMNS))
        }
        identity = next(
            (payload for payload in payloads.values() if payload is not None),
            None,
        )
        if identity is None:
            continue
        key = (str(identity.get('env_name', 'unknown')), str(identity.get('variant', 'unknown')))
        seeds[key].add(int(identity.get('seed', -1)))

        for column, filename in _SUCCESS_COLUMNS:
            payload = payloads.get(filename)
            if payload is None:
                continue
            value = payload.get('evaluation/overall_success')
            if value is not None:
                grouped[key][column].append(float(value))

        diagnostics = payloads.get('diagnostics.json')
        if diagnostics is not None:
            for column, json_key in _DIAGNOSTIC_COLUMNS:
                if json_key in diagnostics:
                    grouped[key][column].append(float(diagnostics[json_key]))

    rows: list[dict[str, Any]] = []
    for (env_name, variant), metrics in sorted(grouped.items()):
        row: dict[str, Any] = {
            'env_name': env_name,
            'variant': variant,
            'num_seeds': len(seeds[(env_name, variant)]),
        }
        for column, _ in _SUCCESS_COLUMNS:
            mean, std = _mean_std(metrics.get(column, []))
            row[f'{column}_success_mean'] = mean
            row[f'{column}_success_std'] = std
        for column, _ in _DIAGNOSTIC_COLUMNS:
            mean, std = _mean_std(metrics.get(column, []))
            row[f'{column}_mean'] = mean
            row[f'{column}_std'] = std
        rows.append(row)
    return rows


def _format(value: Any, precision: int) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return '-'
        return f'{value:.{precision}f}'
    return str(value)


def _render_table(rows: list[dict[str, Any]], precision: int) -> list[str]:
    headers = [
        'env_name',
        'variant',
        'num_seeds',
        'direct_goal_success_mean',
        'direct_goal_success_std',
        'latent_flow_success_mean',
        'latent_flow_success_std',
        *[f'{column}_mean' for column, _ in _DIAGNOSTIC_COLUMNS],
    ]
    table = [headers]
    table.extend(
        [_format(row.get(header), precision) for header in headers] for row in rows
    )
    widths = [max(len(row[index]) for row in table) for index in range(len(headers))]
    return [
        '  '.join(cell.ljust(widths[index]) for index, cell in enumerate(row)).rstrip()
        for row in table
    ]


def write_outputs(rows: list[dict[str, Any]], output_dir: Path, precision: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ['env_name', 'variant', 'num_seeds']

    csv_path = output_dir / 'summary.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    headers = [
        'env_name',
        'variant',
        'num_seeds',
        'direct_goal_success_mean',
        'direct_goal_success_std',
        'latent_flow_success_mean',
        'latent_flow_success_std',
        *[f'{column}_mean' for column, _ in _DIAGNOSTIC_COLUMNS],
    ]
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

    rows = collect(root)
    if not rows:
        raise SystemExit(f'No LatentBridger result files found under {root}.')

    print('\n'.join(_render_table(rows, args.precision)))
    write_outputs(rows, Path(args.output_dir or root), args.precision)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
