#!/usr/bin/env python3
"""Aggregate the long LatentBridger sweep into learning curves and a report.

The sweep's question is whether the 100k ranking was about sample efficiency
or about asymptotic performance, so the primary output is a curve per
(environment, variant, metric) over checkpoint steps, averaged across seeds.

Three files and six plots are written into the sweep root::

    summary_learning_curves.csv   every metric at every evaluated checkpoint
    summary_final.csv             the last checkpoint each variant reached
    summary.md                    a readable table plus the 100k -> final delta
    plots/*.png

Example::

    python scripts/summarize_latent_sweep.py --sweep_dir=exp/sweep24h
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

_EVAL_NAME = re.compile(
    r'^eval_(?P<mode>direct_goal|latent_flow|slerp_bridge)'
    r'(?:_r(?P<interval>\d+))?_(?P<step>\d+)(?:_ep(?P<episodes>\d+))?\.json$'
)
_DIAGNOSTIC_NAME = re.compile(r'^diagnostics_(?P<step>\d+)\.json$')

# (column, json key).  Scalars that summarize one aspect of the method.
DIAGNOSTIC_COLUMNS = (
    ('recall_at_1', 'diagnostics/retrieval/recall_at_1'),
    ('recall_at_5', 'diagnostics/retrieval/recall_at_5'),
    ('positive_rank', 'diagnostics/retrieval/mean_positive_rank'),
    ('positive_negative_gap', 'diagnostics/retrieval/score_gap'),
    ('p_data_gt_shuffled', 'diagnostics/action_sensitivity/p_data_gt_shuffled'),
    ('p_data_gt_uniform', 'diagnostics/action_sensitivity/p_data_gt_uniform'),
    ('p_data_gt_hard', 'diagnostics/action_sensitivity/p_data_gt_hard'),
    ('margin_shuffled', 'diagnostics/action_sensitivity/margin_shuffled'),
    ('margin_uniform', 'diagnostics/action_sensitivity/margin_uniform'),
    ('margin_hard', 'diagnostics/action_sensitivity/margin_hard'),
    ('bc_mse', 'diagnostics/actor/bc_mse'),
    ('action_saturation', 'diagnostics/actor/saturation_fraction'),
    ('actor_advantage', 'diagnostics/actor/critic_score_advantage'),
    ('prefix_latent_mse', 'diagnostics/flow/prefix_latent_mse'),
    ('flow_min_lsnr', 'diagnostics/flow/min_lsnr'),
    ('flow_mean_prefix_cosine', 'diagnostics/flow/mean_prefix_cosine'),
)

# Prefixes whose per-horizon members are discovered from the data, because the
# valid horizons depend on the environment's planning horizon.
DIAGNOSTIC_FAMILIES = (
    ('goalsens_delta', 'diagnostics/actor_goal_sensitivity/action_delta_'),
    ('goalsens_gain', 'diagnostics/actor_goal_sensitivity/critic_gain_'),
    ('latent_D', 'diagnostics/latent_geometry/distance_delta_'),
    ('flow_D', 'diagnostics/flow/D_h'),
    ('flow_E', 'diagnostics/flow/E_h'),
    ('flow_lsnr', 'diagnostics/flow/lsnr_h'),
    ('flow_retrieval', 'diagnostics/flow/retrieval_h'),
)

PLOTS = (
    ('success_vs_updates', 'direct_goal_success', 'Task success'),
    ('recall1_vs_updates', 'recall_at_1', 'Recall@1'),
    ('phard_vs_updates', 'p_data_gt_hard', 'P(data > hard)'),
    ('actor_gain_vs_updates', 'actor_advantage', 'Actor critic advantage'),
    ('lsnr_vs_updates', 'flow_min_lsnr', 'min LSNR over waypoints'),
    ('flow_prefix_error_vs_updates', 'prefix_latent_mse', 'Prefix latent MSE'),
)


def _read_json(path: Path):
    try:
        with path.open('r', encoding='utf-8') as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def _mean_std(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return float('nan'), float('nan')
    return float(np.mean(finite)), float(np.std(finite))


def collect(sweep_root: Path):
    """Gather every metric into ``cell[(env, variant, step)][column] -> [per seed]``."""

    cells: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    counts: dict[tuple, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0])
    )
    seeds: dict[tuple, set[int]] = defaultdict(set)
    family_columns: set[str] = set()
    success_columns: set[str] = set()

    for results_dir in sorted(sweep_root.rglob('results')):
        if not results_dir.is_dir():
            continue
        seed_dir = results_dir.parent
        variant_dir = seed_dir.parent
        env_key = variant_dir.parent.name
        variant = variant_dir.name
        try:
            seed = int(seed_dir.name.removeprefix('seed'))
        except ValueError:
            continue

        for path in sorted(results_dir.glob('*.json')):
            payload = _read_json(path)
            if payload is None:
                continue

            match = _EVAL_NAME.fullmatch(path.name)
            if match is not None:
                # The denser final evaluation is a separate column; averaging
                # it with the 250-episode runs would blur two sample sizes.
                episodes = match.group('episodes')
                column = match.group('mode')
                if match.group('interval'):
                    column += f'_r{match.group("interval")}'
                if episodes:
                    column += f'_ep{episodes}'
                column += '_success'
                step = int(match.group('step'))
                key = (env_key, variant, step)
                value = payload.get('evaluation/overall_success')
                if value is None:
                    continue
                cells[key][column].append(float(value))
                success_columns.add(column)
                seeds[key].add(seed)
                episode_outcomes = payload.get('evaluation/episodes')
                if isinstance(episode_outcomes, list):
                    counts[key][column][0] += int(sum(episode_outcomes))
                    counts[key][column][1] += len(episode_outcomes)
                continue

            match = _DIAGNOSTIC_NAME.fullmatch(path.name)
            if match is None:
                continue
            step = int(match.group('step'))
            key = (env_key, variant, step)
            seeds[key].add(seed)
            for column, json_key in DIAGNOSTIC_COLUMNS:
                if json_key in payload:
                    cells[key][column].append(float(payload[json_key]))
            for label, prefix in DIAGNOSTIC_FAMILIES:
                for json_key, value in payload.items():
                    if not json_key.startswith(prefix):
                        continue
                    suffix = json_key[len(prefix):]
                    if not suffix.isdigit():
                        continue
                    column = f'{label}_{suffix}'
                    cells[key][column].append(float(value))
                    family_columns.add(column)

    return cells, counts, seeds, sorted(success_columns), _sorted_families(family_columns)


def _sorted_families(columns: set[str]) -> list[str]:
    order = {label: index for index, (label, _) in enumerate(DIAGNOSTIC_FAMILIES)}

    def sort_key(column: str):
        label, _, suffix = column.rpartition('_')
        return (order.get(label, 99), label, int(suffix))

    return sorted(columns, key=sort_key)


def build_rows(cells, counts, seeds, success_columns, family_columns):
    columns = (
        list(success_columns)
        + [column for column, _ in DIAGNOSTIC_COLUMNS]
        + list(family_columns)
    )
    rows = []
    for (env_key, variant, step) in sorted(cells):
        row = {
            'env': env_key,
            'variant': variant,
            'step': step,
            'num_seeds': len(seeds[(env_key, variant, step)]),
        }
        for column in columns:
            values = cells[(env_key, variant, step)].get(column, [])
            mean, std = _mean_std(values)
            row[f'{column}_mean'] = mean
            row[f'{column}_std'] = std
            if column in success_columns:
                successes, episodes = counts[(env_key, variant, step)][column]
                row[f'{column}_raw'] = f'{successes}/{episodes}' if episodes else ''
        rows.append(row)
    return rows, columns


def _write_csv(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (f'{value:.6g}' if isinstance(value, float) else value)
                    for key, value in row.items()
                }
            )


def _format(value, precision=4):
    if isinstance(value, float):
        return '-' if math.isnan(value) else f'{value:.{precision}f}'
    return '' if value is None else str(value)


def _markdown_table(rows, headers):
    lines = ['| ' + ' | '.join(headers) + ' |']
    lines.append('| ' + ' | '.join('---' for _ in headers) + ' |')
    for row in rows:
        lines.append('| ' + ' | '.join(_format(row.get(header)) for header in headers) + ' |')
    return '\n'.join(lines)


def write_report(sweep_root: Path, rows, success_columns):
    """A readable summary whose headline is the change from 100k to the end."""

    by_variant: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_variant[(row['env'], row['variant'])].append(row)

    final_rows = []
    delta_rows = []
    for (env_key, variant), variant_rows in sorted(by_variant.items()):
        variant_rows.sort(key=lambda row: row['step'])
        evaluated = [row for row in variant_rows if row.get('direct_goal_success_mean') is not None]
        if not evaluated:
            continue
        last = evaluated[-1]
        final_rows.append(last)
        first = next((row for row in evaluated if row['step'] == 100_000), evaluated[0])
        delta = {
            'env': env_key,
            'variant': variant,
            'first_step': first['step'],
            'final_step': last['step'],
        }
        for column in success_columns + ['recall_at_1', 'p_data_gt_hard', 'flow_min_lsnr']:
            key = f'{column}_mean'
            if key in first and key in last:
                delta[f'{column}_first'] = first[key]
                delta[f'{column}_final'] = last[key]
                delta[f'{column}_delta'] = last[key] - first[key]
        delta_rows.append(delta)

    success_headers = [f'{column}_mean' for column in success_columns]
    success_headers += [f'{column}_raw' for column in success_columns]
    lines = [
        '# LatentBridger long sweep',
        '',
        'Every variant, seed, and checkpoint is evaluated on one paired episode',
        'manifest per (environment, seed), so a success difference between two',
        'rows is a policy difference and not a difference in which episodes each',
        'policy faced.  Success columns are the mean over seeds of the per-seed',
        'success rate; `_raw` is the pooled successes over pooled episodes.',
        '',
        '## Final checkpoint reached',
        '',
        _markdown_table(
            final_rows,
            ['env', 'variant', 'step', 'num_seeds', *success_headers],
        ),
        '',
        '## Change from 100k to the final checkpoint',
        '',
        '`*_first` is the 100k point, `*_final` the last checkpoint evaluated.',
        'A variant whose success is flat while its diagnostics improve is',
        'limited by something other than representation quality.',
        '',
        _markdown_table(
            delta_rows,
            [
                'env',
                'variant',
                'first_step',
                'final_step',
                'direct_goal_success_first',
                'direct_goal_success_final',
                'direct_goal_success_delta',
                'recall_at_1_first',
                'recall_at_1_final',
                'p_data_gt_hard_first',
                'p_data_gt_hard_final',
                'flow_min_lsnr_first',
                'flow_min_lsnr_final',
                'flow_min_lsnr_delta',
            ],
        ),
        '',
        '## Learning curves',
        '',
        'Full per-checkpoint numbers are in `summary_learning_curves.csv`;',
        'plots are in `plots/`.',
        '',
    ]
    (sweep_root / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return final_rows


def write_plots(sweep_root: Path, rows) -> list[Path]:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_dir = sweep_root / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    written = []

    environments = sorted({row['env'] for row in rows})
    for name, column, ylabel in PLOTS:
        key = f'{column}_mean'
        std_key = f'{column}_std'
        panels = [
            env_key
            for env_key in environments
            if any(
                row['env'] == env_key and not math.isnan(row.get(key, float('nan')))
                for row in rows
            )
        ]
        if not panels:
            continue
        figure, axes = plt.subplots(
            1, len(panels), figsize=(5.2 * len(panels), 4.0), squeeze=False
        )
        for axis, env_key in zip(axes[0], panels):
            variants = sorted({row['variant'] for row in rows if row['env'] == env_key})
            for variant in variants:
                series = sorted(
                    (
                        row
                        for row in rows
                        if row['env'] == env_key
                        and row['variant'] == variant
                        and not math.isnan(row.get(key, float('nan')))
                    ),
                    key=lambda row: row['step'],
                )
                if not series:
                    continue
                steps = np.array([row['step'] for row in series]) / 1e6
                means = np.array([row[key] for row in series])
                stds = np.nan_to_num(np.array([row.get(std_key, 0.0) for row in series]))
                axis.plot(steps, means, marker='o', label=variant)
                axis.fill_between(steps, means - stds, means + stds, alpha=0.15)
            axis.set_title(env_key)
            axis.set_xlabel('updates (millions)')
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.3)
            if column == 'flow_min_lsnr':
                # LSNR = 1 is the line where the generated waypoint is no
                # better than standing still.
                axis.axhline(1.0, color='k', linestyle='--', linewidth=1)
            axis.legend(fontsize=8)
        figure.tight_layout()
        path = plot_dir / f'{name}.png'
        figure.savefig(path, dpi=140)
        plt.close(figure)
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep_dir', default='exp/sweep24h')
    parser.add_argument('--no_plots', action='store_true')
    args = parser.parse_args()

    sweep_root = Path(args.sweep_dir).resolve()
    if not sweep_root.is_dir():
        raise SystemExit(f'No such sweep directory: {sweep_root}')

    cells, counts, seeds, success_columns, family_columns = collect(sweep_root)
    if not cells:
        raise SystemExit(f'No results found under {sweep_root}.')
    rows, columns = build_rows(cells, counts, seeds, success_columns, family_columns)

    fieldnames = ['env', 'variant', 'step', 'num_seeds']
    for column in columns:
        fieldnames.append(f'{column}_mean')
        fieldnames.append(f'{column}_std')
        if column in success_columns:
            fieldnames.append(f'{column}_raw')

    _write_csv(sweep_root / 'summary_learning_curves.csv', rows, fieldnames)
    final_rows = write_report(sweep_root, rows, success_columns)
    _write_csv(sweep_root / 'summary_final.csv', final_rows, fieldnames)
    print(f'{len(rows)} (env, variant, checkpoint) rows -> {sweep_root}')

    if not args.no_plots:
        for path in write_plots(sweep_root, rows):
            print(f'plot {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
