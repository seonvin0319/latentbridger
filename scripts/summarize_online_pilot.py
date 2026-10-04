#!/usr/bin/env python
"""Summarize the cube-single online bridge pilot into tables and plots.

Evaluation points at or below the warm-start step come from the shared
warm-up run, so all three variants report the identical number there by
construction.  Points beyond it come from each branch.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

VARIANTS = (
    'online_sgcrl',
    'online_sgcrl_det_bridge',
    'online_sgcrl_rf_bridge',
)
LABELS = {
    'online_sgcrl': 'SGCRL',
    'online_sgcrl_det_bridge': 'SGCRL + deterministic bridge',
    'online_sgcrl_rf_bridge': 'SGCRL + RF bridge',
}
WARMUP_STEPS = 100_000
TABLE_POINTS = (
    100_000, 200_000, 300_000, 500_000, 800_000, 1_000_000,
    2_000_000, 4_000_000, 6_000_000, 8_000_000,
)
CURVE_POINTS = (
    10_000, 50_000, 100_000, 200_000, 300_000, 500_000, 800_000, 1_000_000,
    2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000, 7_000_000, 8_000_000,
)


def load_point(run_dir: Path, point: int) -> dict | None:
    path = run_dir / f'eval_{point}.json'
    if not path.is_file():
        return None
    with path.open(encoding='utf-8') as file:
        return json.load(file)


def collect(root: Path, seeds: list[int]) -> dict:
    """``records[variant][seed][point] -> evaluation payload``."""

    records: dict[str, dict[int, dict[int, dict]]] = {v: {} for v in VARIANTS}
    for seed in seeds:
        seed_dir = root / f'seed{seed}'
        warmup_dir = seed_dir / 'warmup'
        for variant in VARIANTS:
            per_point: dict[int, dict] = {}
            for point in CURVE_POINTS:
                # Shared history: read the warm-up's own evaluation rather
                # than duplicating it in each branch.
                source = warmup_dir if point <= WARMUP_STEPS else seed_dir / variant
                payload = load_point(source, point)
                if payload is not None:
                    per_point[point] = payload
            if per_point:
                records[variant][seed] = per_point
    return records


def success_matrix(records, variant, point, seeds):
    values, counts = [], []
    for seed in seeds:
        payload = records[variant].get(seed, {}).get(point)
        if payload is None:
            continue
        values.append(float(payload['evaluation/success']))
        counts.append(
            (
                int(payload['evaluation/num_successes']),
                int(payload['evaluation/num_episodes']),
            )
        )
    return np.asarray(values), counts


def diagnostic_series(records, variant, key, seeds):
    points, means, stds = [], [], []
    for point in CURVE_POINTS:
        values = [
            float(records[variant][seed][point][key])
            for seed in seeds
            if point in records[variant].get(seed, {})
            and key in records[variant][seed][point]
        ]
        if values:
            points.append(point)
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values)))
    return np.asarray(points), np.asarray(means), np.asarray(stds)


def write_success_table(records, seeds, output: Path) -> str:
    header = ['variant'] + [f'{p // 1000}k' for p in TABLE_POINTS]
    lines = ['| ' + ' | '.join(header) + ' |']
    lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')
    rows = []
    for variant in VARIANTS:
        cells = [LABELS[variant]]
        for point in TABLE_POINTS:
            values, counts = success_matrix(records, variant, point, seeds)
            if values.size == 0:
                cells.append('-')
                continue
            raw = ', '.join(f'{s}/{n}' for s, n in counts)
            cells.append(
                f'{100 * values.mean():.1f} ± {100 * values.std():.1f} ({raw})'
            )
            rows.append(
                {
                    'variant': variant,
                    'env_steps': point,
                    'success_mean': float(values.mean()),
                    'success_std': float(values.std()),
                    'num_seeds': int(values.size),
                    'raw_counts': raw,
                }
            )
        lines.append('| ' + ' | '.join(cells) + ' |')

    with output.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                'variant',
                'env_steps',
                'success_mean',
                'success_std',
                'num_seeds',
                'raw_counts',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return '\n'.join(lines)


def write_bridge_table(records, seeds) -> str:
    columns = [
        ('online_sgcrl_det_bridge', 'diagnostics/det_waypoint_mse',
         'diagnostics/det_final_goal_score', 'diagnostics/det_waypoint_score'),
        ('online_sgcrl_rf_bridge', 'diagnostics/rf_waypoint_mse',
         'diagnostics/rf_final_goal_score', 'diagnostics/rf_waypoint_score'),
    ]
    header = [
        'variant',
        'bridge error (MSE)',
        'final-goal critic score',
        'waypoint critic score',
    ]
    lines = ['| ' + ' | '.join(header) + ' |']
    lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')

    baseline = [
        float(records['online_sgcrl'][seed][1_000_000][
            'diagnostics/final_goal_score_direct'])
        for seed in seeds
        if 1_000_000 in records['online_sgcrl'].get(seed, {})
        and 'diagnostics/final_goal_score_direct'
        in records['online_sgcrl'][seed][1_000_000]
    ]
    lines.append(
        '| '
        + ' | '.join(
            [
                LABELS['online_sgcrl'],
                '-',
                f'{np.mean(baseline):.3f}' if baseline else '-',
                '-',
            ]
        )
        + ' |'
    )

    for variant, mse_key, goal_key, waypoint_key in columns:
        cells = [LABELS[variant]]
        for key in (mse_key, goal_key, waypoint_key):
            values = [
                float(records[variant][seed][1_000_000][key])
                for seed in seeds
                if 1_000_000 in records[variant].get(seed, {})
                and key in records[variant][seed][1_000_000]
            ]
            cells.append(
                f'{np.mean(values):.4g} ± {np.std(values):.2g}' if values else '-'
            )
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def plot_curve(records, seeds, keys, title, ylabel, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    drew = False
    for variant, key in keys:
        points, means, stds = diagnostic_series(records, variant, key, seeds)
        if points.size == 0:
            continue
        drew = True
        axis.plot(points, means, marker='o', label=LABELS[variant])
        axis.fill_between(points, means - stds, means + stds, alpha=0.18)
    if not drew:
        plt.close(figure)
        return
    axis.axvline(
        WARMUP_STEPS, color='grey', linestyle='--', linewidth=1,
        label='branch point',
    )
    axis.set_xlabel('environment interactions')
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=REPO / 'exp' / 'online_pilot')
    parser.add_argument('--seeds', type=str, default='0,1,2')
    arguments = parser.parse_args()

    root = Path(arguments.root).resolve()
    seeds = [int(piece) for piece in arguments.seeds.split(',') if piece.strip()]
    records = collect(root, seeds)

    output = root / 'summary'
    plots = output / 'plots'
    plots.mkdir(parents=True, exist_ok=True)

    success_table = write_success_table(records, seeds, output / 'summary_success.csv')
    bridge_table = write_bridge_table(records, seeds)

    plot_curve(
        records,
        seeds,
        [(v, 'evaluation/success') for v in VARIANTS],
        'cube-single task 1: success vs environment interactions',
        'evaluation success rate',
        plots / 'success_vs_env_steps.png',
    )
    plot_curve(
        records,
        seeds,
        [('online_sgcrl_det_bridge', 'diagnostics/det_waypoint_mse')],
        'Deterministic bridge: held-out waypoint MSE',
        'MSE to the observed waypoint',
        plots / 'det_bridge_mse.png',
    )
    plot_curve(
        records,
        seeds,
        [('online_sgcrl_rf_bridge', 'diagnostics/rf_waypoint_mse')],
        'Rectified-flow bridge: held-out waypoint MSE',
        'MSE to the observed waypoint',
        plots / 'rf_bridge_mse.png',
    )
    plot_curve(
        records,
        seeds,
        [('online_sgcrl_rf_bridge', 'diagnostics/rf_pairwise_distance')],
        'Rectified-flow bridge: waypoint sample diversity',
        'mean pairwise distance over 8 samples',
        plots / 'rf_diversity.png',
    )
    plot_curve(
        records,
        seeds,
        [
            ('online_sgcrl_det_bridge', 'diagnostics/det_waypoint_score'),
            ('online_sgcrl_rf_bridge', 'diagnostics/rf_waypoint_score'),
        ],
        'Waypoint reachability: C(s, pi(s, w), w)',
        'critic score of the predicted waypoint',
        plots / 'waypoint_reachability.png',
    )

    with (output / 'summary.md').open('w', encoding='utf-8') as file:
        file.write('# cube-single task 1 online bridge pilot\n\n')
        file.write(
            f'Seeds {seeds}; shared SGCRL warm start to {WARMUP_STEPS:,} '
            'environment steps, then three branches to 1,000,000.\n'
            'Points at or below the branch step are the shared warm-up run, '
            'so the three rows are identical there by construction.\n\n'
        )
        file.write('## Success rate (%, mean ± std over seeds)\n\n')
        file.write(success_table + '\n\n')
        file.write('## Bridge quality at 1M environment steps\n\n')
        file.write(bridge_table + '\n')
    print((output / 'summary.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    main()
