"""Aggregate task-1 goal-abstraction curves with the reused references.

New runs live under ``exp/goal_abstraction_task1``.  Raw SGCRL, the oracle
cube run, and the two bridges are read from the directories that already
hold them.  Nothing here writes into those directories.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'exp' / 'goal_abstraction_task1'
MILESTONES = (1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000, 6_000_000, 7_000_000, 8_000_000)

ROWS = (
    ('SGCRL raw', 'exp/online_pilot/seed{seed}/online_sgcrl', (0, 1, 2)),
    ('psi-goal', 'exp/goal_abstraction_task1/sgcrl_psi_goal/seed{seed}', (0, 1, 2)),
    ('state-dependent latent', 'exp/goal_abstraction_task1/sgcrl_state_goal/seed{seed}', (0, 1, 2)),
    ('state-dependent mask', 'exp/goal_abstraction_task1/sgcrl_state_mask/seed{seed}', (0, 1, 2)),
    ('oracle xyz', 'exp/cube_oracle/seed{seed}/online_sgcrl', (0,)),
    ('deterministic bridge', 'exp/online_pilot/seed{seed}/online_sgcrl_det_bridge', (0, 1, 2)),
    ('RF bridge', 'exp/online_pilot/seed{seed}/online_sgcrl_rf_bridge', (0, 1, 2)),
)


def successes(directory: Path) -> dict[int, int | None]:
    found = {}
    for step in MILESTONES:
        path = directory / f'eval_{step}.json'
        if not path.exists():
            found[step] = None
            continue
        payload = json.loads(path.read_text())
        found[step] = int(payload['evaluation/num_successes'])
    return found


def normalized_area(values: list[int | None]) -> float | None:
    """Trapezoid over 1M..8M, divided by the span, in successes out of 100."""

    if any(value is None for value in values):
        return None
    xs = np.asarray(MILESTONES, dtype=np.float64)
    ys = np.asarray(values, dtype=np.float64)
    return float(np.trapz(ys, xs) / (xs[-1] - xs[0]))


def collect() -> list[dict]:
    table = []
    for label, pattern, seeds in ROWS:
        per_seed = []
        for seed in seeds:
            directory = REPO / pattern.format(seed=seed)
            counts = successes(directory)
            per_seed.append({'seed': seed, 'counts': counts, 'auc': normalized_area([counts[step] for step in MILESTONES])})
        table.append({'label': label, 'seeds': per_seed})
    return table


def write_csv(table: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'summary.csv'
    header = ['variant', 'seed'] + [f'{step // 1_000_000}M' for step in MILESTONES] + ['auc']
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in table:
            for item in row['seeds']:
                writer.writerow(
                    [row['label'], item['seed']]
                    + ['' if item['counts'][step] is None else item['counts'][step] for step in MILESTONES]
                    + ['' if item['auc'] is None else f'{item["auc"]:.2f}']
                )
            means = []
            for step in MILESTONES:
                present = [item['counts'][step] for item in row['seeds'] if item['counts'][step] is not None]
                if not present:
                    means.append('')
                    continue
                means.append(f'{np.mean(present):.2f} ± {np.std(present):.2f}')
            areas = [item['auc'] for item in row['seeds'] if item['auc'] is not None]
            area = '' if not areas else f'{np.mean(areas):.2f} ± {np.std(areas):.2f}'
            writer.writerow([row['label'], 'mean'] + means + [area])


def write_markdown(table: list[dict]) -> None:
    lines = [
        '# Goal abstraction, cube-single task 1',
        '',
        'Successes out of 100. Mean ± population std across the seeds that have a number. AUC is the trapezoidal area from 1M to 8M divided by 7M, so it stays on the same 0–100 scale. The best checkpoint is not the headline.',
        '',
        'SGCRL raw, oracle xyz, and both bridges are the existing runs, not retrained here. Oracle xyz is seed 0 only.',
        '',
        '| variant | 1M | 2M | 3M | 4M | 5M | 6M | 7M | 8M | AUC |',
        '|---|---|---|---|---|---|---|---|---|---|',
    ]
    for row in table:
        cells = []
        for step in MILESTONES:
            present = [item['counts'][step] for item in row['seeds'] if item['counts'][step] is not None]
            raw = ', '.join(str(item['counts'][step]) if item['counts'][step] is not None else '-' for item in row['seeds'])
            if not present:
                cells.append(f'- ({raw})')
            else:
                cells.append(f'{np.mean(present):.1f} ± {np.std(present):.1f} ({raw})')
        areas = [item['auc'] for item in row['seeds'] if item['auc'] is not None]
        area = '-' if not areas else f'{np.mean(areas):.1f} ± {np.std(areas):.1f}'
        lines.append('| ' + ' | '.join([row['label'], *cells, area]) + ' |')
    lines.append('')
    (OUT / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')


def plot_success(table: list[dict]) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    xs = [step / 1_000_000 for step in MILESTONES]
    for row in table:
        ys = []
        for step in MILESTONES:
            present = [item['counts'][step] for item in row['seeds'] if item['counts'][step] is not None]
            ys.append(np.mean(present) if present else np.nan)
        axis.plot(xs, ys, marker='o', label=row['label'])
    axis.set_xlabel('environment steps (millions)')
    axis.set_ylabel('successes / 100')
    axis.set_title('cube-single task 1')
    axis.legend(fontsize=8)
    figure.tight_layout()
    plots = OUT / 'plots'
    plots.mkdir(parents=True, exist_ok=True)
    figure.savefig(plots / 'success_vs_env_steps.png', dpi=140)
    plt.close(figure)


def plot_diagnostics() -> None:
    plots = OUT / 'plots'
    plots.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    drew = False
    for variant, label in (
        ('sgcrl_psi_goal', 'psi-goal'),
        ('sgcrl_state_goal', 'state latent'),
        ('sgcrl_state_mask', 'state mask'),
    ):
        xs, sensitivity, residual = [], [], []
        for step in MILESTONES:
            values_s, values_r = [], []
            for seed in (0, 1, 2):
                path = OUT / variant / f'seed{seed}' / f'eval_{step}.json'
                if not path.exists():
                    continue
                payload = json.loads(path.read_text())
                if 'diagnostics/state_sensitivity' in payload:
                    values_s.append(payload['diagnostics/state_sensitivity'])
                if 'diagnostics/residual_norm' in payload:
                    values_r.append(payload['diagnostics/residual_norm'])
            if values_s:
                xs.append(step / 1_000_000)
                sensitivity.append(float(np.mean(values_s)))
                residual.append(float(np.mean(values_r)) if values_r else np.nan)
        if xs:
            axis.plot(xs, sensitivity, marker='o', label=f'{label} state sensitivity')
            drew = True
        if xs and variant == 'sgcrl_state_goal':
            axis.plot(xs, residual, marker='s', label='state latent residual norm')
    axis.set_xlabel('environment steps (millions)')
    axis.set_ylabel('representation distance')
    axis.set_title('goal representation diagnostics')
    if drew:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plots / 'goal_representation_diagnostics.png', dpi=140)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    drew_mask = False
    for seed in (0, 1, 2):
        latest = None
        for step in reversed(MILESTONES):
            path = OUT / 'sgcrl_state_mask' / f'seed{seed}' / f'eval_{step}.json'
            if path.exists():
                latest = json.loads(path.read_text())
                break
        profile = None if latest is None else latest.get('mask_profile')
        if not profile:
            continue
        axis.plot(profile['mean'], label=f'seed {seed}')
        drew_mask = True
    axis.set_xlabel('observation dimension')
    axis.set_ylabel('mean mask')
    axis.set_title('state mask by dimension')
    if drew_mask:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(plots / 'state_mask_by_dimension.png', dpi=140)
    plt.close(figure)


def main() -> None:
    table = collect()
    write_csv(table)
    write_markdown(table)
    plot_success(table)
    plot_diagnostics()
    print(f'wrote {OUT / "summary.md"}')


if __name__ == '__main__':
    main()
