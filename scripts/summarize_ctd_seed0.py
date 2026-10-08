"""Build seed-0 CTD result and diagnostic tables from immutable artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'exp' / 'ctd_pathbridger'
CPB = Path('/home/shchoi/latentbridger/exp/contrastive_pathbridger')
ENVS = ('cube_double', 'puzzle_3x3', 'antmaze_medium', 'cube_single')
METHODS = (
    'ctd_weighted',
    'dtrl_weighted',
    'ctd_pathnce_weighted',
    'ctd_uniform',
    'dtrl_uniform',
)
LABELS = {
    'ctd_weighted': 'CTD-W',
    'dtrl_weighted': 'DTRL-W',
    'ctd_pathnce_weighted': 'PathNCE-W',
    'ctd_uniform': 'CTD-U',
    'dtrl_uniform': 'DTRL-U',
}
STEPS = (100_000, 300_000, 500_000, 800_000, 1_000_000)
COMPARE = ('ctd_weighted', 'dtrl_weighted', 'ctd_pathnce_weighted')


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ['env'])
        writer.writeheader()
        writer.writerows(rows)


def pct(value):
    return f'{100 * float(value):.1f}'


def evaluation_rows():
    rows = []
    for env in ENVS:
        for method in METHODS:
            run = OUT / env / method / 'seed0'
            for step in STEPS:
                for horizon in (5, 2, 1):
                    record = load(run / f'evaluation_{step}_h{horizon}.json')
                    if record is None:
                        continue
                    rows.append({
                        'Environment': env,
                        'Method': method,
                        'checkpoint': step,
                        'h': horizon,
                        'success': record['overall_success'],
                        **{f'task{i}': record[f'task_{i}_success'] for i in range(1, 6)},
                        'provenance': str(run / f'evaluation_{step}_h{horizon}.json'),
                    })
    for env in ENVS:
        for variant, label in (
            ('cpb_rank_only', 'CPB rank-only'),
            ('cpb_full', 'CPB full'),
        ):
            for horizon in (5, 2, 1):
                path = CPB / env / variant / 'seed0' / f'evaluation_1000000_h{horizon}.json'
                record = load(path)
                if record is None:
                    continue
                rows.append({
                    'Environment': env,
                    'Method': label,
                    'checkpoint': 1_000_000,
                    'h': horizon,
                    'success': record['overall_success'],
                    **{f'task{i}': record[f'task_{i}_success'] for i in range(1, 6)},
                    'provenance': str(path),
                })
    return rows


def diagnostic_rows():
    all_rows = []
    for env in ENVS:
        for method in METHODS:
            for step in STEPS:
                record = load(OUT / env / method / 'seed0' / f'diagnostics_{step}.json')
                if record is not None:
                    all_rows.append({'env': env, 'method': method, 'step': step, **record})
    return all_rows


def select(rows, prefixes, *, cube_only=False):
    return [
        {
            'env': row['env'],
            'method': row['method'],
            'step': row['step'],
            **{key: value for key, value in row.items() if key.startswith(prefixes)},
        }
        for row in rows
        if not cube_only or row['env'] == 'cube_double'
    ]


def success_cell(env, method, horizon=5, step=1_000_000):
    if method.startswith('CPB'):
        variant = 'cpb_rank_only' if 'rank' in method else 'cpb_full'
        record = load(CPB / env / variant / 'seed0' / f'evaluation_{step}_h{horizon}.json')
    else:
        record = load(OUT / env / method / 'seed0' / f'evaluation_{step}_h{horizon}.json')
    return '-' if record is None else pct(record['overall_success'])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evaluations = evaluation_rows()
    diagnostics = diagnostic_rows()
    write_csv(OUT / 'results_seed0.csv', evaluations)
    write_csv(OUT / 'cube_double_drift.csv', select(
        diagnostics,
        ('proposer/', 'weight/'),
        cube_only=True,
    ))
    write_csv(OUT / 'distance_diagnostics.csv', select(
        diagnostics,
        ('distance/', 'triangle/', 'representation/'),
    ))
    write_csv(OUT / 'nce_diagnostics.csv', select(diagnostics, ('nce/',)))
    write_csv(OUT / 'pathnce_diagnostics.csv', select(diagnostics, ('path/',)))

    lines = [
        '# CTD PathBridger — seed 0',
        '',
        'Primary protocol: 1M updates, 5 tasks × 50 episodes, seed 0 only.',
        'CPB rows are existing files under `exp/contrastive_pathbridger` (not retrained).',
        'BridgeGeo and remaining all-env ablations are deferred.',
        '',
        '## Main comparison: CPB rank / CTD-W / DTRL-W / PathNCE-W',
        '',
        '| Environment | Method | h=5 | h=2 | h=1 |',
        '| --- | --- | ---: | ---: | ---: |',
    ]
    display = (
        ('CPB rank-only', 'CPB rank-only'),
        ('ctd_weighted', 'CTD-W'),
        ('dtrl_weighted', 'DTRL-W'),
        ('ctd_pathnce_weighted', 'PathNCE-W'),
    )
    for env in ENVS:
        for method, label in display:
            cells = [success_cell(env, method, horizon) for horizon in (5, 2, 1)]
            lines.append(f"| {env} | {label} | {' | '.join(cells)} |")

    lines += [
        '',
        '## Puzzle emphasis',
        '',
        '| Method | 100k | 300k | 500k | 800k | 1M |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for method, label in display:
        cells = [success_cell('puzzle_3x3', method, 5, step) for step in STEPS]
        lines.append(f"| {label} | {' | '.join(cells)} |")
    pathnce_puzzle = success_cell('puzzle_3x3', 'ctd_pathnce_weighted')
    lines += [
        '',
        f'PathNCE-W puzzle 1M h=5 = **{pathnce_puzzle}**.',
        'Scheduling rule (exploratory only): >=70 keep PathNCE family; 50-70 puzzle-only PathNCE-U; <50 defer PathNCE-U/BridgeGeo.',
        '',
        '## cube-double checkpoint trajectory (h=5)',
        '',
        '| Method | 100k | 300k | 500k | 800k | 1M |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for method in COMPARE + ('dtrl_uniform', 'ctd_uniform'):
        cells = [success_cell('cube_double', method, 5, step) for step in STEPS]
        lines.append(f"| {LABELS.get(method, method)} | {' | '.join(cells)} |")

    lines += [
        '',
        '## cube-double proposer diagnostics (W/U)',
        '',
        '| Method | Step | success h5 | pairwise | best-of-N endpoint | goal-space best-of-N | nearest-data | ESS |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for method in ('dtrl_weighted', 'dtrl_uniform', 'ctd_weighted', 'ctd_uniform'):
        for step in (300_000, 500_000, 800_000, 1_000_000):
            diag = load(OUT / 'cube_double' / method / 'seed0' / f'diagnostics_{step}.json')
            succ = success_cell('cube_double', method, 5, step)
            if diag is None:
                lines.append(f"| {LABELS[method]} | {step} | {succ} | - | - | - | - | - |")
                continue
            ess = diag.get('weight/ess_fraction')
            ess_s = '-' if ess is None else f'{ess:.3f}'
            lines.append(
                f"| {LABELS[method]} | {step} | {succ} | "
                f"{diag['proposer/pairwise_distance']:.3f} | "
                f"{diag['proposer/min_true_distance']:.3f} | "
                f"{diag['proposer/min_goal_distance']:.3f} | "
                f"{diag['proposer/nearest_reference']:.3f} | {ess_s} |"
            )

    lines += [
        '',
        '## FutureNCE scale diagnostics',
        '',
        '| Env | Method | Step | temporal MAE | d mean | d p90 | d p99 | self mean | NCE@1 |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for env in ('cube_double', 'puzzle_3x3'):
        for method in COMPARE:
            for step in (500_000, 1_000_000):
                diag = load(OUT / env / method / 'seed0' / f'diagnostics_{step}.json')
                if diag is None:
                    lines.append(f"| {env} | {LABELS[method]} | {step} | - | - | - | - | - | - |")
                    continue
                nce = diag.get('nce/recall_at_1')
                nce_s = '-' if nce is None else f'{nce:.3f}'
                def num(key, default=float('nan')):
                    value = diag.get(key, default)
                    return float('nan') if value is None else float(value)

                lines.append(
                    f"| {env} | {LABELS[method]} | {step} | "
                    f"{num('distance/temporal_mae'):.1f} | "
                    f"{num('distance/direct_mean'):.1f} | "
                    f"{num('distance/direct_p90'):.1f} | "
                    f"{num('distance/direct_p99'):.1f} | "
                    f"{num('distance/self_mean'):.3g} | {nce_s} |"
                )

    sweep = load(OUT / 'puzzle_n_sweep.json') or []
    if sweep:
        lines += [
            '',
            '## Puzzle candidate-count sweep (1M, h=5)',
            '',
            '| Method | N | success | nearest-data | diversity | path cost | top1-top2 margin |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
        ]
        for row in sweep:
            lines.append(
                f"| {LABELS.get(row['method'], row['method'])} | {row['N']} | "
                f"{pct(row['success'])} | {row['selected_nearest_data_distance']:.3f} | "
                f"{row['candidate_diversity']:.3f} | {row['selected_path_cost']:.3f} | "
                f"{row['top1_top2_path_cost_margin']:.3f} |"
            )

    lines += [
        '',
        '## Scientific comparisons',
        '',
        '- CTD-W vs DTRL-W: future-occupancy discrimination beyond TRL.',
        '- DTRL-W vs DTRL-U / CTD-W vs CTD-U: PB-style proposer weighting vs structured distance alone.',
        '- PathNCE-W vs CTD-W: observed-intermediate ranking beyond FutureNCE, especially on puzzle.',
        '- BridgeGeo deferred until high-level geometry/puzzle questions are resolved.',
        '',
    ]
    (OUT / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
