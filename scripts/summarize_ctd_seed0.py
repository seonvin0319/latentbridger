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
    'ctd_pathnce_uniform',
    'ctd_pathnce_weighted_bridgegeo',
    'ctd_pathnce_uniform_bridgegeo',
)
STEPS = (100_000, 300_000, 500_000, 800_000, 1_000_000)


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or ['env'])
        writer.writeheader()
        writer.writerows(rows)


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

    by_key = {
        (row['Environment'], row['Method'], row['h']): row
        for row in evaluations
        if row['checkpoint'] == 1_000_000
    }
    lines = [
        '# CTD PathBridger — seed 0',
        '',
        'All CTD rows are seed 0 only, 1M updates, 5 tasks × 50 episodes.',
        'CPB rows are existing files under `exp/contrastive_pathbridger`; they were not retrained.',
        'No original PBF evaluation artifact with the same protocol was found, so PBF is marked unavailable rather than copied from prose.',
        '',
        '## Main result',
        '',
        '| Environment | Method | h=5 | h=2 | h=1 |',
        '| --- | --- | ---: | ---: | ---: |',
    ]
    display = (
        'Original PBF',
        'CPB rank-only',
        'CPB full',
        *METHODS,
    )
    for env in ENVS:
        for method in display:
            values = []
            for horizon in (5, 2, 1):
                row = by_key.get((env, method, horizon))
                values.append('unavailable' if row is None else f"{100 * float(row['success']):.1f}")
            lines.append(f"| {env} | {method} | {' | '.join(values)} |")
    lines += [
        '',
        '## Scientific comparisons',
        '',
        '- CTD-W vs DTRL-W: future-occupancy discrimination beyond TRL.',
        '- CTD-W vs CTD-U: PB-style proposer weighting and long-training coverage.',
        '- CTD-PathNCE-W vs CTD-W: observed-intermediate ranking beyond FutureNCE.',
        '- CTD-PathNCE-U vs CTD-U: the same PathNCE comparison with uniform proposer training.',
        '- BridgeGeo rows are interpreted only after methods 1–6 complete.',
        '',
        'Conclusions must remain separated into geometry learning, proposal distribution, and explicit bridge execution.',
    ]
    (OUT / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
