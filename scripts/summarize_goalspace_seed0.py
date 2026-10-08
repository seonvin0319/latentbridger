"""Build SUMMARY_GOALSPACE.md from immutable seed-0 artifacts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'exp' / 'goalspace_transitive_distance'
SUMMARY = ROOT / 'SUMMARY_GOALSPACE.md'
METHODS = (
    ('gsdtrl_weighted', 'GSDTRL-W'),
    ('gsctd_learned_temp', 'GSCTD-learned-temp'),
    ('gsctd_fixed', 'GSCTD-fixed'),
)
STEPS = (100_000, 300_000, 500_000, 800_000, 1_000_000)
REFERENCE = {
    ('puzzle_3x3', 'CPB-rank'): 91.6,
    ('puzzle_3x3', 'DTRL-W full'): 18.8,
    ('puzzle_3x3', 'CTD-W full'): 45.6,
    ('cube_double', 'DTRL-W full'): 84.4,
}


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def success(env: str, method: str, step=1_000_000, horizon=5):
    record = load(OUT / env / method / 'seed0' / f'evaluation_{step}_h{horizon}.json')
    return None if record is None else 100.0 * float(record['overall_success'])


def cell(value, digits=1):
    return '—' if value is None else f'{float(value):.{digits}f}'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    preflight = load(OUT / 'preflight.json') or {}
    smoke_run = OUT / 'smoke_final' / 'puzzle_3x3' / 'gsctd_learned_temp' / 'seed0'
    smoke_complete = load(smoke_run / 'complete.json')
    smoke_diag = load(smoke_run / 'diagnostics_1.json') or {}
    complete = []
    for env, method in (
        ('puzzle_3x3', 'gsdtrl_weighted'),
        ('cube_double', 'gsdtrl_weighted'),
        ('puzzle_3x3', 'gsctd_learned_temp'),
        ('cube_double', 'gsctd_learned_temp'),
        ('puzzle_3x3', 'gsctd_fixed'),
    ):
        record = load(OUT / env / method / 'seed0' / 'complete.json')
        complete.append(bool(record and int(record.get('steps', 0)) == 1_000_000))

    lines = [
        '# Goal-Space Temporal Quasimetric — seed 0',
        '',
        f'Pilot status: **{"complete" if all(complete) else "incomplete"}** '
        f'({sum(complete)}/5 required 1M runs complete).',
        '',
        f"Host preflight: `{preflight.get('status', 'not recorded')}`; "
        f"devices={preflight.get('devices', [])}.",
        (
            'CPU smoke: complete (one puzzle update, checkpoint round-trip, diagnostics, '
            f"and 5×1 rollout evaluation); alpha={smoke_diag.get('nce/alpha', '—')}, "
            f"effective temperature={smoke_diag.get('nce/effective_temperature', '—')}."
            if smoke_complete
            else 'CPU smoke: not recorded.'
        ),
        '',
        'Protocol: 1M updates; checkpoints 100k/300k/500k/800k/1M; '
        'five tasks × 50 episodes; execution h=5/2/1. No success threshold was imposed.',
        '',
        '## Main table (1M, h=5, success %)',
        '',
        '| Environment | CPB-rank | DTRL-W full | CTD-W full | GSDTRL-W | GSCTD-learned-temp | GSCTD-fixed |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for env in ('puzzle_3x3', 'cube_double'):
        values = [
            REFERENCE.get((env, 'CPB-rank')),
            REFERENCE.get((env, 'DTRL-W full')),
            REFERENCE.get((env, 'CTD-W full')),
            success(env, 'gsdtrl_weighted'),
            success(env, 'gsctd_learned_temp'),
            success(env, 'gsctd_fixed'),
        ]
        lines.append(f"| {env} | {' | '.join(cell(value) for value in values)} |")
    lines += [
        '',
        'Full-state baseline numbers shown above are the supplied comparison values; missing comparisons remain blank.',
        '',
        '## Checkpoint rollout success (%)',
        '',
        '| Environment | Method | h | 100k | 300k | 500k | 800k | 1M |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for env in ('puzzle_3x3', 'cube_double'):
        for method, label in METHODS:
            if env == 'cube_double' and method == 'gsctd_fixed':
                continue
            for horizon in (5, 2, 1):
                values = [success(env, method, step, horizon) for step in STEPS]
                lines.append(
                    f"| {env} | {label} | {horizon} | "
                    + ' | '.join(cell(value) for value in values)
                    + ' |'
                )

    lines += [
        '',
        '## Temporal and FutureNCE diagnostics (1M)',
        '',
        '| Environment | Method | temporal MAE | d median | d p90 | d p99 | FutureNCE recall@1 | alpha | effective temperature |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for env in ('puzzle_3x3', 'cube_double'):
        for method, label in METHODS:
            if env == 'cube_double' and method == 'gsctd_fixed':
                continue
            diag = load(OUT / env / method / 'seed0' / 'diagnostics_1000000.json') or {}
            fields = (
                diag.get('distance/temporal_mae'),
                diag.get('distance/direct_median'),
                diag.get('distance/direct_p90'),
                diag.get('distance/direct_p99'),
                diag.get('nce/recall_at_1'),
                diag.get('nce/alpha'),
                diag.get('nce/effective_temperature'),
            )
            lines.append(f"| {env} | {label} | {' | '.join(cell(value, 3) for value in fields)} |")

    lines += [
        '',
        '## Proposer diversity (1M)',
        '',
        '| Environment | Method | candidate pairwise distance | nearest-data distance | selected displacement | weight ESS fraction |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for env in ('puzzle_3x3', 'cube_double'):
        for method, label in METHODS:
            if env == 'cube_double' and method == 'gsctd_fixed':
                continue
            diag = load(OUT / env / method / 'seed0' / 'diagnostics_1000000.json') or {}
            fields = (
                diag.get('proposer/pairwise_distance'),
                diag.get('proposer/nearest_reference'),
                diag.get('proposer/selected_displacement'),
                diag.get('weight/ess_fraction'),
            )
            lines.append(f"| {env} | {label} | {' | '.join(cell(value, 3) for value in fields)} |")

    sweep = load(OUT / 'puzzle_n_sweep.json') or []
    lines += [
        '',
        '## Puzzle candidate-N sensitivity (1M, h=5)',
        '',
        '| Method | N | success % | diversity | nearest-data distance | selected path cost | top1–top2 margin |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in sweep:
        lines.append(
            f"| {dict(METHODS).get(row['method'], row['method'])} | {row['N']} | "
            f"{100.0 * row['success']:.1f} | {row['candidate_diversity']:.3f} | "
            f"{row['selected_nearest_data_distance']:.3f} | {row['selected_path_cost']:.3f} | "
            f"{row['top1_top2_path_cost_margin']:.3f} |"
        )
    if not sweep:
        lines.append('| — | — | — | — | — | — | — |')

    lines += [
        '',
        '## Interpretation',
        '',
    ]
    if all(complete):
        lines.append('Interpretation must be written from the completed numerical artifacts; no threshold-based verdict is precommitted.')
    else:
        lines.append(
            'No scientific conclusion yet: the required run set is incomplete. '
            'The one-step smoke validates execution plumbing only and is not performance evidence.'
        )
    lines.append('')
    SUMMARY.write_text('\n'.join(lines))
    print(f'wrote {SUMMARY}', flush=True)


if __name__ == '__main__':
    main()
