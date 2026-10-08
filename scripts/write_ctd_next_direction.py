"""Write NEXT_DIRECTION.md from finished seed-0 CTD artifacts."""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'exp' / 'ctd_pathbridger'
CPB = Path('/home/shchoi/latentbridger/exp/contrastive_pathbridger')


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def pct(record):
    return None if record is None else 100 * float(record['overall_success'])


def fmt(value):
    return 'pending' if value is None else f'{value:.1f}'


def main():
    puzzle = {
        'CPB-rank': pct(load(CPB / 'puzzle_3x3/cpb_rank_only/seed0/evaluation_1000000_h5.json')),
        'CTD-W': pct(load(OUT / 'puzzle_3x3/ctd_weighted/seed0/evaluation_1000000_h5.json')),
        'DTRL-W': pct(load(OUT / 'puzzle_3x3/dtrl_weighted/seed0/evaluation_1000000_h5.json')),
        'PathNCE-W': pct(load(OUT / 'puzzle_3x3/ctd_pathnce_weighted/seed0/evaluation_1000000_h5.json')),
    }
    pathnce = puzzle['PathNCE-W']
    if pathnce is None:
        case = 'PathNCE-W puzzle unfinished'
        action = 'Finish PathNCE-W puzzle before scheduling PathNCE-U / BridgeGeo.'
    elif pathnce >= 70:
        case = 'Case 1: PathNCE-W >= 70%'
        action = 'Run PathNCE-U on puzzle and cube-double to separate PathNCE from weighting.'
    elif pathnce >= 50:
        case = 'Case 2: 50% <= PathNCE-W < 70%'
        action = 'PathNCE helps partially; run PathNCE-U on puzzle only as a diagnostic.'
    else:
        case = 'Case 3: PathNCE-W < 50%'
        action = 'Defer PathNCE-U and BridgeGeo; focus on representation / weighting diagnosis.'

    lines = [
        '# CTD next direction',
        '',
        'Generated after the priority queue: PathNCE-W finish → DTRL-U/CTD-U cube-double+puzzle → puzzle N sweep.',
        'BridgeGeo remains deferred.',
        '',
        '## PathNCE puzzle decision',
        '',
        f'- CPB-rank puzzle h5: {fmt(puzzle["CPB-rank"])}',
        f'- CTD-W puzzle h5: {fmt(puzzle["CTD-W"])}',
        f'- DTRL-W puzzle h5: {fmt(puzzle["DTRL-W"])}',
        f'- PathNCE-W puzzle h5: {fmt(puzzle["PathNCE-W"])}',
        f'- Decision: **{case}**',
        f'- Action: {action}',
        '',
        'This threshold is an exploratory scheduling rule, not a paper selection criterion.',
        '',
        '## Weighting causal check (cube-double)',
        '',
        'Compare DTRL-W vs DTRL-U and CTD-W vs CTD-U at 300k/500k/800k/1M using',
        '`SUMMARY.md` proposer diagnostics. Interpret only after both U runs finish:',
        '',
        '- W stable / U collapses → PB-style transitive weighting is the main late-drift protection.',
        '- both stable → structured distance itself is sufficient.',
        '- both collapse → apparent stability was incidental; redesign proposer objective.',
        '',
        '## Puzzle N-sweep',
        '',
        'See `puzzle_n_sweep.csv`. If N=4/8 >> N=32, suspect optimizer\'s curse in large-N ranking.',
        'If N=1 is already poor, representation/proposer is primary.',
        '',
        '## Deferred',
        '',
        '- `ctd_pathnce_weighted_bridgegeo`',
        '- `ctd_pathnce_uniform_bridgegeo`',
        '- remaining all-env ablations not listed in the priority queue',
        '',
        'Do not change the FutureNCE training formulation until scale diagnostics in SUMMARY are reviewed.',
        '',
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'NEXT_DIRECTION.md').write_text('\n'.join(lines))
    print(f'wrote {OUT / "NEXT_DIRECTION.md"}', flush=True)


if __name__ == '__main__':
    main()
