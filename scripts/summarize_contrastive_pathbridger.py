"""Aggregate observed results only; missing runs never count as zero success."""
import argparse
import csv
import json
from pathlib import Path


def table_csv(path, rows, fields=None):
    fields = fields or sorted({key for row in rows for key in row})
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='exp/contrastive_pathbridger')
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    evaluations, diagnostics = [], []
    for path in sorted(root.glob('*/*/seed*/evaluation_*.json')):
        d = json.loads(path.read_text())
        evaluations.append(dict(env=d['env'], variant=d['variant'], seed=d['seed'], checkpoint=d['checkpoint'], h=d['h'],
                                success=d['overall_success'], success_count=d['success_count'], N=d['N'], temperature=d['temperature']))
    for path in sorted(root.glob('*/*/seed*/diagnostics_*.json')):
        d = json.loads(path.read_text())
        d.update(env=path.parents[2].name, variant=path.parents[1].name, seed=int(path.parent.name[4:]))
        diagnostics.append(d)
    table_csv(root / 'evaluation_summary.csv', evaluations, ['env', 'variant', 'seed', 'checkpoint', 'h', 'success', 'success_count', 'N', 'temperature'])
    table_csv(root / 'diagnostic_summary.csv', diagnostics, None if diagnostics else ['env', 'variant', 'seed', 'step'])
    lines = ['# Contrastive PathBridger', '', 'Primary: 1M updates, h=2, 5 tasks × 50 episodes per seed.', '',
             '| Environment | Variant | Seed | Checkpoint | h | Success |', '|---|---|---:|---:|---:|---:|']
    for d in evaluations:
        lines.append(f"| {d['env']} | {d['variant']} | {d['seed']} | {d['checkpoint']} | {d['h']} | {d['success']:.4f} |")
    if not evaluations:
        lines.append('\nNo completed checkpoint evaluations yet.')
    lines += ['', '## 1M h=2 across seeds', '', '| Environment | Variant | Completed seeds | Mean ± std |', '|---|---|---:|---|']
    import statistics
    for env in ('cube-single-play-v0','cube-double-play-v0','puzzle-3x3-play-v0','antmaze-medium-navigate-v0'):
        for variant in ('cpb_rank_only', 'cpb_full'):
            vals=[d['success'] for d in evaluations if d['env']==env and d['variant']==variant and d['checkpoint']==1000000 and d['h']==2]
            lines.append(f'| {env} | {variant} | {len(vals)} | '+(f'{statistics.mean(vals):.4f} ± {statistics.pstdev(vals):.4f}' if vals else 'pending')+' |')
    lines += ['', '## Reference evidence', '', 'Original PathBridger and latent endpoint results require matching protocol and provenance; no baseline is retrained automatically.']
    # Inventory locally available evidence without inventing or pooling incomparable settings.
    for reference in (Path('/home/shchoi/PathBridger/exp'),):
        paths = sorted(reference.rglob('*summary*.csv')) + sorted(reference.rglob('eval.csv')) + sorted(reference.rglob('*summary*.json'))
        lines.append(f'Local reference root: `{reference}`; {len(paths)} result files found.')
        for path in paths[:40]:
            lines.append(f'- `{path}`')
    lines += ['', '## Research questions', '',
              'Q1 (TRL replacement), Q2 (ranking), Q3 (progress weights), Q4 (explicit bridge versus latent chunks), and Q5 (collapse/stability) remain pending until the corresponding runs and matched references are available.',
              'N=1 on cube-single cannot identify a candidate-ranking benefit. Rank-only versus full isolates proposer weighting; latent effective rank and per-dimension variance diagnose collapse.',
              'Nearest-state diagnostics use the complete training dataset. Other diagnostic probes use held-out states and are not environment-rollout measurements.']
    smoke = root / 'smoke/diagnostics_2000.json'
    if smoke.exists():
        d = json.loads(smoke.read_text())
        lines += ['', '## 2k smoke (not a 1M result)', '',
                  f"Held-out recall@1: {d['critic/recall_at_1']:.4f} (chance 1/128). "
                  f"Bridge prefix L1: {d['bridge/five_step_prefix_error']:.4f}; "
                  f"IDM action MSE: {d['idm/action_mse']:.6f}.",
                  f"Effective ranks: phi {d['phi/effective_rank']:.2f}, psi {d['psi/effective_rank']:.2f}."]
    for name in ('status.json','failure.json'):
        if (root / name).exists():
            lines += ['', f'{name}: `{(root / name).read_text()}`']
    (root / 'SUMMARY.md').write_text('\n'.join(lines)+'\n')

if __name__ == '__main__':
    main()
