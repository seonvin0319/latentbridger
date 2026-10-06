"""Aggregate calibrated results only; report missing evidence explicitly."""
import argparse
import csv
import json
import statistics
from pathlib import Path

ENVS=('cube-single-play-v0','cube-double-play-v0','puzzle-3x3-play-v0','antmaze-medium-navigate-v0')
VARIANTS=('pathbridger_original','cpb_rank_only','cpb_full')


def table_csv(path,rows,fields=None):
    fields=fields or sorted({key for row in rows for key in row})
    with path.open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def aggregate(root):
    evaluations,diagnostics,curves=[],[],[]
    for path in sorted(root.glob('*/*/seed*/evaluation_[0-9]*_h*.json')):
        d=json.loads(path.read_text())
        evaluations.append(dict(env=d['env'],variant=d['variant'],seed=d['seed'],checkpoint=d['checkpoint'],h=d['h'],
            success=d['overall_success'],success_count=d['success_count'],N=d['N'],temperature=d['temperature']))
    for path in sorted(root.glob('*/*/seed*/diagnostics_*.json')):
        d=json.loads(path.read_text())
        config=json.loads((path.parent/'config.json').read_text())
        d.update(env=config['agent']['env_name'],variant=config['agent']['variant'],seed=config['runtime']['seed'])
        diagnostics.append(d)
    for path in sorted(root.glob('*/*/seed*/train.jsonl')):
        config=json.loads((path.parent/'config.json').read_text())
        for line in path.read_text().splitlines():
            try:
                d=json.loads(line)
            except json.JSONDecodeError:
                continue  # The active logger may be writing its last line.
            d.update(env=config['agent']['env_name'],variant=config['agent']['variant'],seed=config['runtime']['seed'])
            curves.append(d)
    table_csv(root/'evaluation_summary.csv',evaluations,['env','variant','seed','checkpoint','h','success','success_count','N','temperature'])
    table_csv(root/'diagnostic_summary.csv',diagnostics,None if diagnostics else ['env','variant','seed','step'])
    table_csv(root/'learning_curves.csv',curves,None if curves else ['env','variant','seed','step'])
    lines=['# Calibrated Contrastive PathBridger','','Primary: 1M updates, h=5. Secondary: h=2 and h=1. Each point uses five tasks × 50 episodes per seed.',
           'Raw-score experiments are archived separately and are never pooled here.','','## Main comparison (1M, h=5)','',
           '| Environment | PathBridger | CPB rank-only | CPB full |','|---|---|---|---|']
    def result(env,variant,h):
        rows=[d for d in evaluations if d['env']==env and d['variant']==variant and d['checkpoint']==1000000 and d['h']==h]
        values=[d['success'] for d in rows]
        if not values:
            return 'reference unavailable' if variant=='pathbridger_original' else 'pending'
        return f'{statistics.mean(values):.4f} ± {statistics.pstdev(values):.4f} (seeds {",".join(str(d["seed"]) for d in rows)}; h={h})'
    for env in ENVS:
        lines.append('| '+env+' | '+' | '.join(result(env,v,5) for v in VARIANTS)+' |')
    lines+=['','## CPB full execution ablation (1M)','','| Environment | h=5 primary | h=2 | h=1 |','|---|---|---|---|']
    for env in ENVS:
        lines.append('| '+env+' | '+' | '.join(result(env,'cpb_full',h) for h in (5,2,1))+' |')
    lines+=['','## Checkpoint evaluations','','| Environment | Variant | Seed | Updates | h | Success | Count / 250 |','|---|---|---:|---:|---:|---:|---:|']
    for d in evaluations:
        lines.append(f"| {d['env']} | {d['variant']} | {d['seed']} | {d['checkpoint']} | {d['h']} | {d['success']:.4f} | {d['success_count']} |")
    lines+=['','## Checkpoint diagnostics','','| Environment | Variant | Seed | Updates | Recall@1 | Phi / psi effective rank | Bridge L1 | IDM MSE |','|---|---|---:|---:|---:|---|---:|---:|']
    for d in diagnostics:
        if 'phi/effective_rank' not in d:
            continue
        lines.append(f"| {d['env']} | {d['variant']} | {d['seed']} | {d['step']} | {d['critic/recall_at_1']:.4f} | {d['phi/effective_rank']:.2f} / {d['psi/effective_rank']:.2f} | {d['bridge/five_step_prefix_error']:.4f} | {d.get('training/idm/action_mse',0):.6f} |")
    completed=[json.loads(path.read_text()) for path in root.glob('*/*/seed*/complete.json')]
    lines+=['',f"Completed 1M runs: {sum(d['steps']==1000000 for d in completed)}/14. Completed-run wall time: {sum(d['wall_seconds'] for d in completed)/3600:.2f} hours.",
            'Failed runs: see failure.json.' if (root/'failure.json').exists() else 'Failed calibrated runs: none recorded.',
            'The 100k full seed0 sanity prelude is resumed as the same 1M run, not retrained.',
            '', '## Scientific interpretation','',
            'Q1. The primary table measures calibrated CPB at the original h=5 execution horizon. Matched original TRL results are required to conclude whether reachability replaces TRL without loss; no matched local baseline has been found.',
            'Q2. Candidate Cbar, top1–top2 margins and target-ranking agreement are in diagnostic_summary.csv. A causal ranking benefit needs a random/unranked comparison. Cube-single N=1 does not test ranking.',
            'Q3. Matched seed0 full-minus-rank-only differences at 1M h=5:']
    for env in ENVS[:2]:
        paired={d['variant']:d['success'] for d in evaluations if d['env']==env and d['seed']==0 and d['checkpoint']==1000000 and d['h']==5}
        if {'cpb_full','cpb_rank_only'}<=paired.keys():
            lines.append(f"- {env}: {paired['cpb_full']-paired['cpb_rank_only']:+.4f}; one paired seed, not a significance claim.")
        else:
            lines.append(f'- {env}: pending both 1M evaluations.')
    lines+=['','Q4. Cube-double and puzzle results appear in the primary table. The current latent-endpoint AWR direct/guided results are unavailable locally; older latent-flow or online SGCRL runs are not substituted.',
            'Q5. Calibration effects on candidate ordering and target-progress estimates:']
    if diagnostics:
        for env in ENVS:
            rows=[d for d in diagnostics if d['env']==env]
            ranked=[d for d in rows if d.get('calibration/ranking_correlation_defined_fraction',0)>0]
            if ranked:
                lines.append(f"- {env}: mean raw/calibrated candidate Spearman {statistics.mean(d['calibration/ranking_spearman'] for d in ranked):.4f}; top1 agreement {statistics.mean(d['calibration/raw_calibrated_top1_agreement'] for d in ranked):.4f}.")
            progress=[d['calibration/progress_pearson'] for d in rows if d.get('calibration/progress_correlation_defined',0)]
            if progress:
                lines.append(f"- {env}: mean raw/calibrated target-progress Pearson {statistics.mean(progress):.4f} across available checkpoints; this is descriptive, not a raw-score training ablation.")
    else:
        lines.append('Pending production checkpoint probes.')
    lines+=['','Q6. Finite metrics are checked each update. Effective rank, per-dimension std, norms and eigenvalues are recorded at checkpoints.']
    for encoder in ('phi','psi'):
        ranks=[d[f'{encoder}/effective_rank'] for d in diagnostics if f'{encoder}/effective_rank' in d]
        if ranks:
            lines.append(f'Observed {encoder} effective ranks: {min(ranks):.2f}–{max(ranks):.2f}/64. Nonzero ranks alone do not establish task sufficiency.')
    lines+=['','Q7. Paired CPB full h=2 minus h=5 at 1M:']
    for env in ENVS:
        paired=[]
        for seed in (0,1,2):
            values={d['h']:d['success'] for d in evaluations if d['env']==env and d['variant']=='cpb_full' and d['seed']==seed and d['checkpoint']==1000000}
            if {2,5}<=values.keys():
                paired.append(values[2]-values[5])
        lines.append(f'- {env}: '+(f'{statistics.mean(paired):+.4f} over {len(paired)} paired seeds.' if paired else 'pending.'))
    lines+=['','Diagnostic candidates use held-out states; nearest-state search covers the complete training set. These are not rollout measurements. N=1 ranking correlations are marked undefined, not interpreted as evidence.',
            'The fixed reference bank uses training future-goal marginal samples only. Validation states are used solely for diagnostics.']
    smoke=root/'smoke/diagnostics_2000.json'
    if smoke.exists():
        d=json.loads(smoke.read_text())
        lines+=['','## Smoke (2k, not a main result)','',f"Recall@1 {d['critic/recall_at_1']:.4f}; phi/psi ranks {d['phi/effective_rank']:.2f}/{d['psi/effective_rank']:.2f}; calibrated delta std {d['calibration/calibrated_delta_std']:.4f}."]
    for name in ('status.json','failure.json'):
        if (root/name).exists():
            lines+=['',f'{name}: `{(root/name).read_text()}`']
    (root/'SUMMARY.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='exp/contrastive_pathbridger');args=p.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True);aggregate(root)

if __name__=='__main__':
    main()
