"""Read-only evidence aggregation; never substitute supplied numbers for runs."""
import csv
import fcntl
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'exp/goalspace_ablation'
CORE = ('puzzle_3x3', 'cube_double', 'antmaze_medium', 'cube_single')
HARD = ('puzzle_4x4', 'cube_triple', 'antmaze_large', 'scene')
METHODS = (
    ('pbf', 'Original PBF'), ('cpb_rank', 'CPB rank'),
    ('dtrl_weighted', 'full-state DTRL-W'), ('gsdtrl_weighted', 'GSDTRL-W'),
    ('gs_trl_weighted', 'GS-TRL-W'), ('gsdtrl_uniform', 'GSDTRL-U'),
    ('gsdtrl_no_transitive_weighted', 'GSDTRL-NoTransitive-W'),
    ('gs_symmetric_weighted', 'GS-Symmetric-W'),
)


def load(path):
    return json.loads(path.read_text()) if path.exists() else None


def write_csv(name, rows, fields):
    path = OUT / name
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'summary.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        roots = [OUT, OUT / 'external/choi',
                 Path('/home/shchoi/latentbridger_ctd/exp/ctd_pathbridger'),
                 Path('/home/shchoi/latentbridger/exp/contrastive_pathbridger')]
        records = []; seen = set(); diagnostics = []
        for root in roots:
            for path in sorted(root.glob('*/*/seed*/evaluation_*_h*.json')):
                if '_N' in path.stem: continue
                row = load(path)
                if row.get('num_tasks') != 5 or row.get('episodes_per_task') != 50: continue
                env, method, seed = path.parts[-4], path.parts[-3], int(path.parts[-2][4:])
                method = {'cpb_rank_only': 'cpb_rank'}.get(method, method)
                if method not in dict(METHODS): continue
                step = int(row.get('checkpoint', path.stem.split('_')[1]))
                h = int(row.get('h', path.stem.split('_h')[1]))
                if step not in (100000,300000,500000,800000,1000000): continue
                key = (env,method,seed,step,h)
                if key in seen: raise ValueError(f'Duplicate evidence for {key}: {path}')
                seen.add(key)
                config = load(path.parent/'config.json')
                signature = hashlib.sha256(json.dumps([config['agent'],row.get('N'),row.get('temperature')],sort_keys=True).encode()).hexdigest() if config and 'agent' in config else ''
                records.append(dict(config_signature=signature,env=env,method=method,seed=seed,step=step,h=h,
                                    success_percent=100*row['overall_success'],provenance=str(path)))
            for path in sorted(root.glob('*/*/seed*/diagnostics_*.json')):
                row = load(path)
                for key,value in row.items():
                    if '/' in key:
                        diagnostics.append(dict(env=path.parts[-4],method=path.parts[-3],
                            seed=int(path.parts[-2][4:]),step=row['step'],metric=key,value=value,provenance=str(path)))
        fields = ['env','method','seed','step','h','success_percent','provenance','config_signature']
        write_csv('ablation_results.csv',[r for r in records if r['env'] in CORE],fields)
        write_csv('harder_env_results.csv',[r for r in records if r['env'] in HARD],fields)
        write_csv('diagnostics.csv',diagnostics,['env','method','seed','step','metric','value','provenance'])
        multiseed=[]
        for env in CORE+HARD:
            for h in (5,2,1):
                found = {r['seed']:r for r in records if r['env']==env and r['method']=='gsdtrl_weighted' and r['step']==1000000 and r['h']==h}
                complete = all(seed in found for seed in (0,1,2))
                compatible = complete and bool(found[0]['config_signature']) and len({found[s]['config_signature'] for s in (0,1,2)}) == 1
                values = [found[s]['success_percent'] for s in (0,1,2) if s in found]
                multiseed.append(dict(env=env,h=h,n_seeds=len(values),
                    seed0=found.get(0,{}).get('success_percent',''),seed1=found.get(1,{}).get('success_percent',''),seed2=found.get(2,{}).get('success_percent',''),
                    mean=statistics.mean(values) if compatible else '',sample_std=statistics.stdev(values) if compatible else '',
                    status='complete' if compatible else 'config verification required' if complete else 'pending',
                    provenance=';'.join(found[s]['provenance'] for s in (0,1,2) if s in found)))
        write_csv('gsdtrl_multiseed.csv',multiseed,list(multiseed[0]))
        sweep=[]
        sweep_root=OUT/'external/choi/puzzle_3x3/gsdtrl_weighted/seed0'
        for n in (1,4,8,16,32):
            path=sweep_root/f'evaluation_1000000_h5_N{n}.json'
            row=load(path)
            if row and (row.get('checkpoint'),row.get('h'),row.get('N'),row.get('seed'),row.get('variant'),row.get('num_tasks'),row.get('episodes_per_task')) == (1000000,5,n,0,'gsdtrl_weighted',5,50):
                sweep.append(dict(N=n,success_percent=100*row['overall_success'],provenance=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        write_csv('puzzle_n_sweep_provenance.csv',sweep,['N','success_percent','provenance','sha256'])
        sweep_status='merged Choi evidence; no duplicate evaluation' if len(sweep)==5 else f'awaiting Choi provenance ({len(sweep)}/5 N values available); no duplicate evaluation started'
        status = load(OUT/'queue_status.json') or {'status':'not started'}
        lines=['# GSDTRL ablations and generalization','',
               f"Queue: {status.get('status')}. Active: {status.get('active',[])}.",'',
               'Base: da37867d31d47b02a14dcbc56dd8f267062c02aa. Branch: goalspace-ablation-generalization.',
               'Protocol: 1M updates; 100k/300k/500k/800k/1M; h=5 primary, h=2/1 secondary; 5 tasks × 50 episodes.',
               'Existing evidence directories are read-only. Missing results are pending, not zero.', '',
               '| Method | Puzzle-3x3 | Cube-double | AntMaze-medium | Cube-single |',
               '| --- | ---: | ---: | ---: | ---: |']
        for method,label in METHODS:
            cells=[]
            for env in CORE:
                row=next((r for r in records if (r['env'],r['method'],r['seed'],r['step'],r['h'])==(env,method,0,1000000,5)),None)
                cells.append(f"[{row['success_percent']:.1f}]({row['provenance']})" if row else 'pending')
            lines.append('| '+label+' | '+' | '.join(cells)+' |')
        lines += ['', 'Table: seed0, 1M, h=5, success %. CSVs retain all checkpoints and horizons.',
                  'Supplied Choi puzzle references (not locally verified): CPB-rank 91.6, full DTRL-W 18.8, full CTD-W 45.6, GSDTRL-W 97.2.',
                  'No causal conclusion is available until matched ablation runs finish.', '',
                  'Three-seed mean and sample standard deviation are emitted only when seeds 0, 1, 2 all exist.',
                  'Choi seed0 files may be placed under external/choi/<env>/gsdtrl_weighted/seed0/.',
                  f'Puzzle N-sweep: {sweep_status}.', '',
                  'Harder configurations: see harder_env_provenance.json. No new hyperparameters or phi features.',
                  'Scheduler: see queue_status.json; launch history: launch_history.jsonl.', '']
        tmp=OUT/'SUMMARY_ABLATION.md.tmp';tmp.write_text('\n'.join(lines));tmp.replace(OUT/'SUMMARY_ABLATION.md')


if __name__=='__main__':main()
