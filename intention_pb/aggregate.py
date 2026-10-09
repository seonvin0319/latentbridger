"""Aggregate tokenizer / diagnostics / control-eval outputs into CSV tables and ``summary.md``."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from intention_pb.common import (
    EXP_ROOT,
    FINAL_STEP,
    H_EXECS,
    METHODS,
    NUM_CODES,
    REF_METHOD,
    SEEDS,
    TASK_ORDER,
    TOKENIZER_SAVE_STEPS,
    diag_dir,
    read_json,
    tokenizer_dir,
)

PAIRS = (('I-SG', 'PB'), ('Shared-I', 'PB'), ('Shared-I', 'I-SG'), ('Shared-I', 'Shuffled-I'))
EFFECT = 0.05  # 5 success-rate points: threshold used by the rule-based verdict helper


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted({k for r in rows for k in r})
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in fields})


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample std (ddof=1; nan for a single value)."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float('nan'), float('nan')
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else float('nan')


def paired_deltas(scores: dict[tuple, float], tasks, seeds, h_execs) -> list[dict]:
    """``scores[(task, seed, method, h)] -> success``; returns one row per (task, seed, h, pair)."""
    rows = []
    for t in tasks:
        for h in h_execs:
            for s in seeds:
                for a, b in PAIRS:
                    if (t, s, a, h) in scores and (t, s, b, h) in scores:
                        rows.append(dict(task=t, seed=s, h_exec=h, pair=f'{a} - {b}', delta=scores[(t, s, a, h)] - scores[(t, s, b, h)]))
    return rows


def collect_eval(root: Path = EXP_ROOT) -> list[dict]:
    rows = []
    for p in sorted((root / 'eval').glob('*/step*/*/summary.json')):
        s = read_json(p)
        r = {k: s.get(k) for k in ('task', 'seed', 'step', 'method', 'h_exec', 'success_rate', 'mean_return', 'mean_length',
                                   'mean_replans', 'mean_selected_score', 'num_episodes', 'switch_frequency',
                                   'code_entropy_steps', 'code_entropy_episodes', 'dominant_code_share', 'temperature',
                                   'num_candidates')}
        r['_summary'] = s
        rows.append(r)
    return rows


def _fmt(m: float, s: float) -> str:
    if np.isnan(m):
        return 'n/a'
    return f'{100 * m:.1f} ± {100 * s:.1f}' if not np.isnan(s) else f'{100 * m:.1f}'


def rule_based_verdict(delta_means: dict[tuple, float], offline_gap_positive: bool | None) -> tuple[str, str]:
    """Mechanical helper; the final verdict in the report is written by hand from all evidence."""
    cells = sorted({(t, h) for (t, h, _) in delta_means})
    if not cells:
        return 'n/a', 'no paired results'
    def frac(pair, cond):
        vals = [delta_means.get((t, h, pair)) for t, h in cells]
        vals = [v for v in vals if v is not None and not np.isnan(v)]
        return (sum(cond(v) for v in vals) / len(vals)) if vals else 0.0
    sp_pos = frac('Shared-I - PB', lambda v: v > EFFECT)
    sp_neg = frac('Shared-I - PB', lambda v: v < -EFFECT)
    sc_pos = frac('Shared-I - Shuffled-I', lambda v: v > EFFECT)
    ss_pos = frac('Shared-I - I-SG', lambda v: v > 0)
    ip_pos = frac('I-SG - PB', lambda v: v > EFFECT)
    detail = (f'cells={len(cells)} frac(Shared-I-PB>+{EFFECT})={sp_pos:.2f} frac(<-{EFFECT})={sp_neg:.2f} '
              f'frac(Shared-I-Shuffled-I>+{EFFECT})={sc_pos:.2f} frac(Shared-I-I-SG>0)={ss_pos:.2f} '
              f'frac(I-SG-PB>+{EFFECT})={ip_pos:.2f} offline_gap_positive={offline_gap_positive}')
    if sp_pos >= 0.75 and sc_pos >= 0.75 and ss_pos >= 0.75:
        return 'A', detail
    if sp_neg >= 0.5:
        return 'D', detail
    if sp_pos > 0 or ip_pos > 0 or sc_pos > 0:
        return 'B', detail
    return 'C', detail


def aggregate(root: Path = EXP_ROOT) -> dict[str, Any]:
    out = root / 'aggregate'
    out.mkdir(parents=True, exist_ok=True)
    ev = collect_eval(root)
    seed_rows = [{k: v for k, v in r.items() if k != '_summary'} for r in ev]
    _write_csv(out / 'seed_scores.csv', seed_rows,
               ['task', 'seed', 'step', 'method', 'h_exec', 'success_rate', 'mean_return', 'mean_length', 'mean_replans',
                'mean_selected_score', 'num_episodes', 'switch_frequency', 'code_entropy_steps', 'code_entropy_episodes',
                'dominant_code_share', 'temperature', 'num_candidates'])
    _write_csv(out / 'learning_curves.csv',
               [dict(task=r['task'], seed=r['seed'], method=r['method'], h_exec=r['h_exec'], step=r['step'],
                     success_rate=r['success_rate'], num_episodes=r['num_episodes']) for r in sorted(ev, key=lambda r: (r['task'], r['seed'], r['method'], r['h_exec'], r['step']))],
               ['task', 'seed', 'method', 'h_exec', 'step', 'success_rate', 'num_episodes'])

    final = {(r['task'], r['seed'], r['method'], r['h_exec']): r for r in ev if r['step'] == FINAL_STEP}
    scores = {k: float(v['success_rate']) for k, v in final.items()}
    tasks = [t for t in TASK_ORDER if any(k[0] == t for k in final)]
    final_rows = []
    for t in tasks:
        for m in (*METHODS, REF_METHOD):
            for h in H_EXECS:
                vals = [(s, scores[(t, s, m, h)]) for s in SEEDS if (t, s, m, h) in scores]
                if not vals:
                    continue
                mu, sd = mean_std([v for _, v in vals])
                final_rows.append(dict(task=t, method=m, h_exec=h, mean=mu, std=sd, n_seeds=len(vals),
                                       raw=';'.join(f's{s}={v:.3f}' for s, v in vals)))
    _write_csv(out / 'final_scores.csv', final_rows, ['task', 'method', 'h_exec', 'mean', 'std', 'n_seeds', 'raw'])

    deltas = paired_deltas(scores, tasks, SEEDS, H_EXECS)
    _write_csv(out / 'paired_deltas.csv', deltas, ['task', 'seed', 'h_exec', 'pair', 'delta'])
    delta_means = {}
    for t in tasks:
        for h in H_EXECS:
            for a, b in PAIRS:
                vals = [d['delta'] for d in deltas if d['task'] == t and d['h_exec'] == h and d['pair'] == f'{a} - {b}']
                if vals:
                    delta_means[(t, h, f'{a} - {b}')] = float(np.mean(vals))

    # Tokenizer metrics
    tok_rows = []
    for t in TASK_ORDER:
        for s in SEEDS:
            for st in TOKENIZER_SAVE_STEPS:
                p = tokenizer_dir(t, s) / 'metrics' / f'step_{st}.json'
                if p.is_file():
                    m = read_json(p)
                    tok_rows.append(dict(task=t, seed=s, step=st, split='val', perplexity=m['perplexity'], max_usage=m['max_usage'],
                                         action_recon_mse=m['action_recon_mse'], delta_recon_mse=m['delta_recon_mse'],
                                         collapsed=m['collapsed'], usage=' '.join(f'{u:.3f}' for u in m['usage'])))
    _write_csv(out / 'tokenizer_metrics.csv', tok_rows,
               ['task', 'seed', 'step', 'split', 'perplexity', 'max_usage', 'action_recon_mse', 'delta_recon_mse', 'collapsed', 'usage'])

    # Offline diagnostics (long format)
    diag_rows, diags = [], {}
    for t in TASK_ORDER:
        for s in SEEDS:
            p = diag_dir(t, s) / 'diagnostics.json'
            if not p.is_file():
                continue
            d = read_json(p)
            diags[(t, s)] = d
            for grp in ('bridge',):
                for var, mets in d[grp].items():
                    for k, v in mets.items():
                        diag_rows.append(dict(task=t, seed=s, group=f'bridge/{var}', metric=k, value=v))
            for k, v in d['bridge_gap_shuffled_minus_true'].items():
                diag_rows.append(dict(task=t, seed=s, group='bridge/gap_shuffled_minus_true', metric=k, value=v))
            for tn, mets in d['subgoal'].items():
                for k, v in mets.items():
                    if not isinstance(v, list):
                        diag_rows.append(dict(task=t, seed=s, group=f'subgoal/{tn}', metric=k, value=v))
            for var, mets in d['conditional_variance'].items():
                if isinstance(mets, dict):
                    for k, v in mets.items():
                        diag_rows.append(dict(task=t, seed=s, group=f'cond_var/{var}', metric=k, value=v))
            for tn, mets in d['subgoal_diversity'].items():
                for k, v in mets.items():
                    diag_rows.append(dict(task=t, seed=s, group=f'diversity/{tn}', metric=k, value=v))
            for k, v in d['bridge_controllability'].items():
                diag_rows.append(dict(task=t, seed=s, group='bridge_controllability', metric=k, value=v))
            diag_rows.append(dict(task=t, seed=s, group='tokenizer_val', metric='perplexity', value=d['tokenizer_val']['perplexity']))
            diag_rows.append(dict(task=t, seed=s, group='tokenizer_val', metric='max_usage', value=d['tokenizer_val']['max_usage']))
    _write_csv(out / 'offline_diagnostics.csv', diag_rows, ['task', 'seed', 'group', 'metric', 'value'])

    # Intention usage (final step)
    use_rows = []
    for (t, s, m, h), r in sorted(final.items()):
        summ = r['_summary']
        if 'code_histogram' not in summ:
            continue
        for k in range(NUM_CODES):
            pc = summ['per_code_success_majority'][str(k)]
            use_rows.append(dict(task=t, seed=s, method=m, h_exec=h, code=k, selected_share=summ['code_histogram'][k],
                                 majority_episodes=pc['n'], majority_success=pc['success'],
                                 switch_frequency=summ['switch_frequency'], code_entropy_steps=summ['code_entropy_steps']))
    _write_csv(out / 'intention_usage.csv', use_rows,
               ['task', 'seed', 'method', 'h_exec', 'code', 'selected_share', 'majority_episodes', 'majority_success',
                'switch_frequency', 'code_entropy_steps'])

    offline_gaps = [d['bridge_gap_shuffled_minus_true']['path_mse'] for d in diags.values()]
    offline_pos = (bool(np.mean([g > 0 for g in offline_gaps]) >= 0.75) if offline_gaps else None)
    verdict, verdict_detail = rule_based_verdict(delta_means, offline_pos)
    md = _summary_md(tasks, final_rows, deltas, delta_means, tok_rows, diags, final, verdict, verdict_detail)
    (out / 'summary.md').write_text(md, encoding='utf-8')
    return dict(num_eval=len(ev), num_final=len(final), verdict=verdict, verdict_detail=verdict_detail)


def _summary_md(tasks, final_rows, deltas, delta_means, tok_rows, diags, final, verdict, verdict_detail) -> str:
    L = ['# Intention-Conditioned PathBridger — aggregate summary', '',
         'Success rate in %, mean ± sample std over seeds (raw per-seed values in `final_scores.csv`). '
         'Final checkpoint 1M, 100 episodes (20 x tasks 1-5) per cell. Equal budget N=16 '
         '(intention methods: 2 per code x K=8). PB-ref = PB at its original best setting (not equal budget).', '']
    fr = {(r['task'], r['method'], r['h_exec']): r for r in final_rows}
    for h in H_EXECS:
        L += [f'## Final success, h_exec={h}', '', '| Task | PB | I-SG | Shared-I | Shuffled-I | PB-ref | seeds |', '|---|---|---|---|---|---|---|']
        for t in tasks:
            cells = [_fmt(fr[(t, m, h)]['mean'], fr[(t, m, h)]['std']) if (t, m, h) in fr else 'n/a' for m in (*METHODS, REF_METHOD)]
            n = max([fr[(t, m, h)]['n_seeds'] for m in METHODS if (t, m, h) in fr] or [0])
            L.append(f'| {t} | ' + ' | '.join(cells) + f' | {n} |')
        L.append('')
        L += [f'Raw per-seed values (h_exec={h}):', '']
        for t in tasks:
            for m in (*METHODS, REF_METHOD):
                if (t, m, h) in fr:
                    L.append(f'- {t} {m}: {fr[(t, m, h)]["raw"]}')
        L.append('')
    L += ['## Paired deltas (success points, mean over seeds; per-seed in `paired_deltas.csv`)', '',
          '| Task | h | I-SG − PB | Shared-I − PB | Shared-I − I-SG | Shared-I − Shuffled-I (consistency) |', '|---|---|---|---|---|---|']
    for t in tasks:
        for h in H_EXECS:
            vals = []
            for a, b in PAIRS:
                ds = [d['delta'] for d in deltas if d['task'] == t and d['h_exec'] == h and d['pair'] == f'{a} - {b}']
                mu, sd = mean_std(ds)
                vals.append((f'{100 * mu:+.1f}' + (f' ± {100 * sd:.1f}' if not np.isnan(sd) else '')) if ds else 'n/a')
            L.append(f'| {t} | {h} | ' + ' | '.join(vals) + ' |')
    L.append('')
    L += ['## Offline bridge: true c vs shuffled c (held-out val, true z)', '',
          '| Task | seed | path MSE true-c | path MSE shuffled-c | Δ(shuf − true) | first-step MSE true / shuf | IDM action MSE true / shuf | PB bridge path MSE |',
          '|---|---|---|---|---|---|---|---|']
    for (t, s), d in sorted(diags.items(), key=lambda x: (TASK_ORDER.index(x[0][0]), x[0][1])):
        b = d['bridge']
        L.append(f"| {t} | {s} | {b['true_c']['path_mse']:.5f} | {b['shuffled_c']['path_mse']:.5f} | "
                 f"{d['bridge_gap_shuffled_minus_true']['path_mse']:+.5f} | {b['true_c']['first_step_mse']:.5f} / {b['shuffled_c']['first_step_mse']:.5f} | "
                 f"{b['true_c']['idm_action_mse']:.4f} / {b['shuffled_c']['idm_action_mse']:.4f} | {b['pb_bridge']['path_mse']:.5f} |")
    L.append('')
    L += ['## Conditional variance (kNN k=64 in standardised [s,g]; ratio = Var(.|s,g,c)/Var(.|s,g); perm = chance baseline)', '',
          '| Task | seed | subgoal Δ ratio (perm) | first action ratio (perm) | action chunk ratio (perm) | prefix path ratio (perm) |', '|---|---|---|---|---|---|']
    for (t, s), d in sorted(diags.items(), key=lambda x: (TASK_ORDER.index(x[0][0]), x[0][1])):
        cv = d['conditional_variance']
        cells = [f"{cv[k]['ratio_cond']:.3f} ({cv[k]['ratio_perm']:.3f})" for k in ('subgoal_displacement', 'first_action', 'action_chunk', 'prefix_path')]
        L.append(f'| {t} | {s} | ' + ' | '.join(cells) + ' |')
    L.append('')
    L += ['## Subgoal diversity and bridge controllability', '',
          '| Task | seed | between-code (t=1) | within-code (t=1) | between-code (eval t) | within-code (eval t) | bridge pairwise / motion |', '|---|---|---|---|---|---|---|']
    for (t, s), d in sorted(diags.items(), key=lambda x: (TASK_ORDER.index(x[0][0]), x[0][1])):
        dv = d['subgoal_diversity']
        bc = d['bridge_controllability']
        L.append(f"| {t} | {s} | {dv['t1']['between_code']:.3f} | {dv['t1']['within_code']:.3f} | {dv['eval']['between_code']:.3f} | "
                 f"{dv['eval']['within_code']:.3f} | {bc['pairwise_prefix_dist']:.3f} / {bc['prefix_motion_norm']:.3f} |")
    L.append('')
    L += ['## Subgoal error (normalised RMS to the true s_{t+K}; eval temperature)', '',
          '| Task | seed | PB selected | PB best-of-16 | I-SG enum selected | I-SG enum best-of-16 | teacher-c mean | matching-code / non-matching |', '|---|---|---|---|---|---|---|---|']
    for (t, s), d in sorted(diags.items(), key=lambda x: (TASK_ORDER.index(x[0][0]), x[0][1])):
        e = d['subgoal']['eval']
        L.append(f"| {t} | {s} | {e['pb_selected_err']:.3f} | {e['pb_best_of_n_err']:.3f} | {e['isg_enum_selected_err']:.3f} | "
                 f"{e['isg_enum_best_of_n_err']:.3f} | {e['isg_teacher_mean_err']:.3f} | {e['per_code_err_matching']:.3f} / {e['per_code_err_nonmatching']:.3f} |")
    L.append('')
    L += ['## Tokenizer (validation split, 200k)', '', '| Task | seed | perplexity | max usage | action MSE | delta MSE | collapsed | usage |', '|---|---|---|---|---|---|---|---|']
    for r in tok_rows:
        if r['step'] == TOKENIZER_SAVE_STEPS[-1]:
            L.append(f"| {r['task']} | {r['seed']} | {r['perplexity']:.2f} | {r['max_usage']:.3f} | {r['action_recon_mse']:.3f} | "
                     f"{r['delta_recon_mse']:.3f} | {r['collapsed']} | {r['usage']} |")
    L.append('')
    L += ['## Selected intention usage at 1M (Shared-I)', '', '| Task | seed | h | dominant code share | step entropy (max ln8=2.08) | switch freq |', '|---|---|---|---|---|---|']
    for (t, s, m, h), r in sorted(final.items()):
        if m == 'Shared-I':
            summ = r['_summary']
            L.append(f"| {t} | {s} | {h} | {summ['dominant_code_share']:.3f} | {summ['code_entropy_steps']:.3f} | {summ['switch_frequency']:.3f} |")
    L.append('')
    L += ['## Rule-based verdict helper', '', f'Suggested: **{verdict}** — {verdict_detail}', '',
          f'Rule: A if Shared-I − PB > {EFFECT}, Shared-I − Shuffled-I > {EFFECT} and Shared-I − I-SG > 0 in >= 75% of task x h cells; '
          f'D if Shared-I − PB < −{EFFECT} in >= 50% of cells; B if any of Shared-I − PB, I-SG − PB, Shared-I − Shuffled-I exceeds {EFFECT} somewhere; else C. '
          'The final verdict in INTENTION_PATHBRIDGER.md is written by hand from all evidence.', '']
    return '\n'.join(L)
