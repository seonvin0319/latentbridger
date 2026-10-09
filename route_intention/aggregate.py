"""Write the route-intention aggregate tables from whatever stages have finished."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from route_intention.common import EXP_ROOT, KNN_DEFAULT, LOCAL_SUBGOAL_VARIANCE_RATIO, SEEDS, TASK_ORDER, aggregate_dir
from route_intention.gates import representation_gate


def _read(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def _rows(pattern: str) -> list[dict]:
    return [_read(p) for p in sorted(EXP_ROOT.glob(pattern)) if p.is_file()]


def write_aggregate() -> Path:
    dest = aggregate_dir()
    dest.mkdir(parents=True, exist_ok=True)
    tok_rows = []
    for task in TASK_ORDER:
        for seed in SEEDS:
            metrics = _read(EXP_ROOT / 'tokenizer' / f'{task}_seed{seed}' / 'metrics' / 'step_200000.json')
            complete = _read(EXP_ROOT / 'tokenizer' / f'{task}_seed{seed}' / 'COMPLETE.json')
            if metrics is None:
                continue
            tok_rows.append(dict(
                task=task, seed=seed, perplexity=metrics.get('perplexity'), max_usage=metrics.get('max_usage'),
                collapsed=metrics.get('collapsed'), recon_mse=metrics.get('recon_mse'),
                between_endpoint=metrics.get('between_endpoint'), within_endpoint=metrics.get('within_endpoint'),
                runtime_s=None if complete is None else complete.get('runtime_s'),
            ))
    _write_csv(dest / 'tokenizer.csv', tok_rows)

    var_src = EXP_ROOT / 'diagnostics' / 'subgoal_variance.csv'
    if var_src.is_file():
        (dest / 'variance_diagnostics.csv').write_text(var_src.read_text())
    else:
        _write_csv(dest / 'variance_diagnostics.csv', [])

    oracle_rows = [r for r in _rows('oracle_subgoal/*/eval.json') if r]
    _write_csv(dest / 'oracle_subgoal.csv', oracle_rows)
    _write_csv(dest / 'bridge_consistency.csv', [r for r in _rows('conditioned/*/bridge.json') if r])
    _write_csv(dest / 'control_scores.csv', [r for r in _rows('evaluation/**/summary.json') if r])
    _write_csv(dest / 'seed_scores.csv', [r for r in _rows('evaluation/**/summary.json') if r])

    decisions = []
    lines = [
        '# Route-Level Intention PathBridger',
        '',
        'Engineering gates were fixed before these numbers were computed. See `ROUTE_INTENTION_PATHBRIDGER.md`.',
        '',
        '## A. Local-intention postmortem',
        '',
        'Seed-0 local intention did not explain subgoal choice '
        f'(variance ratios {LOCAL_SUBGOAL_VARIANCE_RATIO}).',
        'Full write-up: `exp/intention_pathbridger/aggregate/local_intention_postmortem.md`.',
        '',
        '## B. Route tokenizer',
        '',
    ]
    if not tok_rows:
        lines.append('No tokenizer has finished.')
    else:
        lines += ['| Task | seed | perplexity | max usage | collapsed | between / within endpoint |',
                  '|---|---|---|---|---|---|']
        for r in tok_rows:
            lines.append(
                f"| {r['task']} | {r['seed']} | {r['perplexity']:.2f} | {r['max_usage']:.3f} | {r['collapsed']} | "
                f"{r['between_endpoint']:.3f} / {r['within_endpoint']:.3f} |"
            )
    lines += ['', '## C. Subgoal variance ratio', '',
              '| Task | seed | Local (seed 0) | Route | k |',
              '|---|---|---|---|---|']
    var_rows = []
    if var_src.is_file():
        with var_src.open(encoding='utf-8') as f:
            var_rows = [r for r in csv.DictReader(f) if int(r['knn_k']) == KNN_DEFAULT]
    if not var_rows:
        lines.append('| — | — | — | — | — |')
    for r in var_rows:
        lines.append(
            f"| {r['task']} | {r['seed']} | {r['local_variance_ratio']} | {float(r['variance_ratio']):.3f} | {r['knn_k']} |"
        )
    lines += ['', '## D. Oracle route-c subgoal error (best-of-16, lower is better)', '']
    if not oracle_rows:
        lines.append('Not run.')
    else:
        lines += ['| Task | seed | PB best | Route-c best |', '|---|---|---|---|']
        for r in oracle_rows:
            lines.append(f"| {r['task']} | {r['seed']} | {r['pb_best_error']:.4f} | {r['route_best_error']:.4f} |")
    lines += ['', '## E. Correct-c vs shuffled-c bridge error', '',
              'Not run unless a task passes the representation gate and a route-conditioned bridge is trained.',
              '', '## F. Gate', '']
    for task in TASK_ORDER:
        ratios, ppl, usage, route_err, base_err = [], [], [], [], []
        for seed in SEEDS:
            metrics = _read(EXP_ROOT / 'tokenizer' / f'{task}_seed{seed}' / 'metrics' / 'step_200000.json')
            var = _read(EXP_ROOT / 'diagnostics' / f'{task}_seed{seed}' / 'variance.json')
            ev = _read(EXP_ROOT / 'oracle_subgoal' / f'{task}_seed{seed}' / 'eval.json')
            if metrics is None or var is None:
                continue
            ppl.append(metrics['perplexity'])
            usage.append(metrics['max_usage'])
            ratios.append(var['variance_ratio'])
            if ev:
                route_err.append(ev['route_best_error'])
                base_err.append(ev['pb_best_error'])
        if not ratios:
            decisions.append(dict(task=task, gate='PENDING'))
            lines.append(f'- {task}: PENDING')
            continue
        gate = representation_gate(
            perplexity=float(np_min_or_mean(ppl)),
            max_usage=float(max(usage)),
            variance_ratio=float(sum(ratios) / len(ratios)),
            task=task,
            route_subgoal_error=None if not route_err else float(sum(route_err) / len(route_err)),
            baseline_subgoal_error=None if not base_err else float(sum(base_err) / len(base_err)),
        )
        # A task passes only when every finished seed would pass on its own.
        per_seed = []
        for seed in SEEDS:
            metrics = _read(EXP_ROOT / 'tokenizer' / f'{task}_seed{seed}' / 'metrics' / 'step_200000.json')
            var = _read(EXP_ROOT / 'diagnostics' / f'{task}_seed{seed}' / 'variance.json')
            ev = _read(EXP_ROOT / 'oracle_subgoal' / f'{task}_seed{seed}' / 'eval.json')
            if metrics is None or var is None:
                continue
            per_seed.append(representation_gate(
                perplexity=metrics['perplexity'], max_usage=metrics['max_usage'],
                variance_ratio=var['variance_ratio'], task=task,
                route_subgoal_error=None if ev is None else ev['route_best_error'],
                baseline_subgoal_error=None if ev is None else ev['pb_best_error'],
            ))
        if per_seed and all(g == 'PASS' for g in per_seed):
            gate = 'PASS'
        elif per_seed and any(g == 'VARIANCE_FAIL' or g == 'FAIL' for g in per_seed) and not any(g == 'NEED_ORACLE' for g in per_seed):
            gate = 'FAIL' if all(g in ('FAIL', 'VARIANCE_FAIL', 'ORACLE_FAIL') for g in per_seed) else gate
        decisions.append(dict(task=task, gate=gate, per_seed=per_seed))
        lines.append(f"- {task}: **{gate}** ({', '.join(per_seed) if per_seed else 'no seeds'})")
    lines += ['', '## G. Control success', '', 'Not run. 1M control is launched only for tasks whose gate is PASS.',
              '', '## H. Paired deltas', '', 'Not run.',
              '', '## I. Conclusion', '']
    if decisions and all(d['gate'] in ('FAIL', 'VARIANCE_FAIL', 'ORACLE_FAIL') for d in decisions):
        lines.append(
            'NO SUPPORT. The route code did not pass the representation gate on any task, '
            'so no 1M control run was started.'
        )
    elif not decisions or any(d['gate'] == 'PENDING' for d in decisions):
        lines.append('Screening is incomplete.')
    elif any(d['gate'] == 'PASS' for d in decisions):
        lines.append('At least one task passed the representation gate. Control results are filled in after those runs.')
    else:
        lines.append('See the per-task gate above. Thresholds were not changed after seeing these numbers.')
    text = '\n'.join(lines) + '\n'
    (dest / 'summary.md').write_text(text)
    (dest / 'gate.json').write_text(json.dumps(decisions, indent=2) + '\n')
    return dest / 'summary.md'


def np_min_or_mean(xs):
    return sum(xs) / len(xs)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('')
        return
    fields = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
