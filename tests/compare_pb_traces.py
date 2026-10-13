"""Compare two probe_pb_update_trace pickles for bit-exact / numeric match."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def _max_abs(a, b) -> float:
    leaves_a = _leaves(a)
    leaves_b = _leaves(b)
    if len(leaves_a) != len(leaves_b):
        return float('inf')
    m = 0.0
    for x, y in zip(leaves_a, leaves_b):
        x, y = np.asarray(x), np.asarray(y)
        if x.shape != y.shape:
            return float('inf')
        if np.issubdtype(x.dtype, np.number):
            m = max(m, float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))))
        elif x.tobytes() != y.tobytes():
            return float('inf')
    return m


def _leaves(obj):
    out = []

    def rec(x):
        if isinstance(x, dict):
            for k in sorted(x.keys(), key=str):
                rec(x[k])
        elif isinstance(x, (list, tuple)):
            for y in x:
                rec(y)
        else:
            out.append(x)

    rec(obj)
    return out


def _exact(a, b) -> bool:
    return _max_abs(a, b) == 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stock', required=True)
    ap.add_argument('--ours', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--atol', type=float, default=0.0)
    args = ap.parse_args()

    with open(args.stock, 'rb') as f:
        stock = pickle.load(f)
    with open(args.ours, 'rb') as f:
        ours = pickle.load(f)

    report = {
        'stock_backend': stock['backend'],
        'ours_backend': ours['backend'],
        'stock_mode': stock.get('mode'),
        'ours_mode': ours.get('mode'),
        'steps': stock['steps'],
        'init': {},
        'updates': [],
        'pass': True,
    }

    for name in ('dynamics', 'critic', 'd_opt', 'c_opt', 'd_rng', 'c_rng'):
        mad = _max_abs(stock['init'][name], ours['init'][name])
        ok = mad <= args.atol
        report['init'][name] = {'max_abs': mad, 'exact': ok}
        report['pass'] = report['pass'] and ok

    n = min(len(stock['updates']), len(ours['updates']))
    for i in range(n):
        s, o = stock['updates'][i], ours['updates'][i]
        entry = {'step': i, 'losses': {}, 'params': {}}
        # Compare overlapping loss keys
        keys = sorted(set(s['info']) & set(o['info']))
        for k in keys:
            diff = abs(float(s['info'][k]) - float(o['info'][k]))
            entry['losses'][k] = {'stock': s['info'][k], 'ours': o['info'][k], 'abs_diff': diff}
            if diff > args.atol:
                report['pass'] = False
        for name in ('dynamics', 'critic', 'd_opt', 'c_opt', 'd_rng', 'c_rng'):
            mad = _max_abs(s['params'][name], o['params'][name])
            ok = mad <= args.atol
            entry['params'][name] = {'max_abs': mad, 'exact': ok}
            report['pass'] = report['pass'] and ok
        # EMA targets specifically
        mad_tv = _max_abs(
            s['params']['critic']['modules_target_value'],
            o['params']['critic']['modules_target_value'],
        )
        mad_tac = _max_abs(
            s['params']['critic']['modules_target_action_critic'],
            o['params']['critic']['modules_target_action_critic'],
        )
        entry['ema'] = {
            'target_value_max_abs': mad_tv,
            'target_action_critic_max_abs': mad_tac,
            'exact': mad_tv <= args.atol and mad_tac <= args.atol,
        }
        report['pass'] = report['pass'] and entry['ema']['exact']
        report['updates'].append(entry)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS' if report['pass'] else 'FAIL', '->', out)
    # Print worst diffs
    worst = []
    for u in report['updates']:
        for k, v in u['losses'].items():
            worst.append((v['abs_diff'], u['step'], k))
        for name, v in u['params'].items():
            worst.append((v['max_abs'], u['step'], f'param/{name}'))
    worst.sort(reverse=True)
    for mad, step, name in worst[:12]:
        print(f'  step{step} {name}: max_abs={mad}')
    raise SystemExit(0 if report['pass'] else 1)


if __name__ == '__main__':
    main()
