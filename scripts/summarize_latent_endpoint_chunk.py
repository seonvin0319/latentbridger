#!/usr/bin/env python3
"""Summarize latent endpoint chunk evaluations and anti-exploitation metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    return parser.parse_args()


def _mean(values):
    return float(np.mean(values)) if values else float('nan')


def main():
    root = Path(parse_args().root).resolve()
    evaluation_rows = []
    for path in sorted(root.rglob('evaluation_*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(payload, dict) or 'inference_mode' not in payload:
            # Paired manifests deliberately share the ``evaluation_`` prefix.
            continue
        run_root = path.parent
        evaluation_rows.append(
            {
                'environment': run_root.parent.parent.name,
                'variant': run_root.parent.name,
                'seed': run_root.name.removeprefix('seed'),
                'inference_mode': payload['inference_mode'],
                'num_proposals': payload.get('num_proposals', 0),
                'execute_h': payload['execute_h'],
                'num_episodes': payload['num_episodes'],
                'success': payload['overall_success'],
                'critic_score': payload.get('planner/selected_critic_score'),
                'support_logprob': payload.get('planner/selected_support_logprob'),
                'support_distance': payload.get('planner/selected_support_distance'),
                'filtered_fraction': payload.get('planner/filtered_fraction'),
                'path': str(path),
            }
        )
    diagnostic_rows = []
    for path in sorted(root.rglob('diagnostics_*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        run_root = path.parent
        diagnostic_rows.append(
            {
                'environment': run_root.parent.parent.name,
                'variant': run_root.parent.name,
                'seed': run_root.name.removeprefix('seed'),
                **payload,
                'path': str(path),
            }
        )

    root.mkdir(parents=True, exist_ok=True)
    if evaluation_rows:
        with (root / 'evaluation_summary.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(evaluation_rows[0]))
            writer.writeheader()
            writer.writerows(evaluation_rows)
    if diagnostic_rows:
        keys = list(dict.fromkeys(key for row in diagnostic_rows for key in row))
        with (root / 'diagnostic_summary.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(diagnostic_rows)

    grouped = defaultdict(list)
    for row in evaluation_rows:
        key = (
            row['environment'],
            row['variant'],
            row['inference_mode'],
            row['num_proposals'],
            row['execute_h'],
        )
        grouped[key].append(row)
    lines = [
        '# Latent endpoint chunk report',
        '',
        'All success estimates use the same 250-episode paired manifest per seed.',
        '',
        '| environment | variant | inference | N | h | success | critic score | support logprob | support distance |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for key, rows in sorted(grouped.items()):
        environment, variant, mode, count, execute_h = key
        lines.append(
            '| {0} | {1} | {2} | {3} | {4} | {5:.4f} | {6:.4f} | {7:.4f} | {8:.4f} |'.format(
                environment,
                variant,
                mode,
                count,
                execute_h,
                _mean([float(row['success']) for row in rows]),
                _mean([float(row['critic_score']) for row in rows if row['critic_score'] is not None]),
                _mean([float(row['support_logprob']) for row in rows if row['support_logprob'] is not None]),
                _mean([float(row['support_distance']) for row in rows if row['support_distance'] is not None]),
            )
        )
    lines.extend(
        [
            '',
            'The N/score/support/success columns are reported together so rising '
            'critic score with falling support or success is visible as offline '
            'critic exploitation.',
            '',
        ]
    )
    (root / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    print(
        json.dumps(
            {
                'root': str(root),
                'evaluation_rows': len(evaluation_rows),
                'diagnostic_rows': len(diagnostic_rows),
                'report': str(root / 'SUMMARY.md'),
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
