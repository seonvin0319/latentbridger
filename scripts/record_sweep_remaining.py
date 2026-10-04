#!/usr/bin/env python3
"""Record what a stopped sweep still had queued.

Writes ``unfinished_queue.json`` next to the sweep's results so a later run --
or a reader asking "how far did this get?" -- can tell completed work from
work that was never started, without re-deriving the plan.

Example::

    python scripts/record_sweep_remaining.py --sweep_dir=exp/sweep24h
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep_dir', default='exp/sweep24h')
    args = parser.parse_args()

    sweep_root = Path(args.sweep_dir).resolve()
    plan_path = sweep_root / 'resume_plan.json'
    manifest_path = sweep_root / 'run_manifest.jsonl'
    if not plan_path.is_file():
        raise SystemExit(f'No resume_plan.json under {sweep_root}.')

    with plan_path.open('r', encoding='utf-8') as file:
        plan = json.load(file)
    records = []
    if manifest_path.is_file():
        with manifest_path.open('r', encoding='utf-8') as file:
            records = [json.loads(line) for line in file if line.strip()]

    trained = {
        (record['env'], record['stage'], record['signature'], record['seed'])
        for record in records
        if record['kind'] == 'train' and record['status'] == 'completed'
    }
    evaluated = defaultdict(set)
    diagnosed = defaultdict(set)
    for record in records:
        if record['status'] != 'completed':
            continue
        key = (record['env'], record.get('variant', ''), record['seed'])
        if record['kind'] == 'eval':
            evaluated[key].add((record['step'], record['mode'], record.get('replan_interval')))
        elif record['kind'] == 'diagnostics':
            diagnosed[key].add(record['step'])

    remaining_stages = [
        job
        for job in plan
        if (job['env'], job['stage'], job['signature'], job['seed']) not in trained
    ]
    by_env: dict[str, int] = defaultdict(int)
    for job in remaining_stages:
        by_env[job['env']] += 1

    payload = {
        'sweep_dir': str(sweep_root),
        'planned_stages': len(plan),
        'trained_stages': len(trained),
        'remaining_stages': len(remaining_stages),
        'remaining_stages_by_env': dict(sorted(by_env.items())),
        'completed_runs_by_kind': {
            kind: sum(
                1
                for record in records
                if record['kind'] == kind and record['status'] == 'completed'
            )
            for kind in ('train', 'eval', 'diagnostics')
        },
        'failed_runs': sum(1 for record in records if record['status'] == 'failed'),
        'evaluated_checkpoints': {
            f'{env}/{variant}/seed{seed}': sorted(step for step, _, _ in entries)
            for (env, variant, seed), entries in sorted(evaluated.items())
        },
        'diagnosed_checkpoints': {
            f'{env}/{variant}/seed{seed}': sorted(steps)
            for (env, variant, seed), steps in sorted(diagnosed.items())
        },
        'remaining_stage_jobs': [
            {
                'env': job['env'],
                'stage': job['stage'],
                'signature': job['signature'],
                'seed': job['seed'],
                'action': job['action'],
                'start_step': job['start_step'],
                'target_step': job['target_step'],
                'shared_by': job['shared_by'],
            }
            for job in remaining_stages
        ],
    }

    output = sweep_root / 'unfinished_queue.json'
    with output.open('w', encoding='utf-8') as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write('\n')
    print(
        f'{payload["trained_stages"]}/{payload["planned_stages"]} stages trained; '
        f'{payload["remaining_stages"]} never started -> {output}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
