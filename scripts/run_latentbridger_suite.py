#!/usr/bin/env python3
"""Run the staged LatentBridger ablation suite for one environment.

For every (variant, seed) pair the runner trains the stages that variant needs,
then runs diagnostics and environment evaluation.  Every variant sees the same
dataset, batch size, seed, and per-stage update budget, so a difference in the
summary table is attributable to the variant's one changed knob.

A failed subprocess aborts the suite.  Partial results are never summarized as
if they were complete.

Example::

    python scripts/run_latentbridger_suite.py \\
        --config configs/latent/cube_single.py \\
        --variants sa_cl_bc,latent_rf --seeds 0 --preset smoke \\
        --save_dir exp/latentbridger_smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from agents.latentbridger import VARIANTS, VARIANT_SETTINGS  # noqa: E402

PRESETS: dict[str, dict[str, int]] = {
    'smoke': dict(
        critic_steps=2_000,
        actor_steps=2_000,
        flow_steps=2_000,
        eval_episodes=2,
        batch_size=256,
        log_interval=500,
        diagnostic_batches=4,
    ),
    'pilot': dict(
        critic_steps=100_000,
        actor_steps=100_000,
        flow_steps=100_000,
        eval_episodes=20,
        batch_size=1024,
        log_interval=5_000,
        diagnostic_batches=8,
    ),
    'full': dict(
        critic_steps=1_000_000,
        actor_steps=1_000_000,
        flow_steps=1_000_000,
        eval_episodes=50,
        batch_size=1024,
        log_interval=10_000,
        diagnostic_batches=16,
    ),
}


def stage_plan(variant: str) -> tuple[str, ...]:
    """Stages the research protocol runs for ``variant``, in order."""

    settings = VARIANT_SETTINGS[variant]
    stages: list[str] = []
    if settings['critic_type'] != 'none':
        stages.append('critic')
    stages.append('actor')
    if settings['use_flow']:
        stages.append('flow')
    return tuple(stages)


def critic_signature(variant: str) -> str | None:
    """Identify variants whose critic stage is the same computation.

    ``sa_cl``, ``sa_cl_bc``, and ``latent_rf`` differ only after the critic, so
    they must share one trained Module A rather than train three nominally
    identical copies.  Separate processes do not reproduce each other bitwise
    anyway -- XLA's GPU autotuner picks kernels by measured timing, so two
    identically-configured runs drift apart in the low-order bits -- and a
    shared checkpoint removes that confound from the ablation entirely.
    """

    settings = VARIANT_SETTINGS[variant]
    if settings['critic_type'] == 'none':
        return None
    return f'critic_{settings["critic_type"]}_actnce{settings["action_nce_coef"]:g}'


def actor_signature(variant: str) -> str:
    """Identify variants whose actor stage is the same computation.

    ``latent_rf`` is ``sa_cl_bc`` plus Module B, and ``latent_rf_actnce`` is
    ``sa_cl_bc_actnce`` plus Module B.  In both pairs the actor stage is the
    same objective trained against the same frozen critic, so the flow must be
    built on the *same* controller rather than on a separately trained copy.
    """

    settings = VARIANT_SETTINGS[variant]
    critic = critic_signature(variant) or 'nocritic'
    return (
        f'actor_{critic}_{settings["actor_goal_input"]}'
        f'_{settings["actor_objective"]}_bc{settings["actor_bc_coef"]:g}'
    )


def eval_jobs(variant: str, replan_intervals: tuple[int, ...]) -> tuple[tuple[str, int | None], ...]:
    """(mode, replan_interval) pairs to evaluate, as (name, interval)."""

    jobs: list[tuple[str, int | None]] = [('direct_goal', None)]
    if VARIANT_SETTINGS[variant]['use_flow']:
        jobs.extend(('latent_flow', interval) for interval in replan_intervals)
    return tuple(jobs)


def eval_result_name(mode: str, replan_interval: int | None) -> str:
    if replan_interval is None:
        return f'eval_{mode}.json'
    return f'eval_{mode}_r{replan_interval}.json'


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config',
        default='configs/latent/cube_single.py',
        help='LatentBridger environment config file (without a :variant suffix).',
    )
    parser.add_argument(
        '--variants',
        default=','.join(VARIANTS),
        help='Comma-separated variant names.',
    )
    parser.add_argument(
        '--replan_intervals',
        default='5',
        help='Comma-separated latent-flow replan intervals to evaluate.',
    )
    parser.add_argument('--seeds', default='0', help='Comma-separated seeds.')
    parser.add_argument(
        '--preset',
        default='smoke',
        choices=sorted(PRESETS),
        help='Update-budget preset.',
    )
    parser.add_argument('--dataset_dir', default='', help='Optional OGBench dataset directory.')
    parser.add_argument('--save_dir', default='exp/latentbridger', help='Experiment root.')
    parser.add_argument('--use_wandb', action='store_true', help='Enable W&B logging.')
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        help='Reuse finished stages and result files instead of recomputing them.',
    )
    parser.add_argument(
        '--diagnostic_split',
        default='val',
        choices=('val', 'train'),
        help='Dataset split used by the offline diagnostics.',
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Print the command plan without running anything.',
    )
    return parser.parse_args()


def _run(command: list[str], *, log_path: Path, dry_run: bool) -> None:
    printable = ' '.join(command)
    print(f'\n[latentbridger] {printable}', flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as file:
        file.write(f'{printable}\n')
    if dry_run:
        return
    result = subprocess.run(command, cwd=str(_REPO_ROOT), check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f'LatentBridger step failed with exit code {result.returncode}:\n  {printable}'
        )


def _final_checkpoint(stage_dir: Path, steps: int) -> Path:
    return stage_dir / 'checkpoints' / f'params_{steps}.pkl'


def main() -> int:
    args = _parse_args()
    preset = PRESETS[args.preset]

    config_path = args.config
    if ':' in Path(config_path).name:
        raise SystemExit('--config must not carry a :variant suffix; use --variants.')
    if not (_REPO_ROOT / config_path).is_file():
        raise SystemExit(f'Config file not found: {config_path}')

    variants = [name.strip() for name in args.variants.split(',') if name.strip()]
    unknown = [name for name in variants if name not in VARIANTS]
    if unknown:
        raise SystemExit(f'Unknown variants {unknown}; expected a subset of {VARIANTS}.')
    seeds = [int(value) for value in args.seeds.split(',') if value.strip()]
    if not seeds:
        raise SystemExit('--seeds must contain at least one seed.')
    replan_intervals = tuple(
        int(value) for value in args.replan_intervals.split(',') if value.strip()
    )
    if not replan_intervals or any(interval < 1 for interval in replan_intervals):
        raise SystemExit('--replan_intervals must be positive integers.')

    env_stem = Path(config_path).stem
    suite_root = (_REPO_ROOT / args.save_dir / env_stem).resolve()
    command_log = suite_root / 'commands.log'
    stage_steps = {
        'critic': preset['critic_steps'],
        'actor': preset['actor_steps'],
        'flow': preset['flow_steps'],
    }

    suite_started = time.time()
    completed: list[dict[str, object]] = []
    # Stage checkpoints keyed by (stage signature, seed).  Any two variants with
    # the same signature run that stage once and share the result.
    shared_stages: dict[tuple[str, str, int], Path] = {}
    for variant in variants:
        agent_flag = f'{config_path}:{variant}'
        signature = critic_signature(variant)
        signatures = {'critic': str(signature), 'actor': actor_signature(variant)}
        for seed in seeds:
            run_root = suite_root / variant / f'seed{seed}'
            results_dir = run_root / 'results'
            results_dir.mkdir(parents=True, exist_ok=True)
            previous_checkpoint: Path | None = None
            stage_sources: dict[str, str] = {}

            for stage in stage_plan(variant):
                steps = stage_steps[stage]
                stage_signature = signatures.get(stage)
                if stage_signature is not None:
                    stage_dir = (
                        suite_root / '_shared' / stage_signature / f'seed{seed}'
                    )
                    cached = shared_stages.get((stage, stage_signature, seed))
                else:
                    stage_dir = run_root / stage
                    cached = None
                checkpoint = _final_checkpoint(stage_dir, steps)

                if cached is not None:
                    print(f'[latentbridger] share {cached}', flush=True)
                    previous_checkpoint = cached
                    stage_sources[stage] = str(cached)
                    continue
                if args.skip_existing and checkpoint.is_file():
                    print(f'[latentbridger] reuse {checkpoint}', flush=True)
                    previous_checkpoint = checkpoint
                    if stage_signature is not None:
                        shared_stages[(stage, stage_signature, seed)] = checkpoint
                    stage_sources[stage] = str(checkpoint)
                    continue

                command = [
                    sys.executable,
                    'main_latent.py',
                    f'--agent={agent_flag}',
                    f'--stage={stage}',
                    f'--seed={seed}',
                    f'--train_steps={steps}',
                    f'--batch_size={preset["batch_size"]}',
                    f'--log_interval={preset["log_interval"]}',
                    f'--output_dir={stage_dir}',
                    f'--run_group={env_stem}_{variant}',
                    '--eval_interval=0',
                    '--save_interval=0',
                    f'--use_wandb={str(bool(args.use_wandb)).lower()}',
                    '--use_tqdm=true',
                ]
                if args.dataset_dir:
                    command.append(f'--dataset_dir={args.dataset_dir}')
                if previous_checkpoint is not None:
                    command.append(f'--restore_path={previous_checkpoint}')
                _run(command, log_path=command_log, dry_run=args.dry_run)
                previous_checkpoint = checkpoint
                if stage_signature is not None:
                    shared_stages[(stage, stage_signature, seed)] = checkpoint
                stage_sources[stage] = str(checkpoint)

            if previous_checkpoint is None:
                raise RuntimeError(f'Variant {variant!r} produced no checkpoint.')

            diagnostics_path = results_dir / 'diagnostics.json'
            if not (args.skip_existing and diagnostics_path.is_file()):
                command = [
                    sys.executable,
                    'diagnose_latent.py',
                    f'--agent={agent_flag}',
                    f'--checkpoint_dir={previous_checkpoint}',
                    f'--seed={seed}',
                    f'--split={args.diagnostic_split}',
                    f'--num_batches={preset["diagnostic_batches"]}',
                    f'--output_path={diagnostics_path}',
                ]
                if args.dataset_dir:
                    command.append(f'--dataset_dir={args.dataset_dir}')
                _run(command, log_path=command_log, dry_run=args.dry_run)

            for mode, interval in eval_jobs(variant, replan_intervals):
                eval_path = results_dir / eval_result_name(mode, interval)
                if args.skip_existing and eval_path.is_file():
                    continue
                command = [
                    sys.executable,
                    'evaluate_latent.py',
                    f'--agent={agent_flag}',
                    f'--checkpoint_dir={previous_checkpoint}',
                    f'--mode={mode}',
                    f'--episodes={preset["eval_episodes"]}',
                    f'--seed={seed}',
                    f'--output_path={eval_path}',
                ]
                if interval is not None:
                    command.append(f'--replan_interval={interval}')
                if args.dataset_dir:
                    command.append(f'--dataset_dir={args.dataset_dir}')
                _run(command, log_path=command_log, dry_run=args.dry_run)

            record = {
                'variant': variant,
                'seed': seed,
                'preset': args.preset,
                'config': config_path,
                'stages': list(stage_plan(variant)),
                'stage_steps': {
                    stage: stage_steps[stage] for stage in stage_plan(variant)
                },
                'batch_size': preset['batch_size'],
                'eval_episodes': preset['eval_episodes'],
                'final_checkpoint': str(previous_checkpoint),
                'results_dir': str(results_dir),
                'critic_signature': signature,
                'actor_signature': signatures['actor'],
                'stage_checkpoints': stage_sources,
                'replan_intervals': list(replan_intervals),
            }
            completed.append(record)
            if not args.dry_run:
                with (run_root / 'status.json').open('w', encoding='utf-8') as file:
                    json.dump(record, file, indent=2, sort_keys=True)
                    file.write('\n')

    if not args.dry_run:
        manifest = {
            'config': config_path,
            'preset': args.preset,
            'variants': variants,
            'seeds': seeds,
            'elapsed_seconds': time.time() - suite_started,
            'runs': completed,
        }
        with (suite_root / 'suite.json').open('w', encoding='utf-8') as file:
            json.dump(manifest, file, indent=2, sort_keys=True)
            file.write('\n')

    print(
        f'\n[latentbridger] {len(completed)} run(s) finished in '
        f'{time.time() - suite_started:.1f}s under {suite_root}',
        flush=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
