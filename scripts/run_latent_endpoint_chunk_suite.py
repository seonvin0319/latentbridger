#!/usr/bin/env python3
"""Run staged latent endpoint chunk experiments and paired evaluations."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.latent_endpoint_chunk import VARIANTS, VARIANT_SETTINGS  # noqa: E402
from utils.latent_endpoint_chunk_evaluation import (  # noqa: E402
    INFERENCE_MODES,
    episode_manifest,
)

MILESTONES = (100_000, 300_000, 500_000, 800_000, 1_000_000)
CUBE_CONFIG = 'configs/latent_endpoint_chunk/cube_single.py'
POST_CUBE_CONFIGS = (
    'configs/latent_endpoint_chunk/cube_double.py',
    'configs/latent_endpoint_chunk/puzzle_3x3.py',
    'configs/latent_endpoint_chunk/antmaze_medium.py',
)


def latest_checkpoint(directory: Path, limit: int) -> tuple[Path | None, int]:
    best, step = None, 0
    for path in (directory / 'checkpoints').glob('params_*.pkl'):
        try:
            candidate = int(path.stem.removeprefix('params_'))
        except ValueError:
            continue
        if step < candidate <= int(limit):
            best, step = path, candidate
    return best, step


def run_with_retry(command: list[str], log_path: Path) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in (1, 2):
        with log_path.open('a', encoding='utf-8') as log:
            log.write(f'\n[attempt {attempt}] {" ".join(command)}\n')
            log.flush()
            result = subprocess.run(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
        if result.returncode == 0:
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', default=CUBE_CONFIG)
    parser.add_argument('--variants', default=','.join(VARIANTS))
    parser.add_argument('--seeds', default='0,1,2')
    parser.add_argument('--train_steps', type=int, default=1_000_000)
    parser.add_argument('--save_dir', default='exp/latent_endpoint_chunk')
    parser.add_argument('--dataset_dir', default='')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--eval_episodes', type=int, default=50)
    parser.add_argument('--diagnostic_batches', type=int, default=16)
    parser.add_argument('--skip_eval', action='store_true')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument(
        '--promote_after_cube_single',
        action='store_true',
        help='After a sane cube-single result, continue to cube-double, puzzle, and AntMaze.',
    )
    return parser.parse_args()


def _milestones(limit: int) -> tuple[int, ...]:
    points = tuple(value for value in MILESTONES if value <= int(limit))
    return points if int(limit) in points else (*points, int(limit))


def _training_command(args, config, variant, stage, seed, output, parent=None):
    command = [
        args.python,
        str(ROOT / 'main_latent_endpoint_chunk.py'),
        f'--agent={config}:{variant}',
        f'--stage={stage}',
        f'--seed={seed}',
        f'--train_steps={args.train_steps}',
        f'--batch_size={args.batch_size}',
        f'--output_dir={output}',
        '--use_tqdm=false',
        f'--use_wandb={str(args.use_wandb).lower()}',
        '--checkpoint_steps=' + ','.join(str(value) for value in _milestones(args.train_steps)),
    ]
    if args.dataset_dir:
        command.append(f'--dataset_dir={args.dataset_dir}')
    resume_path, resume_step = latest_checkpoint(output, args.train_steps)
    if args.resume and resume_path is not None and resume_step < args.train_steps:
        command.extend([f'--resume_path={resume_path}', f'--resume_step={resume_step}'])
    elif parent is not None:
        command.append(f'--restore_path={parent}')
    return command


def _run_stage(args, failed, *, config, variant, stage, seed, output, parent=None):
    final = output / 'checkpoints' / f'params_{args.train_steps}.pkl'
    if final.is_file() and args.skip_existing:
        return final
    command = _training_command(
        args, config, variant, stage, seed, output, parent=parent
    )
    print('[endpoint-suite]', ' '.join(command), flush=True)
    if args.dry_run:
        return final
    if not run_with_retry(command, output / 'run.log') or not final.is_file():
        failed.append(
            {'config': config, 'variant': variant, 'seed': seed, 'stage': stage}
        )
        return None
    return final


def _evaluate_actor(args, failed, *, config, variant, seed, checkpoint, run_root, manifest):
    records = []
    diagnostics = run_root / f'diagnostics_{args.train_steps}.json'
    for execute_h in (1, 2, 5):
        for mode in INFERENCE_MODES:
            output = run_root / (
                f'evaluation_{args.train_steps}_{mode}_h{execute_h}.json'
            )
            command = [
                args.python,
                str(ROOT / 'evaluate_latent_endpoint_chunk.py'),
                f'--agent={config}:{variant}',
                f'--checkpoint={checkpoint}',
                f'--manifest={manifest}',
                f'--seed={seed}',
                f'--execute_h={execute_h}',
                f'--inference_mode={mode}',
                f'--output={output}',
            ]
            if mode == 'direct' and execute_h == 2:
                command.extend(
                    [
                        f'--diagnostics_output={diagnostics}',
                        f'--diagnostic_batches={args.diagnostic_batches}',
                    ]
                )
            if args.dataset_dir:
                command.append(f'--dataset_dir={args.dataset_dir}')
            print('[endpoint-suite]', ' '.join(command), flush=True)
            if not args.dry_run and not (output.is_file() and args.skip_existing):
                if not run_with_retry(command, run_root / f'eval_{mode}_h{execute_h}.log'):
                    failed.append(
                        {
                            'config': config,
                            'variant': variant,
                            'seed': seed,
                            'stage': f'eval_{mode}_h{execute_h}',
                        }
                    )
            records.append(
                {'inference_mode': mode, 'execute_h': execute_h, 'path': str(output)}
            )
    (run_root / 'run_manifest.json').write_text(
        json.dumps(
            {
                'config': config,
                'variant': variant,
                'seed': seed,
                'checkpoint': str(checkpoint),
                'paired_manifest': str(manifest),
                'paired_manifest_episodes': 250,
                'evaluations': records,
                'diagnostics': str(diagnostics),
            },
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )


def run_configs(args, configs, failed):
    variants = tuple(item.strip() for item in args.variants.split(',') if item.strip())
    seeds = tuple(int(item) for item in args.seeds.split(',') if item.strip())
    save_root = (ROOT / args.save_dir).resolve()
    save_root.mkdir(parents=True, exist_ok=True)
    for config in configs:
        env_key = Path(config).stem
        env_root = save_root / env_key
        for seed in seeds:
            manifest = env_root / f'evaluation_manifest_seed{seed}_250.json'
            if not manifest.exists():
                manifest.parent.mkdir(parents=True, exist_ok=True)
                manifest.write_text(
                    json.dumps(
                        episode_manifest(
                            episodes_per_task=args.eval_episodes, seed=seed
                        ),
                        indent=2,
                    )
                    + '\n',
                    encoding='utf-8',
                )
            rows = json.loads(manifest.read_text(encoding='utf-8'))
            if not args.skip_eval and len(rows) != 250:
                raise ValueError(
                    f'Full evaluation requires a 250-episode manifest; {manifest} has {len(rows)}.'
                )

            proposal_dir = env_root / 'shared_proposal' / f'seed{seed}'
            proposal = _run_stage(
                args,
                failed,
                config=config,
                variant='latent_endpoint_awr',
                stage='proposal',
                seed=seed,
                output=proposal_dir,
            )
            if proposal is None:
                continue
            critic_dir = env_root / 'shared_critic' / f'seed{seed}'
            critic = _run_stage(
                args,
                failed,
                config=config,
                variant='latent_endpoint_awr',
                stage='critic',
                seed=seed,
                output=critic_dir,
                parent=proposal,
            )
            if critic is None:
                continue
            for variant in variants:
                policy_type = VARIANT_SETTINGS[variant]['policy_type']
                stage = f'policy_{policy_type}'
                run_root = env_root / variant / f'seed{seed}'
                actor_dir = run_root / stage
                checkpoint = _run_stage(
                    args,
                    failed,
                    config=config,
                    variant=variant,
                    stage=stage,
                    seed=seed,
                    output=actor_dir,
                    parent=critic,
                )
                if checkpoint is None or args.skip_eval:
                    continue
                _evaluate_actor(
                    args,
                    failed,
                    config=config,
                    variant=variant,
                    seed=seed,
                    checkpoint=checkpoint,
                    run_root=run_root,
                    manifest=manifest,
                )
        if not args.dry_run:
            subprocess.run(
                [
                    args.python,
                    str(ROOT / 'scripts/summarize_latent_endpoint_chunk.py'),
                    f'--root={env_root}',
                ],
                cwd=ROOT,
                check=False,
            )


def cube_single_is_sane(root: Path, variants, seeds) -> tuple[bool, str]:
    """Require completed outputs and finite, above-chance retrieval diagnostics."""

    for variant in variants:
        for seed in seeds:
            run = root / 'cube_single' / variant / f'seed{seed}'
            diagnostics = run / 'diagnostics_1000000.json'
            evaluation = run / 'evaluation_1000000_direct_h2.json'
            if not diagnostics.is_file() or not evaluation.is_file():
                return False, f'missing diagnostics/evaluation for {variant} seed{seed}'
            values = json.loads(diagnostics.read_text(encoding='utf-8'))
            required = (
                'endpoint/recall_at_1',
                'state_goal/recall_at_1',
                'action/p_positive_gt_q',
                'policy/support_logprob',
            )
            if any(key not in values or not math.isfinite(float(values[key])) for key in required):
                return False, f'non-finite diagnostic for {variant} seed{seed}'
            if values['endpoint/recall_at_1'] <= 1.0 / 256.0:
                return False, f'endpoint retrieval is at chance for {variant} seed{seed}'
    return True, 'all cube-single checkpoints, evaluations, and diagnostics are sane'


def main():
    args = parse_args()
    variants = tuple(item.strip() for item in args.variants.split(',') if item.strip())
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise ValueError(f'Unknown variants: {sorted(unknown)}')
    seeds = tuple(int(item) for item in args.seeds.split(',') if item.strip())
    configs = tuple(item.strip() for item in args.configs.split(',') if item.strip())
    failed: list[dict[str, object]] = []
    started = time.time()
    run_configs(args, configs, failed)
    save_root = (ROOT / args.save_dir).resolve()
    if args.promote_after_cube_single and configs == (CUBE_CONFIG,) and not failed:
        sane, reason = cube_single_is_sane(save_root, variants, seeds)
        print(f'[endpoint-suite] cube-single sanity: {sane} ({reason})', flush=True)
        if sane:
            run_configs(args, POST_CUBE_CONFIGS, failed)
    (save_root / 'failed_runs.json').write_text(
        json.dumps(failed, indent=2) + '\n', encoding='utf-8'
    )
    (save_root / 'run_summary.json').write_text(
        json.dumps(
            {
                'wall_clock_seconds': time.time() - started,
                'configs': configs,
                'variants': variants,
                'seeds': seeds,
                'updates_per_stage': args.train_steps,
                'paired_evaluation_episodes': 250,
                'failed_runs': failed,
            },
            indent=2,
        )
        + '\n',
        encoding='utf-8',
    )
    if not args.dry_run:
        subprocess.run(
            [
                args.python,
                str(ROOT / 'scripts/summarize_latent_endpoint_chunk.py'),
                f'--root={save_root}',
            ],
            cwd=ROOT,
            check=False,
        )
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
