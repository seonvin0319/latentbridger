#!/usr/bin/env python3
"""Run the LatentBridger scaling sweep to 1M updates under a wall-clock budget.

The question this sweep answers is whether the 100k pilot ranking is a ranking
of *sample efficiency* or of *asymptotic performance*.  Nothing is stopped
early: a variant at 0% success at 100k still trains to 1M, because the point is
to see the curve, not to pick a winner from its first point.

Three properties make the output trustworthy.

**Shared stages.**  Variants whose critic or actor stage is literally the same
computation train it once and share the checkpoint, verified by hashing the
file each consumer reads.  ``latent_rf_sparse`` is therefore provably
``actnce_multihorizon`` plus a flow module, not a re-trained lookalike.

**Real continuation.**  An existing 100k checkpoint is extended to 1M by
restoring the optimizer, RNG, and sampler state and running 900k further
updates -- not by restarting Adam from a warm parameter vector, which is a
different optimization problem.  A resume is only allowed when the seven-field
stage fingerprint matches exactly.

**Paired evaluation.**  Every variant, seed, and checkpoint is evaluated on one
episode manifest per environment and seed, which pins the task, the initial
state (``env.np_random``), and the goal (the action space's own generator).  A
success difference is then a policy difference.

Interrupting the sweep is safe: the running stage checkpoints and exits, and
rerunning the same command resumes from where it stopped.

Example::

    python scripts/run_24h_latent_sweep.py --save_dir=exp/sweep24h --dry_run
    python scripts/run_24h_latent_sweep.py --save_dir=exp/sweep24h
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.latent_sweep_lib import (  # noqa: E402
    StageJob,
    actor_signature,
    critic_signature,
    file_digest,
    fingerprint_matches,
    latest_checkpoint,
    read_fingerprint,
    stage_fingerprint,
    stage_plan,
    stage_signature,
    write_fingerprint,
)
from utils.latent_evaluation import DEFAULT_TASK_IDS, episode_manifest  # noqa: E402

TARGET_STEP = 1_000_000
SAVE_INTERVAL = 100_000
# Checkpoints the learning curve is measured at.
EVAL_STEPS = (100_000, 300_000, 500_000, 800_000, 1_000_000)
# Diagnostics are cheap but not free; these three carry the curve's shape.
REQUIRED_DIAGNOSTIC_STEPS = (100_000, 500_000, 1_000_000)
BATCH_SIZE = 1024
LOG_INTERVAL = 10_000
EVAL_EPISODES = 50
DIAGNOSTIC_BATCHES = 16

CORE_VARIANTS = ('gcbc', 'actnce_local', 'actnce_multihorizon', 'latent_rf_sparse')
EXTRA_VARIANTS = ('sa_cl_bc_actnce', 'latent_rf_actnce')

# (phase label, env key, config file, variants).  Phases run in order; a phase
# is only started when the remaining budget can plausibly hold some of it.
PHASES = (
    ('cube_single_core', 'cube_single', 'configs/latent/cube_single.py', CORE_VARIANTS),
    ('cube_double_core', 'cube_double', 'configs/latent/cube_double.py', CORE_VARIANTS),
    ('cube_single_extra', 'cube_single', 'configs/latent/cube_single.py', EXTRA_VARIANTS),
    ('antmaze_medium_core', 'antmaze_medium', 'configs/latent/antmaze_medium.py', CORE_VARIANTS),
    ('puzzle_3x3_core', 'puzzle_3x3', 'configs/latent/puzzle_3x3.py', CORE_VARIANTS),
)

SEEDS = (0, 1, 2)
# r=1 first: it is the primary result for the sparse bridge, and r=5 is the one
# to drop if the budget runs short.
REPLAN_INTERVALS = (1, 5)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _git_commit() -> str:
    try:
        return subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def _resolve_config(config_path: str, variant: str) -> dict:
    """Resolve an agent config the way the agent does at construction time.

    ``apply_variant`` has already written every variant-owned key, so the only
    derived field left is the flow's target offsets, resolved here with the
    same rule the agent uses.
    """

    import importlib.util

    from utils.latent_datasets import sparse_prefix_offsets

    spec = importlib.util.spec_from_file_location(
        f'_latent_config_{Path(config_path).stem}', _REPO_ROOT / config_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    resolved = dict(module.get_config(variant))
    action_horizon = int(resolved['action_horizon'])
    if str(resolved['flow_target_mode']) == 'sparse':
        resolved['flow_target_offsets'] = sparse_prefix_offsets(
            int(resolved['horizon']), action_horizon
        )
    else:
        resolved['flow_target_offsets'] = tuple(range(1, action_horizon + 1))
    return resolved


# ----------------------------------------------------------------------
# Queue construction
# ----------------------------------------------------------------------
class LegacyIndex:
    """Every pre-existing stage directory, indexed by its fingerprint.

    This is how the sweep picks up the v0 and v1 pilots: a directory with a
    checkpoint and a recorded config is a candidate continuation point, and
    the fingerprint decides whether it is really the same computation.
    """

    def __init__(self, sweep_root: Path, search_roots: list[Path], min_step: int):
        self.exact: dict[tuple, tuple[Path, int]] = {}
        self.entries: list[tuple[Path, int, dict]] = []
        for root in search_roots:
            if not root.is_dir():
                continue
            for flags_path in root.rglob('flags.json'):
                stage_dir = flags_path.parent
                if sweep_root in stage_dir.parents or stage_dir == sweep_root:
                    continue
                _, step = latest_checkpoint(stage_dir)
                # The learning curve's first point is 100k, so a shorter pilot
                # (a smoke run) is not a useful continuation point and would
                # only import an optimizer state from a different regime.
                if step < min_step:
                    continue
                fingerprint = read_fingerprint(stage_dir)
                if fingerprint is None:
                    continue
                self.entries.append((stage_dir, step, fingerprint))
                key = _fingerprint_key(fingerprint)
                if key not in self.exact or step > self.exact[key][1]:
                    self.exact[key] = (stage_dir, step)

    def __len__(self) -> int:
        return len(self.exact)

    def get(self, key):
        return self.exact.get(key)

    def near_matches(self, fingerprint: dict) -> list[tuple[Path, int, list[str]]]:
        """Pilots that agree on the objective and seed but differ elsewhere."""

        near = []
        for stage_dir, step, candidate in self.entries:
            if (
                candidate['objective_signature'] != fingerprint['objective_signature']
                or int(candidate['seed']) != int(fingerprint['seed'])
            ):
                continue
            _, mismatched = fingerprint_matches(candidate, fingerprint)
            if mismatched:
                near.append((stage_dir, step, mismatched))
        return sorted(near, key=lambda entry: len(entry[2]))


def _fingerprint_key(fingerprint: dict) -> tuple:
    return (
        fingerprint['config_hash'],
        fingerprint['objective_signature'],
        int(fingerprint['repr_dim']),
        float(fingerprint['action_nce']),
        tuple(fingerprint['actor_goal_offsets']),
        tuple(fingerprint['flow_target_offsets']),
        int(fingerprint['seed']),
        int(fingerprint.get('batch_size', 1024)),
        int(fingerprint.get('parent_step', 0)),
    )


def build_stage_jobs(sweep_root: Path, phases, legacy) -> list[StageJob]:
    """One job per distinct (environment, stage, signature, seed)."""

    jobs: dict[tuple, StageJob] = {}
    ordered: list[StageJob] = []
    for _, env_key, config_path, variants in phases:
        for seed in SEEDS:
            for variant in variants:
                parent: StageJob | None = None
                for stage in stage_plan(variant):
                    signature = stage_signature(variant, stage)
                    key = (env_key, stage, signature, seed)
                    existing = jobs.get(key)
                    if existing is not None:
                        if variant not in existing.shared_by:
                            existing.shared_by.append(variant)
                        parent = existing
                        continue

                    resolved = _resolve_config(config_path, variant)
                    job = StageJob(
                        env_key=env_key,
                        config_path=config_path,
                        stage=stage,
                        signature=signature,
                        seed=seed,
                        variant=variant,
                        stage_dir=sweep_root / env_key / '_stages' / signature / f'seed{seed}',
                        target_step=TARGET_STEP,
                        fingerprint=stage_fingerprint(
                            resolved,
                            variant,
                            stage,
                            seed,
                            BATCH_SIZE,
                            0 if parent is None else TARGET_STEP,
                        ),
                        parent=parent,
                        shared_by=[variant],
                    )
                    _plan_stage(job, legacy)
                    jobs[key] = job
                    ordered.append(job)
                    parent = job
    return ordered


def _plan_stage(job: StageJob, legacy: dict) -> None:
    """Decide SKIP / RESUME / TRAIN for one stage, and from where."""

    # Work already done inside this sweep root takes precedence.
    own_fingerprint = read_fingerprint(job.stage_dir)
    _, own_step = latest_checkpoint(job.stage_dir)
    if own_step > 0 and own_fingerprint is not None:
        compatible, mismatched = fingerprint_matches(own_fingerprint, job.fingerprint)
        if not compatible:
            job.action = 'TRAIN'
            job.source = f'incompatible existing dir ({",".join(mismatched)})'
            return
        if own_step >= job.target_step:
            job.action = 'SKIP'
            job.start_step = own_step
            job.source = str(job.stage_dir / 'checkpoints' / f'params_{own_step}.pkl')
            return
        job.action = 'RESUME'
        job.start_step = own_step
        job.source = str(job.stage_dir / 'checkpoints' / f'params_{own_step}.pkl')
        return

    candidate = legacy.get(_fingerprint_key(job.fingerprint))
    if candidate is not None:
        source_dir, step = candidate
        job.action = 'REUSE' if step >= job.target_step else 'RESUME'
        job.start_step = step
        job.source = str(source_dir / 'checkpoints' / f'params_{step}.pkl')
        return

    job.action = 'TRAIN'
    job.source = ''
    # Say why a pilot that nearly matches was not continued, so the plan can be
    # audited rather than taken on trust.
    near = legacy.near_matches(job.fingerprint)
    if near:
        source_dir, step, mismatched = near[0]
        job.source = (
            f'not continued from {source_dir}@{step}: '
            f'{",".join(mismatched)} differ'
        )


def _seed_pilot(job: StageJob, log) -> None:
    """Copy a pilot checkpoint into the sweep so the pilot tree stays read-only."""

    source = Path(job.source)
    destination = job.stage_dir / 'checkpoints' / source.name
    if destination.is_file():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    for extra in ('train.csv', 'flags.json'):
        extra_path = source.parent.parent / extra
        if extra_path.is_file() and not (job.stage_dir / extra).is_file():
            shutil.copy2(extra_path, job.stage_dir / extra)
    log(
        f'  seeded {job.signature}/seed{job.seed} from {source} '
        f'(sha256:{file_digest(destination)})'
    )
    job.source = str(destination)


# ----------------------------------------------------------------------
# Evaluation and diagnostic jobs
# ----------------------------------------------------------------------
def eval_modes(variant: str) -> list[tuple[str, int | None]]:
    from agents.latentbridger import VARIANT_SETTINGS

    settings = VARIANT_SETTINGS[variant]
    jobs: list[tuple[str, int | None]] = [('direct_goal', None)]
    if settings['use_flow']:
        if settings['flow_target_mode'] == 'sparse':
            # A sparse bridge is only defined at single-step replanning.
            jobs.append(('latent_flow', 1))
        else:
            jobs.extend(('latent_flow', interval) for interval in REPLAN_INTERVALS)
    return jobs


def eval_filename(mode: str, interval: int | None, step: int) -> str:
    suffix = '' if interval is None else f'_r{interval}'
    return f'eval_{mode}{suffix}_{step}.json'


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------
class Sweep:
    def __init__(self, args):
        self.args = args
        self.sweep_root = Path(args.save_dir).resolve()
        self.sweep_root.mkdir(parents=True, exist_ok=True)
        self.log_root = self.sweep_root / 'logs'
        self.log_root.mkdir(parents=True, exist_ok=True)
        self.deadline = time.time() + args.budget_hours * 3600.0
        self.started = time.time()
        self.commit = _git_commit()
        self.manifest_path = self.sweep_root / 'run_manifest.jsonl'
        self.failed_path = self.sweep_root / 'failed_runs.json'
        self.failures: list[dict] = []
        self.stopping = False
        self.child: subprocess.Popen | None = None
        self.completed = 0
        self.skipped = 0
        self._console = (self.sweep_root / 'sweep.log').open('a', encoding='utf-8')

    # -- logging -------------------------------------------------------
    def log(self, message: str) -> None:
        line = f'[{_now()}] {message}'
        print(line, flush=True)
        self._console.write(line + '\n')
        self._console.flush()

    def record(self, entry: dict) -> None:
        with self.manifest_path.open('a', encoding='utf-8') as file:
            json.dump(entry, file, sort_keys=True)
            file.write('\n')

    # -- budget --------------------------------------------------------
    @property
    def remaining(self) -> float:
        return self.deadline - time.time()

    def have_time(self, estimate_seconds: float) -> bool:
        return not self.stopping and self.remaining > estimate_seconds

    def handle_signal(self, signal_number, _frame):
        name = signal.Signals(signal_number).name
        if self.stopping:
            self.log(f'{name} again; aborting without waiting for the child.')
            if self.child is not None:
                self.child.kill()
            return
        self.stopping = True
        self.log(f'{name} received; finishing the running stage and stopping.')
        if self.child is not None:
            # main_latent.py checkpoints on SIGTERM, so the stage is not lost.
            self.child.send_signal(signal.SIGTERM)

    # -- subprocess ----------------------------------------------------
    def run_command(self, command: list[str], log_path: Path, label: str) -> bool:
        """Run one command, retrying once, and never killing the sweep."""

        for attempt in (1, 2):
            if self.stopping:
                return False
            log_path.parent.mkdir(parents=True, exist_ok=True)
            start = time.time()
            self.log(f'START {label} (attempt {attempt})')
            with log_path.open('a', encoding='utf-8') as file:
                file.write(f'\n==== attempt {attempt} @ {_now()} ====\n')
                file.write(' '.join(command) + '\n')
                file.flush()
                self.child = subprocess.Popen(
                    command,
                    cwd=_REPO_ROOT,
                    stdout=file,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                )
                returncode = self.child.wait()
                self.child = None
            elapsed = time.time() - start
            if returncode == 0:
                self.log(f'OK    {label} in {elapsed / 60:.1f} min')
                return True
            if self.stopping:
                self.log(f'STOP  {label} interrupted after {elapsed / 60:.1f} min')
                return False
            self.log(f'FAIL  {label} rc={returncode} after {elapsed / 60:.1f} min')
            if attempt == 2:
                # A second failure is recorded and stepped over; one broken
                # run must not take the other 100 down with it.
                self.failures.append(
                    {
                        'label': label,
                        'command': command,
                        'returncode': returncode,
                        'log': str(log_path),
                        'time': _now(),
                    }
                )
                with self.failed_path.open('w', encoding='utf-8') as handle:
                    json.dump(self.failures, handle, indent=2)
                    handle.write('\n')
        return False

    # -- stage training ------------------------------------------------
    def train_stage(self, job: StageJob) -> bool:
        if job.action == 'SKIP':
            self.skipped += 1
            self.log(f'[SKIP]   {job.env_key}/{job.signature}/seed{job.seed} @ {job.start_step}')
            return True
        if job.action == 'REUSE':
            _seed_pilot(job, self.log)
            self.skipped += 1
            self.log(f'[REUSE]  {job.env_key}/{job.signature}/seed{job.seed} @ {job.start_step}')
            return True

        write_fingerprint(job.stage_dir, job.fingerprint)
        if job.action == 'RESUME' and not Path(job.source).is_relative_to(self.sweep_root):
            _seed_pilot(job, self.log)

        remaining_updates = job.target_step - job.start_step
        estimate = remaining_updates * self.args.seconds_per_update + 180.0
        if not self.have_time(estimate):
            self.log(
                f'[DEFER]  {job.env_key}/{job.signature}/seed{job.seed}: '
                f'needs ~{estimate / 60:.0f} min, {self.remaining / 60:.0f} min left'
            )
            return False

        label = f'train {job.env_key}/{job.stage}/{job.signature}/seed{job.seed}'
        command = [
            sys.executable,
            'main_latent.py',
            f'--agent={job.config_path}:{job.variant}',
            f'--stage={job.stage}',
            f'--seed={job.seed}',
            f'--train_steps={job.target_step}',
            f'--batch_size={BATCH_SIZE}',
            f'--log_interval={LOG_INTERVAL}',
            f'--save_interval={SAVE_INTERVAL}',
            f'--output_dir={job.stage_dir}',
            f'--run_group={job.env_key}_{job.signature}',
            '--eval_interval=0',
            '--use_wandb=false',
            '--use_tqdm=false',
        ]
        if self.args.dataset_dir:
            command.append(f'--dataset_dir={self.args.dataset_dir}')
        if job.action == 'RESUME':
            command.extend(
                [f'--resume_path={job.stage_dir / "checkpoints"}', f'--resume_step={job.start_step}']
            )
        elif job.parent is not None:
            parent_checkpoint = job.parent.stage_dir / 'checkpoints' / f'params_{job.target_step}.pkl'
            if not parent_checkpoint.is_file():
                self.log(f'[BLOCK]  {label}: parent checkpoint {parent_checkpoint} missing')
                return False
            command.extend(
                [f'--restore_path={parent_checkpoint}', f'--restore_step={job.target_step}']
            )

        log_path = self.log_root / job.env_key / f'{job.signature}_seed{job.seed}_{job.stage}.log'
        start = _now()
        ok = self.run_command(command, log_path, label)
        final = job.final_checkpoint
        self.record(
            {
                'kind': 'train',
                'env': job.env_key,
                'variant': job.variant,
                'shared_by': job.shared_by,
                'seed': job.seed,
                'stage': job.stage,
                'signature': job.signature,
                'action': job.action,
                'start_step': job.start_step,
                'target_step': job.target_step,
                'start_time': start,
                'end_time': _now(),
                'checkpoint': str(final) if final.is_file() else '',
                'checkpoint_sha256': file_digest(final) if final.is_file() else '',
                'config_hash': job.fingerprint['config_hash'],
                'git_commit': self.commit,
                'status': 'completed' if (ok and final.is_file()) else 'failed',
                'log': str(log_path),
            }
        )
        if ok and final.is_file():
            self.completed += 1
            return True
        return False

    # -- evaluation ----------------------------------------------------
    def manifest_for(self, env_key: str, seed: int, episodes: int) -> Path:
        path = self.sweep_root / env_key / 'manifests' / f'seed{seed}_ep{episodes}.json'
        if not path.is_file():
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'seed': int(seed),
                'task_ids': list(DEFAULT_TASK_IDS),
                'episodes_per_task': int(episodes),
                'episodes': episode_manifest(DEFAULT_TASK_IDS, episodes, seed),
            }
            with path.open('w', encoding='utf-8') as file:
                json.dump(payload, file, indent=2)
                file.write('\n')
        return path

    def evaluate_variant(
        self,
        env_key: str,
        config_path: str,
        variant: str,
        seed: int,
        final_job: StageJob,
        episodes: int = EVAL_EPISODES,
        steps: tuple[int, ...] = EVAL_STEPS,
    ) -> None:
        results_dir = self.sweep_root / env_key / variant / f'seed{seed}' / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = final_job.stage_dir / 'checkpoints'
        manifest = self.manifest_for(env_key, seed, episodes)

        for step in steps:
            checkpoint = checkpoint_dir / f'params_{step}.pkl'
            if not checkpoint.is_file():
                continue
            for mode, interval in eval_modes(variant):
                name = eval_filename(mode, interval, step)
                if episodes != EVAL_EPISODES:
                    name = name.replace('.json', f'_ep{episodes}.json')
                output = results_dir / name
                if output.is_file():
                    continue
                estimate = self.args.seconds_per_episode * episodes * 5 + 60.0
                if not self.have_time(estimate):
                    self.log(f'[DEFER]  eval {variant}/seed{seed}@{step}: out of budget')
                    return
                command = [
                    sys.executable,
                    'evaluate_latent.py',
                    f'--agent={config_path}:{variant}',
                    f'--checkpoint_dir={checkpoint_dir}',
                    f'--checkpoint_step={step}',
                    f'--mode={mode}',
                    f'--episodes={episodes}',
                    f'--seed={seed}',
                    f'--manifest_path={manifest}',
                    f'--output_path={output}',
                ]
                if interval is not None:
                    command.append(f'--replan_interval={interval}')
                if self.args.dataset_dir:
                    command.append(f'--dataset_dir={self.args.dataset_dir}')
                label = f'eval {env_key}/{variant}/seed{seed}@{step} {mode}'
                if interval is not None:
                    label += f' r={interval}'
                log_path = self.log_root / env_key / f'{variant}_seed{seed}_eval.log'
                start = _now()
                ok = self.run_command(command, log_path, label)
                self.record(
                    {
                        'kind': 'eval',
                        'env': env_key,
                        'variant': variant,
                        'seed': seed,
                        'stage': final_job.stage,
                        'step': step,
                        'mode': mode,
                        'replan_interval': interval,
                        'episodes_per_task': episodes,
                        'manifest': str(manifest),
                        'start_time': start,
                        'end_time': _now(),
                        'checkpoint': str(checkpoint),
                        'checkpoint_sha256': file_digest(checkpoint),
                        'config_hash': final_job.fingerprint['config_hash'],
                        'git_commit': self.commit,
                        'status': 'completed' if ok else 'failed',
                        'log': str(log_path),
                    }
                )

    def diagnose_variant(
        self,
        env_key: str,
        config_path: str,
        variant: str,
        seed: int,
        final_job: StageJob,
    ) -> None:
        results_dir = self.sweep_root / env_key / variant / f'seed{seed}' / 'results'
        results_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = final_job.stage_dir / 'checkpoints'
        for step in EVAL_STEPS:
            checkpoint = checkpoint_dir / f'params_{step}.pkl'
            if not checkpoint.is_file():
                continue
            output = results_dir / f'diagnostics_{step}.json'
            if output.is_file():
                continue
            required = step in REQUIRED_DIAGNOSTIC_STEPS
            estimate = self.args.seconds_per_diagnostic
            if not self.have_time(estimate if required else estimate * 2):
                if not required:
                    continue
                self.log(f'[DEFER]  diag {variant}/seed{seed}@{step}: out of budget')
                return
            command = [
                sys.executable,
                'diagnose_latent.py',
                f'--agent={config_path}:{variant}',
                f'--checkpoint_dir={checkpoint_dir}',
                f'--checkpoint_step={step}',
                f'--seed={seed}',
                f'--split={self.args.diagnostic_split}',
                f'--num_batches={DIAGNOSTIC_BATCHES}',
                f'--output_path={output}',
            ]
            if self.args.dataset_dir:
                command.append(f'--dataset_dir={self.args.dataset_dir}')
            log_path = self.log_root / env_key / f'{variant}_seed{seed}_diag.log'
            start = _now()
            ok = self.run_command(command, log_path, f'diag {env_key}/{variant}/seed{seed}@{step}')
            self.record(
                {
                    'kind': 'diagnostics',
                    'env': env_key,
                    'variant': variant,
                    'seed': seed,
                    'stage': final_job.stage,
                    'step': step,
                    'start_time': start,
                    'end_time': _now(),
                    'checkpoint': str(checkpoint),
                    'config_hash': final_job.fingerprint['config_hash'],
                    'git_commit': self.commit,
                    'status': 'completed' if ok else 'failed',
                    'log': str(log_path),
                }
            )


def _print_plan(sweep: Sweep, jobs: list[StageJob]) -> None:
    sweep.log('=' * 78)
    sweep.log('RESUME PLAN')
    sweep.log('=' * 78)
    by_env: dict[str, list[StageJob]] = {}
    for job in jobs:
        by_env.setdefault(job.env_key, []).append(job)
    for env_key, env_jobs in by_env.items():
        sweep.log(f'-- {env_key}')
        for job in env_jobs:
            shared = ','.join(job.shared_by)
            if job.action == 'RESUME':
                detail = f'{job.start_step // 1000}k -> {job.target_step // 1000}k'
            elif job.action in ('REUSE', 'SKIP'):
                detail = f'@ {job.start_step // 1000}k'
            else:
                detail = f'0 -> {job.target_step // 1000}k'
            sweep.log(
                f'   [{job.action:<6}] {job.stage:<6} seed{job.seed} '
                f'{job.signature:<58} {detail:<16} shared_by={shared}'
            )
            if job.source:
                prefix = 'source' if job.action != 'TRAIN' else 'note  '
                sweep.log(f'            {prefix}: {job.source}')
    sweep.log('=' * 78)

    # A stage must never be planned twice, and a dependent stage must have its
    # parent in the queue; either would mean duplicated or misrooted training.
    seen = set()
    for job in jobs:
        assert job.key not in seen, f'Duplicate stage job {job.key}'
        seen.add(job.key)
        if job.parent is not None:
            assert job.parent.key in seen, f'{job.key} precedes its parent'
    sweep.log(f'{len(jobs)} distinct stages, no duplicates, dependencies ordered.')


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--save_dir', default='exp/sweep24h')
    parser.add_argument('--dataset_dir', default='')
    parser.add_argument('--budget_hours', type=float, default=24.0)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--diagnostic_split', default='val')
    parser.add_argument(
        '--phases',
        default='',
        help='Comma-separated phase labels; defaults to all, in priority order.',
    )
    parser.add_argument(
        '--min_reuse_step',
        type=int,
        default=EVAL_STEPS[0],
        help='Ignore pilot checkpoints shorter than the first curve point.',
    )
    parser.add_argument('--seconds_per_update', type=float, default=0.0012)
    parser.add_argument('--seconds_per_episode', type=float, default=0.25)
    parser.add_argument('--seconds_per_diagnostic', type=float, default=120.0)
    parser.add_argument(
        '--final_episodes',
        type=int,
        default=100,
        help='Extra episodes/task for the 1M checkpoint, budget permitting.',
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    phases = PHASES
    if args.phases:
        wanted = [label.strip() for label in args.phases.split(',') if label.strip()]
        phases = tuple(phase for phase in PHASES if phase[0] in wanted)
        if not phases:
            raise SystemExit(f'No phases matched {wanted}.')

    sweep = Sweep(args)
    sweep.log(f'git commit {sweep.commit}')
    sweep.log(f'budget {args.budget_hours} h, deadline {datetime.fromtimestamp(sweep.deadline)}')

    search_roots = [_REPO_ROOT / 'exp']
    legacy = LegacyIndex(sweep.sweep_root, search_roots, args.min_reuse_step)
    sweep.log(f'{len(legacy)} reusable pilot stages indexed under exp/')

    jobs = build_stage_jobs(sweep.sweep_root, phases, legacy)
    _print_plan(sweep, jobs)
    if args.dry_run:
        plan_path = sweep.sweep_root / 'resume_plan.json'
        with plan_path.open('w', encoding='utf-8') as file:
            json.dump(
                [
                    {
                        'env': job.env_key,
                        'stage': job.stage,
                        'signature': job.signature,
                        'seed': job.seed,
                        'action': job.action,
                        'start_step': job.start_step,
                        'target_step': job.target_step,
                        'source': job.source,
                        'shared_by': job.shared_by,
                        'fingerprint': job.fingerprint,
                    }
                    for job in jobs
                ],
                file,
                indent=2,
                sort_keys=True,
            )
            file.write('\n')
        sweep.log(f'Dry run only; plan written to {plan_path}')
        return 0

    for signal_number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signal_number, sweep.handle_signal)

    jobs_by_key = {job.key: job for job in jobs}
    for label, env_key, config_path, variants in phases:
        if sweep.stopping:
            break
        sweep.log(f'### phase {label} ({sweep.remaining / 3600:.1f} h left)')
        for seed in SEEDS:
            if sweep.stopping:
                break
            # Train every stage this (env, seed) needs, parents first.
            for job in jobs:
                if (job.env_key, job.seed) != (env_key, seed):
                    continue
                if not any(variant in job.shared_by for variant in variants):
                    continue
                sweep.train_stage(job)

            # Then measure, so an (env, seed) unit is complete before moving on.
            for variant in variants:
                if sweep.stopping:
                    break
                stages = stage_plan(variant)
                final_key = (env_key, stages[-1], stage_signature(variant, stages[-1]), seed)
                final_job = jobs_by_key.get(final_key)
                if final_job is None:
                    continue
                sweep.evaluate_variant(env_key, config_path, variant, seed, final_job)
                sweep.diagnose_variant(env_key, config_path, variant, seed, final_job)

    # The denser 1M evaluation is a luxury: only run it once everything that
    # the learning curve needs is already on disk.
    if not sweep.stopping and args.final_episodes > EVAL_EPISODES:
        for label, env_key, config_path, variants in phases:
            for seed in SEEDS:
                for variant in variants:
                    if sweep.stopping:
                        break
                    stages = stage_plan(variant)
                    final_key = (env_key, stages[-1], stage_signature(variant, stages[-1]), seed)
                    final_job = jobs_by_key.get(final_key)
                    if final_job is None:
                        continue
                    sweep.evaluate_variant(
                        env_key,
                        config_path,
                        variant,
                        seed,
                        final_job,
                        episodes=args.final_episodes,
                        steps=(TARGET_STEP,),
                    )

    elapsed = (time.time() - sweep.started) / 3600.0
    sweep.log(
        f'sweep finished after {elapsed:.2f} h: {sweep.completed} stages trained, '
        f'{sweep.skipped} reused, {len(sweep.failures)} failed runs'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
