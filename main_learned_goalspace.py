"""Train and evaluate frozen learned-goalspace PathBridger agents."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MUJOCO_GL', 'egl')

import jax
import numpy as np

from envs.env_utils import make_env_and_datasets
from learned_goalspace.checkpoints import load_goal_encoder
from learned_goalspace.downstream import FrozenLearnedGoalspaceAgent, METHODS
from learned_goalspace.pretrain import VARIANT
from utils.contrastive_pathbridger_evaluation import evaluate
from utils.datasets import PathBridgerDataset
from utils.flax_utils import restore_agent, save_agent

CHECKPOINTS = (100_000, 300_000, 500_000, 800_000, 1_000_000)
EXECUTE_H = (5, 2, 1)
ENVS = ('puzzle_3x3', 'cube_double')


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def _latest(run_dir: Path) -> Path | None:
    values = list((run_dir / 'checkpoints').glob('params_*.pkl'))
    return max(values, key=lambda path: int(path.stem.split('_')[-1])) if values else None


def config_for(env: str, method: str):
    return importlib.import_module(f'configs.learned_goalspace.{env}').get_config(method)


def default_paths(env: str, method: str, seed: int) -> tuple[Path, Path]:
    pretrain = (
        Path('exp/learned_goalspace/pretrain') / env / VARIANT / f'seed{seed}' / 'checkpoints' / 'params_500000.pkl'
    )
    downstream = Path('exp/learned_goalspace/downstream') / env / method / f'seed{seed}'
    return pretrain, downstream


def _write_results(run_dir: Path) -> None:
    rows = []
    for path in sorted(run_dir.glob('evaluation_*_h*.json')):
        rows.append(json.loads(path.read_text()))
    if (run_dir / 'smoke_evaluation.json').is_file():
        rows.append(json.loads((run_dir / 'smoke_evaluation.json').read_text()))
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    output = run_dir / 'downstream_results.csv'
    temporary = output.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def _evaluation_file_valid(
    path: Path,
    step: int,
    execute_h: int,
    expected: dict | None = None,
) -> bool:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    try:
        matches_identity = int(payload.get('checkpoint', -1)) == step and int(payload.get('h', -1)) == execute_h
    except (TypeError, ValueError):
        return False
    if not matches_identity:
        return False
    return expected is None or all(payload.get(key) == value for key, value in expected.items())


def _evaluation_complete(
    run_dir: Path,
    step: int,
    *,
    smoke: bool,
    expected: dict | None = None,
) -> bool:
    if smoke:
        return _evaluation_file_valid(
            run_dir / 'smoke_evaluation.json',
            step,
            5,
            expected,
        )
    return all(
        _evaluation_file_valid(
            run_dir / f'evaluation_{step}_h{execute_h}.json',
            step,
            execute_h,
            expected,
        )
        for execute_h in EXECUTE_H
    )


def run(args) -> None:
    method = args.method.upper()
    config = config_for(args.env, method)
    default_pretrain, default_run = default_paths(args.env, method, args.seed)
    pretrain_checkpoint = Path(args.pretrain_checkpoint or default_pretrain)
    run_dir = Path(args.run_dir or default_run)
    run_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    env, train, _ = make_env_and_datasets(config.env_name, dataset_dir=args.dataset_dir or None)
    try:
        dataset = PathBridgerDataset(train, config)
        example = dataset.sample(2)
        encoder_params, encoder_metadata = load_goal_encoder(
            pretrain_checkpoint,
            env_name=config.env_name,
            obs_dim=example['observations'].shape[-1],
            step=args.pretrain_step,
            allow_nonproduction_step=args.allow_nonproduction_pretrain,
        )
        agent = FrozenLearnedGoalspaceAgent.create(
            args.seed,
            example['observations'],
            example['actions'],
            config.to_dict(),
            encoder_params,
        )
        identity = {
            'agent': config.to_dict(),
            'pretrain': encoder_metadata,
            'seed': args.seed,
            'batch_size': args.batch_size,
            'episodes': args.episodes,
            'dataset_dir': (str(Path(args.dataset_dir).resolve()) if args.dataset_dir else None),
            'steps': args.steps,
            'smoke': args.smoke,
        }
        identity = json.loads(json.dumps(identity))
        config_path = run_dir / 'config.json'
        if config_path.exists() and json.loads(config_path.read_text()) != identity:
            raise ValueError('Resume configuration does not match this downstream run.')
        _write_json(config_path, identity)

        checkpoint = _latest(run_dir)
        start = 0
        if checkpoint:
            if not args.resume:
                raise ValueError('Existing downstream checkpoint found; pass --resume.')
            agent = restore_agent(agent, checkpoint)
            start = int(agent.network.step) - 1
        final_step = min(args.steps, args.stop_after or args.steps)
        log_path = run_dir / 'train.jsonl'
        if start and log_path.exists():
            rows = []
            for line in log_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # The atomic checkpoint remains authoritative if the last
                    # JSONL write was interrupted.
                    continue
            log_path.write_text(''.join(json.dumps(row) + '\n' for row in rows if int(row['step']) <= start))
        manifest = [
            {
                'task_id': task,
                'episode': episode,
                'env_seed': args.seed * 1_000_000 + task * 10_000 + episode,
            }
            for task in (1, 2, 3, 4, 5)
            for episode in range(args.episodes)
        ]
        _write_json(run_dir / 'evaluation_manifest.json', manifest)
        evaluation_identity = {
            'env': config.env_name,
            'method': method,
            'variant': VARIANT,
            'seed': args.seed,
            'episodes_per_task': args.episodes,
            'N': int(config.eval_num_candidates),
            'temperature': float(config.eval_temperature),
        }

        def process_checkpoint(current_agent, step):
            # Evaluation must not perturb the sampler stream. A resumed run
            # restores the RNG state captured before evaluation, so an
            # uninterrupted run must continue from that same state.
            numpy_state = np.random.get_state()
            python_state = random.getstate()
            try:
                if args.smoke:
                    result = evaluate(
                        current_agent,
                        env,
                        episodes_per_task=args.episodes,
                        execute_h=5,
                        num_candidates=int(config.eval_num_candidates),
                        temperature=float(config.eval_temperature),
                        seed=args.seed,
                    )
                    result.update(
                        env=config.env_name,
                        method=method,
                        variant=VARIANT,
                        seed=args.seed,
                        checkpoint=step,
                    )
                    _write_json(run_dir / 'smoke_evaluation.json', result)
                else:
                    for execute_h in EXECUTE_H:
                        output = run_dir / f'evaluation_{step}_h{execute_h}.json'
                        if _evaluation_file_valid(
                            output,
                            step,
                            execute_h,
                            evaluation_identity,
                        ):
                            continue
                        result = evaluate(
                            current_agent,
                            env,
                            episodes_per_task=args.episodes,
                            execute_h=execute_h,
                            num_candidates=int(config.eval_num_candidates),
                            temperature=float(config.eval_temperature),
                            seed=args.seed,
                        )
                        result.update(
                            env=config.env_name,
                            method=method,
                            variant=VARIANT,
                            seed=args.seed,
                            checkpoint=step,
                        )
                        _write_json(output, result)
                _write_results(run_dir)
            finally:
                np.random.set_state(numpy_state)
                random.setstate(python_state)

        # Repair interrupted evaluations from their matching checkpoints before
        # continuing. Never evaluate an older step with the latest parameters.
        host_numpy_state = np.random.get_state()
        host_python_state = random.getstate()
        try:
            required_steps = (start,) if args.smoke and start else tuple(step for step in CHECKPOINTS if step <= start)
            for evaluation_step in required_steps:
                if _evaluation_complete(
                    run_dir,
                    evaluation_step,
                    smoke=args.smoke,
                    expected=evaluation_identity,
                ):
                    continue
                evaluation_checkpoint = run_dir / 'checkpoints' / f'params_{evaluation_step}.pkl'
                if not evaluation_checkpoint.is_file():
                    raise FileNotFoundError(
                        f'Missing checkpoint required to repair evaluation: {evaluation_checkpoint}'
                    )
                evaluation_agent = restore_agent(agent, evaluation_checkpoint)
                process_checkpoint(evaluation_agent, evaluation_step)
        finally:
            np.random.set_state(host_numpy_state)
            random.setstate(host_python_state)

        with log_path.open('a') as log:
            for step in range(start + 1, final_step + 1):
                batch = {key: jax.numpy.asarray(value) for key, value in dataset.sample(args.batch_size).items()}
                agent, info = agent.update(batch)
                save = step in CHECKPOINTS or step == final_step
                if step % args.log_interval == 0 or save:
                    row = {
                        'step': step,
                        **{key: float(np.asarray(value)) for key, value in info.items()},
                    }
                    log.write(json.dumps(row, allow_nan=False) + '\n')
                    log.flush()
                    print(json.dumps(row), flush=True)
                if save:
                    checkpoint_path = save_agent(agent, run_dir / 'checkpoints', step)
                    restored = restore_agent(agent, checkpoint_path)
                    for left, right in zip(
                        jax.tree_util.tree_leaves(agent),
                        jax.tree_util.tree_leaves(restored),
                    ):
                        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))
                    process_checkpoint(agent, step)
        marker = 'complete.json' if final_step == args.steps else 'paused.json'
        _write_json(
            run_dir / marker,
            {'steps': final_step, 'method': method, 'seed': args.seed},
        )
    finally:
        env.close()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument('--env', choices=ENVS, required=True)
    result.add_argument('--method', type=str.upper, choices=METHODS, required=True)
    result.add_argument('--seed', type=int, default=0)
    result.add_argument('--steps', type=int, default=1_000_000)
    result.add_argument('--stop-after', type=int, default=0)
    result.add_argument('--batch-size', type=int, default=1024)
    result.add_argument('--episodes', type=int, default=50)
    result.add_argument('--log-interval', type=int, default=1000)
    result.add_argument('--dataset-dir', default='')
    result.add_argument('--pretrain-checkpoint', default='')
    result.add_argument('--pretrain-step', type=int, default=500_000)
    result.add_argument('--run-dir', default='')
    result.add_argument('--resume', action='store_true')
    result.add_argument('--smoke', action='store_true')
    result.add_argument('--allow-nonproduction-pretrain', action='store_true')
    return result


if __name__ == '__main__':
    arguments = parser().parse_args()
    if arguments.steps < 1 or arguments.batch_size < 1 or arguments.episodes < 1:
        raise ValueError('Invalid run size.')
    run(arguments)
