"""Joint calibrated CPB training with deterministic checkpoint continuation."""
import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('MUJOCO_GL', 'egl')
import jax
import numpy as np
from agents.contrastive_pathbridger import ContrastivePathBridgerAgent
from envs.env_utils import make_env_and_datasets
from utils.datasets import PathBridgerDataset
from utils.flax_utils import save_agent, restore_agent, resolve_checkpoint
from utils.contrastive_pathbridger_evaluation import evaluate
from utils.cpb_diagnostics import diagnostics
from utils.cpb_reference_bank import make_reference_goal_bank, checkpoint_reference_bank

CHECKPOINTS = (100000, 300000, 500000, 800000, 1000000)
EXECUTE_H = (5, 2, 1)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def config_for(env, variant):
    path = Path(env)
    name = path.stem if path.suffix == '.py' else env
    if name not in ('cube_single', 'cube_double', 'puzzle_3x3', 'antmaze_medium', 'antmaze_large', 'humanoid_medium', 'humanoid_large'):
        raise ValueError(f'Unknown CPB config: {env}')
    config = importlib.import_module('configs.cpb.' + name).get_config()
    config.variant = variant
    return config


def latest_checkpoint(run_dir):
    paths = list((Path(run_dir) / 'checkpoints').glob('params_*.pkl'))
    return max(paths, key=lambda p: int(p.stem.split('_')[-1])) if paths else None


def finite_metrics(info):
    return {k: float(np.asarray(v)) for k, v in info.items()}


def run(args):
    config = config_for(args.env, args.variant)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    existing = run_dir / 'config.json'
    if existing.exists():
        saved = json.loads(existing.read_text())
        if saved['agent'] != json.loads(json.dumps(config.to_dict())):
            raise ValueError('Resume config does not match saved calibrated run')
        for key in ('seed', 'batch_size', 'dataset_dir', 'episodes'):
            if saved['runtime'][key] != getattr(args, key):
                raise ValueError(f'Resume must preserve {key}')
    else:
        write_json(existing, dict(agent=config.to_dict(), runtime=vars(args)))
    restore_path = args.restore or (str(latest_checkpoint(run_dir) or '') if args.resume else '')
    if latest_checkpoint(run_dir) and not restore_path:
        raise ValueError('Existing checkpoint: use --resume instead of overwriting')
    random.seed(args.seed)
    np.random.seed(args.seed)
    env, train, val = make_env_and_datasets(config.env_name, dataset_dir=args.dataset_dir or None)
    dataset, validation = PathBridgerDataset(train, config), PathBridgerDataset(val, config)
    bank = None
    if args.variant != 'pathbridger_original':
        bank = checkpoint_reference_bank(restore_path) if restore_path else make_reference_goal_bank(dataset, args.seed, config.reference_bank_size)
    example = dataset.sample(1)
    agent = ContrastivePathBridgerAgent.create(args.seed, example['observations'], example['actions'], config, reference_goal_bank=bank)
    start = 0
    if restore_path:
        agent = restore_agent(agent, restore_path)
        start = int(agent.network.step) - 1
        if start != resolve_checkpoint(restore_path)[1]:
            raise ValueError('Checkpoint counter does not match filename')
    final_step = min(args.steps, args.stop_after or args.steps)
    if start > final_step:
        raise ValueError('Restored update is beyond requested stopping point')
    git = shutil.which('git') or str(Path.home() / 'miniconda3/bin/git')
    provenance = dict(commit=subprocess.check_output([git, 'rev-parse', 'HEAD'], text=True).strip(),
        packages={name: importlib.metadata.version(name) for name in ('jax','flax','optax','mujoco','ogbench','gymnasium')},
        devices=[str(device) for device in jax.devices()], restored_from=restore_path,
        reference_bank_sha256=hashlib.sha256(np.asarray(bank).tobytes()).hexdigest() if bank is not None else None)
    write_json(run_dir / f'provenance_from_{start}.json', provenance)
    manifest = [dict(task_id=task, episode=episode, env_seed=args.seed*1000000+task*10000+episode)
                for task in (1,2,3,4,5) for episode in range(args.episodes)]
    write_json(run_dir / 'evaluation_manifest.json', manifest)
    log_path = run_dir / 'train.jsonl'
    history = []
    if log_path.exists():
        history = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        history = [row for row in history if row['step'] <= start]
        log_path.write_text(''.join(json.dumps(row)+'\n' for row in history))
    previous_wall = history[-1]['wall_seconds'] if history else 0.
    for marker in ('paused.json', 'complete.json'):
        if (run_dir / marker).exists():
            record = json.loads((run_dir / marker).read_text())
            if record['steps'] == start:
                previous_wall = max(previous_wall, record['wall_seconds'])
    started = time.time()
    metrics = history[-1] if history else {}
    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(project='ContrastivePathBridger', id=hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:16],
                              resume='allow', dir=str(run_dir), config=config.to_dict())

    def process_checkpoint(agent, step):
        state = np.random.get_state()
        python_state = random.getstate()
        try:
            if args.variant != 'pathbridger_original':
                agent = agent.with_reference_cache()
            diag_path = run_dir / f'diagnostics_{step}.json'
            vb = validation.sample(128)
            if not diag_path.exists():
                _, heldout = agent.value_loss(vb, agent.network.params)
                diag = finite_metrics(heldout)
                if args.variant != 'pathbridger_original':
                    diag.update(diagnostics(agent, vb, train['observations'], env.action_space.low, env.action_space.high))
                diag.update({f'training/{k}': v for k,v in metrics.items() if k != 'step'})
                diag.update(step=step, **{'critic/batch_size':len(vb['observations'])})
                write_json(diag_path, diag)
            else:
                diag = json.loads(diag_path.read_text())
            if not all(np.isfinite(v) for v in diag.values()):
                raise FloatingPointError('Nonfinite checkpoint diagnostic')
            if args.smoke:
                result = evaluate(agent, env, episodes_per_task=1, execute_h=5,
                                  num_candidates=config.eval_num_candidates, temperature=config.eval_temperature, seed=args.seed)
                write_json(run_dir / 'smoke_evaluation.json', result)
            else:
                for h in EXECUTE_H:
                    path = run_dir / f'evaluation_{step}_h{h}.json'
                    if path.exists():
                        continue
                    result = evaluate(agent, env, episodes_per_task=args.episodes, num_candidates=config.eval_num_candidates,
                                      temperature=config.eval_temperature, seed=args.seed, execute_h=h)
                    result.update(env=config.env_name, variant=args.variant, seed=args.seed, checkpoint=step, calibration=config.calibration)
                    write_json(path, result)
            if step == 100000 and args.variant == 'cpb_full' and args.env == 'cube_single':
                checks = dict(recall_above_chance=diag['critic/recall_at_1'] > 1/diag['critic/batch_size'],
                              phi_not_collapsed=diag['phi/effective_rank'] > 2., psi_not_collapsed=diag['psi/effective_rank'] > 2.,
                              weights_not_all_capped=diag['heldout/progress/weight_mean'] < 5.,
                              calibrated_delta_finite=bool(np.isfinite(diag['calibration/calibrated_delta_std'])))
                write_json(run_dir / 'sanity_100000.json', checks)
                if not all(checks.values()):
                    raise RuntimeError(f'100k representation/sampling sanity failure: {checks}')
            return agent
        finally:
            np.random.set_state(state)
            random.setstate(python_state)

    # Finish interrupted checkpoint diagnostics/evaluations before any more training.
    if start:
        agent = process_checkpoint(agent, start)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool, log_path.open('a') as log:
            future = pool.submit(dataset.sample, args.batch_size) if start < final_step else None
            for step in range(start+1, final_step+1):
                batch = future.result()
                save = step in CHECKPOINTS or step == final_step
                if not save:
                    future = pool.submit(dataset.sample, args.batch_size)
                agent, info = agent.update(batch)
                if not bool(np.asarray(jnp_finite(info))):
                    raise FloatingPointError(f'Nonfinite training metric at update {step}')
                if step % args.log_interval == 0 or save:
                    metrics = finite_metrics(info)
                    metrics.update(step=step, wall_seconds=previous_wall+time.time()-started)
                    log.write(json.dumps(metrics,allow_nan=False)+'\n');log.flush()
                    print(json.dumps(metrics),flush=True)
                    if wandb_run is not None:
                        wandb_run.log(metrics,step=step)
                if save:
                    checkpoint = save_agent(agent,run_dir/'checkpoints',step)
                    restored = restore_agent(agent,checkpoint)
                    for x,y in zip(jax.tree_util.tree_leaves(agent),jax.tree_util.tree_leaves(restored)):
                        np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
                    agent = process_checkpoint(agent,step)
                    if step < final_step:
                        future = pool.submit(dataset.sample,args.batch_size)
        name = 'complete.json' if final_step == args.steps else 'paused.json'
        write_json(run_dir/name,dict(steps=final_step,wall_seconds=previous_wall+time.time()-started,smoke=args.smoke,
                                    calibration=config.calibration,reference_bank_sha256=provenance['reference_bank_sha256']))
    finally:
        env.close()
        if wandb_run is not None:
            wandb_run.finish()


@jax.jit
def jnp_finite(info):
    import jax.numpy as jnp
    return jnp.all(jnp.stack([jnp.isfinite(x) for x in info.values()]))


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--env',default='cube_single')
    p.add_argument('--variant',choices=('cpb_full','cpb_rank_only','pathbridger_original'),default='cpb_full')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--steps','--train_steps',type=int,default=1000000)
    p.add_argument('--stop_after',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=1024)
    p.add_argument('--episodes',type=int,default=50)
    p.add_argument('--log-interval',type=int,default=1000)
    p.add_argument('--run-dir','--save_dir',required=True)
    p.add_argument('--dataset_dir',default='')
    p.add_argument('--restore',default='')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--use_wandb',action='store_true')
    p.add_argument('--smoke',action='store_true')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.steps<1 or args.batch_size<2 or args.episodes<1:
        raise ValueError('Invalid run size')
    run(args)
