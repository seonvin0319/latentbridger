"""Evaluate CPB checkpoints on CPU, one process per task, without blocking training.

The trainer skips a horizon when ``evaluation_<step>_h<h>.json`` already exists.
This watcher replaces those deferred placeholders once ``params_<step>.pkl`` is saved.
Action noise is keyed by (seed, task, episode, replan) so the five tasks can run together.
Environment reset seeds match the in-process evaluator.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

ROOT = Path("/home/choi/latentbridger_cpb/exp/contrastive_pathbridger")
ENVS = ("antmaze_large", "humanoid_medium", "humanoid_large")
STEPS = (100000, 300000, 500000, 800000, 1000000)
HORIZONS = (5, 2, 1)
TASKS = (1, 2, 3, 4, 5)
EPISODES = 50


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _deferred(path: Path) -> bool:
    if not path.exists():
        return True
    payload = json.loads(path.read_text())
    return bool(payload.get("deferred")) or "overall_success" not in payload


def _worker(spec: dict) -> str:
    import os
    import sys
    os.nice(10)
    sys.path.insert(0, "/home/choi/latentbridger_cpb")
    import jax.numpy as jnp
    import ogbench

    from agents.contrastive_pathbridger import ContrastivePathBridgerAgent
    from utils.contrastive_pathbridger_evaluation import evaluate
    from utils.cpb_reference_bank import checkpoint_reference_bank
    from utils.flax_utils import restore_agent

    run = Path(spec["run"])
    saved = json.loads((run / "config.json").read_text())
    config = saved["agent"]
    env = ogbench.make_env_and_datasets(config["env_name"], env_only=True)
    agent = ContrastivePathBridgerAgent.create(
        saved["runtime"]["seed"],
        jnp.zeros((1, *env.observation_space.shape)),
        jnp.zeros((1, *env.action_space.shape)),
        config,
        reference_goal_bank=checkpoint_reference_bank(run / "checkpoints", spec["checkpoint"]),
    )
    agent = restore_agent(agent, run / "checkpoints", spec["checkpoint"])
    agent = agent.with_reference_cache()
    result = evaluate(
        agent,
        env,
        task_ids=(spec["task"],),
        episodes_per_task=spec["episodes"],
        num_candidates=int(config["eval_num_candidates"]),
        temperature=float(config["eval_temperature"]),
        seed=int(saved["runtime"]["seed"]),
        execute_h=spec["h"],
        rng_mode="per_episode",
    )
    env.close()
    out = Path(spec["out"])
    _write_json(out, result)
    print(
        f"task {spec['task']} h={spec['h']} step={spec['checkpoint']} "
        f"success={result['overall_success']:.3f}",
        flush=True,
    )
    return str(out)


def _merge(run: Path, step: int, h: int, parts: list[Path]) -> None:
    saved = json.loads((run / "config.json").read_text())
    config = saved["agent"]
    by_task = {}
    for path in parts:
        payload = json.loads(path.read_text())
        by_task.update({key: value for key, value in payload.items() if key.startswith("task_")})
    rates = [float(by_task[f"task_{task}_success"]) for task in TASKS]
    success_count = 0
    for path in parts:
        success_count += int(json.loads(path.read_text())["success_count"])
    merged = {
        **{f"task_{task}_success": rates[task - 1] for task in TASKS},
        "success_count": success_count,
        "h": h,
        "N": int(config["eval_num_candidates"]),
        "temperature": float(config["eval_temperature"]),
        "overall_success": float(sum(rates) / len(rates)),
        "num_tasks": len(TASKS),
        "episodes_per_task": EPISODES,
        "env": config["env_name"],
        "variant": config["variant"],
        "seed": int(saved["runtime"]["seed"]),
        "checkpoint": step,
        "calibration": config["calibration"],
        "rng_mode": "per_episode",
        "device": "cpu",
    }
    _write_json(run / f"evaluation_{step}_h{h}.json", merged)
    print(
        f"merged {run.parent.parent.name} step={step} h={h} "
        f"success={merged['overall_success']:.3f}",
        flush=True,
    )


def _next_job() -> tuple[Path, int, int] | None:
    for name in ENVS:
        run = ROOT / name / "cpb_rank_only" / "seed0"
        for step in STEPS:
            checkpoint = run / "checkpoints" / f"params_{step}.pkl"
            if not checkpoint.is_file() or not (run / "config.json").is_file():
                continue
            for h in HORIZONS:
                if _deferred(run / f"evaluation_{step}_h{h}.json"):
                    return run, step, h
    return None


def _finished() -> bool:
    for name in ENVS:
        run = ROOT / name / "cpb_rank_only" / "seed0"
        for step in STEPS:
            for h in HORIZONS:
                if _deferred(run / f"evaluation_{step}_h{h}.json"):
                    return False
    return True


def main() -> None:
    print("cpu eval watcher started", flush=True)
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=len(TASKS), mp_context=context) as pool:
        while not _finished():
            job = _next_job()
            if job is None:
                time.sleep(5)
                continue
            run, step, h = job
            print(f"start {run.parent.parent.name} step={step} h={h}", flush=True)
            part_dir = run / "eval_parts"
            specs = []
            for task in TASKS:
                out = part_dir / f"{step}_h{h}_task{task}.json"
                if out.exists() and "overall_success" in json.loads(out.read_text()):
                    continue
                specs.append(dict(
                    run=str(run), checkpoint=step, h=h, task=task,
                    episodes=EPISODES, out=str(out),
                ))
            futures = [pool.submit(_worker, spec) for spec in specs]
            for future in as_completed(futures):
                future.result()
            _merge(run, step, h, [part_dir / f"{step}_h{h}_task{task}.json" for task in TASKS])


if __name__ == "__main__":
    main()
