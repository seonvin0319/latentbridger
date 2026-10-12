"""Common Scene h5 evaluation for six existing 1M checkpoints.

Uses the training entrypoint's evaluation function unchanged: five tasks,
50 episodes, h=5, and info['success'] on any step. Every checkpoint shares
one reset and policy-RNG stream. Results are written only under --output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MUJOCO_GL", "egl")

from agents.contrastive_transitive_distance_pathbridger import (
    ContrastiveTransitiveDistanceAgent,
)
from envs.env_utils import make_env_and_datasets
from main_ctd_pathbridger import config_for
from utils.contrastive_pathbridger_evaluation import evaluate
from utils.datasets import PathBridgerDataset
from utils.flax_utils import restore_agent


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def manifest_for(pin: dict) -> list[dict]:
    spec = pin["eval"]
    seed = int(spec["common_eval_seed"])
    rows = []
    for task_id in spec["task_ids"]:
        for episode in range(int(spec["episodes_per_task"])):
            rows.append(
                {
                    "task_id": int(task_id),
                    "episode": int(episode),
                    "env_seed": seed * 1_000_000 + int(task_id) * 10_000 + int(episode),
                    "goal_source": "info['goal'] from env.reset",
                    "policy_rng": (
                        "jax.random.PRNGKey(common_eval_seed), split once per replan, "
                        "carried across episodes by evaluate()"
                    ),
                }
            )
    return rows


def sample_mean_sd(values: list[float]) -> tuple[float, float]:
    count = len(values)
    if count < 2:
        raise ValueError("sample standard deviation needs at least two seeds")
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return mean, math.sqrt(variance)


def percent(rate: float) -> float:
    return 100.0 * float(rate)


def load_agent(pin: dict, variant: str, seed: int, checkpoint: Path):
    config = config_for(pin["env"], variant)
    if variant == pin["variant"] and json.loads(json.dumps(config.to_dict())) != pin["agent"]:
        raise RuntimeError("GSDTRL config drifted from the seed0 pin")
    env, train, _val = make_env_and_datasets(config.env_name)
    example = PathBridgerDataset(train, config).sample(2)
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    agent = ContrastiveTransitiveDistanceAgent.create(
        seed,
        example["observations"],
        example["actions"],
        config.to_dict(),
    )
    agent = restore_agent(agent, checkpoint)
    np.random.set_state(numpy_state)
    random.setstate(python_state)
    if int(agent.network.step) != pin["eval"]["checkpoint"] + 1:
        raise RuntimeError(
            f"{checkpoint} network.step={int(agent.network.step)} "
            f"is not the {pin['eval']['checkpoint']} update"
        )
    spec = pin["eval"]
    if int(config.eval_num_candidates) != int(spec["num_candidates"]):
        raise RuntimeError("eval candidate count drifted")
    if float(config.eval_temperature) != float(spec["temperature"]):
        raise RuntimeError("eval temperature drifted")
    return agent, env, config


def evaluate_run(pin: dict, variant: str, seed: int, run_dir: Path) -> dict:
    spec = pin["eval"]
    checkpoint = run_dir / "checkpoints" / f"params_{int(spec['checkpoint'])}.pkl"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    agent, env, config = load_agent(pin, variant, seed, checkpoint)
    try:
        result = evaluate(
            agent,
            env,
            task_ids=tuple(spec["task_ids"]),
            episodes_per_task=int(spec["episodes_per_task"]),
            num_candidates=int(spec["num_candidates"]),
            temperature=float(spec["temperature"]),
            seed=int(spec["common_eval_seed"]),
            execute_h=int(spec["execute_h"]),
        )
    finally:
        env.close()
    result.update(
        env=config.env_name,
        variant=variant,
        train_seed=seed,
        checkpoint=int(spec["checkpoint"]),
        method="scene_common_h5",
        common_eval_seed=int(spec["common_eval_seed"]),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        run_dir=str(run_dir),
    )
    expected = (
        int(spec["execute_h"]),
        len(spec["task_ids"]),
        int(spec["episodes_per_task"]),
        int(spec["num_candidates"]),
        float(spec["temperature"]),
    )
    got = (
        result["h"],
        result["num_tasks"],
        result["episodes_per_task"],
        result["N"],
        float(result["temperature"]),
    )
    if got != expected:
        raise RuntimeError(f"evaluation metadata drifted: {got} != {expected}")
    return result


def archived_success(run_dir: Path) -> float | None:
    path = run_dir / "evaluation_1000000_h5.json"
    if not path.exists():
        return None
    return percent(json.loads(path.read_text())["overall_success"])


def format_pm(mean: float, sd: float) -> str:
    return f"{mean:.1f} ± {sd:.1f}"


def write_table(path: Path, pin: dict, rows: list[dict]) -> None:
    lines = [
        "# Scene common-protocol h5",
        "",
        "Primary checkpoint: 1M. h=5. 5 tasks × 50 episodes. "
        "Success is any step with info['success'].",
        f"Common eval seed: {pin['eval']['common_eval_seed']}.",
        "Archived training-time scores stay in the original run directories.",
        "",
    ]
    task_ids = pin["eval"]["task_ids"]
    for variant in ("gsdtrl_weighted", "gs_trl_weighted"):
        selected = [row for row in rows if row["variant"] == variant]
        lines.append(f"## {variant}")
        lines.append("")
        header = "| seed | common h5 % | archived h5 % | " + " | ".join(
            f"task {task}" for task in task_ids
        ) + " |"
        rule = "|---:|---:|---:|" + "|".join("---:" for _ in task_ids) + "|"
        lines.extend([header, rule])
        for row in selected:
            tasks = " | ".join(
                f"{percent(row[f'task_{task}_success']):.1f}" for task in task_ids
            )
            archived = row["archived_h5_percent"]
            archived_text = "" if archived is None else f"{archived:.1f}"
            lines.append(
                f"| {row['train_seed']} | {percent(row['overall_success']):.1f} | "
                f"{archived_text} | {tasks} |"
            )
        mean, sd = sample_mean_sd([percent(row["overall_success"]) for row in selected])
        task_summary = " | ".join(
            format_pm(
                *sample_mean_sd(
                    [percent(row[f"task_{task}_success"]) for row in selected]
                )
            )
            for task in task_ids
        )
        lines.append(
            f"| mean ± sample SD | {format_pm(mean, sd)} |  | {task_summary} |"
        )
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pin = json.loads(Path(args.config).read_text())
    output = Path(args.output).resolve()
    runs = []
    for variant, seeds in pin["historical_runs"].items():
        for seed, run_dir in seeds.items():
            runs.append((variant, int(seed), Path(run_dir)))
    new_root = Path(pin["output_root"])
    if not new_root.is_absolute():
        new_root = Path(args.config).resolve().parents[1] / new_root
    for seed in pin["seeds"]:
        runs.append(("gsdtrl_weighted", int(seed), new_root / "gsdtrl_weighted" / f"seed{seed}"))
    for _variant, _seed, run_dir in runs:
        resolved_run = run_dir.resolve()
        if output == resolved_run or output.is_relative_to(resolved_run):
            raise SystemExit(f"refusing to write evaluation into {run_dir}")
    rows_spec = manifest_for(pin)
    if len(rows_spec) != len(pin["eval"]["task_ids"]) * int(pin["eval"]["episodes_per_task"]):
        raise SystemExit("manifest length drifted")
    write_json(
        output / "rng_manifest.json",
        {
            "common_eval_seed": pin["eval"]["common_eval_seed"],
            "execute_h": pin["eval"]["execute_h"],
            "episodes_per_task": pin["eval"]["episodes_per_task"],
            "num_candidates": pin["eval"]["num_candidates"],
            "temperature": pin["eval"]["temperature"],
            "success": "any environment step with info['success']",
            "episodes": rows_spec,
        },
    )
    results = []
    for variant, seed, run_dir in runs:
        result = evaluate_run(pin, variant, seed, run_dir)
        result["archived_h5_percent"] = archived_success(run_dir)
        write_json(output / variant / f"seed{seed}.json", result)
        results.append(result)
        print(
            f"{variant} seed{seed} common_h5="
            f"{percent(result['overall_success']):.1f}",
            flush=True,
        )
    write_table(output / "results.md", pin, results)
    write_json(output / "results.json", results)
    write_json(output / "DONE.json", {"runs": len(results), "checkpoint": 1_000_000})


if __name__ == "__main__":
    main()
