# Goal-Space Temporal Quasimetric pilot

This branch tests whether the full-state temporal quasimetric is the source of
the puzzle failure. The only high-level geometry change is

`d_phi(s, g) = d(phi(s), phi(g))`,

where `phi` is the existing `utils.goal_representation.goal_representation`
mapping. Puzzle uses button states and cube uses cube positions. No new goal
features are defined.

The endpoint proposer remains the original PBF proposer and therefore consumes
the full current state together with `phi(goal)`. The bridge and IDM remain
unchanged. PB weighting and best-of-N ranking use the goal-space metric.

The locked variants are:

- `gsdtrl_weighted`: original TRL plus transitive proposer weighting.
- `gsctd_learned_temp`: the same model plus FutureNCE logits
  `b(g) - alpha * d_phi`, with global `alpha = softplus(raw_alpha)`, initialized
  to `1 / horizon` and safety-clipped to `[1e-3, 10]`.
- `gsctd_fixed`: the same goal-space model with `b(g) - d_phi / horizon`.

Run only the prescribed seed-0 queue:

```bash
conda create --prefix .venv python=3.11 pip
.venv/bin/pip install -r requirements-goalspace-cpu.lock.txt  # CPU validation only
.venv/bin/pytest -q
.venv/bin/python scripts/run_goalspace_suite.py
```

For the 1M runs, install the JAX build matching the machine's accelerator in
the isolated environment before invoking the suite. The suite refuses a long
CPU run unless explicitly overridden.

The runner executes, sequentially: GSDTRL puzzle, GSDTRL cube-double,
GSCTD-learned puzzle, GSCTD-learned cube-double, and GSCTD-fixed puzzle. It then
runs the puzzle `N=1,4,8,16,32` sweep and writes `SUMMARY_GOALSPACE.md`. It does
not schedule AntMaze or cube-single.
