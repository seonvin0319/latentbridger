# Contrastive PathBridger

CPB replaces the scalar TRL critic with full-state phi and task-goal psi encoders
(512×3, layer norm, 64-dimensional output). It directly reuses the original PBF
proposer, eight-step flow sampler, endpoint-pinned explicit bridge, five-state
prefix objective and inverse dynamics. `agents/pathbridger.py` is unchanged.

InfoNCE uses terminal-bounded geometric future states from `PathBridgerDataset`,
with diagonal positives, temperature 1, and squared logsumexp regularization .01.
Target phi/psi use .005 EMA after each joint update. Candidate ranking uses only
online C(z,g); N=1 skips ranking. Scores are raw dot products; temperature only
scales training logits. The default temperature is 1.

Progress weights use the target-score difference with the same final goal,
divided by batch std + 1e-6, clipped to [-2,2], exponentiated, capped at 5 and
stopped from propagating gradients. Weighting is off through 100k and ramps
linearly to 200k. `cpb_rank_only` always uses unit weights. TRL
`endpoint_value_scale` and `value_distance_weight_power` are unused/deprecated
for CPB and absent from its configs. `pathbridger_original` explicitly dispatches
to the untouched original agent and original tuned config.

Original loss reductions are preserved: bridge L1 sums state coordinates then
averages examples and prefix steps; IDM sums action squared errors before
averaging examples. The separately logged action MSE averages coordinates too.
The sampler may choose a closer-than-K endpoint when its original conditioning
goal is nearer; CPB preserves this original behavior, including prefix clamping.

## Run

Use Python 3.11 and the versions recorded in `requirements-cpb.lock.txt`:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-cpb.lock.txt
.venv/bin/python scripts/run_contrastive_pathbridger_suite.py
```

The suite runs `pytest -q`, a 2k cube-single full smoke with checkpoint round-trip,
held-out recall > 1/128, finite losses/actions, and a five-task one-episode h=2
rollout. It then runs cube-single rank-only seed0 and full seeds0–2, cube-double
rank-only seed0 and full seeds0–2, puzzle full seeds0–2, and antmaze full seeds0–2.
Each is one joint 1M run, with checkpoints at 100k/300k/500k/800k/1M. Each checkpoint
is evaluated on all five tasks, 50 episodes per task, separately at h=1/2/5.
Low success never stops a run. A suite lock prevents duplicate launches.

A dedicated venv avoids altering existing training environments. GPU memory
preallocation is disabled. Original branches/processes are never operated on.

```bash
.venv/bin/python evaluate_contrastive_pathbridger.py \
  --run-dir exp/contrastive_pathbridger/cube_single/cpb_full/seed0 \
  --checkpoint 1000000 --h 2
.venv/bin/python scripts/summarize_contrastive_pathbridger.py
```

Checkpoint payloads include optimizer, targets, update counter and random states.
Training can resume with `main_contrastive_pathbridger.py --restore <exact-pkl>`.
The suite refuses silent partial-run overwrites: recover missing evaluations with
the evaluation CLI, then resume training at the last checkpoint. Completed runs
are skipped on relaunch. Diagnostics restore the host RNG before resuming batches.

## Diagnostics and interpretation

`train.jsonl` records training metrics every 1000 updates and at checkpoints.
`diagnostics_<step>.json` additionally records held-out recall, per-dimension std,
norms, all covariance eigenvalues, entropy effective rank, positive/random cosine,
endpoint candidates and exact nearest-state distance over the complete training
set. Diagnostic candidates use held-out states; these are not rollout statistics.
Evaluation JSONs record success counts, rate, h, N and temperature.

`SUMMARY.md`, `evaluation_summary.csv`, and `diagnostic_summary.csv` are generated
under `exp/contrastive_pathbridger`. Missing results remain pending. Local result
files are inventoried as reference evidence; unrelated latent methods or unmatched
training/evaluation protocols are not silently presented as the requested baseline.
Cube-single has N=1, so it cannot establish the utility of candidate ranking.

Experiments, checkpoints and datasets are never committed or pushed.
