# PathBridger: preserved 3999f37 core

This branch packages the training and inference computation from
`PathBridger_legacy` commit `3999f3759fe6bfce1e62966bb38f0ce277add843` behind the
current Python config, CLI, CSV, and checkpoint interfaces. PBF and PBG remain
available. The acceptance criterion is exact fixed-input computation under the
same JAX/runtime/device settings. Full paper success-rate reproduction is a
separate experiment.

## Computation contract

`pathbridger_core/` contains the original dynamics, critic, samplers, networks,
and training states. Only internal import prefixes changed; `PROVENANCE.json`
records the upstream and packaged SHA-256 hashes. `agents/pathbridger.py` is a
thin facade that calls the original dynamics update before the original critic
update, using the pre-update target value. It does not enclose both in a new JIT.

The original initialization names/order, unused sampler draws, Flow key splits,
separate value forwards, target clipping to `[1e-6, 1]`, Adam states and EMA are
retained. Transitive samples include all distances greater than one. Bridge
training keeps rows `[0, 1, 2, 3, 4, 5, K]`, original schedule arithmetic and the
mean of the five prefix errors plus the zero endpoint error. The full original
trajectory batch and inference plan remain intact.

The original action-critic and target heads are retained, including their
initialization, loss, optimizer and EMA work, even though the deployed IDM
policy uses the value head. No SPI actor is trained. Removing these heads or
simplifying equivalent formulas requires a separate equivalence check.

## Install and run

Use Python >= 3.10 and the JAX build for your accelerator:

```bash
pip install -e .
python main.py --agent=configs/pbf/puzzle_3x3.py --seed=1 --run_group=preserved3999
```

`pathbridger-train` and `pathbridger-eval` are equivalent installed entry points.
Defaults: 1M updates, batch 1024, save/evaluate every 100k, 25 episodes on each of
five tasks. For a short training-only check:

```bash
python main.py --agent=configs/pbf/puzzle_3x3.py --seed=1 \
  --train_steps=16 --log_interval=4 --save_interval=8 \
  --eval_interval=0 --noeval_final
```

Training sets deterministic GPU operations and autotune level 0 unless explicit
XLA options already select otherwise. Compare only matching runtime options.
The verified environment is recorded in `validation/runtime.json`; different
JAX/compiler/hardware versions are not promised to produce identical bits.

One worker prefetches batches by default (`--noasync_prefetch` disables it).
Sampling stops at a save boundary before host RNG serialization, preserving the
next batch on resume. `--dataset_dir` retains compact OGBench shard support.

## Checkpoints and provenance

A run stores `flags.json`, `provenance.json`, `train.csv`, `eval.csv`, and
`checkpoints/params_<step>.pkl`. Provenance includes resolved original configs,
code hashes, runtime versions, device, and compiler flags. The versioned
`pathbridger/3999f37/v1` checkpoint contains both original agents, all Adam/EMA
state, model RNGs, NumPy/Python RNG state, public config and input dimensions.
Writes are atomic; restore rejects changed configs and incompatible formats.
CSV retains `value/loss`, `endpoint/loss`, `bridge/loss`, and `idm/loss` aliases
for comparing component curves. Original dynamics/critic metrics remain available
under their prefixes. The critic total also includes the preserved action-Q
loss, so it should not be equated to the earlier four-component total.

```bash
python main.py --agent=configs/pbf/puzzle_3x3.py --seed=1 \
  --restore_path=/path/to/checkpoints/params_100000.pkl --train_steps=1000000
```

Earlier unified checkpoints cannot resume this core exactly because the
original action-critic/optimizer state is missing. They are rejected explicitly.
No existing checkpoint is overwritten or silently converted.

## CPU evaluation

Every evaluation checkpoint enters a durable `cpu_eval/job_*.json` queue. One
separate CPU process drains it while GPU training continues, including after
training finishes. Results are written to `cpu_eval/result_*.json` and `eval.csv`.
`worker.log`, per-job `.error` files and `WORKER_DONE.json` expose failures.
To resume an interrupted evaluation queue:

```bash
python utils/eval_worker.py /path/to/run/cpu_eval
python evaluate.py --checkpoint_dir=/path/to/params_100000.pkl --episodes=25
```

The worker and standalone evaluator force the CPU backend and hide CUDA. They
load the checkpoint metadata without loading training datasets. Standalone
`--agent` remains optional and, if supplied, must match the checkpoint config.

Both implementations share the agreed `3999-idm-strict-reset-v2` evaluator:
original IDM chunking/selection, five-step actions, clipping at environment
step, any-step `info['success']`, full episode limit, reset seed `730000+i`, and
action-space seed `930000+i`. ManipSpace's Box is cached during evaluation to
make full goals reproducible. Candidate RNG is episode index within each task,
reused at every replan, matching the agreed evaluator. Training seed does not
change evaluation inputs. Episode observation/goal hashes and successes are
saved. The original ratio candidate score is retained.

## Verification

`tests/check_core_equivalence.py` accepts an independent original source archive.
It verifies source hashes, initial parameters, full sampled batches and RNG,
loss/gradient equality, repeated native updates (including Adam and EMA), and
checkpoint resume. `tests/check_inference.py` checks 125 CPU states using the
same reset observations and complete goals. `tests/benchmark_core.py` measures
warm update throughput and allocator memory in separate processes.

See `validation/REPORT.md` for results, scope and exact commands. Source files
in `pathbridger_core/` are kept verbatim apart from import prefixes; refactoring
or formatting them is intentionally deferred.

## Original scene limitation

The `scene-play-v0` observation has 40 dimensions. The original `3999f37` goal
layout parser rejects it as ambiguous for both PBF and PBG. This release retains
that same rejection; scene is **not supported** by this preserved core. The
other 14 presets are verified below. Fixing scene would be a separate, explicitly
changed computation path rather than a claim of original-code equivalence.

## Public configuration presets

The following environment presets are retained from the existing distribution.
Their numeric settings are not retuned by this change. `discount` controls the
critic; the dynamics sampler retains its original 0.99 default. Mixtures use
`(current, geometric future, uniform future, random)`; `dynamics_p=(0,0,1,0)` and
`critic_p=(0,1,0,0)` map directly to the original sampling options.

### PBF

| Config | Environment | $K$ | $\gamma$ | $c_{\mathrm{sg}}$ | $\lambda$ | $N$ | $T$ |
|---|---|---:|---:|---:|---:|---:|---:|
| `pbf/antmaze_medium.py` | antmaze-medium-navigate-v0 | 25 | 0.99 | 10 | 0.0 | 8 | 0.25 |
| `pbf/antmaze_large.py` | antmaze-large-navigate-v0 | 25 | 0.995 | 10 | 0.0 | 32 | 0.5 |
| `pbf/cube_single.py` | cube-single-play-v0 | 40 | 0.99 | 5 | 0.7 | 1 | 0 |
| `pbf/cube_double.py` | cube-double-play-v0 | 40 | 0.99 | 10 | 1.0 | 8 | 0.25 |
| `pbf/cube_triple.py` | cube-triple-play-v0 | 40 | 0.995 | 10 | 1.0 | 1 | 0 |
| `pbf/puzzle_3x3.py` | puzzle-3x3-play-v0 | 25 | 0.99 | 10 | 0.5 | 32 | 1.0 |
| `pbf/puzzle_4x4.py` | puzzle-4x4-play-v0 | 25 | 0.99 | 10 | 2.0 | 16 | 1.0 |
| `pbf/scene.py` | scene-play-v0 | 25 | 0.99 | 5 | 1.0 | 8 | 0.5 |

### PBG

| Config | Environment | $K$ | $\gamma$ | $c_{\mathrm{sg}}$ | $\lambda$ | $N$ | $T$ |
|---|---|---:|---:|---:|---:|---:|---:|
| `pbg/antmaze_medium.py` | antmaze-medium-navigate-v0 | 25 | 0.99 | 10 | 0.0 | 1 | 0 |
| `pbg/antmaze_large.py` | antmaze-large-navigate-v0 | 25 | 0.995 | 10 | 0.0 | 1 | 0 |
| `pbg/cube_single.py` | cube-single-play-v0 | 25 | 0.99 | 10 | 0.7 | 1 | 0 |
| `pbg/cube_double.py` | cube-double-play-v0 | 25 | 0.99 | 10 | 1.0 | 1 | 0 |
| `pbg/cube_triple.py` | cube-triple-play-v0 | 25 | 0.995 | 10 | 1.0 | 1 | 0 |
| `pbg/puzzle_3x3.py` | puzzle-3x3-play-v0 | 40 | 0.99 | 10 | 0.5 | 2 | 0.25 |
| `pbg/puzzle_4x4.py` | puzzle-4x4-play-v0 | 40 | 0.995 | 10 | 2.0 | 32 | 0.5 |
| `pbg/scene.py` | scene-play-v0 | 40 | 0.99 | 5 | 1.0 | 16 | 0.5 |
