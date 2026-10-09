# Offline learned goalspace

## Provenance and locked experiment

This branch is based on
`da37867d31d47b02a14dcbc56dd8f267062c02aa`. The authoritative environment
semantics come from `configs/gsctd/puzzle_3x3.py` and
`configs/gsctd/cube_double.py`, which in turn preserve the original PBF
PathBridger settings.

- Puzzle 3x3: `puzzle-3x3-play-v0`, flow endpoint, horizon 25, discount 0.99,
  endpoint scale 10, value-distance power 0.5, 32 candidates, temperature 1.
- Cube double: `cube-double-play-v0`, flow endpoint, horizon 40, discount 0.99,
  endpoint scale 10, value-distance power 1, 8 candidates, temperature 0.25.

Both use the original full-state endpoint proposer and its existing phi goal
conditioning, plus the unchanged full-state endpoint-pinned bridge and IDM.

## Offline pretraining

`fullobs_future_nce` uses only compact full observations and terminals.
Geometric future positives use the environment discount and cannot cross an
episode terminal. Query and goal augmentations are independent. The query and
goal encoders have separate parameter subtrees; the exported representation is
the unnormalized 16-dimensional pre-normalization output of `E`.

The pretraining loss and package do not import or call oracle phi, rewards,
tasks, or task masks. Checkpoints are written at 100k, 300k, and 500k. The
downstream checkpoint is predetermined as **500k**, regardless of probe or
success metrics.

## Frozen transfer methods

- `LGS_TRL_W_FROZEN`: scalar high-level
  `V(E(s), E(g))`, original self/base/transitive expectile objective and EMA
  target, original proposer weighting, and candidate score
  `V(E(s),E(z)) * V(E(z),E(g))`.
- `LGSDTRL_W_FROZEN`: temporal-quasimetric high-level distance over frozen
  embeddings, `V=gamma^d`, the same TRL objective/EMA/proposer weighting, and
  candidate score `d(E(s),E(z)) + d(E(z),E(g))`.

`E` is loaded once into an explicit `modules_goal_encoder` subtree. Its output
is defensively stop-gradient and its optimizer partition uses
`optax.set_to_zero`, so its parameters receive exactly zero updates. Every
high-level state role (`s`, `z`, and `g`) passes through this encoder. The
endpoint proposer, bridge, and IDM definitions are unchanged.

Downstream checkpoints are 100k, 300k, 500k, 800k, and 1M. Evaluation uses five
tasks times 50 episodes, with execution horizon 5 primary and horizons 2 and 1
secondary, using the deterministic PathBridger manifest seed formula.

## Probe boundary

Oracle phi is confined to `learned_goalspace/probes.py` and
`main_learned_goal_probes.py`. Probes receive frozen host NumPy embeddings
after `jax.device_get`; there is no gradient route back to `E`. Every pretrain
checkpoint is compared with PCA-16 and a deterministic fixed random
projection. Probe metrics never select a checkpoint.

Oracle-method numbers are reference metadata only: puzzle GSDTRL about 97.2%,
GS-TRL about 98%; cube both about 81%, and full-state DTRL about 84%. They are
not thresholds and are never used for selection.

## Outputs and execution

- Pretrain: `exp/learned_goalspace/pretrain/<env>/fullobs_future_nce/seed0`
- Probes: `exp/learned_goalspace/probes/<env>/fullobs_future_nce/seed0`
- Downstream: `exp/learned_goalspace/downstream/<env>/<method>/seed0`

The drivers write `representation_metrics.csv`, `probe_metrics.csv`, and
`downstream_results.csv`; the queue also aggregates these files at the output
root with environment, variant, seed, checkpoint, and method metadata.
`scripts/run_learned_goalspace_queue.py` runs the six jobs sequentially,
supports safe resume/skip and `--dry-run`, and has a separate smoke output
root. Child processes apply the Error-21 thread pins, verify a real CUDA
backend, and use `taskset`.

Production should be started through:

```bash
scripts/run_learned_goalspace_16g.sh
```

The wrapper always re-executes itself under a user systemd unit with
`MemoryHigh=15G`, `MemoryMax=16G`, `MemorySwapMax=0`, `OOMPolicy=kill`, and
`KillMode=control-group`. Inside that unit it verifies `memory.high`,
`memory.max`, `memory.swap.max`, and cgroup-wide OOM killing before running. It
refuses to launch if the full contract cannot be established.
