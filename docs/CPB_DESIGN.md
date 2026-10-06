# Calibrated Contrastive PathBridger

This implementation follows the revised calibrated CPB request. It replaces
only PathBridger's high-level TRL signal. Earlier raw-score experiments are
archived in `exp/contrastive_pathbridger_raw_archive` and excluded from CPB tables.

## Authoritative release semantics

Read and preserved: `agents/pathbridger.py`, `utils/datasets.py`,
`utils/evaluation.py`, `utils/goal_representation.py`, `configs/_base.py`, and
all four requested `configs/pbf` files. `agents/latent_endpoint_chunk.py` was
also inspected; its action-conditioned critic, AWR/TD3BC actors and chunk
proposal models are not used.

The original source files remain unchanged. CPB inherits the original
`FlowEndpointProposer`, `BridgeResidual`, `InverseDynamics`, eight Euler steps,
state-space interpolation, exact pins, five-state bridge prefix and IDM losses.
All networks use the released GELU MLP, three 512-wide hidden layers and LayerNorm.
Adam's learning rate is 3e-4; target EMA is .005 after each joint update.

Released bridge loss sums absolute state-coordinate errors, then averages over
examples and five prefix states. Released IDM loss sums squared action errors,
then averages examples; a separately logged MSE averages action coordinates.
These reductions are preserved. Only endpoint flow-matching examples receive
progress weights. No actor, reward, return, separate training stages or weighted
bridge/IDM objective is introduced.

`PathBridgerDataset` samples anchors from starts with a complete K-step episode
window. Critic positive offsets are geometric with p=1−gamma and **clamped** at
the episode terminal, accumulating tail mass there. This is not a renormalized
truncated geometric distribution. Endpoints use the existing trajectory-future
goal distribution, and targets are clipped to a conditioning goal closer than K;
bridge-prefix targets are clipped identically. All sampler behavior is reused.

Task-goal projection is full-state → XY for AntMaze (29→2), cube positions
(28→3 or 37→6), and nine puzzle button fields (55→9). Phi sees the full state;
psi sees only this projection. K/gamma/N/temperature remain 40/.99/1/0 for
cube-single, 40/.99/8/.25 for cube-double, 25/.99/32/1 for puzzle, and
25/.99/8/.25 for AntMaze. TRL `endpoint_value_scale` and
`value_distance_weight_power` are unused/deprecated and absent from CPB config.

## Critic and calibration

Raw `C(s,g)=phi(s)·psi(g)` trains with diagonal-positive, row-wise InfoNCE,
temperature 1, repr_dim 64, no representation normalization, and .01 times mean
squared logsumexp. Calibration does not alter this training objective.

Raw InfoNCE permits an anchor-dependent additive constant. Cross-anchor ranking
or progress therefore uses `Cbar(s,g)=C(s,g)−logZ(s)`, where
`logZ(s)=logsumexp(C(s,bank))−log(M)`. This is a finite-bank estimate, not an exact
population normalizer. The bank contains M=512 projected training future goals,
sampled once by the original training sampler under an independent deterministic
seed. It matches the training positive-goal/negative marginal, including valid
anchor support and terminal clamping; it is not uniform-state sampling.

The explicit APIs are `raw_score`, `log_partition`, `calibrated_score` and
`target_calibrated_score`. Bank contents are an agent pytree field serialized
with the network, optimizer, EMA parameters and agent RNG. NumPy and Python RNGs
are included by the existing checkpoint utility. Resume reads the saved bank,
not a newly sampled bank. A hash appears in run provenance. `psi(bank)` can be
cached for an immutable evaluation snapshot; every update invalidates the cache.

Inference N>1 ranks only online `Cbar(z,g)`. N=1 bypasses the critic. The proposer
models local support, while calibrated reachability ranks remaining progress.
Neither `C(s,z)+C(z,g)` nor its calibrated equivalent is used.

Endpoint weights use target `Delta=Cbar(z,g)−Cbar(s,g)` with identical final g,
divide by stop-gradient batch std + 1e-6, clip to [-2,2], exponentiate with scale
1, and cap at 5. All weights stop gradients. Guidance is zero at/before 100k,
linear to one at 200k, then one. Rank-only always uses unit weights. There is one
joint 1M objective with unit coefficients for critic, endpoint, bridge and IDM.

## Evaluation and experiment order

Primary execution h=5 matches released PathBridger. Secondary h=2 and h=1 alter
only execution before replanning, never training. Evaluation retains clipping,
max episode length, the five predefined tasks, and success if `info['success']`
is true on any step. Paired deterministic reset seeds and saved episode manifests
extend the original evaluator without modifying it. Each checkpoint uses 50
episodes per task per h; ordering is h=5,2,1.

After full tests and 2k cube-single full seed0 smoke, a 100k full seed0 sanity
prelude checks finite calibrated progress, held-out recall above 1/B, effective
ranks above 2 (a deliberately minimal pathological-collapse gate), finite samples
and weights not all capped. This checkpoint is retained, then resumed as the
same full seed0 run. It is not an extra independently trained stage.

After this required prelude the full-run order is cube-single rank-only seed0,
full seed0 resumed to 1M, full seeds1/2; cube-double rank-only seed0 and full
seeds0/1/2; puzzle full seeds0/1/2; AntMaze full seeds0/1/2. Environments never
train simultaneously. Low task success does not stop the suite.

## Recovery and diagnostics

The runner supports configs, variants, seeds, train_steps, dataset_dir, save_dir,
resume, skip_existing and optional W&B. Exact continuation resumes optimizer,
parameters, EMA, all RNGs, sampler stream and fixed bank. Sampling prefetch pauses
at checkpoints; diagnostic/evaluation RNG effects are restored. Missing checkpoint
evaluations finish before training resumes. No raw checkpoint can be treated as
a calibrated checkpoint.

Diagnostics include collapse metrics, raw/calibrated rank Spearman and top1
agreement, target-progress correlation and percentiles, log-partition statistics,
endpoint displacement/support distance, progress weights, bridge errors/pins and
action saturation. Puzzle compares nested candidate pools N=1/8/32 against exact
nearest training-state distance. These diagnostics use held-out anchors; the bank
always comes from training. Undefined constant-vector correlations have a zero
placeholder and an explicit validity flag, and are excluded from interpretation.

The report produces evaluation_summary.csv, diagnostic_summary.csv,
learning_curves.csv and SUMMARY.md, with primary h=5 means/std/seeds, all secondary
h values and evidence-bounded answers to seven research questions. No baseline
numbers or missing runs are fabricated. Original PathBridger is a reference-only
dispatch to the original agent/config, and is not in the default suite.
