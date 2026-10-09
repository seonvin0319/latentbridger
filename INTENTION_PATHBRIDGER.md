# Intention-Conditioned PathBridger (IPathBridger / Shared-I)

Status: design document written **before** implementation. Sections marked
"Implementation note" record decisions taken from the actual code base.
The "Experimental Results" section is appended after the sweep finishes.

Code: `intention_pb/` (new package) and `run_intention_pathbridger.py`
(launcher / CLI). All new outputs live under `exp/intention_pathbridger/`.
Nothing in `agents/`, `main.py`, `checkpoints/`, `runs/` or the
`Pathbridger_InFOM` BOC / InFOM experiments is modified.

---------------------------------------------------------------------------

## 1. Motivation

Original PathBridger (PB) acts as

    (s, g) -> subgoal z -> local bridge Y -> IDM -> actions

In this repository (`agents/dynamics.py`, `DynamicsAgent`):

* `subgoal_net` is a conditional rectified flow over the displacement
  `Delta = s_{t+K} - s_t` (goal fed through the env `phi` projection),
  trained with a value-weighted flow-matching loss
  `w = min(exp(10 * (V(s_{t+K}, g) - V(s_t, g))), 5)`.
* `path_residual_net` is a deterministic, endpoint-preserving bridge:
  `z_i = a_i z_0 + b_i z_K + w_i r(s_t, z_K, i/K)`, `w_i = i(K-i)/K^2`,
  trained with an L1 loss on the prefix `i = 1..5`.
* `idm_net` maps consecutive planned states to actions.
* At evaluation, `N` flow candidates are scored with the TRL critic
  transitive ratio `V(s,z) V(z,g) / (V(s,g) + eps)`; the argmax is bridged
  and the first 5 IDM actions are executed.

The same `(s, g)` (or the same `(s, z)`) admits several local behaviours:
different approach directions, grasp orders, button-press orders. A single
waypoint `z` has to answer both "which intermediate state?" and "how do we
get there?". The flow subgoal model can represent multimodality in state
space, but the deterministic bridge must then average over the different
ways of reaching `z`, which can produce poor first actions.

Hypothesis: a latent behavioural **intention** `c` should jointly coordinate
subgoal proposal and local bridge execution:

    p(z, Y | s, g) = sum_c p(c | s, g) p(z | s, g, c) p(Y | s, z, c)

No `p(c | s, g)` is learned. At inference all discrete intentions are
enumerated with an equal candidate budget and the existing PB value score
selects `(c*, z*)`.

## 2. Difference from InFOM

This is not InFOM. The following components are **not** used:
InFOM occupancy flow, discounted future occupancy, generative value
estimation, upper-expectile critic, InFOM future anchor, BOC contrastive
occupancy ratio, BOC forward/backward product, flow path bridge, TD
InfoNCE, CARL, any reranker.

The only idea borrowed from InFOM is a **latent intention / behaviour
mode**. Unlike InFOM, the intention is defined directly from a *local,
contiguous* behaviour chunk (actions and relative state changes), not from
a future-occupancy latent.

The baseline is the **original PathBridger** in this repository
(`Pathbridger_flow`, flow subgoal + TRL critic + deterministic residual
bridge + IDM). It is *not* the "InFOM-anchor + deterministic bridge"
(`pb_base`) model of `Pathbridger_InFOM/sweep_runs/1m`; that model is not
used here.

## 3. Hypotheses

* **H1** Conditioning subgoal proposal on intention reduces multimodality:
  `H(Z | S, G, C) < H(Z | S, G)`.
* **H2** Conditioning the bridge on the same intention reduces local path
  ambiguity: `H(Y | S, Z, C) < H(Y | S, Z)`.
* **H3** The same intention must coordinate both levels:
  `Shared-I > I-SG` and `Shared-I > Shuffled-I`.
* **H4** Gains are larger on behaviourally multimodal tasks
  (cube-double, puzzle-4x4) than on cube-single.

## 4. Intention definition

Local horizon `h = 5` (= PB `forward_bridge_path_loss_horizon` = critic
`action_chunk_horizon` = IDM execution chunk; verified in
`checkpoints/1m_env_best/*/flags.json`).

For a contiguous same-episode chunk starting at `t`:

    features = ( a_t, ..., a_{t+4},  Delta s_t, ..., Delta s_{t+4} ),
    Delta s_j = s_{j+1} - s_j

Both actions and deltas are z-scored with per-dimension training-set
statistics (std floored at `1e-3`). The encoder never receives the goal,
the absolute state, the absolute endpoint `s_{t+5}`, reward, or success.
The auxiliary decoder receives the (z-scored) absolute `s_t`.

Implementation note: the chunk is the *real* dataset chunk
`a_{t:t+5}, s_{t:t+6}`; PB's `clip_path_to_goal` padding of the bridge
target is not applied to the intention features.

## 5. Discrete VQ intention tokenizer

* `K = 8` codes, embedding dim `32`.
* Encoder `E`: MLP(512, 512, LayerNorm) -> 32.
* Quantisation `c = argmin_j ||e - e_j||^2`, straight-through estimator,
  **EMA codebook** (decay 0.99, Laplace eps 1e-5), codebook initialised
  from `K` random encoder outputs of the first batch.
* Decoder `D(s_t, e_c)`: MLP(512, 512, LayerNorm) -> normalised action
  chunk (5 x A) and normalised deltas (5 x D).
* Loss `L = 1.0 L_action + 1.0 L_delta + 0.25 L_commit + 0.01 L_balance`,
  `L_commit = ||e - sg(e_c)||^2`,
  `L_balance = KL(mean_batch softmax(-||e - e_j||^2) || Uniform(K))`
  (mild; it does not force uniformity). The EMA codebook needs no
  codebook loss term.
* 200k steps, batch 1024, Adam 3e-4; checkpoints at 50k / 100k / 200k.
* Trained **separately** and then **frozen**: goal/subgoal losses never
  shape `c`, so `c` cannot become a hidden endpoint label.
* Metrics (train batches and held-out validation dataset): usage
  histogram, perplexity `exp(H(usage))`, max code probability, action and
  delta reconstruction MSE; per-code table (frequency, mean action vector,
  mean per-step displacement, mean chunk displacement norm).
* **Collapse** (at 200k, validation set): max usage > 0.85 or perplexity
  < 2.0. A collapsed task/seed gets no downstream 1M job; nothing is
  re-tuned.

## 6. Intention-conditioned subgoal model `p(z | s, g, c)`

Same family as PB: the `SubgoalFlowNet` vector field, with a learned
`nn.Embed(K, 32)` intention embedding concatenated to the MLP input.
Same displacement frame, same flow-matching loss, same value weighting
(`gap_scale = 10`, `w_max = 5`), same 8 Euler steps, same goal sampling
(`PathHGCDataset` with the PB run's dynamics config), same batch size and
optimiser (Adam 3e-4, batch 1024), 1M steps.

Subgoal horizon decision (user-selected, `pb_K`): the subgoal is PB's
`z = s_{t+K}` (K = 40 for cube, 25 for puzzle; clipped at the goal as in
PB), so the PB critic score and IDM are reused unchanged. The intention is
defined on the first `h = 5` executed steps of the same segment.

Implementation note: the value weight uses the **frozen** final PB critic
(`modules_target_value`, exactly what PB passes to the dynamics update),
whereas PB itself trained against a co-evolving critic.

## 7. Intention-conditioned deterministic bridge `B(Y | s, z, c)`

Same `PathResidualNet` (MLP 512x3, LayerNorm) plus the intention
embedding broadcast over time; same closed-form forward-bridge mean,
endpoint-preserving weight `w_i`, displacement frame, absolute-state
anchor, L1 loss on prefix `1..5` (and the clamped endpoint). Targets come
from the contiguous `trajectory_segment`. No stochastic or flow bridge.

The conditioned subgoal net and conditioned bridge are trained jointly in
one job (like PB), but they share no parameters.

## 8. IDM and critic

Frozen and shared: for each task/seed the PB checkpoint's `idm_net` and
TRL critic are used by **all** variants. No per-variant copies, no InFOM
critic. Selection score: PB's `score_transitive_subgoals(..., 'ratio')`
with the critic's online params, identical to `infer_subgoal_for_eval`.

## 9. Inference (equal budget)

User-selected budget (`n16_envtemp`): `N = 16` candidates for every method;
for intention methods `M = 2` per code x `K = 8`. Flow noise temperature is
PB's env-best value: cube 0.0, puzzle 0.5.

Note: at temperature 0 the two candidates of a code coincide and PB's 16
candidates coincide (PB N=16 t0 == PB N=1 t0).

    PB          : z_m ~ p_PB(z|s,g), m=1..16; z* = argmax Score; Y = B_PB(s,z*)
    I-SG        : z_{c,m} ~ p(z|s,g,c); (c*,z*) = argmax Score; Y = B_PB(s,z*)
    Shared-I    : same (c*,z*) as I-SG;  Y = B(s,z*,c*)
    Shuffled-I  : same (c*,z*) as I-SG;  Y = B(s,z*,(c*+1) mod K)

Actions: frozen PB IDM on consecutive bridge states, first 5 actions
(`h_exec = 5`, native PB) or only the first action (`h_exec = 1`,
truncated chunk). Each episode uses the PB eval convention
`rng = PRNGKey(subgoal_eval_seed + ep_ix)` at every replan, so I-SG /
Shared-I / Shuffled-I see the same candidate set at the same state.
Environment resets are seeded with `1000 * task_id + ep_ix` so that all
methods face identical initial states (PB's own evaluator does not seed
resets; this only affects pairing, not the expectation).

Reference row (not part of the equal-budget comparison): PB at its
original best setting (cube N=1 t0, which equals PB N=16 t0; puzzle N=32
t0.5).

## 10. Variants

Exactly four methods: PB, I-SG, Shared-I, Shuffled-I (evaluation only).
No K sweep, no BOC / occupancy / InFOM critic / flow bridge / CARL /
contrastive objective / reranker.

## 11. Offline diagnostics (held-out OGBench `-val` dataset)

* Tokenizer usage / perplexity on validation chunks.
* Subgoal error: PB vs conditioned model with the teacher code; per-code
  error; best-of-16 endpoint error and critic-selected error under the
  equal budget (2 per code).
* Bridge with teacher `c` vs shuffled `(c+1) mod K` vs PB bridge, given
  the true `z`: prefix path MSE (raw and per-dim-normalised), first-step
  error, full-prefix error, IDM action MSE against the dataset actions.
* Conditional multimodality: for query `(s, g)` pairs, `k = 64`
  nearest neighbours in standardised `[s, phi(g)]` space within a pool of
  validation samples; pooled within-neighbourhood variance of the
  standardised subgoal displacement (and of the first action / action
  chunk), unconditioned vs pooled within-code (groups with >= 2 members).
  A permutation baseline (codes shuffled within the neighbourhood, same
  group sizes) measures the reduction expected by chance from finite
  grouping.
* Subgoal diversity for fixed `(s, g)`: between-code distance
  `E_{c != c'} ||zbar_c - zbar_c'||` vs within-code distance (two samples
  of the same code) at temperature 1.0 and at the eval temperature.
* Bridge controllability: `||B(s,z,c) - B(s,z,c')||` over the prefix,
  relative to the prefix motion magnitude, and the induced IDM action
  difference.

## 12. Control evaluation

Tasks `cube-double-play-v0`, `puzzle-4x4-play-v0`, `cube-single-play-v0`
(OGBench task ids 1-5); seeds 0, 1, 2; `h_exec in {1, 5}`.
Final (1M): 100 episodes (20 per task). Intermediate (250k, 500k, 750k):
20 episodes (4 per task). Logged per episode: success (any-step
`info['success']`), return, length, replans, selected code sequence,
switch frequency, selected score. Aggregates: code histogram, code
entropy, per-code success (episode attributed to its majority code).

Seed 0 reuses `checkpoints/1m_env_best/{cd_cube-double, p4_puzzle-4x4,
cs_cube-single}` (read-only). Seeds 1 and 2 have no PB checkpoints, so
PB is trained once per task/seed with the identical `config_used.yaml`
(`main.py`, 1M steps, joint SPI actor as in the seed-0 runs) under
`exp/intention_pathbridger/pb_runs/` and shared by all variants.

## 13. Interpretation

* `Shared-I > I-SG` and `Shared-I > Shuffled-I`: intention coordinates
  waypoint selection and realisation (H3).
* I-SG improves, Shared-I does not: intention partitions subgoal modes;
  bridge conditioning is unnecessary or poorly learned.
* `Shared-I ~= Shuffled-I`: the bridge ignores `c` (check offline
  true-c vs shuffled-c gap).
* All ~= PB: local behavioural multimodality is not the bottleneck.
* Shared-I < PB: the discrete decomposition hurts generalisation or the
  tokenizer is not behaviourally meaningful.

Verdict scale: A strong support / B partial support / C no support /
D negative.

## 14. Execution

`run_intention_pathbridger.py launch` builds the job DAG
(PB-train for seeds 1-2 -> tokenizer -> conditioned training -> offline
diagnostics + control evaluation as checkpoints appear -> aggregation),
runs at most 2 GPU training jobs concurrently on the single RTX 5080 and
CPU evaluation jobs in parallel, and is resumable (status files, PID /
start / end / exit code, per-job stdout/stderr, checkpoint validation, no
overwrite, explicit failure on missing checkpoints).

### Implementation notes

* **PB code version.** The seed-0 PB checkpoints were produced by commit
  `21a4042`; the current working tree changed the PB module definitions
  (it removed the 64-d sinusoidal flow-time embedding), so the checkpoints
  do not load with it. All PB code (`agents/`, `utils/`, `main.py`,
  `eval_checkpoint.py`) is imported from an unmodified export of that
  commit at `../Pathbridger_flow_pb21a4042` (marker file `PB_CODE_COMMIT`),
  and seeds 1-2 are trained with that commit's `main.py` (where the actor
  is always trained jointly, as for seed 0). Their `config_used.yaml` is
  byte-identical to seed 0's; only seed / run dir / in-training eval flags
  differ (no in-training or final eval; checkpoints every 100k).
* **Conditioned modules** wrap the exact PB module definitions
  (`SubgoalFlowNet`, `PathResidualNet` cloned from the loaded PB agent) and
  concatenate the code embedding to the first input; with `K = 0` they
  reduce exactly to PB (unit-tested against the loaded checkpoint).
* **Bridge loss** is PB's loss at `21a4042`: interior L1 over the
  `[1..H, N]` indices plus the next-step L1 term.
* **Temperature**: the `21a4042` sampler has no temperature argument; the
  eval temperature is set by overriding `subgoal_temperature` in the
  frozen dynamics config, for PB and for the conditioned sampler alike.
* **Tests**: `intention_pb/tests/test_intention_pb.py` (12 tests, spec
  section 15). The `21a4042` test suite passes (79); the current working
  tree has one pre-existing failure
  (`test_forward_bridge_path_loss_uses_interior_only_once`) unrelated to
  this experiment.
