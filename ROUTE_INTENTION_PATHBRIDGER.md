# Route-Level Intention PathBridger

Method name: **Route-Level Intention PathBridger** (Route-I, RouteInt-PB).

This is not the local-intention experiment in `INTENTION_PATHBRIDGER.md`.
That experiment's outputs stay in `exp/intention_pathbridger/` and are not overwritten.
New outputs go to `exp/route_intention/`.

## 1. Motivation

PathBridger chooses a subgoal `z = s_{t+K}` and then a deterministic bridge from `s` to `z`.
On cube and puzzle tasks the subgoal is a choice of route, not a choice of motor style:
which side of an obstacle, which object first, which puzzle order.
A latent that only describes the next few actions cannot tell those routes apart, so
conditioning both the subgoal model and the bridge on that latent does not help planning.

## 2. Failure of local intention

The local code `c_local` was a VQ over a contiguous 5-step action and state-delta chunk,
and the same code conditioned both

```text
(s, g, c_local) -> subgoal z
(s, z, c_local) -> bridge Y
```

Seed 0, final 1M, 100 episodes (success rate):

| Task | h | PB | I-SG | Shared-I | Shuffled-I |
|---|---|---|---|---|---|
| cube-double | 1 | 46% | 10% | 6% | 5% |
| cube-double | 5 | 79% | 24% | 23% | 30% |
| puzzle-4x4 | 1 | 80% | 67% | 60% | 66% |
| puzzle-4x4 | 5 | 77% | 76% | 73% | 73% |
| cube-single | 1 | 95% | 21% | 6% | 6% |
| cube-single | 5 | 100% | 30% | 32% | 40% |

Shared-I is below PB by more than 5 points in 5 of 6 cells. No cell has Shared-I above
Shuffled-I by more than 5 points. The tokenizer did not collapse (validation perplexity
6.2–6.7, max code usage 25–31%). Offline, the bridge path error is lower with the correct
code than with a shuffled code, so the bridge does read `c_local`. The kNN subgoal
variance ratio `Var(Z | S, G, C_local) / Var(Z | S, G)` is 0.954 (cube-double),
0.984 (puzzle-4x4), and 0.910 (cube-single). Conditioning on `c_local` leaves 91–98% of
the subgoal variance. The write-up is
`exp/intention_pathbridger/aggregate/local_intention_postmortem.md`.

## 3. Local vs route-level intention

`c_local` is how the agent moves for the next few steps.
`c_route` is which longer trajectory it is in the middle of.

The local code can sharpen bridge execution and still be useless for choosing `z`.
The route code is only worth a control run if it predicts which subgoal occurs.

## 4. Model factorization

```text
c_route ~ q_phi(c | trajectory segment of length H_route)
z       ~ p_theta(z | s, g, c_route)
Y       ~ B_psi(Y | s, z, c_route)
a       = IDM(Y)
```

`B` is the existing deterministic PathBridger bridge (`forward_bridge_residual`).
It is not a flow bridge. The IDM, critic, candidate score, normalisation, and evaluation
protocol are the frozen PathBridger ones.

## 5. Route intention learning

`h_local` in the failed experiment is the 5-step chunk. PathBridger's planning horizon
in the seed-0 flags is 40 for both cube tasks and 25 for puzzle-4x4. Episodes are 1001
steps. `H_route = 25` would end before the cube subgoal `s_{t+40}`, so the common
default is **`H_route = 50`** (the longer of the candidate set {20, 50}). Windows never
cross an episode boundary.

The encoder sees, at each step `j = 0 .. H_route-1`,

```text
a_{t+j}    and    s_{t+j+1} - s_t
```

both z-scored with training-set statistics. It does not see absolute `s_t`, the goal,
reward, success, or a goal id. The relative endpoint `s_{t+H} - s_t` is inside that
relative sequence; it is not a separate goal label.

The encoder is a two-layer temporal convolution (kernel 5, width 64) with mean pooling,
then a linear map to a 64-d embedding. A transformer is unnecessary at this width.
The codebook is the same EMA vector quantiser as the local tokenizer: `K = 8`,
straight-through estimator, decay 0.99. The code is `c = argmin_j ||e - e_j||^2`.

From `(s_t, c)` the decoder predicts z-scored summaries, not a reconstruction of the
raw window:

- relative endpoints at horizons 5, 10, 20, and `H_route`
- relative waypoints at `H/4`, `H/2`, `3H/4`
- the mean action over the window

```text
L = 1.0 L_multi + 1.0 L_path + 1.0 L_act + 0.25 L_commit + 0.01 L_balance
```

Targets are z-scored, so the three reconstruction terms have the same scale. The commit
and balance weights match the local tokenizer. They are not tuned per task.

Training: 200k Adam steps, batch 1024, learning rate 3e-4, three tasks times three seeds.
Checkpoints at 50k, 100k, and 200k. Collapse (validation) is max usage > 0.85 or
perplexity < 2. A collapsed task/seed gets no downstream model.

## 6. Inference

At evaluation the future segment is not available. A classifier `p_eta(c | s, g)` is
trained on the tokenizer codes as pseudo-labels, and only for tasks that pass the
representation gate. Inference uses the top `L = min(4, K) = 4` codes. Each code
proposes `budget / L` subgoals. With the PathBridger budget of 16 that is 4 candidates
per code. The total number of candidates stays 16. The existing value model ranks them.
The selected code is the one that produced the chosen subgoal, and the same code
conditions the bridge in the Route-Shared variant.

## 7. Representation diagnostics

Before any 1M run, on the held-out split, with PathBridger's own goal sampling:

```text
s = s_t
g = high_actor_goals
z = high_actor_targets     # s_{t+K}, K = planning horizon
c = q_phi(segment starting at t, length H_route)
```

Rows whose `H_route` window would cross an episode boundary are dropped.

The conditional variance estimator is the same kNN estimator as the local diagnostic
(the query is included in its neighbourhood; variance is the mean over dimensions with
`ddof=1`). Neighbourhood sizes 32, 64, and 128 are all written down. The gate reads
**k = 64 only**. It does not pick the k that looks best.

Also recorded: code usage, perplexity, dominant-code fraction, mean displacement,
endpoint centroid, between-code vs within-code endpoint distance, and between-code vs
within-code subgoal distance. The desired direction is between > within.

The comparison that matters is

| Task | Local intention variance ratio (seed 0) | Route intention variance ratio |
|---|---|---|

Local ratios are 0.954 / 0.984 / 0.910 for cube-double / puzzle-4x4 / cube-single.

## 8. Variance-reduction gate

This is an engineering gate for whether to spend a 1M control run. It is not a
statistical claim for a paper.

Primary:

```text
variance_ratio = Var(Z | NN(s, g), c_route) / Var(Z | NN(s, g))  <  0.70
```

That is a 30% drop. A secondary rule, also fixed before any route result, counts as
a pass when the ratio is below 0.85 and at least 0.10 below that task's seed-0 local
ratio. Otherwise the variance component fails and the oracle subgoal model is not trained.

## 9. Bridge consistency gate

Only after a route-conditioned bridge is trained:

```text
BridgeError(correct c) / BridgeError(shuffled c)  <  0.80
```

using normalised path MSE. If the route code helps the subgoal and not the bridge, the
Route-SG-only variant is the one that gets a control comparison. The bridge is still the
deterministic PathBridger bridge. Its target is the local segment `s_t ... s_{t+K}`,
even though `c` was read from the longer window.

## 10. Control experiment plan

Entered only for tasks whose representation gate is PASS on the finished seeds.

Variants, equal candidate budget:

| Variant | Subgoal | Bridge |
|---|---|---|
| PB | original PathBridger | original |
| Route-SG | route code | original |
| Route-Shared | route code `c` | same `c` |
| Route-Shuffled | same `z` as Route-Shared | `c' != c` |

Optional and only if it is cheap: Route-Bridge, original subgoal with a route-conditioned
bridge.

Comparisons: Route-SG vs PB, Route-Shared vs Route-SG, Route-Shared vs Route-Shuffled.

## 11. Ablations

Not a sweep. The first run uses one `H_route` (50), one `K` (8), and one encoder.
The endpoint feature is part of the relative trajectory, not a separate switch.
If the variance gate fails, there is no ablation that turns into a 1M run.

## 12. Possible failure modes

- The code collapses onto position even without an absolute-state input, because a long
  relative path still identifies where the agent is. Usage and endpoint centroids are
  checked for that.
- `H_route` is long enough to contain the subgoal, so a code could memorize `z` instead
  of a strategy. The variance ratio would then look good while the inference predictor
  `p(c | s, g)`, which cannot see the future, would be near chance. That still fails the
  spirit of a control experiment; top-1 accuracy is reported before a 1M run.
- The code reduces variance but the oracle flow does not, because the subgoal network
  cannot use the embedding. That is `ORACLE_FAIL` and stops the task.
- Route conditioning can also make control worse. That is a negative result, not a cue
  to change the gate.

## 13. Relationship to InFOM

Taken from InFOM: the idea of a latent behaviour intention.

Not taken:

- occupancy flow
- generative Q
- expectile critic
- discounted future occupancy
- the InFOM actor
- InFOM future anchors
- BOC
- TD InfoNCE

`c_route` is a discrete route/strategy latent for PathBridger's existing hierarchy
(subgoal, deterministic bridge, IDM). It is not an occupancy model and it does not
replace the critic.

## 14. Experimental status

Implementation and unit tests are in `route_intention/` and
`route_intention/tests/test_route_intention.py`.
Screening order: 9 tokenizers (200k), held-out variance diagnostics, then a 250k
oracle subgoal model only on task/seeds whose variance component passes, then the gate.
If every task fails, no 1M control job is launched and section I of
`exp/route_intention/aggregate/summary.md` records that.

Thresholds in `route_intention/common.py` are not to be edited after the diagnostics
are in.

## Training budget

| Stage | What | Budget |
|---|---|---|
| A | route tokenizer | 200k, 3 tasks x 3 seeds |
| B | variance diagnostic | no learning |
| C | oracle-c subgoal flow only | 250k, only if stage B passes |
| D | `p(c \| s, g)` | 150k, only if the full gate passes |
| E | route-conditioned subgoal + bridge | 1M, only on passing tasks, 3 seeds |

## Critical difference from the failed local experiment

```text
FAILED:  c_local = 5-step movement style. It can help the bridge. It does not pick z.
NEW:     c_route = 50-step trajectory strategy. It must predict z before any 1M run.
```

The change is the timescale, not a larger network.

## Verdict labels (filled only after the runs that the gate allows)

- **STRONG SUPPORT** — variance drops, the oracle subgoal improves, Route-Shared beats PB and beats Route-Shuffled.
- **SUBGOAL-ONLY SUPPORT** — variance drops and Route-SG improves, with no extra gain from the bridge.
- **REPRESENTATION SUPPORT, NO CONTROL** — the variance drop is real and control does not improve.
- **NO SUPPORT** — the route code does not meaningfully cut subgoal variance.
- **NEGATIVE** — route conditioning hurts both the representation and control.

The gate is not revised after control numbers exist.
