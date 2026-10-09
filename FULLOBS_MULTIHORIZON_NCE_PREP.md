# Prep: `fullobs_multihorizon_nce`

Status: **prepared, not launched**.
Trigger: Case C from `LEARNED_PHI_DIAG_300K.md` (puzzle LGS-TRL-W 300k h5 = 22.4%).
Constraint: do not kill the live `fullobs_future_nce` → LGS-TRL-W 1M queue.

## Spec (locked by parent plan)

- Shared goal encoder \(E\): same MLP 512³ → latent 16 (pre-norm export).
- Query towers (separate projections only):
  - `q_short` for positives with offset in `[1, 5]`
  - `q_medium` for `[6, 20]`
  - `q_long` for `[21, +∞)` clipped by episode terminal
- Score: normalized \(q_\cdot^\top \mathrm{normalize}(E(g)) / \tau\), \(\tau=0.1\)
- Loss: \(L_\mathrm{short}+L_\mathrm{medium}+L_\mathrm{long}\) (row-wise InfoNCE each)
- Do **not** tie \(E(s)=E(g)\)
- Same aug (noise 0.01·std, feature dropout 0.05), Adam 3e-4, batch 1024→512 fallback
- Env order: puzzle-3x3 seed0 500k, then frozen LGS-TRL-W downstream 1M
- Checkpoints: 100k / 300k / 500k; downstream uses predetermined 500k

## Implementation plan (when approved)

1. Extend `learned_goalspace/dataset.py` with band-aware geometric / band sampling that never crosses terminals.
2. Add `FutureMultiHorizonNCE` trainer next to `FutureNCEPretrainer` (three query modules, one goal encoder).
3. Wire `main_learned_goal_pretrain.py --variant fullobs_multihorizon_nce`.
4. Reuse probes + `LGS_TRL_W_FROZEN` freeze path unchanged.
5. Tests: band bounds, terminal clip, distinct query heads, frozen E zero-update, existing suite green.
6. Launch only under the 16GiB wrapper, after the current queue cell finishes or on a free GPU — never in parallel with the live LGS run on this single GPU.

## Non-goals until diagnosis accepted

- No Off2Online GPU
- No kill / soft-restart of current LGS-TRL-W
- No oracle φ in the multihorizon loss
