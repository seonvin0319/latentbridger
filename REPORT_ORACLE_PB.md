# Oracle PB on latest official PathBridger — report

**Method name:** Oracle PB (formerly called LatentBridger / GS-TPB in older notes)  
**Backbone pin:** `SChoish/PathBridger` `main` @ `e592665923973655696889d0882c245bc7d2b314`  
**Worktree:** `/home/shchoi/PathBridger-paper`  
**Train suite output:** `/home/shchoi/oracle-pb-on-pb-e592665-20261014/` (does not overwrite prior / legacy 3999 runs)

## 1. Prior definition (old names → Oracle PB)

Source of truth for the *definition* transplanted here:

- Repo: `latentbridger-goalspace-ablation` (historical)
- Variant: **`gs_trl_weighted`** (= Oracle PB)
- Mechanism: `GoalSpaceScalarValue` — PB scalar MLP on **`φ(s), φ(g)`** only
- Not transplanted: CTD / GSDTRL / contrastive / quasimetric / symmetric variants

| Piece | Prior PB | Prior Oracle PB (`gs_trl_weighted`) |
| --- | --- | --- |
| Value | full-state scalar TRL | `V_φ(φ(s), φ(g))` |
| Proposer | full-state endpoints | same |
| Scoring / proposer weights | full-state V | goal-space V |
| Bridge / IDM / actor | full-state | full-state |

## 2. Official PB `e592665` vs older local PB stacks

Latest pushed PB already includes the documented TRL / flow / IDM / actor stack used by paper reproduction. Relative to older local checkouts, treat **`e592665` as the loss/backbone SSOT** (value-distance weighting, target/online value selection, clipping, transitive normalization, flow matching, IDM/actor, goal mixes, env hypers). This port does **not** retune those; it only adds `pathbridger_mode`.

## 3. Code changes on latest PB (this port)

| File | Change |
| --- | --- |
| `pathbridger_core/agents/critic.py` | `ScalarValueNet.state_representation`; `pathbridger_mode` → `phi/phi` value (action critics stay full) |
| `pathbridger_core/agents/dynamics.py` | Borrowed value head respects `subgoal_value_state_representation` |
| `utils/core_config.py` | Wires `pathbridger_mode` → value reps for critic + proposer weighting |
| `agents/pathbridger.py` | Public default `pathbridger_mode='pb'` |
| `configs/oracle_pb/*.py` | Paper PBF hypers + `pathbridger_mode='oracle_pb'` for all envs |
| `tests/test_oracle_pb.py`, `tests/test_pb_equivalence_mode.py` | Correctness / PB-mode checks |
| `ORACLE_PB.md`, `scripts/run_oracle_pb_train_suite.sh` | Docs + wait-for-free-GPU train suite |

**Modes**

- `pathbridger_mode=pb` — identical semantics to stock PB (full-state V)
- `pathbridger_mode=oracle_pb` — goal-space V + scoring + value-dependent proposer weights; proposer/bridge/IDM/actor unchanged

## 4. PB numerical equivalence (`mode=pb` vs stock `e592665`)

**Result: PASS, bit-exact** (max_abs = 0 on all compared tensors).

Protocol (`tests/probe_pb_update_trace.py` + `tests/compare_pb_traces.py`):

- Stock worktree: `/home/shchoi/pb-stock-e592665-equiv` @ `e592665`
- Ours: `/home/shchoi/PathBridger-paper` with `pathbridger_mode=pb`
- Env: `puzzle-3x3-play-v0`, dataset `~/.ogbench/data`, seed=1, batch=256, CPU
- Steps: **16** consecutive `agent.update` calls
- Compared: init params, Adam opt_state, RNGs, per-step scalar losses, post-update params, EMA `modules_target_value` / `modules_target_action_critic`

Artifacts: `/home/shchoi/oracle-pb-on-pb-e592665-20261014/equiv/pb_mode_vs_stock_e592665_16.json`  
**Code fix required: none.**

Structural PB checks also pass (`tests/test_pb_equivalence_mode.py`).

## 5. Oracle PB correctness + gradient tests

Structural (`tests/test_oracle_pb.py`): all PASS.

Gradient / path checks (`tests/test_oracle_pb_gradients.py`): all PASS.

1. TRL value grads nonzero under Oracle PB; value invariant to non-`φ` channels
2. Proposer weighting uses `stop_gradient` on value params (no value update via weighting); weighting signal still `φ`-sensitive
3. Action critic remains full-state (robot channels affect Q)
4. Bridge trajectory last-dim = full `obs_dim`; IDM actions from full-state path
5. PB value remains sensitive to robot channels

## 6. Environment results (PB vs Oracle PB)

**Status:** suite PID waiting on legacy GPU (`pb-3999-puzzle-1m-seed0`, ~978k/956k @ check time). Does not interrupt those jobs.

Order when free: smoke 2k (p33/p44 × PB/Oracle PB) → seed0 1M all envs → puzzle seeds 1–3.

Output root: `/home/shchoi/oracle-pb-on-pb-e592665-20261014/`  
Provenance: `manifests/suite_provenance.json` + per-run `run_manifest.json` (commit, config path, seed, eval protocol).

**Prior** runs under the old name stay in `latentbridger-goalspace-ablation` — do not mix tables.

## 7. Ablation triage (run after seed0 mains)

From prior goalspace ablation notes (`gs_trl_weighted` = Oracle PB; CTD/GSDTRL/quasimetric = out of scope):

| Priority | Ablation | Why |
| --- | --- | --- |
| P0 | A. PB vs Oracle PB | Core claim; already in suite |
| P1 | B. Scoring-only GS value vs scoring+proposer weighting | Separates the two Oracle PB deltas; highest info value after A |
| P2 | Candidate N coverage (existing puzzle4 N-sweep) | Reuse prior; re-run only if seed0 Oracle PB disagrees |
| defer | State aliasing deep dive | Prior diagnostics exist; only if puzzle gap unexplained |

Do **not** reintroduce contrastive / quasimetric / GSDTRL variants.

## 8. Projection pros / aliasing limits

**Pros:** high-level transitive relations evaluated in task-relevant goal coordinates (buttons / cube poses / maze xy) without changing low-level generation.

**Limits:** states that share `φ(s)` but differ in robot configuration are aliased in V; low-level path still sees full state, so execution can disambiguate, but high-level selection cannot.

---

## Execution log

- Canvas / monitoring loops for unrelated legacy suite: stopped for this work
- Correctness + PB-mode tests: **PASS**
- Train suite: `nohup scripts/run_oracle_pb_train_suite.sh` → waits for free GPU
