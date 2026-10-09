# Learned goalspace comparison (phase-2)

Status: skeleton — fill after phase-2 queue artifacts land.
Live constraint: do **not** kill or overwrite `LGS_TRL_W_FROZEN` under
`exp/learned_goalspace/downstream/puzzle_3x3/LGS_TRL_W_FROZEN`.

**Oracle task features are used only for post-hoc representation diagnostics
and never affect training or experimental selection.**

No oracle probe is used for:

- method run/stop gates
- checkpoint selection (always predetermined **500k**)
- hyperparameter tuning

## Output layout

| Path | Contents |
| --- | --- |
| `exp/learned_goalspace/old_future_nce/` | Pointer note to archival FutureNCE pretrain + LGS downstream |
| `exp/learned_goalspace/pretrain/puzzle_3x3/fullobs_future_nce/seed0/` | Archival FutureNCE 500k |
| `exp/learned_goalspace/downstream/puzzle_3x3/LGS_TRL_W_FROZEN/seed0/` | Archival frozen scalar TRL 1M (live / do not overwrite) |
| `exp/learned_goalspace/pca16/puzzle_3x3/seed0/` | PCA-16 fit (`representation.pkl`) |
| `exp/learned_goalspace/random16/puzzle_3x3/seed0/` | Random-16 fit (`representation.pkl`) |
| `exp/learned_goalspace/downstream/puzzle_3x3/PCA16_GS_TRL_W/seed0/` | PCA frozen scalar TRL 1M |
| `exp/learned_goalspace/downstream/puzzle_3x3/RANDOM16_GS_TRL_W/seed0/` | Random frozen scalar TRL 1M |
| `exp/learned_goalspace/multihorizon/pretrain/.../fullobs_multihorizon_nce/seed0/` | Multihorizon NCE 500k |
| `exp/learned_goalspace/multihorizon/probes/.../` | Multihorizon probe CSVs + diagnostic `probe_summary.json` |
| `exp/learned_goalspace/downstream/puzzle_3x3/MH_LGS_TRL_W_FROZEN/seed0/` | MH frozen scalar TRL 1M (**unconditional** after successful MH pretrain) |

## Phase-2 order

1. Fit PCA16 on TRAIN offline observations only
2. `PCA16_GS_TRL_W` puzzle seed0 1M
3. Fit deterministic Random16
4. `RANDOM16_GS_TRL_W` puzzle seed0 1M
5. `fullobs_multihorizon_nce` puzzle seed0 500k
6. Post-hoc probes at 100k / 300k / 500k (diagnostic only)
7. `MH_LGS_TRL_W_FROZEN` puzzle seed0 1M **unconditionally**
   (skip only on technical failure: NaN, corrupt checkpoint, numerical collapse, implementation error)

All representation downstream transfers use the **predeclared** encoder checkpoint
`pretrain step = 500k`.

## Probe diagnostics (explanatory only)

| Representation | checkpoint | button_exact_state_accuracy | button_accuracy_mean | nuisance_r2 | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| FutureNCE `learned_E` | 500k | 0.0145 | 0.660 | 0.078 | archival; **over-invariance**, not collapse |
| Multihorizon `learned_E` | 500k |  |  |  | pending |
| PCA-16 (probe baseline) | — | 1.000 | 1.000 | 0.799 | from FutureNCE probe CSV |
| Random-16 (probe baseline) | — | 0.864 | 0.976 | 0.670 | from FutureNCE probe CSV |

Lower nuisance R² is **not** automatically better. Useful representations must
keep task-relevant state while discarding irrelevant variation. Old FutureNCE
has low nuisance R² **and** extremely low task exact accuracy → it discarded
useful button configuration together with proprio nuisance.

## Downstream success (puzzle-3x3, seed0, h=5)

| Method | encoder | 100k | 300k | 500k | 800k | 1M | notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Oracle phi + GS-TRL-W | oracle |  |  |  |  | ~0.98 | reference metadata |
| Full-state GS-TRL baseline | full |  |  |  |  |  | if available |
| `LGS_TRL_W_FROZEN` | FutureNCE E | 0.212 | 0.224 |  |  |  | archival / live to 1M |
| `PCA16_GS_TRL_W` | PCA-16 |  |  |  |  |  | phase-2 |
| `RANDOM16_GS_TRL_W` | Random-16 |  |  |  |  |  | phase-2 |
| `MH_LGS_TRL_W_FROZEN` | Multihorizon E |  |  |  |  |  | unconditional |

## Launch

```bash
# Preferred: wait for archival LGS, stop old queue (skip LGSDTRL/cube), start phase-2.
PYTHON=/home/svcho/anaconda3/envs/offrl/bin/python \
  nohup scripts/handoff_lgs_to_phase2.sh \
  > exp/learned_goalspace/phase2_handoff.log 2>&1 &

# Or, after LGS is already complete:
PYTHON=/home/svcho/anaconda3/envs/offrl/bin/python \
  scripts/run_learned_goalspace_phase2_16g.sh
```

Off2Online GPU remains paused until Old FutureNCE / PCA / Random / MH
downstream 1M results are all available.
