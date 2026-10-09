# Learned goalspace comparison (phase-2)

Status: skeleton — fill after phase-2 queue artifacts land.
Live constraint: do **not** kill or overwrite `LGS_TRL_W_FROZEN` under
`exp/learned_goalspace/downstream/puzzle_3x3/LGS_TRL_W_FROZEN`.

## Output layout

| Path | Contents |
| --- | --- |
| `exp/learned_goalspace/old_future_nce/` | Pointer note (see below) to archival FutureNCE pretrain + LGS downstream |
| `exp/learned_goalspace/pretrain/puzzle_3x3/fullobs_future_nce/seed0/` | Archival FutureNCE 500k (existing) |
| `exp/learned_goalspace/downstream/puzzle_3x3/LGS_TRL_W_FROZEN/seed0/` | Archival frozen scalar TRL 1M (live / do not overwrite) |
| `exp/learned_goalspace/pca16/puzzle_3x3/seed0/` | PCA-16 fit (`representation.pkl`) |
| `exp/learned_goalspace/random16/puzzle_3x3/seed0/` | Random-16 fit (`representation.pkl`) |
| `exp/learned_goalspace/downstream/puzzle_3x3/PCA16_GS_TRL_W/seed0/` | PCA frozen scalar TRL 1M |
| `exp/learned_goalspace/downstream/puzzle_3x3/RANDOM16_GS_TRL_W/seed0/` | Random frozen scalar TRL 1M |
| `exp/learned_goalspace/multihorizon/pretrain/.../fullobs_multihorizon_nce/seed0/` | Multihorizon NCE 500k |
| `exp/learned_goalspace/multihorizon/probes/.../` | Multihorizon probe CSVs + `probe_summary.json` |
| `exp/learned_goalspace/downstream/puzzle_3x3/MH_LGS_TRL_W_FROZEN/seed0/` | MH frozen scalar TRL 1M (gated) |

`exp/learned_goalspace/old_future_nce/README.md` documents the archival paths
without relocating live run directories.

## MH downstream gate

After multihorizon pretrain + probes, write `multihorizon/probe_summary.json`.
Do **not** auto-start `MH_LGS_TRL_W_FROZEN` unless:

1. `--force-mh-downstream`, or
2. gate file `multihorizon/ALLOW_MH_DOWNSTREAM` exists, or
3. `learned_E` `button_exact_state_accuracy` at 500k is **materially > 0.05**.

## Probe summary (puzzle-3x3, seed0)

| Representation | checkpoint | button_exact_state_accuracy | button_accuracy_mean | nuisance_r2 | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| FutureNCE `learned_E` | 500k | 0.0145 | 0.660 | 0.078 | archival; not collapse |
| Multihorizon `learned_E` | 500k |  |  |  | pending |
| PCA-16 (probe baseline) | — | 1.000 | 1.000 | 0.799 | from FutureNCE probe CSV |
| Random-16 (probe baseline) | — | 0.864 | 0.976 | 0.670 | from FutureNCE probe CSV |

## Downstream success (puzzle-3x3, seed0, h=5)

| Method | encoder | 100k | 300k | 500k | 800k | 1M | notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `LGS_TRL_W_FROZEN` | FutureNCE E | 0.212 | 0.224 |  |  |  | archival / live to 1M |
| `PCA16_GS_TRL_W` | PCA-16 |  |  |  |  |  | phase-2 |
| `RANDOM16_GS_TRL_W` | Random-16 |  |  |  |  |  | phase-2 |
| `MH_LGS_TRL_W_FROZEN` | Multihorizon E |  |  |  |  |  | gated |

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
