# Non-oracle goalspace summary

Host: svcho  
Mission: **oracle goal abstraction removal**  
Branch: `nonoracle-goalspace-trl`

Oracle probes are post-hoc diagnostics only and never affect training or selection.

## Main table (puzzle h5 @1M)

| Representation | HL oracle? | Proposer oracle? | Puzzle h5 | button exact | nuisance R² | NN purity |
|---|---|---|---:|---:|---:|---:|
| Oracle phi GS-TPB | yes | yes | ~98% (ref) | — | — | — |
| PCA16 fixed + GS-TRL-W | no | **yes (leak)** | 57.2% | 1.000 | 0.799 | 0.978 |
| Random16 + GS-TRL-W | no | **yes (leak)** | 30.4% | 0.864 | 0.670 | 0.931 |
| Old FutureNCE + GS-TRL-W | no | **yes (leak)** | 29.2% | 0.015 | 0.078 | 0.841 |
| MultiHorizon + GS-TRL-W | no | **yes (leak)** | 20.8% | 0.717 | 0.395 | 0.928 |
| **BTRL16** | **no** | **no** | **17.6%** | TBD post-hoc | TBD | TBD |
| **PCA-BTRL16** | **no** | **no** | pending | pending | pending | pending |

BTRL16 h5 tasks @1M: `[0.84, 0.04, 0.00, 0.00, 0.00]` — **failed** as a general non-oracle abstraction. Artifacts preserved; do not rerun.

## Decision rule (after PCA-BTRL16 1M h5)

- **Case A (≥70%)**: cube-double PCA-BTRL16; no ACPR yet  
- **Case B (50–70%)**: cube-double PCA-BTRL16 + prepare ACPR  
- **Case C (<50%)**: no cube-double; proceed to ACPR  

## Queue

1. ~~BTRL16 puzzle seed0 1M~~ **done (17.6%)**
2. PCA-BTRL16 puzzle seed0 1M ← next
3. Case A/B/C automatic next step
4. Off2Online: **not on svcho** (Choi / oracle track)

## Artifacts

- BTRL16: `exp/nonoracle_goalspace/btrl16/puzzle_3x3/seed0/`
- PCA-BTRL16: `exp/nonoracle_goalspace/pca_btrl16/puzzle_3x3/seed0/`
- Audit: `ORACLE_LEAKAGE_AUDIT.md`
