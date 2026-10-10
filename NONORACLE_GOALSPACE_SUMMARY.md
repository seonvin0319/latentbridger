# Non-oracle goalspace summary

Host: svcho
Mission: **oracle goal abstraction removal**

Oracle probes are post-hoc diagnostics only and never affect training or selection.

## Oracle leakage (phase-2)

See `ORACLE_LEAKAGE_AUDIT.md`.

Phase-2 FutureNCE / PCA / Random / MultiHorizon downstream methods use learned
`E` for high-level value/ranking but **still feed authoritative `phi` into
`FlowEndpointProposer`**. They are **oracle-assisted**, not non-oracle.

## Main comparison table (puzzle h5 @1M)

| Representation | HL oracle? | Proposer oracle? | Puzzle h5@1M | Cube | Notes |
|---|---|---|---:|---:|---|
| Oracle phi GS-TPB | yes | yes | ~98% (ref) | TBD | intentional oracle baseline |
| Old FutureNCE + GS-TRL-W | no | **yes** | 29.2% | — | leakage via proposer |
| PCA16 fixed + GS-TRL-W | no | **yes** | 57.2% | — | leakage via proposer |
| Random16 + GS-TRL-W | no | **yes** | 30.4% | — | leakage via proposer |
| MultiHorizon + GS-TRL-W | no | **yes** | 20.8% | — | leakage via proposer |
| **BTRL16** | **no** | **no** | TBD | TBD | TRL-only E, learned-goal proposer |
| PCA-BTRL16 | no | no | TBD | TBD | after BTRL16 |
| ACPR | no | no | TBD | TBD | only if TRL cases fail decision rule |

## Queue

1. BTRL16 puzzle-3x3 seed0 1M
2. PCA-BTRL16 puzzle-3x3 seed0 1M
3. Decision A/B → cube-double best TRL **or** ACPR
4. Off2Online only after BTRL16 + PCA-BTRL16 complete

## Branches

- `nonoracle-goalspace-trl` — BTRL16 / PCA-BTRL16
- `nonoracle-action-predictive` — ACPR (later)
- Off2Online kept separate (not mixed into these commits)
