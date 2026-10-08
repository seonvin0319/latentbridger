# Goal-Space Temporal Quasimetric — seed 0

Pilot status: **incomplete** (0/5 required 1M runs complete).

Host preflight: `blocked_no_accelerator`; devices=['TFRT_CPU_0'].
CPU smoke: complete (one puzzle update, checkpoint round-trip, diagnostics, and 5×1 rollout evaluation); alpha=0.040011756122112274, effective temperature=24.99265480041504.

Protocol: 1M updates; checkpoints 100k/300k/500k/800k/1M; five tasks × 50 episodes; execution h=5/2/1. No success threshold was imposed.

## Main table (1M, h=5, success %)

| Environment | CPB-rank | DTRL-W full | CTD-W full | GSDTRL-W | GSCTD-learned-temp | GSCTD-fixed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| puzzle_3x3 | 91.6 | 18.8 | 45.6 | — | — | — |
| cube_double | — | 84.4 | — | — | — | — |

Full-state baseline numbers shown above are the supplied comparison values; missing comparisons remain blank.

## Checkpoint rollout success (%)

| Environment | Method | h | 100k | 300k | 500k | 800k | 1M |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| puzzle_3x3 | GSDTRL-W | 5 | — | — | — | — | — |
| puzzle_3x3 | GSDTRL-W | 2 | — | — | — | — | — |
| puzzle_3x3 | GSDTRL-W | 1 | — | — | — | — | — |
| puzzle_3x3 | GSCTD-learned-temp | 5 | — | — | — | — | — |
| puzzle_3x3 | GSCTD-learned-temp | 2 | — | — | — | — | — |
| puzzle_3x3 | GSCTD-learned-temp | 1 | — | — | — | — | — |
| puzzle_3x3 | GSCTD-fixed | 5 | — | — | — | — | — |
| puzzle_3x3 | GSCTD-fixed | 2 | — | — | — | — | — |
| puzzle_3x3 | GSCTD-fixed | 1 | — | — | — | — | — |
| cube_double | GSDTRL-W | 5 | — | — | — | — | — |
| cube_double | GSDTRL-W | 2 | — | — | — | — | — |
| cube_double | GSDTRL-W | 1 | — | — | — | — | — |
| cube_double | GSCTD-learned-temp | 5 | — | — | — | — | — |
| cube_double | GSCTD-learned-temp | 2 | — | — | — | — | — |
| cube_double | GSCTD-learned-temp | 1 | — | — | — | — | — |

## Temporal and FutureNCE diagnostics (1M)

| Environment | Method | temporal MAE | d median | d p90 | d p99 | FutureNCE recall@1 | alpha | effective temperature |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| puzzle_3x3 | GSDTRL-W | — | — | — | — | — | — | — |
| puzzle_3x3 | GSCTD-learned-temp | — | — | — | — | — | — | — |
| puzzle_3x3 | GSCTD-fixed | — | — | — | — | — | — | — |
| cube_double | GSDTRL-W | — | — | — | — | — | — | — |
| cube_double | GSCTD-learned-temp | — | — | — | — | — | — | — |

## Proposer diversity (1M)

| Environment | Method | candidate pairwise distance | nearest-data distance | selected displacement | weight ESS fraction |
| --- | --- | ---: | ---: | ---: | ---: |
| puzzle_3x3 | GSDTRL-W | — | — | — | — |
| puzzle_3x3 | GSCTD-learned-temp | — | — | — | — |
| puzzle_3x3 | GSCTD-fixed | — | — | — | — |
| cube_double | GSDTRL-W | — | — | — | — |
| cube_double | GSCTD-learned-temp | — | — | — | — |

## Puzzle candidate-N sensitivity (1M, h=5)

| Method | N | success % | diversity | nearest-data distance | selected path cost | top1–top2 margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| — | — | — | — | — | — | — |

## Interpretation

No scientific conclusion yet: the required run set is incomplete. The one-step smoke validates execution plumbing only and is not performance evidence.
