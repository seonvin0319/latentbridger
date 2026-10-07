# Contrastive-Transitive Distance PathBridger

CTD-PB keeps the PathBridger endpoint proposer, endpoint-pinned state bridge,
inverse dynamics, dataset goal mixtures, and transitive value losses. The free
scalar `V(s, g)` is replaced by a directed temporal quasimetric.

```text
d(x, y) = ||h(x) - h(y)||_2 + mean_r ReLU(p(y)_r - p(x)_r)
V_d(s, g) = gamma ^ d(s, g)
```

`h` and `p` come from one tied encoder. State and goal do not have separate
networks, because `d(s, z) + d(z, g)` has to live in one geometry. There is no
extra triangle penalty: the L2 term and the coordinate-wise ReLU potential
satisfy the triangle inequality by construction.

## What TRL and InfoNCE each do

The original TRL losses stay. A short future at offset `k` still has target
`gamma^k`, which asks `d(s, s_{t+k})` to sit near `k`. The transitive target
is still `V(s, g) <- V(s, z) V(z, g)`. In distance that is
`d(s, g) ~= d(s, z) + d(z, g)`. The expectile, the validity mask for offsets
above 5, the target EMA at `tau = 0.005`, and
`value_distance_weight_power` are unchanged.

FutureNCE reads this same `d` and adds one scalar goal-only nuisance head:

```text
F(s, g) = b(g) - d(s, g) / tau_C
```

The head absorbs goal-marginal effects without becoming a second critic.
`b(g)` is used only by FutureNCE: never by TRL, proposer weighting, ranking,
PathNCE, or the bridge. The positive is the existing geometric
same-trajectory future goal (`critic_p = (0, 1, 0, 0)`). `tau_C` is the local
bridge horizon: 40 on the cube tasks and 25 on puzzle and antmaze. The
reported term is row-wise InfoNCE divided by `log(B)`.

TRL anchors temporal scale and transitive composition. InfoNCE additionally
pushes the distance to reflect discounted future occupancy relative to the
background goal marginal in the batch. The intended reading of `d` is an
occupancy-grounded transitive temporal distance. InfoNCE by itself is not a
claim that `gamma^d` equals a literal visitation probability.

## PathNCE

PathNCE is a multi-positive ranking objective over observed intermediate
states. For fixed `(s, g)`, it scores each candidate `y` by

```text
S(s, y, g) = -[d(s, y) + d(y, g)] / 5
```

It does not regress the path residual to zero. The residual
`d(s,y)+d(y,g)-d(s,g)` is diagnostic only, since demonstrations may contain
detours. Up to four strictly internal states are sampled per row. Any
in-batch candidate from the same episode and inside the anchor's ordered
interval is promoted to the positive set rather than used as a false
negative. PathNCE touches only the temporal metric.

## Seed-0 method order

Every method runs all environments in the order cube-double, puzzle-3x3,
antmaze-medium, cube-single before the next method starts:

1. `ctd_weighted`
2. `dtrl_weighted`
3. `ctd_pathnce_weighted`
4. `ctd_uniform`
5. `dtrl_uniform`
6. `ctd_pathnce_uniform`
7. `ctd_pathnce_weighted_bridgegeo`
8. `ctd_pathnce_uniform_bridgegeo`

Weighted variants use the original PB formula
`min(5, exp(endpoint_value_scale * (V_target(z,g)-V_target(s,g))))`.
Uniform variants use exactly one. All proposers train through 1M; there is no
freeze or early stopping.

BridgeGeo is reserved for methods 7–8. It keeps original bridge L1 active,
uses the actual clipped endpoint offset, freezes metric parameters on this
loss path, and keeps gradients through predicted bridge states. Its fixed
coefficient is 0.1.

Inference samples the original flow proposer and, when `N > 1`, chooses
`argmin_z d(s, z) + d(z, g)`. That order matches
`argmax_z V_d(s, z) V_d(z, g)`. The selected endpoint goes through the original
bridge prefix and IDM.

## Explicitly unchanged or excluded

Environment settings match the PBF configs: horizon, discount, `N`, evaluation
temperature, `endpoint_value_scale`, and `value_distance_weight_power`.

This experiment does not add an actor, an action-conditioned distance, a
separate contrastive critic, a Cbar bank, quasimetric distillation, a triangle
penalty, I-invariance, T-invariance, or a second proposer.

## Scientific questions

- Geometry learning: does nuisance-normalized FutureNCE improve TRL distance,
  and does observed-intermediate PathNCE add beyond it?
- Proposal distribution: does original PB-style target-value weighting retain
  useful endpoint coverage from 500k to 1M?
- Explicit bridge execution: after methods 1–6 settle high-level behavior,
  does frozen-metric BridgeGeo improve first-state, prefix, or IDM errors?

Claims stay conditional. Same-trajectory intermediates are observed valid
states, not asserted optimal subgoals. FutureNCE supplies discrimination while
TRL anchors temporal scale and transitivity. Weighting is not called
mode-preserving unless the measured candidate coverage supports that claim.
