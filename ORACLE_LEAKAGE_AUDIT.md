# Oracle leakage audit — learned-goalspace phase-2

Date: **2026-10-10**
Host: svcho
Branch base: `learned-goalspace-multihorizon` @ `2d10de3`
Scope: `LGS_TRL_W_FROZEN`, `PCA16_GS_TRL_W`, `RANDOM16_GS_TRL_W`, `MH_LGS_TRL_W_FROZEN`

**Verdict: phase-2 “learned” GS-TRL-W methods are oracle-assisted via the endpoint proposer.**

They must **not** be called non-oracle. High-level value/ranking use frozen `E`, but
`FlowEndpointProposer` / `GaussianEndpointProposer` still call
`goal_representation(..., 'phi', ...)`.

---

## Claim rule (this project)

A method is **non-oracle** only if **both**:

1. high-level planner / value / candidate ranking never use authoritative `phi`
2. endpoint proposer goal conditioning never uses authoritative `phi`

Bridge/IDM may use full state. Full state ≠ oracle.
Oracle `phi` may appear only in post-hoc probes/reporting.

---

## Component table — `LGS_TRL_W_FROZEN` (also PCA16 / Random16 / MH frozen DS)

| Component | Input used | Source |
|---|---|---|
| High-level value `V` | **learned representation** `E(·)` (16-D) | `FrozenLearnedGoalspaceAgent.value_loss` encodes obs/goals with frozen `goal_encoder`, then `LatentScalarValue` |
| Target value (EMA) | **learned representation** | same latent value head |
| Endpoint proposer goal conditioning | **authoritative phi** | `FlowEndpointProposer.__call__` → `goal_representation(goals, 'phi', phi_indices, env_name=...)` in `agents/pathbridger.py` |
| Endpoint proposer current state | **full observation** | concatenated with `phi(g)` |
| Proposer weighting | **learned representation** | overridden `_endpoint_weights` uses `E(s), E(g), E(z)` + target value |
| Candidate ranking (eval/act) | **learned representation** | `sample_action_chunks` scores with `V(E(s),E(z))*V(E(z),E(g))` |
| Bridge | **full observation** | `BridgeResidual(s_full, …)` |
| IDM | **full observation** | `InverseDynamics(s_full, s'_full)` |
| Dataset preprocessing | **full observation** | OGBench compact states; no phi filtering of trajectories |
| Evaluation policy goal input to agent | **full observation** goal from env `info['goal']` | then proposer internally maps goal → **phi** |
| Encoder `E` during downstream | frozen (lr=0) | FutureNCE / MH / PCA / Random params |

Identical proposer leakage applies to:

- `PCA16_GS_TRL_W`
- `RANDOM16_GS_TRL_W`
- `MH_LGS_TRL_W_FROZEN`
- `LGSDTRL_W_FROZEN`

---

## Evidence (proposer)

```133:138:agents/pathbridger.py
        goal_inputs = goal_representation(
            goals,
            'phi',
            self.phi_goal_obs_indices,
            env_name=self.env_name,
        )
```

`FrozenLearnedGoalspaceAgent.create` still constructs:

```258:262:learned_goalspace/downstream.py
        if endpoint_distribution == 'gaussian':
            endpoint = GaussianEndpointProposer(state_dim, env_name, phi_indices)
        elif endpoint_distribution == 'flow':
            endpoint = FlowEndpointProposer(state_dim, env_name, phi_indices)
```

and asserts phi indices (`infer_phi_goal_obs_indices` / `assert_phi_goal_obs_indices`).

`endpoint_loss` passes **full** `batch['endpoint_goals']` into `network.select('endpoint')`; the module then applies oracle phi internally. Overriding weighting/ranking to use `E` does **not** remove this path.

---

## Oracle GS-TRL / GS-TPB (ablation `gs_trl_weighted`) — for contrast

| Component | Input |
|---|---|
| High-level value | **authoritative phi** (`GoalSpaceScalarValue` → `goal_representation(..., 'phi')`) |
| Proposer goal conditioning | **authoritative phi** |
| Weighting / ranking | **authoritative phi** (scalar PB products) |
| Bridge / IDM / current state | **full observation** |

This is the intentional oracle baseline, not a leakage bug.

---

## Implications for reported phase-2 numbers

Puzzle h5 @1M (preserved):

| Method | h5@1M | High-level oracle? | Proposer oracle? | Truly non-oracle? |
|---|---:|---|---|---|
| Old FutureNCE + GS-TRL-W | 29.2% | no (`E`) | **yes (phi)** | **no** |
| PCA16 + GS-TRL-W | 57.2% | no (`PCA16`) | **yes (phi)** | **no** |
| Random16 + GS-TRL-W | 30.4% | no | **yes (phi)** | **no** |
| MultiHorizon + GS-TRL-W | 20.8% | no (`E`) | **yes (phi)** | **no** |

So PCA’s 57.2% is **oracle-assisted control** (learned/unsupervised goal embedding for value + oracle phi for proposal). It remains useful evidence about representation quality for value, but **not** a non-oracle method claim.

---

## Required fix (PROJECT: NonOracle-GS-TPB)

Replace proposer goal path with learned `E(g)` (typically `stop_gradient` for BTRL primary):

- `LearnedGoalFlowEndpointProposer(s_full, E(g), noise, t)` — **no** `goal_representation(...,'phi')`
- High-level `V(E(s), E(g))`
- Ranking / weighting on `E` only

First methods: `BTRL16`, then `PCA-BTRL16`.
