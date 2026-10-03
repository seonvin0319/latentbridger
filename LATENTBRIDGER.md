# LatentBridger

**LatentBridger is an experimental successor/ablation branch. It does not modify
the released PathBridger method.** `agents/pathbridger.py`, `utils/datasets.py`,
`utils/evaluation.py`, `configs/pbf/`, `configs/pbg/`, `main.py`, and
`evaluate.py` are byte-for-byte unchanged, so every published PathBridger
command, checkpoint, and number remains reproducible. LatentBridger adds
parallel files only.

## The hypothesis

PathBridger reaches the environment through an explicit chain in state space:
a scalar transitive value, a transitive-relabelling (TRL) ranking over sampled
endpoint candidates, an endpoint-pinned residual bridge, and an inverse-dynamics
model (IDM) that decodes each state transition into an action.

LatentBridger asks whether that whole chain can be deleted and replaced by a
learned latent interface:

1. learn an action-conditioned temporal contrastive critic
   `C(s, a, g) = phi_sa(s, a)^T psi(g) / tau`,
2. use `psi(g)` itself as the control interface for a latent-conditioned actor,
3. let a rectified flow generate a short *latent* trajectory prefix instead of
   proposing, ranking, and decoding explicit state subgoals.

There is no endpoint proposer, no TRL candidate ranking, no explicit
intermediate state subgoal, and no IDM anywhere in this branch.

## Objectives, exactly as implemented

### Module A, goal InfoNCE

For a batch of anchors `(s_i, a_i)` and trajectory-future positives
`g_i = s_{i + Delta_i}` drawn from the same episode,

```
L_ij        = phi_sa(s_i, a_i)^T psi(g_j) / tau
L_goal_nce  = mean_i [ -log( exp(L_ii) / sum_j exp(L_ij) ) ]
```

Positives are the matrix diagonal. With `repr_norm=True` (default) both `phi`
and `psi` are L2-normalized before the dot product, so every logit lies in
`[-1/tau, 1/tau]`. With `repr_norm=False` the embeddings are unnormalized and an
optional SGCRL-style penalty `logsumexp_coef * mean(logsumexp_j(L_ij)^2)` keeps
the logits bounded.

`Delta` is geometric by default, `P(Delta = k) proportional to gamma^k` with the
environment's configured `discount`, and is recorded in the batch as
`contrastive_offsets`. `future_sampling='uniform'` is the ablation.

The goal encoder uses the same `full` goal representation semantics as
PathBridger's `ScalarTransitiveValue`. The reduced `phi` endpoint representation
is reachable only by setting `goal_representation_mode` explicitly.

### State-only contrastive ablation

`C_state(s, g) = phi_s(s)^T psi(g)` replaces the anchor encoder for the
`state_cl` variant. Nothing tries to optimize actions through `C_state`; that
variant's actor is plain behaviour cloning conditioned on `psi(g)`.

### Module A, latent-conditioned actor

```
z_goal   = stopgrad( psi(g_actor) )
a_hat    = pi(s, z_goal)
L_actor  = -C_frozen(s, a_hat, g_actor) + lambda_BC * || a_hat - a_data ||^2
```

The actor consumes the *latent* `z_goal`, not the raw goal, because Module B
will later generate `z` directly. Actions are squashed with `tanh` and affinely
mapped onto the real `env.action_space.low/high`, so the actor cannot win
contrastive score by leaving the action box.

**The frozen-critic detail matters.** `C_frozen` is evaluated at the agent's
current critic parameters, which are *not* the differentiated parameters. The
critic output is never wrapped in `stop_gradient`; only the parameters are held
fixed. Gradient therefore flows `L_actor -> C -> a_hat -> pi`, which is the
entire point of the objective. Wrapping `C` in `stop_gradient` instead would
silently reduce the actor to pure BC.

`g_actor` defaults to `s_{t+1}` (`actor_goal_max_offset=1`), matching what
Module B emits: a local next latent target. `actor_goal_max_offset=5` is the
ablation that lets `g_actor` be any state within five transitions.

### Action-contrastive auxiliary loss

Goal InfoNCE contrasts *futures*, and a state usually determines its own future
distribution well enough that `phi_sa` can ignore `a` and still score perfectly.
The optional action-NCE loss contrasts *actions* directly. For each positive
`(s_i, a_i, g_i)` it builds `K` negatives by cyclically shifting the batch's
actions (`K = num_action_negatives`, default 16) and applies a `(1+K)`-way
softmax with the data action at index 0. No `B x B x ...` action grid is ever
materialized; the cost is `K` extra encoder rows.

```
L_critic = L_goal_nce + lambda_action_nce * L_action_nce
```

`action_nce_coef` is `0.0` for the plain SA-CL variants and `1.0` only in
`sa_cl_bc_actnce`.

### Module B, latent rectified flow

With `H_a = 5`, encode

```
z_s      = psi(s_t)
z_g      = psi(g)
Z_target = [ psi(s_{t+1}), ..., psi(s_{t+5}) ]        shape [B, 5, d]
```

and train a joint rectified flow over the whole `5 x d` prefix:

```
X_0       ~ N(0, I)
X_1       = Z_target
u         ~ Uniform(0, 1)
X_u       = (1 - u) X_0 + u X_1
V_target  = X_1 - X_0
L_flow    = mean || v_eta(X_u, u | z_s, z_g) - V_target ||^2
```

`v_eta` flattens the noisy prefix, concatenates `z_s`, `z_g`, and the flow time,
runs `3 x 512` GELU + LayerNorm, emits `5 * d`, and reshapes back to `[B, 5, d]`.
`psi` is stop-gradiented during the staged flow stage
(`flow_stop_psi_gradient=True`).

Inference integrates eight Euler steps from one `N(0, I)` draw scaled by
`flow_noise_scale`. When `repr_norm=True`, generated latents are renormalized
(`flow_renormalize=True`) so the actor is conditioned on the same unit-sphere
manifold it was trained on.

If the sampled goal occurs before `t+5`, the prefix is clipped at the goal and
the remaining entries are padded with it -- PathBridger's existing close-goal
behaviour. The conditioning goal uses ordinary trajectory-future sampling by
default, matching the semantic role of the released `dynamics_p` supervision.

## Why BC regularization is used

`L_actor` without BC maximizes `C(s, a_hat, g)` over a critic that was fit on the
offline action distribution. Outside that support the critic is extrapolating,
and the actor is free to walk into whatever region the extrapolation happens to
favour. The `lambda_BC || a_hat - a_data ||^2` term is a support constraint: it
keeps `a_hat` where the critic's scores were actually estimated.

The smoke numbers show the effect directly. `sa_cl` and `sa_cl_bc` share a
bit-identical critic and differ only in `actor_bc_coef`, yet the BC-free actor
sits at `0.417` mean-squared distance from the dataset action while the
regularized one sits at `0.017`.

## Why action sensitivity is a failure mode

A critic can reach excellent future-retrieval recall while `dC/da` is
approximately zero, because knowing *which future* a state leads to rarely
requires knowing *which action* was taken. Such a critic reports a healthy
contrastive loss and healthy recall, and gives the actor no usable gradient at
all; the actor collapses to whatever the BC term alone wants.

`diagnose_latent.py` therefore gates Module B on `action_sensitivity`, not on
retrieval. For a fixed `(s, g)` it compares `C(s, a_data, g)` against shuffled
dataset actions and against uniform actions inside the real action box. An
action-blind critic lands at probability `0.5` with a zero margin. The 2k-step
smoke run shows `0.614` / `0.736` for `sa_cl_bc` and `0.978` / `1.000` once
action-NCE is switched on, which is exactly the comparison
`sa_cl_bc_actnce` exists to make.

## beta-occupancy versus pi-occupancy

The contrastive positives come from the offline behaviour policy `beta`:
`g+ = s_{t+Delta}` is a sample from the *dataset's* discounted future state
occupancy, not from the occupancy of the policy being learned. `C(s, a, g)`
therefore scores "how characteristic is `g` of `beta`'s future after taking `a`
in `s`", which is a representation-learning signal, not a value function for
`pi`.

Two consequences are built into the design. Maximizing `C` cannot be read as
policy improvement, so the actor objective is support-constrained rather than
trusted as an off-policy value; and the latent flow is trained on `beta`'s own
five-step prefixes, so the latent targets it produces are in-distribution for
the controller rather than optimistic extrapolations.

## Why the flow predicts only a five-step prefix

The latent prefix is a *control* signal consumed one step at a time under
receding-horizon replanning, not a plan to be executed open-loop. Five steps is
PathBridger's existing action horizon, so the comparison is like-for-like, and
it is short enough that the flow stays inside the span where `beta`'s prefixes
are well determined by `(s_t, g)`. The horizon also bounds compounding error: a
fresh prefix is generated from the actual state at least every five actions.

## Receding-horizon latent control

`utils/latent_evaluation.py` exists because the released evaluator assumes the
agent emits a whole action chunk from one observation. Both LatentBridger modes
are closed-loop at every environment step.

- `direct_goal` — `a_t = pi(s_t, psi(g))` with `g` the final task goal, at every
  step. This is the Module-A test: no generated latent is anywhere in the loop,
  so a failure here cannot be blamed on the flow. It is also the practical
  "oracle latent target" test available without arbitrary environment state
  reset; nothing in this branch touches private simulator internals to fabricate
  one.
- `latent_flow` — generate `z_1..z_5` from the *current* observation, execute
  `a_i = pi(s_i, z_i)` against the *actual* updated state `s_i`, replan after at
  most `replan_interval` actions. Actions are never precomputed from a stale
  state: only the latent *target* is open-loop within a chunk.

## Architecture

| Module | Signature | Shape |
| --- | --- | --- |
| `phi_sa` | `StateActionEncoder(s, a)` | `[B, d]` |
| `phi_s` | `StateEncoder(s)` | `[B, d]` |
| `psi` | `GoalEncoder(g)` | `[B, d]` |
| `actor` | `LatentConditionedActor(s, z_goal)` | `[B, action_dim]` |
| `flow` | `LatentPrefixFlow(X_u, u, z_s, z_g)` | `[B, 5, d]` |

All five use `hidden_dims=(512, 512, 512)`, GELU, LayerNorm, and the repository's
existing `utils.networks.MLP`, matching PathBridger's network scale. Default
`repr_dim = 64`. All five live in one unified checkpoint regardless of variant,
so staged restore never has to reconcile checkpoint structures.

## Training stages

| Stage | Trains | Frozen |
| --- | --- | --- |
| `--stage=critic` | `phi_sa`, `phi_s`, `psi` | `actor`, `flow` |
| `--stage=actor` | `actor` | `phi_sa`, `phi_s`, `psi`, `flow` |
| `--stage=flow` | `flow` | everything else |
| `--stage=joint` | everything | — |

The default research protocol is A1 (critic) then A2 (restore, freeze critic,
train actor) then B (restore, freeze critic and actor, train flow). Joint
fine-tuning is optional and only meaningful after the staged experiments work.

Freezing is enforced by the optimizer, not by zeroing gradients: non-stage
modules are routed through `optax.set_to_zero` via `optax.multi_transform`. That
is strictly stronger than a zero gradient, which an Adam state carried in from a
previous stage would still turn into drift. Consequently `--restore_path`
restores **module parameters only** and each stage builds a fresh optimizer.
`tests/test_latentbridger.py` asserts the frozen subtrees are bit-identical
after an update, and the smoke run confirms it on real checkpoints.

## Experiment variants

| Variant | Critic | Actor input | Actor objective | `actor_bc_coef` | `action_nce_coef` | Flow | Tests |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `gcbc` | none | raw goal | BC | — | 0.0 | no | the no-representation floor |
| `state_cl` | `phi_s(s)^T psi(g)` | `psi(g)` | BC | — | 0.0 | no | does the representation alone help? |
| `sa_cl` | `phi_sa(s,a)^T psi(g)` | `psi(g)` | contrastive | 0.0 | 0.0 | no | contrastive actor without support control |
| `sa_cl_bc` | `phi_sa(s,a)^T psi(g)` | `psi(g)` | contrastive | 10.0 | 0.0 | no | **primary Module-A candidate** |
| `sa_cl_bc_actnce` | `phi_sa(s,a)^T psi(g)` | `psi(g)` | contrastive | 10.0 | 1.0 | no | does action sensitivity need explicit supervision? |
| `latent_rf` | `phi_sa(s,a)^T psi(g)` | generated `z_i` | contrastive | 10.0 | 0.0 | **yes** | does latent RF bridging add anything? |
| `latent_rf_actnce` | `phi_sa(s,a)^T psi(g)` | generated `z_i` | contrastive | 10.0 | 1.0 | **yes** | does Module B need an action-sensitive critic underneath? |

The table is built so each neighbouring comparison moves one knob.
`sa_cl` and `sa_cl_bc` differ only in `actor_bc_coef`; `sa_cl_bc` and
`sa_cl_bc_actnce` differ only in `action_nce_coef`; `latent_rf` differs from
`sa_cl_bc` only by adding Module B; `latent_rf_actnce` differs from
`sa_cl_bc_actnce` only by adding Module B, and from `latent_rf` only in
`action_nce_coef`.
`test_variant_table_isolates_one_knob_per_comparison` asserts this against
`VARIANT_SETTINGS` so the property cannot rot.

### Fair-ablation rules in force

Same dataset, same train/validation split, same batch size, same seed, same
per-stage update budget, same architecture scale, same five OGBench task IDs.
No online data, no reward, no TRL target.

One subtlety the runner handles for you. Several variants share whole stages:
`sa_cl`, `sa_cl_bc`, and `latent_rf` have identical *critic* stages, and
`latent_rf` additionally has the same *actor* stage as `sa_cl_bc` (likewise
`latent_rf_actnce` and `sa_cl_bc_actnce`). Separate processes do not reproduce
each other bitwise, because XLA's GPU autotuner selects kernels by measured
timing and the resulting low-order differences compound over 100k updates.

Rather than train nominally-identical stages that drift apart, the runner keys
each stage by a signature and trains it once:

- critic signature: `(critic_type, action_nce_coef)`
- actor signature: `(critic signature, actor_goal_input, actor_objective, actor_bc_coef)`

Shared checkpoints land in `_shared/<signature>/seed<N>/`. `latent_rf`
therefore provably reuses the exact Module-A representation **and controller**
that `sa_cl_bc` was evaluated with, so any difference between them is Module B
and nothing else.

### Replan interval

`latent_flow` evaluation takes `--replan_interval` (config default
`replan_interval=5`, the full action horizon). It is an evaluation-time knob,
not a training one: the same trained flow is driven at different replanning
rates.

- `replan_interval=5` executes the whole generated prefix before replanning.
- `replan_interval=1` keeps only `z_1` from each prefix and regenerates from
  the new actual state every step.

Comparing the two separates "the flow produces a useful *next* latent" from
"the flow produces a useful five-step *prefix*". If `r=1` matches or beats
`r=5`, the later prefix entries are not carrying their weight and the joint
five-step formulation is not earning its complexity.

## Diagnostics

`diagnose_latent.py` runs on the held-out validation split by default.

1. **Future retrieval** — rank the true future goal against the other in-batch
   goals: `Recall@{1,5,10}`, mean and median positive rank, positive-negative
   score gap.
2. **Action sensitivity** — `P(C_data > C_shuffled)`, `P(C_data > C_uniform)`,
   and both mean margins. **Required before Module B is trusted.**
3. **Actor imitation/support** — BC MSE, action saturation fraction, mean
   distance from the dataset action, and the critic score of the actor action
   versus the data action.
4. **Latent geometry** — mean `psi(s_t)^T psi(s_{t+Delta})` (cosine when
   normalized) for `Delta in {1, 2, 4, 8, 16, 32}`, against a random-state
   baseline. Anchors are restricted to starts whose `Delta`-step successor stays
   in the same episode.
5. **Flow reconstruction** — latent prefix MSE, per-step cosine similarity, and
   endpoint-conditioned retrieval consistency. Latent targets are never decoded
   back into raw states for this metric.

## CLI

Variants are selected with the ml_collections config-file argument
(`<config>:<variant>`), so the variant is fixed before any `--agent.*` override
is applied.

```bash
# Stage A1: contrastive critic.
python main_latent.py \
    --agent=configs/latent/cube_single.py:sa_cl_bc \
    --stage=critic --seed=0 --train_steps=100000 \
    --output_dir=exp/lb/cube_single/sa_cl_bc/seed0/critic

# Stage A2: restore the critic, freeze it, train the actor.
python main_latent.py \
    --agent=configs/latent/cube_single.py:sa_cl_bc \
    --stage=actor --seed=0 --train_steps=100000 \
    --restore_path=exp/lb/cube_single/sa_cl_bc/seed0/critic/checkpoints/params_100000.pkl \
    --output_dir=exp/lb/cube_single/sa_cl_bc/seed0/actor

# Stage B: restore critic + actor, freeze both, train the latent flow.
python main_latent.py \
    --agent=configs/latent/cube_single.py:latent_rf \
    --stage=flow --seed=0 --train_steps=100000 \
    --restore_path=exp/lb/cube_single/latent_rf/seed0/actor/checkpoints/params_100000.pkl \
    --output_dir=exp/lb/cube_single/latent_rf/seed0/flow

# Module-A evaluation (no generated latent in the loop).
python evaluate_latent.py \
    --agent=configs/latent/cube_single.py:sa_cl_bc \
    --checkpoint_dir=exp/lb/cube_single/sa_cl_bc/seed0/actor/checkpoints/params_100000.pkl \
    --mode=direct_goal --episodes=50 --seed=0 \
    --output_path=exp/lb/cube_single/sa_cl_bc/seed0/results/eval_direct_goal.json

# Receding-horizon latent-flow evaluation (replan every 5 actions).
python evaluate_latent.py \
    --agent=configs/latent/cube_single.py:latent_rf \
    --checkpoint_dir=exp/lb/cube_single/latent_rf/seed0/flow/checkpoints/params_100000.pkl \
    --mode=latent_flow --replan_interval=5 --episodes=50 --seed=0 \
    --output_path=exp/lb/cube_single/latent_rf/seed0/results/eval_latent_flow_r5.json

# Same flow, replanning every single action.
python evaluate_latent.py \
    --agent=configs/latent/cube_single.py:latent_rf \
    --checkpoint_dir=exp/lb/cube_single/latent_rf/seed0/flow/checkpoints/params_100000.pkl \
    --mode=latent_flow --replan_interval=1 --episodes=50 --seed=0 \
    --output_path=exp/lb/cube_single/latent_rf/seed0/results/eval_latent_flow_r1.json

# Offline diagnostics on the held-out split.
python diagnose_latent.py \
    --agent=configs/latent/cube_single.py:sa_cl_bc \
    --checkpoint_dir=exp/lb/cube_single/sa_cl_bc/seed0/actor/checkpoints/params_100000.pkl \
    --split=val --num_batches=8 --seed=0 \
    --output_path=exp/lb/cube_single/sa_cl_bc/seed0/results/diagnostics.json
```

### Suite runner and summarizer

```bash
# Smoke: 2k updates per stage, 2 eval episodes per task.
python scripts/run_latentbridger_suite.py \
    --config configs/latent/cube_single.py --seeds 0 \
    --preset smoke --save_dir exp/latentbridger_smoke

# Pilot: 100k updates per stage, 20 eval episodes per task.
python scripts/run_latentbridger_suite.py \
    --config configs/latent/cube_single.py --seeds 0,1,2 \
    --preset pilot --save_dir exp/latentbridger_pilot

# Full: 1M updates per stage, 50 eval episodes per task.
python scripts/run_latentbridger_suite.py \
    --config configs/latent/cube_single.py --seeds 0,1,2 \
    --preset full --save_dir exp/latentbridger_full

python scripts/summarize_latentbridger.py --root exp/latentbridger_pilot
```

Runner flags: `--config`, `--variants`, `--seeds`, `--preset`, `--dataset_dir`,
`--save_dir`, `--use_wandb`, `--skip_existing`, plus `--replan_intervals`,
`--diagnostic_split`, and `--dry_run`. A failed subprocess aborts the suite;
partial results are never summarized as complete.

```bash
# Both replanning rates for the flow variants.
python scripts/run_latentbridger_suite.py \
    --config configs/latent/cube_double.py --seeds 0,1,2 \
    --preset pilot --replan_intervals 1,5 --save_dir exp/
```

| Preset | critic | actor | flow | eval episodes | batch |
| --- | --- | --- | --- | --- | --- |
| `smoke` | 2,000 | 2,000 | 2,000 | 2 | 256 |
| `pilot` | 100,000 | 100,000 | 100,000 | 20 | 1024 |
| `full` | 1,000,000 | 1,000,000 | 1,000,000 | 50 | 1024 |

## Output layout

```
<save_dir>/<config-stem>/
    suite.json                         # manifest for the whole invocation
    commands.log
    summary.csv                        # written by summarize_latentbridger.py
    summary.md
    _shared/<stage-signature>/seed<N>/  # one critic / actor per signature
        checkpoints/params_<steps>.pkl
        train.csv
        flags.json
    <variant>/seed<N>/
        status.json                    # stages, budgets, stage provenance
        flow/{checkpoints,train.csv,eval.csv,flags.json}
        results/diagnostics.json
        results/eval_direct_goal.json
        results/eval_latent_flow_r<I>.json   # flow variants, one per interval
```

`status.json` records the critic and actor signatures and the exact checkpoint
each stage consumed, so a run's provenance is readable without re-deriving it.

## Configuration

`configs/latent/` covers all eight PathBridger environments: `antmaze_medium`,
`antmaze_large`, `cube_single`, `cube_double`, `cube_triple`, `puzzle_3x3`,
`puzzle_4x4`, `scene`. Each reuses only that environment's PathBridger
`env_name`, `horizon`, and `discount`. `endpoint_value_scale`,
`value_distance_weight_power`, `eval_num_candidates`, and `eval_temperature` are
deliberately **not** carried over, because explicit endpoint candidate selection
is removed.

```python
repr_dim                = 64
hidden_dims             = (512, 512, 512)
layer_norm              = True
repr_norm               = True
contrastive_temperature = 0.1
logsumexp_coef          = 0.0      # SGCRL-style penalty for repr_norm=False

future_sampling         = 'geometric'
bridge_goal_sampling    = 'trajectory'
actor_goal_max_offset   = 1
action_horizon          = 5

actor_bc_coef           = 10.0
action_nce_coef         = 0.0
num_action_negatives    = 16

flow_steps              = 8
flow_noise_scale        = 1.0
flow_renormalize        = True
flow_stop_psi_gradient  = True
replan_interval         = 5        # evaluation-time; 1 <= r <= action_horizon

learning_rate           = 3e-4
```

Every field is editable from a config file or a `--agent.<field>=` flag. The
four *structural* fields (`critic_type`, `actor_goal_input`, `actor_objective`,
`use_flow`) are owned by the variant: a config that disagrees with its variant
raises rather than silently producing an unnamed hybrid.

## Dataset

`utils/latent_datasets.py` provides `LatentBridgerDataset`, separate from
`PathBridgerDataset`. It respects compact OGBench episode boundaries exactly and
emits:

```
observations, next_observations, actions
contrastive_goals, contrastive_offsets
actor_goals,       actor_offsets
bridge_goals,      bridge_goal_offsets,  bridge_targets  # [B, 5, state_dim]
```

The TRL-only fields `base_goals`, `transitive_subgoals`, `transitive_offsets`,
and `transitive_valids` are absent; LatentBridger has no TRL target. Sampling
draws from the global NumPy random state exactly as the released sampler does,
so checkpoint save/restore reproduces the batch stream.

## Tests

```bash
python -m pytest tests/test_latentbridger.py -q
```

Covers: future-positive sampling and bridge targets never crossing episode
boundaries; close-goal clipping and padding; InfoNCE diagonal labels;
state-action and state-only critic shapes (and the state-only critic's action
invariance); actor actions inside configured bounds before and after training;
action-NCE shapes and non-zero finite gradients; latent-RF training shapes;
Euler integration producing `[B, 5, d]`; bit-identical frozen critic during an
actor-only stage; bit-identical frozen critic and actor during a flow-only
stage; the one-knob-per-comparison property of the variant table; and that the
original PathBridger still imports, updates, and emits a `[1, 5, action_dim]`
action chunk.
