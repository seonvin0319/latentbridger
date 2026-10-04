# SGCRL online semantics, and how this port reproduces them

Source: [`graliuce/sgcrl`](https://github.com/graliuce/sgcrl) at tree
`2d1b59db8b8909d25c69d2d6a8a5772bb9dd2509` — files `contrastive/config.py`,
`contrastive/learning.py`, `contrastive/networks.py`, `contrastive/builder.py`,
`contrastive/utils.py`, and the launcher `lp_contrastive.py`.

The original is an Acme + Reverb + Launchpad distributed program. This port
runs single-process on one GPU, so the pieces that exist only to coordinate
four actor processes with a replay server are replaced by their single-process
equivalent. Everything that changes what is *computed* is reproduced exactly.
Where the original's behaviour is a consequence of the distributed machinery,
the equivalent is derived below rather than guessed.

## 1. What the launcher actually runs

`lp_contrastive.main` builds the default experiment as:

| Setting | Value | Where from |
| --- | --- | --- |
| algorithm | `contrastive_cpc` → `use_cpc=True` | `--alg` default |
| `entropy_coefficient` | **`0.0`** | set explicitly in `main` |
| `use_random_actor` | `True` | set explicitly in `main` |
| `fix_goals` | `True` (`--sample_goals` defaults False) | `not FLAGS.sample_goals` |
| `use_td`, `twin_q`, `add_mc_to_td` | `False` | not touched by `contrastive_cpc` |
| `random_goals` | `0.5` | `config.py` default |
| `repr_dim`, `repr_norm` | `64`, `False` | `config.py` default |
| `hidden_layer_sizes` | `(256, 256)` | `config.py` default |
| learning rates | `3e-4` actor and critic, Adam `eps=1e-7` | `builder.make_learner` |
| `discount` | `0.99` | `config.py` default |
| `batch_size` | `256` | `config.py` default |
| `min_replay_size` / `max_replay_size` | `10_000` / `1_000_000` | `config.py` default |

Two of these are easy to get wrong.

**Entropy is off, not adaptive.** `config.py` defaults
`entropy_coefficient=None`, which selects the adaptive-alpha branch in
`learning.py`. But the launcher sets `entropy_coefficient = 0.0`, so
`adaptive_entropy_coefficient` is `False` and `alpha = 0.0`. The actor loss
keeps its `- alpha * entropy` term, multiplied by zero. The actor is still
stochastic (a tanh-Gaussian that it samples from during collection), it simply
gets no entropy bonus. The in-file comment "note this is log alpha" does not
match the code, which uses `config.entropy_coefficient` directly.

**Goals are fixed.** `fix_goals=True` means behaviour collection uses one goal
for the task, not a goal resampled from the environment's goal distribution.
This is the "single goal" condition the port must not break.

## 2. Critic objective

`learning.py:critic_loss`, the `use_td=False` branch with `use_cpc=True`:

```python
logits[i, j] = sa_repr_i · g_repr_j          # einsum('ik,jk->ij'), no norm, no temperature
loss         = softmax_cross_entropy(logits, I) + 0.01 * logsumexp(logits, axis=1) ** 2
```

Three details that matter:

- The softmax is over axis 1, so each anchor `(s_i, a_i)` is classified against
  the batch's goals. It is one-directional InfoNCE, not symmetric.
- `repr_norm=False`, so the logits are raw inner products of unnormalized
  64-dimensional vectors and there is no learned temperature. The
  `0.01 * logsumexp²` penalty is what keeps the logit scale bounded; it is not
  optional decoration.
- No target network is involved. `target_q_params` is only read on the
  `use_td` path, so with `use_cpc` the Polyak update at `tau=0.005` is computed
  and never used.

## 3. Actor objective

`learning.py:actor_loss` with `random_goals=0.5`:

```python
new_state = concat([state, state])                    # 2B rows
new_goal  = concat([goal,  roll(goal, 1, axis=0)])    # half hindsight, half shuffled
action    = sample(pi(new_state, new_goal))
loss      = -diag(Q(new_state, new_goal, action)) - alpha * entropy
```

So the actor is trained on a batch twice the critic's size: half the rows pair
a state with its own hindsight future goal, half pair it with another row's
goal. There is **no behavioural-cloning term anywhere** — the actor's only
signal is the critic score.

`actor_grad` differentiates with respect to `policy_params` only, so the critic
parameters are fixed while the gradient flows through the sampled action into
the policy. There is no `stop_gradient` on the critic output; this is the same
frozen-parameter / live-gradient pattern the offline code already uses.

## 4. Replay and hindsight relabeling

The buffer stores **whole episodes** (`adders_reverb.EpisodeAdder`), sampled
uniformly, with FIFO eviction and capacity
`max_replay_size // max_episode_steps` episodes.

`builder.flatten_fn` relabels a sampled episode:

```python
is_future_mask[t, j] = 1 if j > t else 0
probs[t, j]          = is_future_mask[t, j] * discount ** (j - t)
goal_index[t]        ~ Categorical(probs[t])
```

That is a geometric distribution over the future offset, truncated at the end
of the episode and renormalized, with `Δ >= 1` strictly. Goals never cross an
episode boundary. The last timestep is dropped because it has no future.

Then `transpose_shuffle` (batch → transpose → unbatch → unbatch) rearranges the
stream so that **the rows of a training batch come from different episodes**.
This is not cosmetic: the in-batch negatives of the InfoNCE loss are goals from
other episodes, and sampling a contiguous block of one episode instead would
make the negatives trivially distinguishable.

*Port:* instead of reproducing the TF pipeline, this port samples `B` episodes
uniformly, draws one timestep `t` uniformly within each, and draws `Δ` from the
same truncated geometric. This yields a batch whose rows come from `B` distinct
episodes with the same per-row goal distribution — the property the original
pipeline is constructed to produce.

## 5. Update-to-data ratio

The original's ratio is spread across three settings and a Reverb rate limiter:

- `samples_per_insert = 256` — for each episode inserted, 256 episodes are
  sampled (the table's items are episodes, not transitions).
- each sampled episode becomes `T - 1` transitions in `flatten_fn`.
- `batch_size = 256`, `num_sgd_steps_per_step = 64`, so a learner step consumes
  `256 * 64 = 16384` transitions and performs 64 updates.

Per episode inserted (`T` environment transitions), the learner consumes
`256 * (T - 1) ≈ 256·T` transitions, which is `256·T / 256 = T` gradient
updates. So:

> **one gradient update of batch 256 per environment transition (UTD = 1).**

*Port:* one update per environment step, batch 256. This is logged explicitly
as `train/updates_per_env_step` rather than left implicit, and it is held
identical across all three variants. The bridge module's updates are counted
and logged separately so that adding a bridge never buys variant C extra
critic or actor gradient steps.

## 6. Initially-random actor

`utils.InitiallyRandomActor` takes uniform random actions until the policy
parameters have been updated at least once, which it detects by checking
whether the first linear layer's bias is still all zeros. Since the learner
cannot step until replay holds `min_replay_size` transitions, this is exactly
"act uniformly at random until the replay buffer reaches 10k transitions".

*Port:* uniform random actions while `len(replay) < min_replay_size`. Those
transitions count against the environment interaction budget.

## 7. Networks

```
sa_encoder: MLP([256, 256, 64])  on concat(state, action)     # no final activation
g_encoder : MLP([256, 256, 64])  on goal
policy    : MLP([256, 256], activate_final=True) -> NormalTanhDistribution
```

No layer norm anywhere. Weight init is
`VarianceScaling(1.0, 'fan_avg', 'uniform')` for the encoders and
`VarianceScaling(1.0, 'fan_in', 'uniform')` for the policy trunk. Evaluation
uses the distribution's mode, collection samples from it.

**The policy's scale head is a softplus, not a log-std.** Acme's
`NormalTanhDistribution` emits `scale = softplus(x) + min_scale` with
`min_scale = 1e-3` and a zero-initialized bias, so the policy starts at
`softplus(0) + 1e-3 ≈ 0.694` and can never go below `1e-3`. This is not
cosmetic: with `entropy_coefficient = 0.0` there is nothing in the actor loss
rewarding a wide policy, so the only thing keeping collection stochastic is
that floor. A `exp(clip(log_std, -20, 2))` parameterization starts near `1.0`
and admits scales down to `2e-9`; swapping it in collapses the scale to
`~2e-3` within 2k updates and ends exploration. The port uses softplus.

`TanhTransformedDistribution` clips the event to `±0.99999997` before the
log-det-Jacobian correction. With `alpha = 0.0` the log-probability never
enters a gradient, so the port computes it only as the `actor/entropy`
diagnostic.

## 8. What this port deliberately does not reproduce

| Original | Port | Why |
| --- | --- | --- |
| 4 parallel actor processes | 1 actor | Changes throughput, not the update-to-data ratio or any gradient. |
| Reverb rate limiter | explicit UTD loop | The limiter's purpose is to hold the ratio derived in §5; the loop holds it exactly instead of in expectation. |
| `num_sgd_steps_per_step = 64` batching | 1 update per step | A device-side loop over 64 batches is an efficiency device. Same number of updates on the same data distribution. |
| Polyak target network | omitted | Never read on the `use_cpc` path. |
| `twin_q` | omitted | Off in the launcher. |
| Metaworld / 2D-nav environments | OGBench `cube-single-play-v0`, `task_id=1` | The comparison is against this project's offline results. |

## 9. The three variants under test

All three share §2–§7 exactly. They differ only in what the actor is
conditioned on.

| Variant | Actor conditioning | Purpose |
| --- | --- | --- |
| `online_sgcrl` | raw goal `g` | Faithful SGCRL baseline. |
| `online_sgcrl_latent` | `psi(g)`, the critic's goal encoder | Isolates the cost of the latent interface itself. |
| `online_sgcrl_latent_bridge` | `z_way`, the first waypoint of a rectified-flow prefix from `psi(s)` toward `psi(g)` | Adds the bridge, and nothing else, on top of the latent interface. |

`online_sgcrl_latent` and `online_sgcrl_latent_bridge` branch from one shared
run at 100k environment steps with identical critic parameters, actor
parameters, replay buffer, and RNG state, so the difference in their curves
after that point is attributable to the bridge.
