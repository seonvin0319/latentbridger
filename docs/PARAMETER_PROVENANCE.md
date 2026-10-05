# Latent endpoint chunk parameter provenance

This branch starts from `sgcrl-raw-bridge-online` at `412b775`. It does not
reuse the monolithic action-conditioned critic or pure max-over-N planner from
the earlier chunk experiment.

## Inherited settings

| Parameter | Value | Provenance |
|---|---:|---|
| chunk horizon `H` | 5 | PathBridger `_ACTION_HORIZON` in `agents/pathbridger.py` |
| policy/proposal hidden widths | `(512, 512, 512)` | PathBridger `_HIDDEN_DIMS` |
| policy/proposal layer norm | `True` | PathBridger `_LAYER_NORM` |
| learning rate | `3e-4` | PathBridger `_LEARNING_RATE`; also SGCRL critic/actor default |
| representation dimension | 64 | SGCRL `agents/online_sgcrl.py:get_config` |
| representation normalization | `False` | SGCRL `agents/online_sgcrl.py:get_config` |
| contrastive hidden widths | `(256, 256)` | SGCRL `agents/online_sgcrl.py:get_config` |
| log-sum-exp coefficient | `0.01` | SGCRL `agents/online_sgcrl.py:get_config` |
| discount | `0.99` | SGCRL default and selected OGBench configs |

## NEW ABLATION PARAMETERS

The following values are research choices introduced by this branch. They are
marked together in `agents/latent_endpoint_chunk.py:get_config` and must not be
described as PathBridger or SGCRL defaults.

| Parameter | Default | Role |
|---|---:|---|
| `lambda_action` | 1.0 | Action-NCE coefficient |
| `num_action_negatives` | 16 | `q_beta(A|s,g)` negative samples |
| `proposal_min_scale` | `1e-3` | Numerical floor for proposal scale |
| `awr_beta` | 100.0 | AWR advantage temperature; selected after the 2k cube-single smoke showed weight collapse at 3.0 |
| `awr_weight_max` | 100.0 | AWR weight clipping |
| `td3bc_alpha` | 2.5 | TD3+BC score coefficient (ablation only) |
| `td3bc_bc_coef` | 1.0 | TD3+BC behavior-cloning coefficient |
| `lambda_support` | 0.05 | Behavior-support term in planning score |
| `support_logprob_threshold` | -100.0 | Candidate support filter |
| `score_normalization_eps` | `1e-6` | Candidate-score normalization floor |

`execute_h=2` is only a provisional primary setting. The main report always
contains `h in {1, 2, 5}`; it may be called primary only if cube-single paired
evaluation supports that choice.

## Evaluation protocol

Each seed receives a new manifest containing 50 episodes for each of five task
IDs (250 paired episodes). The older five-episode seed-0 manifest is neither
read nor copied. Direct and all support-guided modes restore the same unified
policy checkpoint. Planning modes add 2, 4, or 8 `q_beta` samples to the direct
candidate and execute only the first `h` actions before replanning.
