# Learned φ diagnosis at puzzle LGS-TRL-W 300k

Updated: **2026-10-10 00:56 KST**
Host: svcho
Live run: **not killed** — `LGS_TRL_W_FROZEN` / puzzle-3x3 / seed0 continues past 300k toward 1M.
Primary method name: **LGS-TRL-W** (LGSDTRL remains secondary).
Off2Online GPU: **not started**.

Oracle task features are used only for post-hoc representation diagnostics
and never affect training or experimental selection.

Frozen encoder under test: pretrain `fullobs_future_nce` **500k** checkpoint
`exp/learned_goalspace/pretrain/puzzle_3x3/fullobs_future_nce/seed0/checkpoints/params_500000.pkl`

---

## 1. Downstream success (puzzle / LGS-TRL-W)

| ckpt | h=5 | h=2 | h=1 |
|---|---:|---:|---:|
| 100k | 0.212 | 0.232 | 0.188 |
| **300k** | **0.224** | **0.288** | **0.284** |

300k task breakdown (h5): `[0.58, 0.14, 0.20, 0.08, 0.12]`

Oracle GS-TRL / GSDTRL puzzle references (~97–98% at 1M) remain far above this.
100k→300k h5 only **+1.2pp** — not a healthy early climb.

### Decision rule

\[
\text{h5}=22.4\% < 30\% \;\Rightarrow\; \textbf{Case C}
\]

- Keep the current frozen LGS-TRL-W run to **1M** (no kill).
- Mark primary `fullobs_future_nce` as **likely insufficient** for oracle-like goal abstraction
  (over-invariance: low nuisance R² with near-zero task exact accuracy — **not** latent collapse).
- Run phase-2: PCA → Random → MH pretrain → MH downstream **unconditionally**
  (no oracle probe gate; predetermined encoder checkpoint 500k).
- Do **not** start Off2Online GPU work yet.

---

## 2. Representation probes (frozen \(E\), post-hoc only)

Baselines: PCA-16 and fixed random-16 on the same full observations.
Oracle φ is used only for scoring; it has no gradient into \(E\).

### A. Oracle task-feature linear probe (button state)

| representation @500k | button_accuracy_mean | button_exact_state_accuracy |
|---|---:|---:|
| **learned_E** | **0.660** | **0.0145** |
| pca16 | 1.000 | 1.000 |
| random16 | 0.976 | 0.864 |

Per-button learned_E accuracies @500k:
`[0.850, 0.526, 0.809, 0.543, 0.525, 0.538, 0.832, 0.545, 0.767]`
Several dimensions sit near chance (~0.5). Exact 9-button configuration recovery is essentially failed.

Pretrain-step trend for learned_E mean button acc: 100k 0.664 → 300k 0.659 → 500k 0.660 (flat).

### B. Nuisance probe (robot proprioception)

| representation @500k | nuisance_r2 | nuisance_mse |
|---|---:|---:|
| **learned_E** | **0.078** | 0.126 |
| pca16 | 0.799 | 0.027 |
| random16 | 0.670 | 0.045 |

Learned \(E\) is **not** strongly encoding proprioceptive identity (nuisance R² low).
That is good relative to a pure physical-state encoder, but it did **not** convert that compression into a usable task quotient.

### C. Nearest-neighbor task-state purity

| representation @500k | NN purity | NN exact purity |
|---|---:|---:|
| **learned_E** | 0.841 | 0.627 |
| pca16 | 0.978 | 0.800 |
| random16 | 0.931 | 0.578 |

Learned neighborhoods are weaker than PCA-16 and only modestly better than random on exact purity.

### D. Representation health @500k (`learned_E`)

| metric | value |
|---|---:|
| effective_rank | 14.23 |
| per_dim_std_mean | 4.73 |
| per_dim_std_min | 3.02 |
| norm_mean | 18.82 |
| norm_std | 3.42 |

Latent is close to full 16-D rank (not collapsed), with growing norm across pretrain (100k→500k). Collapse is **not** the failure mode.

### E. Pretraining InfoNCE retrieval

Batch size 1024; metrics at saved pretrain checkpoints:

| pretrain step | loss | recall@1 | positive_rank | positive_score | negative_score |
|---|---:|---:|---:|---:|---:|
| 100k | 5.076 | 0.123 | 118.1 | 4.379 | 0.357 |
| 300k | 4.754 | 0.172 | 98.1 | 4.735 | 0.096 |
| 500k | 4.712 | 0.191 | 93.1 | 4.846 | 0.077 |

Contrastive retrieval improves, but remains weak for a 1024-way InfoNCE (recall@1 ≈ 0.19).
Contrastive progress **did not** translate into oracle button-state recovery.

---

## 3. Interpretation (required questions)

| If … | Then … | Observed? |
|---|---|---|
| oracle probe poor **and** nuisance high | FutureNCE encodes physical-state identity | **No** — nuisance R² is low |
| oracle probe high **and** downstream low | task info present; geometry/value cannot use it | **No** — oracle probe is poor |
| oracle probe high **and** nuisance high | latent not quotienting nuisance | **No** |

**Actual pattern:**

- Oracle button probe **poor** vs PCA/random (exact-state acc **1.5%**).
- Nuisance probe **low** (R² ≈ 0.08).
- Downstream h5 **stuck ~22%** at 300k.

So standard FutureNCE is **not** mainly leaking proprio. It is failing to form a task-equivalent quotient: contrastive/temporal signal is present, but button-configuration abstraction is not recovered. Geometry/value learning therefore has little usable \(\hat\phi\) to exploit.

Primary claim risk if we stopped here: “learned goal-space PathBridger” would be limited by **representation**, not by scalar TRL vs quasimetric.

---

## 4. Actions

1. **Do not kill** current puzzle LGS-TRL-W 1M run (Case C: preserve + continue).
2. Treat `fullobs_future_nce` as **primary-failure candidate** for oracle-free abstraction on puzzle-3x3.
3. **Prepare** `fullobs_multihorizon_nce` (short/medium/long bands, separate query heads, shared \(E\)); do not launch until this diagnosis is accepted.
4. **Off2Online GPU**: blocked until learned-φ feasibility is clearer.
5. Keep **LGS-TRL-W** as primary downstream name; LGSDTRL secondary only.

### Multihorizon prep checklist (next implementation, not launched)

- Reuse q/E capacity; shared goal encoder \(E\) → 16-D.
- Horizon bands: short 1–5, medium 6–20, long 21+ (terminal-clipped).
- Separate query projections `q_short`, `q_medium`, `q_long`; loss \(L_s+L_m+L_l\).
- Same aug / τ / Adam / 500k / seed0 / puzzle first.
- Export frozen \(E\) into the same LGS-TRL-W downstream path.
- Do not force \(E(s)=E(g)\).

---

## 5. Report snapshot (for go/no-go)

| item | value |
|---|---|
| downstream 300k h5 / h2 / h1 | **0.224 / 0.288 / 0.284** |
| oracle button mean / exact @500k E | **0.660 / 0.0145** |
| nuisance R² @500k E | **0.078** |
| NN purity / exact @500k E | **0.841 / 0.627** |
| InfoNCE recall@1 @500k | **0.191** |
| effective rank @500k E | **14.23** |
| decision | **C (<30%) → continue 1M + prepare multihorizon** |
| live LGS step after eval | ≥304k, queue active |
