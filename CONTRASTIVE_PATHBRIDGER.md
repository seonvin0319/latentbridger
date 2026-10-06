# Calibrated Contrastive PathBridger

See [the design and release-semantics record](docs/CPB_DESIGN.md).

Use the isolated Python 3.11 environment (versions in requirements-cpb.lock.txt):

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-cpb.lock.txt
.venv/bin/python scripts/run_contrastive_pathbridger_suite.py
```

Resume the sequential suite without retraining completed runs:

```bash
.venv/bin/python scripts/run_contrastive_pathbridger_suite.py --resume --skip_existing
```

Other supported options: `--configs`, `--variants`, `--seeds`, `--train_steps`,
`--dataset_dir`, `--save_dir`, and `--use_wandb` (requires the optional wandb package).
Primary evaluation uses h=5; h=2 and h=1 use the same checkpoint.

```bash
.venv/bin/python evaluate_contrastive_pathbridger.py \
  --run-dir exp/contrastive_pathbridger/cube_single/cpb_full/seed0 \
  --checkpoint 1000000 --h 5
.venv/bin/python scripts/summarize_contrastive_pathbridger.py
```

Outputs and checkpoints remain under exp/ and are not committed. The old raw-score
experiment is archived separately; it is not a calibrated CPB result. Existing
PathBridger and latent-endpoint source files and original branches are unchanged.
