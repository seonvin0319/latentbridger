# PCA/PCA full-state handoff

svcho trains a new initialization. This handoff does not contain a downstream checkpoint or the Eexec checkpoint.

The verified trainer is `scripts/run_eexec_fullstate_cross.py`. `ops/run_pca_fullstate_seed.py` calls `train_arm` with `source=pca`, `goal=pca`, latent width 16, and 32 candidates. Bridge and IDM stay on the normalized 55-dimensional observation.

```bash
python ops/run_pca_fullstate_seed.py \
  --handoff /path/to/pca_fullstate_seed1_20261013 \
  --seed 1 \
  --max-updates 400000 \
  --checkpoint-steps 100000 200000 400000 \
  --output /path/to/run \
  --smoke-only
```

`--smoke-only` checks hashes, the PCA projection, frozen encoder parameters, the full-state bridge/IDM tensors, and repeated seeded resets. It does not write a training checkpoint.
