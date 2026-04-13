# DeepPeptide: V7 Precision Propeptide Specialist
Predicting exact propeptide cleavage boundaries in protein sequences.

[![DOI](https://zenodo.org/badge/593202385.svg)](https://zenodo.org/badge/latestdoi/593202385)

DeepPeptide has been completely overhauled from a generalized multi-task sequence tagger into a **Precision Propeptide Specialist (V7)**.

### Key Architectural Upgrades (V7 Baseline)
- **100-State Biological Ruler:** The CRF architecture enforces strict biological length constraints, a minimum propeptide length of 5, and explicit Cleavage Exits.
- **ESM3 Hardening Bottleneck:** Compresses massive 1536d `ESM3` features via `Linear -> LayerNorm -> ReLU -> Dropout(0.3)` before entering sequence modeling.
- **Decoupled Sniper Head:** Two parallel CNN branches (`kernel=5` for propeptide flavor and `kernel=1` for razor-sharp cleavage spikes) eliminate boundary smearing.
- **Auxiliary Focal Loss:** An auxiliary Focal Loss heavily penalizes the network if the exact single-pixel cleavage site is missed. In V7, this has been tuned (alpha=1.0, pos_weight=20.0, gamma=2.0) to prevent CRF destruction.
- **Removed MHSA:** The Multi-Head Self-Attention layer has been removed to reduce trainable parameters and prevent overfitting on small datasets.

### Training the V7 Model
1. Precompute ESM3 embeddings and extract targets.
2. Train the model using the built-in learning rate scheduler and optimized regularization:
```bash
python -m src.train_loop_crf \
  --embeddings_dir PATH/TO/ESM3_EMBEDDINGS \
  --data_file data/labeled_sequences_esm3.csv \
  --partitioning_file data/graphpart_assignments.csv \
  --label_type propeptides_only \
  --lr 1e-5 \
  --weight_decay 0.05 \
  --epochs 50 \
  --out_dir esm3_propeptide_v7
```

### Evaluation & Diagnostics
DeepPeptide V7 features an advanced diagnostic test suite to evaluate pixel-perfect matches versus biological "relaxed" tolerances.
- Validate Model Constraints before training: `python test_v6_sniper.py`
- Analyze Boundary Precision & Recall (Relaxed vs Exact):
```bash
python print_detailed_metrics.py --run_dir esm3_propeptide_v7
```

### Predicting

[See the predictor README](predictor/README.md)