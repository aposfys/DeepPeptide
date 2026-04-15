# DeepPeptide: V8 Precision Propeptide Specialist
Predicting exact propeptide cleavage boundaries in protein sequences.

[![DOI](https://zenodo.org/badge/593202385.svg)](https://zenodo.org/badge/latestdoi/593202385)

DeepPeptide has been completely overhauled from a generalized multi-task sequence tagger into a **Precision Propeptide Specialist (V8)**.

### Key Architectural Upgrades (V8 Baseline)
- **100-State Biological Ruler:** The CRF architecture enforces strict biological length constraints, a minimum propeptide length of 5, and explicit Cleavage Exits.
- **ESM3 Hardening Bottleneck:** Compresses massive 1536d `ESM3` features via `Linear -> LayerNorm -> ReLU -> Dropout(0.3)` before entering sequence modeling.
- **Decoupled Sniper Head:** Two parallel CNN branches (`kernel=5` for propeptide flavor and `kernel=1` for razor-sharp cleavage spikes) eliminate boundary smearing.
- **Auxiliary Focal Loss:** An auxiliary Focal Loss heavily penalizes the network if the exact single-pixel cleavage site is missed. In V8, this uses alpha=2.0 and pos_weight=20.0.
- **Optimizer Stability:** Two parameter-groups to avoid learning starvation (`5e-4` bottleneck / `1e-4` remaining), Linear Warmup with Cosine Annealing, differential gradient clipping to protect the transition matrix, and Stochastic Weight Averaging (SWA).
- **Variance Regularization:** A variance penalty applied to `crf.start_transitions` prevents Viterbi from anchoring falsely to absolute sequence starts.

### Training the V8 Model
1. Precompute ESM3 embeddings and extract targets.
2. Train the model (e.g. V8-A Ablation variant) using the built-in learning rate scheduler and optimized regularization:
```bash
python -m src.train_loop_crf \
  --embeddings_dir PATH/TO/ESM3_EMBEDDINGS \
  --data_file data/labeled_sequences_esm3.csv \
  --partitioning_file data/graphpart_assignments.csv \
  --label_type propeptides_only \
  --epochs 100 \
  --alpha 2.0 \
  --out_dir esm3_propeptide_v8_A
```

### Evaluation & Diagnostics
DeepPeptide V8 features an advanced diagnostic test suite to evaluate pixel-perfect matches versus biological "relaxed" tolerances.
- Validate Model Constraints before training: `python test_v6_sniper.py`
- Inspect positional prior artifacting after training: `python inspect_start_transitions.py --model_path esm3_propeptide_v8_A/model.pt`
- Analyze Boundary Precision & Recall (Relaxed vs Exact):
```bash
python print_detailed_metrics.py --run_dir esm3_propeptide_v8_A
```

### Predicting

[See the predictor README](predictor/README.md)