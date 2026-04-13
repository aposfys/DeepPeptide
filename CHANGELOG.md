# Changelog

## [V7] - Propeptide Cleavage Site Specialist Baseline
### Changed
- **Removed MHSA:** The `MultiheadAttention` and its associated normalizations/residual connections were fully removed from the `LSTMCNN` feature extractor to prevent overfitting on small datasets and reduce parameter count.
- **Dropout Adjustments:** Changed input dropout from 0.3 to 0.2, and conv dropout from 0.3 to 0.15. The ESM3 bottleneck dropout remains fixed at 0.3.
- **Focal Loss Tuning:** Overrode Focal Loss defaults for improved transition matrix learning. Set `alpha=1.0` (down from 5.0) and `pos_weight=20.0` (down from 50.0). `gamma=2.0` is kept.
- **Learning Rate Scheduler:** Increased `ReduceLROnPlateau` patience from 3 to 5 to allow more exploration before decaying learning rate during the 50-epoch runs.
- **Logging:** Updated the training loop to explicitly unpack, accumulate, and separately print the `crf_loss` and the `focal_loss` at the end of each epoch for easier analysis.

### Removed
- **Static Cleavage Bias:** The hardcoded `cleavage_bias = 2.0` added directly to the emissions logit was removed from `CRFBaseModel`. The Focal Loss and CRF transition constraints now handle the class imbalance without this override.

## [V6.1] - Precision Propeptide Specialist
### Added
- **100-State Biological Ruler:** CRF architecture enforcing biological length constraints.
- **ESM3 Hardening Bottleneck:** `Linear(1536, 256) -> LayerNorm -> ReLU -> Dropout` applied directly after frozen ESM3 embedding ingestion.
- **Decoupled Sniper Head:** Two parallel CNN branches (`kernel=5` for propeptide flavor, `kernel=1` for sharp cleavage spikes).
- **Auxiliary BCE/Focal Loss:** "Viterbi Breaker" loss penalizing missed single-pixel cleavage sites.

### Changed
- Shifted dataset and metric scopes exclusively to propeptides, treating active peptides as mature/background (State 0).
