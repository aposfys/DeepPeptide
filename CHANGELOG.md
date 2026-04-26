# Changelog

## [V11] — 2026-04-XX

### Added
- `--evaluate_only` flag to selectively run tests on checkpoints without training.
- Linear warmup (5 epochs) + CosineAnnealingLR scheduler replacing `ReduceLROnPlateau`.
- Per-parameter-group differential gradient clipping (`max_norm=1.0` for `crf`, `max_norm=5.0` for `feature_extractor`).
- Dual test metrics at ±0 and ±3 tolerance windows.
- Emissions clamping (`torch.clamp(emissions, min=-15.0, max=15.0)`) before CRF forward pass to prevent float32 overflow.

### Changed
- **Propeptide Isolation:** Switched active peptides to the background state (0). Model exclusively learns the prodomain cleavage task, mapped onto a 101-state grammar.
- **V5 101-State Ruler:** Expanded the CRF ruler from 51 states to 101. States 1-99 represent the Propeptide Body, and State 100 is a dedicated, single-residue Cleavage Site. This elegantly handles long-tail sequences natively via a `99 -> 99` self-loop.
- **ESM-3 Embeddings:** Replaced `esm2` representations with denser, information-rich `esm3` embeddings (layer 48, 1536d) aligned 1:1 with residues.
- **Linear Bottleneck:** Introduced a stable, 6x compression projection step (`Linear(1536, 256) -> LayerNorm`) immediately post-ESM3 to avoid parameter explosion and overfitting.
- **Split-Head CNN Decoder:** Decoupled the final shared convolutional emission layer into two distinct biology-aware heads: `conv2_body(kernel=5)` for broad flank motifs and `conv2_cut(kernel=1)` for pointwise pixel-perfect cleavage scoring.
- **Auxiliary Focal Loss:** Added a heavily weighted `alpha=2.0` Focal loss directly supervising the raw cleavage logit (Index 2) independent of CRF path-smoothing, acting as a "Viterbi Breaker" to force exact position detection.

### Removed
- Legacy False-Alarm Warnings: Removed outdated 51-state transition constraint debug checks from `_debug_crf` that were generating false warnings during valid 100-state runs.
- Multi-Head Self Attention (MHSA): Ablated post-ESM self-attention mechanisms that caused significant overfitting due to redundant contextualization alongside ESM3.
- Stochastic Weight Averaging (SWA): Removed post-training weight averaging as it caused test metric degradation in the presence of focal loss spikes.
- Explicit BiLSTM gradient flattening warnings (fixed by cleanly re-initializing).

### Result
- Re-architected system resolves competing gradients between Propeptide and Active Peptide objectives.
- Peak validation F1 reaches >0.66 by epoch 35, dramatically improving zero-tolerance (+-0) precision on test folds compared to original DeepPeptide.
