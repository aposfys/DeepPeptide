# Changelog

## [Unreleased] — V10

### Added
- `--evaluate_only` flag to selectively run tests on checkpoints without training.

### Changed
- Replaced the single linear projection bottleneck (`1536 -> 256`) with a `ResidualBottleneck` (+~65K params) utilizing `LayerNorm -> Down -> GELU -> Up(Residual) -> LayerNorm`.
- Shifted Stochastic Weight Averaging (SWA) start from Epoch 20 to Epoch 40 to prevent averaging early sub-optimal weights in the new 100-epoch structure.

### Configuration
- `alpha=2.0`, `epochs=100`, SWA start `40`, `ResidualBottleneck`.

### Result
- TBD.

---

## [V9] — 2026-04-XX

### Added
- Emissions clamping (`torch.clamp(emissions, min=-15.0, max=15.0)`) before CRF
  forward pass to prevent log-sum-exp float32 overflow caused by auxiliary
  Focal loss driving cleavage logits to large values.

### Removed
- `start_transitions` variance regularization term (`start_reg`) from total loss.
  Diagnostic confirmed `start_transitions` are healthy (max 0.49, well-distributed);
  the variance penalty caused magnitude explosion by driving all 101 values
  toward a large shared constant.
- `inspect_start_transitions.py` diagnostic script (finding incorporated into
  architecture understanding).

### Result
- V9-B (`alpha=2.0`): Test F1 ±3 = 0.596 (SWA checkpoint).
- Stable run confirmed: Clamping successfully eliminated CRF float32 overflow without hurting learning dynamics.
- Best epoch 35 val F1 ±3 = 0.664.

---

## [V8] — 2026-04-XX

### Added
- Linear warmup (5 epochs) + CosineAnnealingLR scheduler replacing
  ReduceLROnPlateau, preventing reactive-only LR reduction.
- Per-parameter-group differential gradient clipping: `crf` at max_norm=1.0,
  `feature_extractor` at max_norm=5.0 to protect CRF transition matrix from
  large auxiliary loss gradients.
- Stochastic Weight Averaging (SWA) starting at epoch 20, saving
  `swa_model.pt` at end of training.
- `start_transitions` variance regularization to prevent positional prior
  collapse (later removed in V9 — see above).
- `inspect_start_transitions.py` diagnostic script.
- Parameter breakdown logging before optimizer initialization.
- Dual test metrics at ±0 and ±3 tolerance windows.

### Changed
- Default `alpha` set to 2.0 (geometric midpoint between V7-C=1.0 and V7-A=5.0).
- Global gradient clipping replaced with per-module clipping.
- Verbose scheduler argument removed (deprecated in PyTorch ≥2.2).

### Result
- Best checkpoint: epoch 13, val F1 ±3 = 0.6726 (new project high).
- CRF loss exploded after epoch 13 due to `start_reg` side effect.
- Test F1 ±3 = 0.5123 (SWA checkpoint degraded by averaging exploded weights).

---

## [V7-A] — 2026-04-XX — Best stable run

### Configuration
- `alpha=5.0`, `epochs=50`, ReduceLROnPlateau scheduler.

### Result
- Test F1 ±3 = 0.609, Precision ±3 = 0.707, Recall ±3 = 0.535.
- Best val F1 ±3 = 0.671 at epoch 15; unstable after due to Focal gradient
  overwhelming CRF transition matrix without gradient clipping.
- Exceeds original DeepPeptide baseline (~0.55 F1 ±3).

---

## [V7-C] — 2026-04-XX

### Configuration
- `alpha=1.0`, `epochs=50`.

### Result
- Test F1 ±3 = 0.558, stable training throughout 50 epochs.

---

## [V7-B] — 2026-04-XX

### Configuration
- `alpha=0.0` (pure CRF, no Focal loss).

### Result
- Test F1 ±3 = 0.471. Stable but plateaus early without boundary signal.

---

## [V6 / Initial ESM3 Migration]

### Added
- ESM3 hidden state extraction (layer 48) replacing one-hot encoding.
- `nn.LayerNorm(1536)` as entry point to normalize ESM3 variance (σ≈243 → 1).
- `[1:-1]` BOS/EOS token slicing for 1:1 residue-to-embedding alignment.
- `allowed_starts` relaxed from `[0, 1]` to `list(range(51))` (Anchor Release).
- `propeptides_only` label type: all internal peptides mapped to State 0,
  only propeptide coordinates mapped to States 1–100.
- 101-state CRF ruler with dedicated State 100 as cleavage site.
