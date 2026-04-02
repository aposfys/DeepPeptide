# DeepPeptide: The V6.1 Precision Propeptide Specialist

This document catalogs the complete architectural and mathematical transition from the original **DeepPeptide (V1)** multi-task generalist framework to the **V6.1 Precision Propeptide Specialist**.

---

## 1. Task Formulation: From Generalist to Specialist
**Original:** A multi-task sequence tagger designed to predict both functional peptides and propeptides simultaneously using a 101-state (2-branch) Conditional Random Field (CRF).
**V6.1:** We excised all active peptide coordinates from the training data (`dataset.py`). 100% of the neural network's capacity is now dedicated to the singular task of finding propeptide cleavage sites. Active peptides are mathematically forced to the background `Mature` state (State 0) to eliminate conflicting biological signals.

## 2. The State Space: The 101-State Biological Ruler
**Original:** Prodomains longer than 50 residues stalled in a `50 -> 50` self-loop, blinding the model to the exact biological length of the precursor and causing fuzzy C-terminal boundaries.
**V6.1:** The CRF matrix (`get_crf_constraints`) was rebuilt into a linear 101-state "Ruler":
- **State 0:** Mature Protein (Background).
- **States 1-99:** Propeptide Body. Climbs smoothly up to 99 for massive prodomains.
- **State 100:** The Cleavage Site. This is a dedicated, single-residue state representing the exact biological cut (e.g., the `RR` or `KR` motif).
- **Constraints:** The matrix strictly enforces entry (`0 -> 1`), a minimum biological length of 5 (`i -> 100` only allowed for `i >= 4`), and a mandatory exit *only* through the Cleavage State (`100 -> 0` or `100 -> 1`).

## 3. Embedding Ingestion: The ESM3 Bottleneck
**Original:** Ingested 1280d (ESM-1b) embeddings directly into a 64d CNN/BiLSTM. When faced with dense 1536d ESM3 embeddings, the model instantly overfitted and memorized the training set.
**V6.1:** Implements a "Hardening Bottleneck" (`Linear(1536, 256) -> LayerNorm -> ReLU -> Dropout`). `LayerNorm` stabilizes the massive variance of the LLM embeddings, while the linear projection compresses general protein semantics into 256 propeptide-relevant features before sequence modeling begins.

## 4. Contextual Sequence Modeling: Multi-Head Self-Attention
**Original:** Relied purely on a BiLSTM. LSTMs suffer from catastrophic forgetting over long sequences (e.g., 80+ residues), struggling to connect the N-terminal signal peptide cleavage event with the C-terminal propeptide cut.
**V6.1:** Bolts a PyTorch `MultiheadAttention` layer directly onto the BiLSTM output with a residual connection. This grants the model $O(1)$ global routing, allowing it to directly attend to distant biological motifs simultaneously, bypassing the sequential bottleneck of the LSTM.

## 5. Final Projection: The Decoupled "Split-Head" CNN
**Original:** Used a single `Conv1d(kernel=3)` to project hidden states into logits. This physically smeared the cleavage signal across a 3-residue window, forcing the CRF to guess the peak of a soft probability hill, destroying exact boundary matches.
**V6.1:** The final layer is split into two decoupled, parallel heads (`LSTMCNN.conv2_body` and `conv2_cut`):
- **The Body Head (`kernel=5`):** Uses a wide brush to sense the broad biological "flavor" of the propeptide body and the `-6 to +2` cleavage motifs.
- **The Sniper Head (`kernel=1`):** A pointwise convolution that outputs a razor-sharp, unsmeared Dirac delta spike exactly on the cleavage residue.
These are concatenated into the `[Mature, Body, Cleavage]` 3-class logits.

## 6. The Loss Function: The Auxiliary BCE "Viterbi Breaker"
**Original:** Optimized 100% via the Negative Log-Likelihood of the CRF. A CRF optimizes *global path likelihood*, meaning a cut at residue 49 is mathematically almost identical to a cut at residue 50. It does not punish "off-by-one" errors.
**V6.1:** Introduces an Auxiliary Binary Cross-Entropy (BCE) Loss hooked directly to the raw Cleavage logit *before* it enters the CRF.
- A massive `pos_weight=50.0` is applied to force the model to aggressively hunt for the rare cleavage site (combating the 1:100 class imbalance).
- The BCE Loss is multiplied by `alpha=5.0` and added to the CRF loss.
- **Result:** The CRF enforces the global biological length rules, while the BCE Loss acts as a ruthless sniper, annihilating CNN weights if the cleavage spike isn't pixel-perfect on the ground truth boundary, converting "Relaxed" matches into "Exact" matches.

## 7. Training Dynamics & Regularization
**Original:** Standard AdamW training with static learning rates caused the model to violently bounce out of optimal loss basins in late epochs (validation loss spiking from 4.0 to 9.0).
**V6.1:**
- Heavy regularization (`Dropout=0.3`, `Weight Decay=0.05`) combats ESM3 memorization.
- A highly reactive `ReduceLROnPlateau(patience=3)` scheduler automatically drops the learning rate by 50% the exact moment the validation loss plateaus, ensuring a soft landing in the absolute global minimum for 50-epoch runs.
- The `cleavage_bias` scalar added to the neural network emissions was bumped to `2.0` to further push recall without causing hallucinated false positives.

## 8. Evaluation & Analytics
**Original:** Relied on legacy metric scripts that merged active and propeptide evaluations.
**V6.1:**
- `manuscript_metrics.py` was rebuilt to parse the 100-state space, dynamically inferring boundaries by checking for transitions into State 0 or State 1.
- Pixel-wise accuracy and F1 were added to evaluate the "flavor" detection of the sequence independently of the object-level boundary matches.
- Added `present_v4_results.py` and `print_detailed_metrics.py` to seamlessly evaluate exact (0-tolerance) versus relaxed (±3, ±5 tolerance) boundary matches directly from the `test_outputs.pickle` file.