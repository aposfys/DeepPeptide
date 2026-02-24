# Changelog - Propeptide-Only Prediction Refactor

### Changed

#### **1. Dataset & Labeling Logic (`src/utils/dataset.py`)**
    *   **Class `PrecomputedCSVForOverlapCRFDataset` (Lines 539-543):**
        *   *Before:* Loaded both `peptides` and `propeptides` from the CSV. Sampled from both to create a merged label.
        *   *After:* Explicitly passes an empty list `[]` for peptides to `_sample_from_overlapping_peptides`.
        *   *After:* Calls `peptide_list_to_label_sequence` with `start_state=1` and `max_len=50` for propeptides. This re-maps the propeptide state space from the original `[51-100]` range to the primary `[1-50]` range.
    *   **Class `PrecomputedCSVDataset` (Lines 81-135) & `PrecomputedCSVForCRFDataset` (Lines 325-394):**
        *   *Action:* Commented out code blocks for unused `label_type` options (`binary`, `start_stop`, `multistate`, etc.).
        *   *Reason:* To enforce the single-purpose scope and reduce code complexity/noise. Only `multistate_with_propeptides` is active.

#### **2. Model Architecture (`src/models/crf_models.py`, `predictor/model.py`)**
    *   **Method `get_crf_constraints` (Lines 31-75):**
        *   *Before:* Accepted an `n_branches` argument. If `n_branches=2`, it constructed a 101-state transition matrix (0-50 for Peptide, 51-100 for Propeptide).
        *   *After:* Removed `n_branches` argument. Removed the `if n_branches == 2` block entirely. Changed default `max_len` from 60 to **50**.
        *   *Result:* The method now always produces a 51-state transition matrix corresponding to a single entity type.
    *   **Method `_repeat_emissions` (Lines 81-86):**
        *   *Before:* Handled 3-class input (Background, Peptide, Propeptide).
        *   *After:* Removed the 3-class logic. Now assumes 2-class input (Background, Propeptide) and broadcasts the "Propeptide" emission score to states 1-50.

#### **3. Training Configuration (`src/train_loop_crf.py`)**
    *   **Function `get_model` (Lines 48-77):**
        *   *Change:* Updated constructor calls for `LSTMCNNCRF`, `SimpleLSTMCNNCRF`, and `SelfAttentionCRF`.
        *   *Values:* `num_labels` set to **2** (was 3). `num_states` set to **51** (was 101).
        *   *Reason:* Aligns the network output head and CRF layer with the new dataset schema.

#### **4. Inference & Predictor (`predictor/predict.py`, `predictor/utils.py`)**
    *   **Output Format (`predict.py`):**
        *   *Change:* Default `--output_fmt` is now `json`.
        *   *Change:* Output filename changed to `propeptide_predictions.json`.
        *   *Change:* Plotting logic (`compute_marginals`) defaults to `False`.
    *   **Post-Processing (`predict.py`, Lines 109-110):**
        *   *Logic:* Explicitly sets `peptides = []`. Calls `convert_path_to_peptide_borders` looking for states `1` to `50` and assigns them to `propeptides`.
    *   **Model Loading (`utils.py`, Lines 138, 158):**
        *   *Logic:* Hardcoded `num_states=51` when instantiating the model class for loading checkpoints. This ensures compatibility with the new architecture.
    *   **Lightweighting (`utils.py`, Lines 15, 264+):**
        *   *Action:* Removed/Commented out `matplotlib` and `seaborn` imports and the `plot_predictions` function.
        *   *Reason:* Removes heavy dependencies and speeds up startup/execution for users who only need JSON data.

#### **5. Metrics (`src/utils/manuscript_metrics.py`)**
    *   **Constants (Lines 12-13):**
        *   `PEPTIDE_START_STATE, PEPTIDE_END_STATE` -> `-1, -1` (Disabled).
        *   `PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE` -> `1, 50` (Mapped to new state space).

### Removed
- **Plotting Functionality**: Generating plots for every sequence is now disabled/removed to reduce "noise" and dependencies.
- **Peptide Prediction**: The system no longer predicts or evaluates regular peptides.

