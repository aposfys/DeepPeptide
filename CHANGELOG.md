# Changelog

## [Unreleased] - Propeptide-Only Prediction Refactor

### Changed
- **Labeling System (`src/utils/dataset.py`):**
    - Modified `PrecomputedCSVForOverlapCRFDataset` (and others) to ignore regular peptide coordinates.
    - Mapped propeptide coordinates to label states 1-50 (previously 51-100).
    - States 1-50 now represent the propeptide sequence and cleavage site.
    - Disabled/commented out unused label types (`binary`, `start_stop`, `simple`, `multistate`, `intensity`) to simplify the codebase.

- **Model Architecture (`src/models/crf_models.py`, `predictor/model.py`):**
    - Updated `CRFBaseModel` and `get_crf_constraints` to support a single branch state space (1 background state + 50 propeptide states).
    - Removed legacy multi-branch logic (which supported 101 states for both peptide and propeptide).
    - Updated `_repeat_emissions` to handle the reduced number of labels (2 labels: Background, Propeptide).

- **Training Configuration (`src/train_loop_crf.py`):**
    - Updated model initialization to use `num_labels=2` and `num_states=51` when `with_propeptides` is in the label type.

- **Metrics (`src/utils/manuscript_metrics.py`):**
    - Updated constants: `PEPTIDE_START_STATE, PEPTIDE_END_STATE = -1, -1` (disabled).
    - Updated constants: `PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE = 1, 50`.
    - Metrics now report performance for propeptides using the new state definitions.

- **Predictor Tool (`predictor/`):**
    - **`predict.py`**:
        - Default output format changed to `json` (images disabled by default).
        - Removed `matplotlib` and `seaborn` plotting logic to make the tool lightweight and faster.
        - Output JSON (`propeptide_predictions.json`) now only contains propeptide predictions.
    - **`utils.py`**:
        - Updated model loading to expect 51-state models.
        - Updated `combine_crf` to ensemble 51-state CRFs.
        - Updated post-processing to extract propeptides from states 1-50.
        - Commented out plotting dependencies.
    - **`write_markdown_output.py`**:
        - Removed figure references from the Markdown output.

### Removed
- **Plotting Functionality**: Generating plots for every sequence is now disabled/removed to reduce "noise" and dependencies.
- **Peptide Prediction**: The system no longer predicts or evaluates regular peptides.
