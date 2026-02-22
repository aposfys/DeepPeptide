# DeepPeptide

This is the DeepPeptide predictor, updated to use ESM3 (sm-open-v1).


## Usage

1. Install the requirements in `requirements.txt` using `pip install -r requirements.txt`.
2. Ensure you are in the `predictor` directory. (This is important so the model paths are correct.)
3. Run the predictor using `python3 predict.py -ff fasta_file.fasta -od testrun_outputs/`.
**Note:** You must have valid model checkpoints compatible with the propeptide-only architecture (51 states) in `checkpoints_esm1b/` or `checkpoints_esm2/`.

| Argument | Short | Default | Description |
|---------|-------|-------|-------|
| `--fasta_file` | `-ff` | `None` (required) | Path to the fasta file containing the protein sequences to predict. |
| `--output_dir` | `-od` | `None` (required) | Path to the directory where the output files will be saved. |
| `--batch_size` | `-bs` | `10` | Batch size for prediction. Use this to tune memory usage according to your hardware. In general, larger batches are better, but a batch needs to fit the memory constraints of the given hardware. |
| `--output_fmt` | `-of` | `json` | Output format. `json` (default) is faster. `img` produces plots (if enabled). |
| `--esm` | | `esm2` | Which ESM version to use (`esm2` or `esm1b`). |
| `--esm_pt` | | `None` | Optional path to a local ESM checkpoint (.pt). |

### Input format

The predictor takes fasta-formatted protein sequences as input. Multiline and 1-line formats work.

### Output format

`predict.py` produces the following files in the output directory:
- `propeptide_predictions.json`: A JSON file containing the propeptide predictions for each protein in the input data.
- `sequence_outputs.json`: A JSON file containing the per-position probabilities and predictions for each protein in the input data.
- `output.md`: A markdown file that displays predictions as tables and includes plots. This is the output that is displayed on the webserver.
- `SEQUENCE_NAME.png`: A plot of the predictions for a single protein sequence.
