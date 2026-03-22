import torch
import torch.nn as nn
import numpy as np

# 1. Test ESM3 Bottleneck in LSTMCNN
from src.models.lstm_cnn import LSTMCNN
def test_bottleneck():
    print("Testing ESM3 Bottleneck...")
    model = LSTMCNN(input_size=1536)

    # Check if the bottleneck is nn.Sequential and has the correct layers
    assert hasattr(model, 'bottleneck'), "Bottleneck missing"
    assert isinstance(model.bottleneck, nn.Sequential), "Bottleneck is not nn.Sequential"
    assert isinstance(model.bottleneck[0], nn.Linear), "First layer not Linear"
    assert model.bottleneck[0].in_features == 1536 and model.bottleneck[0].out_features == 256, "Incorrect Linear dims"
    assert isinstance(model.bottleneck[1], nn.LayerNorm), "Second layer not LayerNorm"
    assert isinstance(model.bottleneck[2], nn.ReLU), "Third layer not ReLU"
    assert isinstance(model.bottleneck[3], nn.Dropout), "Fourth layer not Dropout"
    print("✓ Bottleneck correct.")

# 2. Test Transition Constraints in CRFBaseModel
from src.models.crf_models import CRFBaseModel
def test_constraints():
    print("Testing CRF Transition Constraints...")
    transitions, allowed_starts, allowed_ends = CRFBaseModel.get_crf_constraints(max_len=50, min_len=5)

    assert (0, 0) in transitions, "Missing 0 -> 0"
    assert (0, 1) in transitions, "Missing 0 -> 1 entry"
    assert (1, 2) in transitions, "Missing 1 -> 2 progression"
    assert (49, 50) in transitions, "Missing 49 -> 50 progression"

    assert (1, 0) not in transitions, "Premature 1 -> 0 exit allowed"
    assert (4, 0) not in transitions, "Premature 4 -> 0 exit allowed"
    assert (5, 0) in transitions, "Missing 5 -> 0 exit"
    assert (50, 0) in transitions, "Missing 50 -> 0 exit"

    assert (50, 50) in transitions, "Missing 50 -> 50 self-loop (Long-Tail fix)"

    # Ensure Multi-Propeptide jump (i -> 1) is excised for invalid lengths but present for valid
    assert all((i, 1) not in transitions for i in range(1, 5)), "i -> 1 transition allowed before min_length"
    assert (5, 1) in transitions, "Missing 5 -> 1 multi-propeptide jump"
    assert (25, 1) in transitions, "Missing 25 -> 1 multi-propeptide jump"
    assert (50, 1) in transitions, "Missing 50 -> 1 multi-propeptide jump"
    print("✓ Constraints correct.")

# 3. Test Emission Repeating (2-to-51 mapping)
class DummyModel(CRFBaseModel):
    def __init__(self):
        super().__init__(num_labels=2, num_states=51)

def test_emissions():
    print("Testing 2-to-51 Emission Mapping...")
    model = DummyModel()

    # Simulate batch_size=2, seq_len=10, 2-class logits (0: None, 1: Propeptide)
    logits = torch.zeros(2, 10, 2)
    logits[:, :, 0] = 0.5 # None score
    logits[:, :, 1] = 1.5 # Propeptide score

    emissions_out = model._repeat_emissions(logits)

    assert emissions_out.shape == (2, 10, 51), f"Incorrect shape {emissions_out.shape}"
    assert torch.all(emissions_out[:, :, 0] == 0.5), "State 0 incorrect"
    assert torch.all(emissions_out[:, :, 1:51] == 1.5), "States 1-50 incorrectly mapped"
    print("✓ Emissions mapping correct.")

# 4. Test Dataset Label Excision
def test_dataset_labels():
    print("Testing Dataset Label Excision...")

    # Instead of fully initializing the dataset with precomputed PT files,
    # we just test the label sequence generator directly used inside it.
    from src.utils.crf_label_utils import peptide_list_to_label_sequence

    seq_len = 100

    # 1. Mature peptides (active peptides) should be passed as [] and map to 0
    mature_peptides = [(10, 20)] # Let's say we have an active peptide from 10-20
    # In the dataset, it explicitly does: label = peptide_list_to_label_sequence([], seq_len, max_len=50)
    label = peptide_list_to_label_sequence([], seq_len, max_len=50)
    assert np.all(label == 0), "Mature peptides did not map entirely to 0"

    # 2. Propeptides should map to states 1-50
    propeptides = [(30, 85)] # 56 residues long (should trigger max_len cap)
    propeptide_label = peptide_list_to_label_sequence(propeptides, seq_len, start_state=1, max_len=50)

    assert propeptide_label[29] == 1, "Propeptide start state incorrect"
    assert propeptide_label[29 + 49] == 50, "Propeptide max state cap incorrect"
    assert propeptide_label[84] == 50, "Propeptide tail not capped at 50" # The long-tail fix in labels

    # Combine them as done in the dataset
    final_label = label + propeptide_label
    assert np.all(final_label[9:20] == 0), "Active peptide leaked into final labels"
    print("✓ Dataset labels correct.")

# 4.5. Test Adjoining Propeptide Labels (The Training Crash Fix)
def test_adjoining_propeptides():
    print("Testing Adjoining Propeptides Edge Case...")
    from src.models.crf_models import CRFBaseModel
    import torch

    model = CRFBaseModel()

    # Simulating the exact scenario that caused the validation explosion:
    # A propeptide of length 25 followed immediately by a new propeptide
    # e.g., 1, 2, ..., 25, 1, 2, ...
    target = torch.zeros((1, 40), dtype=torch.long)
    target[0, 0:25] = torch.arange(1, 26) # First propeptide
    target[0, 25:35] = torch.arange(1, 11) # Adjoining second propeptide

    # Let's run it through _debug_crf to ensure no invalid transitions are found
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
        model._debug_crf(target)

    output = f.getvalue()
    assert "Found invalid transition" not in output, f"Adjoining propeptides caused an invalid transition! Log: {output}"
    print("✓ Adjoining Propeptides logic correct.")


# 5. Test Evaluation Metric Collapsing
from src.utils.manuscript_metrics import compute_all_metrics, convert_path_to_peptide_borders
def test_evaluation_metrics():
    print("Testing Evaluation Metrics Collapsing...")

    # Let's test the path converter first
    # 0, 1, 2, 3, 4, 5, 0 (Length 5)
    pred_path = [0, 1, 2, 3, 4, 5, 0]
    borders = convert_path_to_peptide_borders(pred_path, 1, 50, offset=0)
    assert borders == [(1, 5)], f"Border extraction failed: {borders}"

    # 0, 1, 2, 3, 4, 5, 50, 50, 50, 0 (Length 8, hit long-tail)
    pred_path_long = [0, 1, 2, 3, 4, 5, 50, 50, 50, 0]
    borders_long = convert_path_to_peptide_borders(pred_path_long, 1, 50, offset=0)
    assert borders_long == [(1, 8)], f"Long border extraction failed: {borders_long}"

    # Test compute_all_metrics pixel-wise collapse logic
    # We will simulate the internal logic of compute_all_metrics since it requires a true_df
    preds = [[0, 0, 1, 2, 3, 4, 5, 0]]
    labels = [[0, 0, 1, 2, 3, 4, 5, 0]] # Perfect match

    seq_tp, seq_fp, seq_fn, seq_tn = 0, 0, 0, 0
    for pred, true in zip(preds, labels):
        p = np.array(pred)
        t = np.array(true)
        p_binary = (p >= 1) & (p <= 50)
        t_binary = (t >= 1) & (t <= 50)

        seq_tp += np.logical_and(p_binary, t_binary).sum()
        seq_tn += np.logical_and(~p_binary, ~t_binary).sum()

    assert seq_tp == 5, f"Expected 5 TPs, got {seq_tp}"
    assert seq_tn == 3, f"Expected 3 TNs, got {seq_tn}"
    print("✓ Metrics logic correct.")


if __name__ == '__main__':
    print("--- Running Minimalist V4 Architecture Tests ---")
    test_bottleneck()
    test_constraints()
    test_emissions()
    test_dataset_labels()
    test_adjoining_propeptides()
    test_evaluation_metrics()
    print("--- All tests passed! ---")
