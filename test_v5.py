import torch
import torch.nn as nn
import numpy as np

# 1. Test ESM3 Bottleneck in LSTMCNN
from src.models.lstm_cnn import LSTMCNN
def test_bottleneck():
    print("Testing ESM3 Bottleneck...")
    model = LSTMCNN(input_size=1536)

    assert hasattr(model, 'bottleneck'), "Bottleneck missing"
    assert isinstance(model.bottleneck, nn.Sequential), "Bottleneck is not nn.Sequential"
    assert isinstance(model.bottleneck[0], nn.Linear), "First layer not Linear"
    assert model.bottleneck[0].in_features == 1536 and model.bottleneck[0].out_features == 256, "Incorrect Linear dims"
    print("✓ Bottleneck correct.")

# 2. Test V5 Transition Constraints in CRFBaseModel
from src.models.crf_models import CRFBaseModel
def test_constraints():
    print("Testing V5 CRF Transition Constraints...")
    transitions, allowed_starts, allowed_ends = CRFBaseModel.get_crf_constraints(max_len=100, min_len=5)

    # Mature
    assert (0, 0) in transitions, "Missing 0 -> 0"
    assert (0, 1) in transitions, "Missing 0 -> 1 entry"

    # Body Ladder
    assert (1, 2) in transitions, "Missing 1 -> 2 progression"
    assert (98, 99) in transitions, "Missing 98 -> 99 progression"
    assert (99, 99) in transitions, "Missing 99 -> 99 overflow safety loop"

    # Jump to Cleavage (i -> 100) requires i >= (min_len - 1), which is 4
    assert (3, 100) not in transitions, "Premature 3 -> 100 cleavage allowed"
    assert (4, 100) in transitions, "Missing 4 -> 100 cleavage jump"
    assert (50, 100) in transitions, "Missing 50 -> 100 cleavage jump"
    assert (99, 100) in transitions, "Missing 99 -> 100 cleavage jump from overflow"

    # Cleavage Exit (100 -> 0 or 100 -> 1)
    assert (100, 0) in transitions, "Missing 100 -> 0 exit to mature"
    assert (100, 1) in transitions, "Missing 100 -> 1 adjacent propeptide jump"
    assert (50, 0) not in transitions, "Old i -> 0 exit still exists!"

    print("✓ V5 Constraints correct.")

# 3. Test 3-Class Emission Mapping
class DummyModel(CRFBaseModel):
    def __init__(self):
        super().__init__(num_labels=3, num_states=101)

def test_emissions():
    print("Testing 3-Class Emission Mapping...")
    model = DummyModel()

    # Simulate batch_size=2, seq_len=10, 3-class logits (0: None, 1: Body, 2: Cleavage)
    logits = torch.zeros(2, 10, 3)
    logits[:, :, 0] = 0.5 # Mature score
    logits[:, :, 1] = 1.5 # Body score
    logits[:, :, 2] = 2.5 # Cleavage score

    emissions_out = model._repeat_emissions(logits)

    assert emissions_out.shape == (2, 10, 101), f"Incorrect shape {emissions_out.shape}"
    assert torch.all(emissions_out[:, :, 0] == 0.5), "State 0 incorrect"
    assert torch.all(emissions_out[:, :, 1:100] == 1.5), "Body states (1-99) incorrectly mapped"
    assert torch.all(emissions_out[:, :, 100] == 2.5), "Cleavage state (100) incorrectly mapped"
    print("✓ V5 Emissions mapping correct.")

# 4. Test Dataset Label Generation
def test_dataset_labels():
    print("Testing V5 Dataset Label Generation...")

    from src.utils.crf_label_utils import peptide_list_to_label_sequence

    seq_len = 120

    # Mature peptides should map to 0
    label = peptide_list_to_label_sequence([], seq_len, max_len=100)
    assert np.all(label == 0), "Mature peptides did not map entirely to 0"

    # Propeptides should map to body (1-99) and end with cleavage (100)
    propeptides = [(30, 115)] # 86 residues long
    propeptide_label = peptide_list_to_label_sequence(propeptides, seq_len, start_state=1, max_len=100, use_cleavage_state=True)

    assert propeptide_label[29] == 1, "Propeptide start state incorrect"
    assert propeptide_label[29 + 84] == 85, "Propeptide body index incorrect"
    assert propeptide_label[114] == 100, "Final residue was not assigned the Cleavage State (100)"

    # Overflow test
    long_propeptides = [(1, 105)] # 105 residues long
    long_label = peptide_list_to_label_sequence(long_propeptides, seq_len, start_state=1, max_len=100, use_cleavage_state=True)
    assert long_label[100] == 99, "Overflow body did not cap at 99"
    assert long_label[104] == 100, "Overflow final residue was not Cleavage State (100)"
    print("✓ V5 Dataset labels correct.")


if __name__ == '__main__':
    print("--- Running Precision V5 Architecture Tests ---")
    test_bottleneck()
    test_constraints()
    test_emissions()
    test_dataset_labels()
    print("--- All tests passed! ---")
