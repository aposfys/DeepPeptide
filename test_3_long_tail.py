import torch
from src.models.crf_models import CRFBaseModel

def run_long_tail_visualizer():
    print("="*60)
    print("TEST 3: The 'Long-Tail' Visualizer")
    print("="*60)

    # Initialize the base CRF model with 101 states
    model = CRFBaseModel(num_labels=3, num_states=101)
    model.eval()

    # Simulate a protein of length 150 (Batch=1, SeqLen=150, Classes=3)
    # The goal is to force a 70-residue propeptide into the Viterbi decoder.
    # We want:
    # [0:20] = Mature (State 0)
    # [20:90] = Propeptide Body (States 1-70)
    # [90] = Cleavage Site (State 100)
    # [91:] = Mature (State 0)

    seq_len = 150
    pro_start, pro_end = 20, 90

    logits = torch.zeros((1, seq_len, 3))

    # Fill Mature (Class 0)
    logits[0, :pro_start, 0] = 50.0
    logits[0, pro_end+1:, 0] = 50.0

    # Fill Propeptide Body (Class 1)
    logits[0, pro_start:pro_end, 1] = 50.0

    # Fill Cleavage Site (Class 2) exactly at pro_end
    logits[0, pro_end, 2] = 50.0

    # Expand to 101 states using the V5 repeat function
    emissions = model._repeat_emissions(logits)
    mask = torch.ones((1, seq_len), dtype=torch.uint8)

    # Decode Viterbi Path
    paths, _ = model.crf.decode(emissions, mask)
    path = paths[0]

    # Validate the ladder climbed accurately to 70 and then hit 100
    highest_state = max([s for s in path if s < 100]) if path else 0
    hit_cleavage = 100 in path

    print(f"Long-Tail Test Path Excerpt [15:95]: {path[15:95]}")

    if hit_cleavage and highest_state == 70:
        print(f"✅ SUCCESS: Model utilized the 100-state ruler perfectly.")
        print(f"   Propeptide scaled up to state {highest_state} and exited via Cleavage State 100.")
    elif highest_state == 50:
        print("⚠️ WARNING: Model stalled at State 50! The V4 limit is still active.")
    elif hit_cleavage:
        print(f"✅ SUCCESS (Partial): Model hit Cleavage State 100, but max body state was {highest_state}.")
    else:
        print("❌ FAILED: Model never reached the Cleavage State (100).")
    print("="*60)

if __name__ == '__main__':
    run_long_tail_visualizer()
