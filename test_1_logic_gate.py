import torch
from src.models.crf_models import LSTMCNNCRF

def run_logic_gate_test():
    print("="*60)
    print("TEST 1: The 'Logic Gate' Unit Test")
    print("="*60)

    # Initialize V5 Specialist
    # num_states=101 implies states 0 to 100
    model = LSTMCNNCRF(input_size=1536, hidden_size=64, num_labels=3, num_states=101)

    # Create dummy logits [Batch=1, Seq=1, Classes=3]
    # 0: None, 1: Internal, 2: Cut
    test_logits = torch.tensor([[[-10.0, -10.0, 20.0]]])

    # Expand them
    emissions = model._repeat_emissions(test_logits)

    # VERIFY: State 100 should have the high score (20.0), State 50 should have -10.0
    state_50_score = emissions[0, 0, 50].item()
    state_100_score = emissions[0, 0, 100].item()

    print(f"Internal State Score (State 50): {state_50_score}")
    print(f"Cleavage State Score (State 100): {state_100_score}")

    if state_100_score == 20.0 and state_50_score == -10.0:
        print("✅ SUCCESS: 3-Class to 101-State mapping is mathematically correct.")
    else:
        print("❌ FAILED: Cut logit was incorrectly mapped.")
    print("="*60)

if __name__ == '__main__':
    run_logic_gate_test()
