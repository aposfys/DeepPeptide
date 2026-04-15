import torch
import argparse
import sys

def inspect_model(checkpoint_path):
    if not torch.os.path.exists(checkpoint_path):
        print(f"Error: {checkpoint_path} not found.")
        sys.exit(1)

    ckpt = torch.load(checkpoint_path, map_location='cpu')

    # Depending on how the model is saved, the state dict might be directly
    # loaded or under a 'model_state_dict' key
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt

    if 'crf.start_transitions' not in state_dict:
        print("Error: 'crf.start_transitions' key not found in the checkpoint.")
        sys.exit(1)

    st = state_dict['crf.start_transitions']
    print(f"Loaded start_transitions from {checkpoint_path}")
    print("="*60)
    print("First 5 Start Transitions (States 0-4):")
    print(st[:5])
    print("="*60)

    # Check for strong dominance in state 0 or 1
    st_max = torch.max(st).item()
    st_0 = st[0].item()
    st_1 = st[1].item()

    if st_0 == st_max or st_1 == st_max:
        print("DIAGNOSTIC: State 0 or 1 is the dominant starting state.")
        # If the gap between max and mean is huge, it's highly peaked
        if st_max - torch.mean(st).item() > 5.0:
            print("WARNING: start_transitions is highly peaked at the beginning!")
            print("Consider initializing start_transitions to uniform (nn.init.zeros_) in V9.")
    else:
        print("DIAGNOSTIC: start_transitions does not appear to be overly biased to the first states.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inspect CRF Start Transitions")
    parser.add_argument("--model_path", type=str, default="esm3_propeptide_v8_A/model.pt", help="Path to the saved model.pt")
    args = parser.parse_args()

    inspect_model(args.model_path)
