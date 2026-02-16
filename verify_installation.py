#!/usr/bin/env python3
import sys
import os
import torch

def check_dependencies():
    missing = []
    try:
        import esm
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein
        print("ESM library installed successfully.")
    except ImportError as e:
        print(f"ESM library issue: {e}")
        missing.append("esm")

    dependencies = ["seaborn", "tabulate", "tqdm", "matplotlib", "torch"]
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"installed")
        except ImportError:
            print(f"not found")
            missing.append(dep)
    
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)

def check_huggingface_login():
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("Hugging Face token found.")
            return True
        else:
            print("No Hugging Face token found.")
            print("Please run `huggingface-cli login` and enter your token.")
            return False
    except ImportError:
        print("huggingface_hub not installed. Cannot verify token explicitly.")
        return False

def verify_esm3():
    check_dependencies()
    
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein

    print("\nAttempting to load ESM3 model (sm-open-v1)...")
    print("This requires access to the gated model: https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1")
    
    try:
        # Check login first
        if not check_huggingface_login():
            print("Attempting to proceed anyway (token might be in env var)...")
            
        model = ESM3.from_pretrained("esm3_sm_open_v1")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print("ESM3 model loaded successfully.")
        
        # Test encoding
        seq = "MKTIIALSYIFCLVFADYKDDDDK"
        protein = ESMProtein(sequence=seq)
        protein_tensor = model.encode(protein)
        
        with torch.no_grad():
            output = model(sequence_tokens=protein_tensor.sequence.unsqueeze(0))
            embedding = output.embeddings.squeeze(0)
            # Embedding size check
            # [L+2, D] -> [10, 1536]
            expected_dim = 1536
            if embedding.shape[-1] == expected_dim:
                print(f"Embedding dimension correct: {embedding.shape[-1]}")
            else:
                print(f"Embedding dimension mismatch: got {embedding.shape[-1]}, expected {expected_dim}")
                
        print("Verification complete! You can now run the predictor.")
        print("Example usage:")
        print("python predictor/predict.py --fastafile your_sequences.fasta --output_dir output/")
        
    except Exception as e:
        print("Failed to load or run ESM3 model.")
        print(f"Error: {e}")
        if "401 Client Error" in str(e) or "gated repo" in str(e):
            print("This error usually means you are not authenticated or do not have access to the model.")
            print("1. Go to https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1 and accept the license.")
            print("2. Run `huggingface-cli login` in your terminal.")
            print("3. Ensure your token has read access.")

if __name__ == "__main__":
    verify_esm3()
