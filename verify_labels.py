
import torch
import pandas as pd
import numpy as np
import os
import sys
from hashlib import md5
import shutil

# Add root directory to python path so we can import src
sys.path.append(os.getcwd())
from src.utils.dataset import PrecomputedCSVForOverlapCRFDataset

def verify_labels():
    print("--- Verifying Dataset Labels ---")
    data_file = 'data/labeled_sequences.csv'
    partitioning_file = 'data/graphpart_assignments.csv'
    embeddings_dir = 'data/embeddings_test_verif'

    # 1. Setup Dummy Environment
    if os.path.exists(embeddings_dir):
        shutil.rmtree(embeddings_dir)
    os.makedirs(embeddings_dir, exist_ok=True)

    # 2. Find Test Candidates
    # We need a protein that definitely has a propeptide to verify labeling works.
    df = pd.read_csv(data_file)

    # We also need the cluster IDs for the partition filter
    part_df = pd.read_csv(partitioning_file)

    # Filter for non-empty propeptide coordinates AND presence in partitioning file
    df_pro = df[df['propeptide_coordinates'].notna() & (df['propeptide_coordinates'] != '')]
    # Ensure protein_id is in partitioning file
    df_pro = df_pro[df_pro['protein_id'].isin(part_df['AC'])]

    if len(df_pro) == 0:
        print("Error: No proteins with propeptides found in CSV (that match partitioning file) to test!")
        return

    # Select 5 random proteins
    n_samples = 5
    if len(df_pro) < n_samples:
        n_samples = len(df_pro)

    print(f"Found {len(df_pro)} valid proteins with propeptides. Testing {n_samples} random samples...")

    # 3. Create Dummy Embeddings & Test
    # We need to mock the embeddings because the dataset loader expects them to exist.
    test_df = df_pro.sample(n_samples)
    test_indices = test_df.index
    valid_clusters = []

    for idx in test_indices:
        seq = df.loc[idx, 'sequence']
        pid = df.loc[idx, 'protein_id']

        # Get cluster
        cluster = part_df.loc[part_df['AC'] == pid, 'cluster'].values[0]
        valid_clusters.append(cluster)

        # Create dummy embedding
        seq_hash = md5(seq.encode()).digest().hex()
        # ESM2 embedding dimension is typically 1280 (for t33), dataset converts to float32
        emb = torch.randn(len(seq), 1280)
        torch.save(emb, os.path.join(embeddings_dir, f'{seq_hash}.pt'))

    # 4. Instantiate Dataset
    # We pass the list of valid clusters so our test proteins are included
    dataset = PrecomputedCSVForOverlapCRFDataset(
        embeddings_dir=embeddings_dir,
        data_file=data_file,
        partitioning_file=partitioning_file,
        partitions=list(set(valid_clusters)),
        label_type='multistate_with_propeptides'
    )

    print(f"Dataset initialized with {len(dataset)} samples.")

    # 5. Check Labels
    # We filter the dataset to only include the names we prepared embeddings for.
    # The dataset object loads everything in the partition, but we only want to iterate over our test set.

    pass_count = 0
    test_names = df.loc[test_indices, 'protein_id'].tolist()

    # Map dataset names to indices
    name_to_idx = {name: i for i, name in enumerate(dataset.names)}

    print(f"\nVerifying {len(test_names)} samples...")

    for pid in test_names:
        if pid not in name_to_idx:
            print(f"Skipping {pid} (not in dataset partition filter)")
            continue

        i = name_to_idx[pid]
        try:
            emb, mask, label, peptides = dataset[i]

            print(f"\nChecking Protein: {pid}")

            # Check Values
            unique_vals = torch.unique(label).tolist()
            unique_vals.sort()
            # print(f"  Label Values Present: {unique_vals}")

            # Validation Checks
            errors = []
            if (label < 0).any():
                errors.append("Negative values found.")
            if (label > 50).any():
                errors.append("Values > 50 found (Old labeling scheme?).")

            if errors:
                print(f"  FAILED: {', '.join(errors)}")
            else:
                # Check if we actually have propeptide labels (values > 0)
                if (label > 0).any():
                    # Find start/end of the non-zero region
                    nonzero_indices = torch.nonzero(label > 0).squeeze()
                    if nonzero_indices.numel() > 0:
                        start, end = nonzero_indices[0].item(), nonzero_indices[-1].item()
                        print(f"  SUCCESS: Propeptide labels (1-50) present at indices {start}-{end}.")
                    else:
                        print("  SUCCESS: Propeptide labels present.")
                    pass_count += 1
                else:
                    print("  WARNING: Only background labels (0) found. (Did we pick a protein without propeptide?)")

        except Exception as e:
            print(f"  ERROR processing sample {pid}: {e}")

    print("\n--- Summary ---")
    print(f"Verified {pass_count}/{len(test_names)} tested samples have correct propeptide labels.")

    # Cleanup
    if os.path.exists(embeddings_dir):
        shutil.rmtree(embeddings_dir)

if __name__ == "__main__":
    verify_labels()
