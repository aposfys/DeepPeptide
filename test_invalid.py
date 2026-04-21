import torch
from src.utils.dataset import PrecomputedCSVDataset, PrecomputedCSVForCRFDataset, PrecomputedCSVForOverlapCRFDataset

# Mock arguments
class Args:
    embeddings_dir = 'data/esm3_embeddings'
    data_file = 'data/labeled_sequences_esm3.csv'
    partitioning_file = 'data/graphpart_assignments.csv'
    label_type = 'propeptides_only'
    partitions = [0, 1, 2, 3, 4]

args = Args()
try:
    dataset = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, args.partitions, args.label_type)
    for i in range(len(dataset)):
        emb, mask, label, peptides = dataset[i]
        label_np = label.numpy()
        for j in range(1, len(label_np)):
            l = int(label_np[j])
            l_prev = int(label_np[j-1])
            # Valid transitions: (0,0), (0,1), (i, i+1), (i, 100), (100, 0), (100, 1)
            # body overflow: (99, 99), (99, 100)
            valid = False
            if l_prev == 0 and l in [0, 1]: valid = True
            elif 1 <= l_prev <= 99 and l == l_prev + 1: valid = True
            elif 1 <= l_prev <= 99 and l == 100: valid = True
            elif l_prev == 99 and l == 99: valid = True
            elif l_prev == 100 and l in [0, 1]: valid = True

            if not valid:
                print(f"Index {i} ({dataset.names[i]}): Invalid transition {l_prev} -> {l} at pos {j-1}->{j}")
                print("Sequence length:", len(label_np))
                print("Peptides:", peptides)
                break
except Exception as e:
    print(e)
