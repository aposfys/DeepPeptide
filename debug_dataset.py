import pandas as pd
from hashlib import md5
import torch
import os

def make_hashes(names):
    hashes = []
    for name in names:
        clean_name = name.strip().replace('\n', '').replace(' ', '')
        hashes.append(md5(clean_name.encode()).digest().hex())
    return hashes

try:
    df = pd.read_csv('data/labeled_sequences_esm3.csv', index_col='protein_id')
    seqs = df['sequence'].tolist()
    hashes = make_hashes(seqs)

    print(f"Total sequences: {len(seqs)}")
    found = 0
    not_found = 0

    for i in range(min(5, len(seqs))):
        seq = seqs[i]
        h = hashes[i]
        path = os.path.join('data/esm3_embeddings', f'{h}.pt')
        if os.path.exists(path):
            found += 1
            emb = torch.load(path)
            print(f"Seq len: {len(seq)}, Emb shape: {emb.shape}")
        else:
            not_found += 1
            print(f"NOT FOUND: {h} for seq {seq[:10]}...")

    print(f"Found: {found}, Not Found: {not_found}")
except Exception as e:
    print(e)
