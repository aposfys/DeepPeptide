'''
Generate one-hot embeddings and save as one 
file per sequence. Use md5 hash of sequence as file name.
We do it this way so that we can just reuse the whole ESM-based setup without
any changes aside from the input dimension.
'''
from hashlib import md5
import torch
import os
import argparse
import pathlib
from tqdm.auto import tqdm
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein

def hash_aa_string(string):
    return md5(string.encode()).digest().hex()

def parse_fasta(fasta_file):
    ids = []
    seqs = []
    with open(fasta_file, 'r') as f:
        header = None
        sequence = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    ids.append(header)
                    seqs.append("".join(sequence))
                header = line[1:]
                sequence = []
            else:
                sequence.append(line)
        if header:
            ids.append(header)
            seqs.append("".join(sequence))
    return list(zip(ids, seqs))

def generate_esm_embeddings(fasta_file, esm_embeddings_dir):
    # Load ESM3 model just for tokenization
    model = ESM3.from_pretrained("esm3_sm_open_v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # model.eval() # Not needed for tokenization strictly but good practice

    dataset = parse_fasta(fasta_file)
    
    print("Starting to generate embeddings")

    for idx, (label, seq) in enumerate(tqdm(dataset)):
        
        # if os.path.isfile(f'{esm_embeddings_dir}/{hash_aa_string(seq)}.pt'):
        #     print("Already processed sequence")
        #     continue

        protein = ESMProtein(sequence=seq)
        protein_tensor = model.encode(protein)
        toks = protein_tensor.sequence # [L]

        # One-hot encode
        # ESM3 vocab size is 64
        seq_embedding = torch.nn.functional.one_hot(toks, num_classes=64) # [L, 64]

        # Remove BOS/EOS
        seq_embedding = seq_embedding[1:-1]

        # Convert to float (optional, but usually embeddings are float)
        seq_embedding = seq_embedding.float()

        output_file = open(f'{esm_embeddings_dir}/{hash_aa_string(seq)}.pt', 'wb')
        torch.save(seq_embedding, output_file)
        output_file.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta_file",
        type=pathlib.Path,
        help="FASTA file on which to extract representations",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    generate_esm_embeddings(args.fasta_file, args.output_dir)

if __name__ == '__main__':
    main()
