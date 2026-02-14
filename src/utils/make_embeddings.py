'''
Generate ESM3 embeddings (per position) and save as one
file per sequence. Use md5 hash of sequence as file name.
Adapted from DeepTMHMM.
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
    # Load ESM3 model
    # We use the small open model
    model = ESM3.from_pretrained("esm3_sm_open_v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataset = parse_fasta(fasta_file)

    print("Starting to generate embeddings")
    
    with torch.no_grad():
        for idx, (label, seq) in enumerate(tqdm(dataset)):
            
            if os.path.isfile(f'{esm_embeddings_dir}/{hash_aa_string(seq)}.pt'):
                # print("Already processed sequence")
                continue
            
            # Create ESMProtein object
            protein = ESMProtein(sequence=seq)
            # Encode to tensor (adds BOS/EOS)
            protein_tensor = model.encode(protein)

            sequence_tokens = protein_tensor.sequence
            # sequence_tokens is on device already because encode puts it there

            # Chunking logic to handle long sequences
            # ESM3 might handle longer sequences, but we stick to a safe window
            # similar to the original code.

            minibatch_max_length = sequence_tokens.size(0)

            tokens_list = []
            end = 0
            # Reshape to [1, L] for model input if needed, but model handles 1D or 2D?
            # ESM3 forward expects [B, L]. protein_tensor.sequence is [L] usually?
            # Let's check encoding.tokenize_sequence. It likely returns [L].
            # But model.forward expects [B, L].

            # Check shape
            if sequence_tokens.dim() == 1:
                sequence_tokens = sequence_tokens.unsqueeze(0) # [1, L]

            minibatch_max_length = sequence_tokens.size(1)

            while end <= minibatch_max_length:
                start = end
                end = start + 1022
                if end <= minibatch_max_length:
                    # we are not on the last one, so make this shorter
                    end = end - 300

                chunk = sequence_tokens[:, start:end]

                # Forward pass
                output = model(sequence_tokens=chunk)

                # Get embeddings
                # output.embeddings is [B, L_chunk, D]
                tokens = output.embeddings
                tokens_list.append(tokens)

            out = torch.cat(tokens_list, dim=1).cpu()

            # set nan to zeros
            out[out!=out] = 0.0

            # Remove batch dim: [L_total, D]
            res = out.squeeze(0)

            # Strip BOS/EOS (first and last)
            # Assumption: The first chunk had BOS, the last had EOS.
            # And we concatenated them.
            # Check if slicing removed them?
            # We sliced sequence_tokens which contained BOS/EOS.
            # So the concatenated output contains embeddings for BOS and EOS at ends.
            seq_embedding = res[1:-1]

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
    # repr_layers argument is removed as it's not applicable to ESM3 in the same way
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    generate_esm_embeddings(args.fasta_file, args.output_dir)

if __name__ == '__main__':
    main()
