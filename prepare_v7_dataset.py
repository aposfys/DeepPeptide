import urllib.request
import pandas as pd
import json
import os
import subprocess
import re
from tqdm import tqdm

def download_uniprot_data(output_file="data/uniprot_propeptides.tsv"):
    """
    Downloads all Reviewed (Swiss-Prot) proteins containing a Propeptide annotation.
    We request TSV format with the Accession, Sequence, and Propeptide coordinates.
    """
    print("="*80)
    print("1. Downloading Swiss-Prot Propeptide Dataset...")
    # The URL explicitly requests format=tsv and the fields for Sequence and Propeptide (ft_propep)
    url = "https://rest.uniprot.org/uniprotkb/stream?format=tsv&fields=accession,sequence,ft_propep&query=%28%28ft_propep%3A*%29+AND+%28reviewed%3Atrue%29%29"

    os.makedirs("data", exist_ok=True)

    try:
        urllib.request.urlretrieve(url, output_file)
        print(f"✅ Successfully downloaded dataset to {output_file}")
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        return None
    return output_file

def parse_propeptide_coordinates(feature_string):
    """
    Parses the messy UniProt feature string into a clean "START-END,START-END" format.
    Example: "PROPEP 25..84; /note=\"Activation peptide\"; PROPEP 105..110;"
    Output: "25-84,105-110"
    """
    if pd.isna(feature_string):
        return ""

    # Regex to find all "PROPEP X..Y" occurrences
    matches = re.findall(r"PROPEP\s+(\d+)\.\.(\d+)", feature_string)

    if not matches:
        return ""

    # Join them into the format expected by our dataset loader
    coords = [f"{start}-{end}" for start, end in matches]
    return ",".join(coords)

def process_dataset(input_tsv="data/uniprot_propeptides.tsv", output_csv="data/labeled_sequences_v7.csv"):
    print("="*80)
    print("2. Parsing Propeptide Coordinates...")

    df = pd.read_csv(input_tsv, sep='\t')
    print(f"Loaded {len(df)} reviewed sequences.")

    # Rename columns to match what DeepPeptide expects
    df = df.rename(columns={
        "Entry": "protein_id",
        "Sequence": "sequence",
        "Propeptide": "raw_features"
    })

    # Parse coordinates
    df['propeptide_coordinates'] = df['raw_features'].apply(parse_propeptide_coordinates)

    # We drop any sequences that failed to parse cleanly
    df = df[df['propeptide_coordinates'] != ""]

    # Add empty active peptide coordinates (since V6.1+ is a Propeptide Specialist)
    df['coordinates'] = ""
    df['organism'] = "Unknown" # Placeholder to satisfy the legacy dataloader

    # Save the cleaned CSV
    df[['protein_id', 'sequence', 'coordinates', 'propeptide_coordinates', 'organism']].to_csv(output_csv, index=False)
    print(f"✅ Saved cleaned dataset with {len(df)} sequences to {output_csv}")

    # Create a FASTA file for MMseqs2 clustering
    fasta_file = "data/uniprot_propeptides.fasta"
    with open(fasta_file, 'w') as f:
        for idx, row in df.iterrows():
            f.write(f">{row['protein_id']}\n{row['sequence']}\n")

    return output_csv, fasta_file

def cluster_with_mmseqs(fasta_file="data/uniprot_propeptides.fasta", out_dir="data/mmseqs_out"):
    """
    Runs MMseqs2 to cluster the dataset at 30% sequence identity to prevent Homology Leakage.
    Requires `mmseqs` to be installed on the system.
    """
    print("="*80)
    print("3. Clustering Dataset to Prevent Homology Leakage...")

    os.makedirs(out_dir, exist_ok=True)
    db_path = os.path.join(out_dir, "seqDB")
    cluster_path = os.path.join(out_dir, "cluDB")
    tsv_path = os.path.join(out_dir, "clusters.tsv")

    try:
        # Create DB
        subprocess.run(["mmseqs", "createdb", fasta_file, db_path], check=True, stdout=subprocess.DEVNULL)

        # Cluster at 30% identity (--min-seq-id 0.3) to guarantee no evolutionary overlap between train and test
        print("Running Deep Clustering at 30% Sequence Identity...")
        subprocess.run(["mmseqs", "cluster", db_path, cluster_path, out_dir, "--min-seq-id", "0.3", "-c", "0.8"], check=True, stdout=subprocess.DEVNULL)

        # Create TSV
        subprocess.run(["mmseqs", "createtsv", db_path, db_path, cluster_path, tsv_path], check=True, stdout=subprocess.DEVNULL)

        print(f"✅ Clustering complete! Saved to {tsv_path}")
        return tsv_path
    except FileNotFoundError:
        print("❌ WARNING: 'mmseqs' command not found. Please install MMseqs2 (e.g., `conda install -c bioconda mmseqs2`).")
        print("Skipping clustering step.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"❌ MMseqs2 failed: {e}")
        return None

def create_partition_file(cluster_tsv, output_file="data/graphpart_assignments_v7.csv"):
    print("="*80)
    print("4. Creating Train/Val/Test Partitions...")

    if not cluster_tsv or not os.path.exists(cluster_tsv):
        print("⚠️ No cluster TSV found. Cannot create partitions safely.")
        return

    df = pd.read_csv(cluster_tsv, sep='\t', header=None, names=["Cluster_Rep", "AC"])

    # Assign clusters to 5 partitions (0-4) sequentially.
    # Because MMseqs guarantees all sequences in a cluster share the same `Cluster_Rep`,
    # we map the `Cluster_Rep` to a partition, ensuring homologous sequences stay together!
    unique_clusters = df["Cluster_Rep"].unique()

    cluster_to_partition = {rep: i % 5 for i, rep in enumerate(unique_clusters)}
    df['cluster'] = df['Cluster_Rep'].map(cluster_to_partition)

    # Save in the format expected by our dataset loader (AC, cluster)
    df[['AC', 'cluster']].to_csv(output_file, index=False)

    print(f"✅ Saved partition file to {output_file}")
    print("="*80)
    print("🎉 V7 DATA PIPELINE COMPLETE")
    print("Next Steps:")
    print("1. Generate ESM3 embeddings for the new sequences using `src/utils/make_embeddings.py`.")
    print("2. Train the V7 model!")

if __name__ == "__main__":
    tsv_file = download_uniprot_data()
    if tsv_file:
        csv_file, fasta_file = process_dataset(tsv_file)
        cluster_tsv = cluster_with_mmseqs(fasta_file)
        create_partition_file(cluster_tsv)
