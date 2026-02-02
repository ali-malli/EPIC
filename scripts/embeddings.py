#!/usr/bin/env python3
"""
- Reads FASTA
- Runs ESM2_650M and ProstT5 in inference mode
- Pools token embeddings (mean/min/max)
- Saves CSV
"""

import os
os.environ["HF_HOME"] = "/scratch/am8992/huggingface"
os.environ["TORCH_HOME"] = "/scratch/am8992/torch_cache"

import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
import torch
from transformers import AutoTokenizer, AutoModel, T5Tokenizer, T5EncoderModel

# Load and prepare sequences
def load_fasta(fasta_path):
    sequences = []
    ids = []
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        ids.append(str(record.id).strip())
    return ids, sequences

# Embedding function 
def get_embeddings(model_name, sequences, pooling="mean"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "facebook/esm2_t33_650M_UR50D":
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        model = AutoModel.from_pretrained(model_name).to(device)
        use_spaced = False
    else:
        # ProstT5
        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prostt5", do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/prostt5").to(device)
        use_spaced = True

    model.eval()
    embeddings = []

    with torch.no_grad():
        for seq in tqdm(sequences, desc=f"Embedding with {model_name}"):
            seq = str(seq).upper()
            if use_spaced:
                seq = " ".join(list(seq))

            inputs = tokenizer(seq, return_tensors="pt", truncation=True).to(device)
            outputs = model(**inputs)

            hidden_states = outputs.last_hidden_state.squeeze(0)  # (L, D)

            if pooling == "mean":
                emb = hidden_states.mean(dim=0).cpu().numpy()
            elif pooling == "min":
                emb = hidden_states.min(dim=0).values.cpu().numpy()
            elif pooling == "max":
                emb = hidden_states.max(dim=0).values.cpu().numpy()
            else:
                raise ValueError("Pooling must be one of 'mean', 'min', or 'max'")

            embeddings.append(emb)

    return np.array(embeddings)

# Main

if __name__ == "__main__":
    fasta_file = "BGLsequences.fasta"
    output_dir = "embeddings_csv"
    os.makedirs(output_dir, exist_ok=True)

    ids, seqs = load_fasta(fasta_file)

    models = {
        "ESM2_650M": "facebook/esm2_t33_650M_UR50D",
        "ProstT5": "Rostlab/prostt5",
    }

    for name, model_id in models.items():
        print(f"\nProcessing {name} ({model_id}) ...")
        try:
            emb = get_embeddings(model_id, seqs, pooling="mean")
            df = pd.DataFrame(emb)
            df.insert(0, "ID", ids)

            out_path = f"{output_dir}/{name}_mean_embeddings.csv"
            df.to_csv(out_path, index=False)
            print(f"Saved embeddings to {out_path}")
        except Exception as e:
            print(f"Failed for {name}: {e}")
