#!/usr/bin/env python3
"""
- Generates all 8000 possible AA trigrams 
- Uses CountVectorizer with fixed vocabulary 
- Converts counts to per-sequence frequencies
- Writes CSV
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from Bio import SeqIO
from itertools import product


AA = list("ACDEFGHIKLMNPQRSTVWY")
NGRAM = 3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", default="Sequences.fasta", help="Input FASTA file")
    parser.add_argument("--out", default="enzyme_trigram_features.csv", help="Output CSV path")
    args = parser.parse_args()


    all_trigrams = ["".join(p) for p in product(AA, repeat=NGRAM)]
    all_trigrams = sorted(all_trigrams)  # stable ordering
    print("Total trigrams:", len(all_trigrams))  # 8000

    ids = []
    sequences = []

    for record in SeqIO.parse(args.fasta, "fasta"):
        ids.append(str(record.id).strip())
        seq = str(record.seq).upper()
        seq = "".join([aa for aa in seq if aa in AA])  # keep only canonical AA
        sequences.append(seq)

    print(f"Loaded {len(sequences)} sequences")

    vectorizer = CountVectorizer(
        analyzer="char",
        ngram_range=(3, 3),
        lowercase=False,
        vocabulary=all_trigrams,
    )

    X = vectorizer.transform(sequences).toarray()

    row_sums = X.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    X = X / row_sums

    X_df = pd.DataFrame(X, columns=all_trigrams)
    X_df.insert(0, "ID", ids)

    X_df.to_csv(args.out, index=False)

    print("Saved:", args.out)
    print("Final shape:", X_df.shape)


if __name__ == "__main__":
    main()
