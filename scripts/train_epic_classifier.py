#!/usr/bin/env python3
"""
Train EPIC classifier (ProstT5 + substrate) on ALL labeled data and save the model.

Inputs:
- Dataset_classification.xlsx with columns: ID, Class, Substrate Concentration (mM)
- ProstT5_embeddings.csv with: first column ID, remaining columns numeric embeddings

Output:
- joblib bundle containing model + metadata

Example:
python train_epic_classifier.py \
  --xlsx Dataset_classification.xlsx \
  --emb ProstT5_embeddings.csv \
  --out models/epic_classifier_prostt5_svm.joblib
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


CLASS_ORDER = ["I", "II", "III", "IV"]
class_map = {c: i for i, c in enumerate(CLASS_ORDER)}
inv_class_map = {i: c for c, i in class_map.items()}

# Optimal params 
def build_model(seed: int = 42):
    svc_params = dict(
        kernel="rbf",
        C=67.7,
        gamma=0.00418,
        class_weight="balanced",
        probability=True,
        random_state=seed,
    )
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(**svc_params)),
    ])
    return clf, svc_params


def load_embeddings(csv_path: str, ids_in_order: np.ndarray) -> np.ndarray:
    """Assumes first column is ID, remaining columns are numeric features"""
    df = pd.read_csv(csv_path)
    id_col = df.columns[0]
    df[id_col] = df[id_col].astype(str).str.strip()
    df = df.set_index(id_col)

    missing = [i for i in ids_in_order if i not in df.index]
    if missing:
        raise ValueError(f"{csv_path}: missing {len(missing)} IDs (first 10): {missing[:10]}")

    X = df.loc[ids_in_order].to_numpy(dtype=float)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="Dataset_classification.xlsx", help="Labeled classification Excel")
    ap.add_argument("--emb", default="ProstT5_embeddings.csv", help="ProstT5 embeddings CSV")
    ap.add_argument("--id_col", default="ID")
    ap.add_argument("--label_col", default="Class")
    ap.add_argument("--sub_col", default="Substrate_conc")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="epic_classifier_prostt5_svm.joblib", help="Output joblib path")
    args = ap.parse_args()

    df = pd.read_excel(args.xlsx)
    need = {args.id_col, args.label_col, args.sub_col}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Excel missing columns: {sorted(list(missing))}")

    df = df[[args.id_col, args.label_col, args.sub_col]].dropna().copy()
    df[args.id_col] = df[args.id_col].astype(str).str.strip()
    df[args.label_col] = df[args.label_col].astype(str).str.strip()

    bad = sorted(set(df[args.label_col].unique()) - set(CLASS_ORDER))
    if bad:
        raise ValueError(f"Unexpected class values {bad}. Expected only {CLASS_ORDER}.")

    df[args.sub_col] = pd.to_numeric(df[args.sub_col], errors="coerce")
    if df[args.sub_col].isna().any():
        med = float(df[args.sub_col].median())
        df[args.sub_col] = df[args.sub_col].fillna(med)
        print(f"[WARN] Missing substrate values; filled with median = {med:g}")

    train_ids = df[args.id_col].to_numpy(dtype=str)
    y = df[args.label_col].map(class_map).to_numpy(dtype=int)
    sub = df[args.sub_col].to_numpy(dtype=float).reshape(-1, 1)

    print("\n=== Train class counts ===")
    print(df[args.label_col].value_counts().reindex(CLASS_ORDER))
    print(f"[INFO] Train n={len(train_ids)}")

    X_emb = load_embeddings(args.emb, train_ids)
    X = np.hstack([X_emb, sub])
    print(f"[INFO] Feature matrix: {X.shape} (ProstT5 + substrate as last column)")

    model, svc_params = build_model(seed=args.seed)
    model.fit(X, y)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    bundle = {
        "model": model,
        "class_order": CLASS_ORDER,
        "class_map": class_map,
        "inv_class_map": inv_class_map,
        "svc_params": svc_params,
        "feature_info": "ProstT5 embeddings + substrate appended as last column",
        "train_ids": train_ids.tolist(),
        "columns": {
            "id_col": args.id_col,
            "label_col": args.label_col,
            "sub_col": args.sub_col,
        },
    }

    joblib.dump(bundle, args.out)
    print(f"[DONE] Saved model bundle: {args.out}")


if __name__ == "__main__":
    main()
