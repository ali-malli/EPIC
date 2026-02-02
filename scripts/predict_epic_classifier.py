#!/usr/bin/env python3
"""
Predict EPIC classes for any new enzymes using a trained model bundle.

Inputs:
- trained model bundle (.joblib) from train_epic_classifier.py
- ProstT5 embeddings CSV for new enzymes: first column ID, rest numeric
- substrate concentrations for new enzymes: a metadata CSV with ID + substrate column

Output:
- CSV with predictions and class probabilities

Example:
python predict_epic_classifier.py \
  --model scripts/epic_classifier_prostt5_svm.joblib \
  --emb yourenzyme_embeddings.csv \
  --meta yourenzyme.csv \
  --sub_col "Substrate_conc" \
  --out predictions/epic_class_predictions.csv
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd


def load_embeddings_any_order(csv_path: str):
    df = pd.read_csv(csv_path)
    id_col = df.columns[0]
    df[id_col] = df[id_col].astype(str).str.strip()
    ids = df[id_col].to_numpy(dtype=str)
    X = df.drop(columns=[id_col]).to_numpy(dtype=float)
    return ids, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model bundle .joblib")
    ap.add_argument("--emb", required=True, help="New ProstT5 embeddings CSV (ID + dims)")

    ap.add_argument("--meta", default=None, help="Optional metadata CSV containing substrate per ID")
    ap.add_argument("--id_col", default="ID", help="ID column in metadata CSV (if used)")
    ap.add_argument("--sub_col", default="Substrate Concentration (mM)", help="Substrate column in metadata CSV")

    ap.add_argument("--substrate_constant", type=float, default=None,
                    help="If meta not provided, use a constant substrate for all IDs")

    ap.add_argument("--out", default="epic_class_predictions.csv", help="Output CSV path")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    class_order = bundle["class_order"]
    inv_class_map = bundle["inv_class_map"]

    ids, X_emb = load_embeddings_any_order(args.emb)

    if args.meta is not None:
        meta = pd.read_csv(args.meta)
        meta.columns = meta.columns.astype(str)

        if args.id_col not in meta.columns or args.sub_col not in meta.columns:
            raise ValueError(
                f"Metadata CSV must contain columns '{args.id_col}' and '{args.sub_col}'. "
                f"Found: {list(meta.columns)}"
            )

        meta[args.id_col] = meta[args.id_col].astype(str).str.strip()
        meta[args.sub_col] = pd.to_numeric(meta[args.sub_col], errors="coerce")

        # collapse if multiple rows per enzyme -> first non-null
        sub_by_id = (
            meta.dropna(subset=[args.sub_col])
            .groupby(args.id_col)[args.sub_col]
            .first()
        )

        missing = [i for i in ids if i not in sub_by_id.index]
        if missing:
            raise ValueError(f"Missing substrate values for {len(missing)} IDs (first 10): {missing[:10]}")

        sub = sub_by_id.loc[ids].to_numpy(dtype=float).reshape(-1, 1)

    else:
        if args.substrate_constant is None:
            raise ValueError(
                "Provide either --meta (with substrate per ID) or --substrate_constant."
            )
        sub = np.full((len(ids), 1), float(args.substrate_constant), dtype=float)

    X = np.hstack([X_emb, sub])

    # --- predict ---
    proba = model.predict_proba(X)  # (n,4)
    pred_int = np.argmax(proba, axis=1)
    pred_class = [inv_class_map[int(i)] for i in pred_int]

    out = pd.DataFrame({"ID": ids, "Predicted_Class": pred_class})
    for j, cls in enumerate(class_order):
        out[f"P({cls})"] = proba[:, j]
    out["Substrate_mM"] = sub.flatten()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"[DONE] Wrote predictions: {args.out}")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
