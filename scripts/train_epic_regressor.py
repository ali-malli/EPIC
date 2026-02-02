#!/usr/bin/env python3
"""
Train EPIC regressor (Trigrams + GBR) on regression rows and save the model.

Inputs:
- Activity_data_for_regression.csv with columns:
    ID, Glucose_conc , Substrate_conc, Activity
- Trigram_embeddings.csv:
    first column = ID, remaining columns = trigram seed features 

Output:
- joblib bundle containing model + metadata 

Example:
python train_epic_regressor.py \
  --activity Activity_data_for_regression.csv \
  --trigrams Trigram_embeddings.csv \
  --out models/epic_regressor_trigrams_gbr.joblib
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


def anchored_pred(y_pred, glucose_vals):
    """Force prediction at glucose==0 to be 1.0"""
    y_pred = np.asarray(y_pred, float).copy()
    g = np.asarray(glucose_vals, float)
    y_pred[np.isclose(g, 0.0)] = 1.0
    return y_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activity", default="Activity_data_for_regression.csv")
    ap.add_argument("--trigrams", default="Trigram_embeddings.csv")
    ap.add_argument("--id_col", default="ID")
    ap.add_argument("--glu_col", default="Glucose_conc ")   # keep trailing space if present
    ap.add_argument("--sub_col", default="Substrate_conc")
    ap.add_argument("--y_col", default="Activity")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="epic_regressor_trigrams_gbr.joblib")
    args = ap.parse_args()

    # Tuned hyperparams 
    gbr_params = dict(
        learning_rate=0.1258,
        max_depth=4,
        max_features=None,
        min_samples_leaf=5,
        min_samples_split=6,
        n_estimators=187,
        subsample=0.897,
        random_state=args.seed,
    )

    act = pd.read_csv(args.activity)
    act.columns = act.columns.astype(str)

    need = {args.id_col, args.glu_col, args.sub_col, args.y_col}
    missing = need - set(act.columns)
    if missing:
        raise ValueError(f"Activity CSV missing columns: {sorted(list(missing))}")

    act = act.dropna(subset=[args.id_col, args.glu_col, args.sub_col, args.y_col]).copy()
    act[args.id_col] = act[args.id_col].astype(str).str.strip()
    act[args.glu_col] = pd.to_numeric(act[args.glu_col], errors="coerce")
    act[args.sub_col] = pd.to_numeric(act[args.sub_col], errors="coerce")
    act[args.y_col] = pd.to_numeric(act[args.y_col], errors="coerce")
    act = act.dropna(subset=[args.glu_col, args.sub_col, args.y_col]).copy()

    # enforce normalization at glucose=0
    act.loc[np.isclose(act[args.glu_col].to_numpy(float), 0.0), args.y_col] = 1.0

    tri = pd.read_csv(args.trigrams)
    tri.columns = tri.columns.astype(str)
    tri_id = tri.columns[0]
    tri[tri_id] = tri[tri_id].astype(str).str.strip()
    tri = tri.rename(columns={tri_id: args.id_col})

    df = pd.merge(act, tri, on=args.id_col, how="left").dropna().reset_index(drop=True)

    trigram_cols = [c for c in tri.columns if c != args.id_col]
    if not trigram_cols:
        raise ValueError("No trigram feature columns found in trigram CSV.")

    df[trigram_cols] = df[trigram_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=trigram_cols + [args.glu_col, args.sub_col, args.y_col]).reset_index(drop=True)

    # Build training matrix (trigrams + glucose + substrate)
    X = np.hstack([
        df[trigram_cols].to_numpy(float),
        df[[args.glu_col, args.sub_col]].to_numpy(float),
    ])
    y = df[args.y_col].to_numpy(float)

    print(f"[INFO] Training rows: {len(df)}")
    print(f"[INFO] Trigram dims: {len(trigram_cols)} | Total dims: {X.shape[1]}")

    model = GradientBoostingRegressor(**gbr_params)
    model.fit(X, y)

    # quick sanity RMSE on train (anchored)
    y_hat = anchored_pred(model.predict(X), df[args.glu_col].to_numpy(float))
    rmse = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    print(f"[INFO] Train RMSE (anchored at glucose=0): {rmse:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    bundle = {
        "model": model,
        "gbr_params": gbr_params,
        "feature_info": "Trigram frequencies + [Glucose_conc, Substrate_conc] appended",
        "trigram_cols": trigram_cols,
        "columns": {
            "id_col": args.id_col,
            "glu_col": args.glu_col,
            "sub_col": args.sub_col,
            "y_col": args.y_col,
        }
    }
    joblib.dump(bundle, args.out)
    print(f"[DONE] Saved model bundle: {args.out}")


if __name__ == "__main__":
    main()
