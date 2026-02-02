#!/usr/bin/env python3
"""
Predict curves with EPIC regressor (Trigrams + GBR) for any dataset

Inputs:
- trained model bundle from train_epic_regressor.py
- new CSV with at least:
    ID, Glucose_conc , Substrate_conc
- trigram features CSV (ID + trigram cols)

Outputs:
- predictions CSV with per-row predicted activity
- optional per-enzyme plots

Example:
python predict_epic_regressor.py \
  --model models/epic_regressor_trigrams_gbr.joblib \
  --data yourenzyme.csv \
  --trigrams enzyme_trigram_features_yourenzyme.csv \
  --out predictions/epic_regression_predictions.csv \
  --plot_dir predictions/plots
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def anchored_pred(y_pred, glucose_vals):
    y_pred = np.asarray(y_pred, float).copy()
    g = np.asarray(glucose_vals, float)
    y_pred[np.isclose(g, 0.0)] = 1.0
    return y_pred


def raw_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def affine_fit_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if len(y_true) < 2:
        return np.nan
    a, b = np.polyfit(y_pred, y_true, 1)
    y_scaled = a * y_pred + b
    return float(np.sqrt(np.mean((y_true - y_scaled) ** 2)))


def safe_spearman(y_true, y_pred):
    rho, _ = spearmanr(y_true, y_pred)
    return float(rho) if rho is not None else np.nan


def plot_curve(eid, g, y_true, y_pred, out_stub):
    # metrics if truth provided
    if y_true is not None:
        rho = safe_spearman(y_true, y_pred)
        rmse = raw_rmse(y_true, y_pred)
        aff = affine_fit_rmse(y_true, y_pred)
        metrics_txt = f"Spearman ρ = {rho:.3f}   |   RMSE = {rmse:.3f}   |   Affine RMSE = {aff:.3f}"
    else:
        metrics_txt = None

    fig, ax = plt.subplots(figsize=(4.4, 3.2), dpi=300)

    # observed points (no connecting lines)
    if y_true is not None:
        ax.plot(g, y_true, marker="o", linestyle="None", markersize=5.5, label="Observed", zorder=3)

    # predicted points (no connecting lines)
    ax.plot(g, y_pred, marker="s", linestyle="None", markersize=5.2, label="Predicted", zorder=4)

    ax.set_xlabel("Glucose concentration (mM)")
    ax.set_ylabel("Normalized activity")

    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.30)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    ax.set_title(f"{eid} — Trigrams + GBR", loc="left", pad=6)

    if metrics_txt is not None:
        fig.text(0.01, 0.01, metrics_txt, ha="left", va="bottom", fontsize=8.2)
        fig.tight_layout(rect=[0.0, 0.06, 0.82, 1.0])
    else:
        fig.tight_layout(rect=[0.0, 0.0, 0.82, 1.0])

    fig.savefig(out_stub + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(out_stub + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Model bundle .joblib from training script")
    ap.add_argument("--data", required=True, help="CSV with ID, Glucose_conc, Substrate_conc")
    ap.add_argument("--trigrams", required=True, help="CSV with ID + trigram columns")
    ap.add_argument("--out", default="epic_regression_predictions.csv")
    ap.add_argument("--plot_dir", default=None, help="If set, saves per-enzyme plots here")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    trigram_cols = bundle["trigram_cols"]
    cols = bundle["columns"]

    ID_COL = cols["id_col"]
    GLU_COL = cols["glu_col"]
    SUB_COL = cols["sub_col"]
    Y_COL = cols["y_col"]

    df = pd.read_csv(args.data)
    df.columns = df.columns.astype(str)

    need = {ID_COL, GLU_COL, SUB_COL}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Data CSV missing required columns: {sorted(list(missing))}")

    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[GLU_COL] = pd.to_numeric(df[GLU_COL], errors="coerce")
    df[SUB_COL] = pd.to_numeric(df[SUB_COL], errors="coerce")
    df = df.dropna(subset=[ID_COL, GLU_COL, SUB_COL]).copy()

    has_truth = (Y_COL in df.columns)
    if has_truth:
        df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")

    # enforce normalization if activity exists
    if has_truth:
        df.loc[np.isclose(df[GLU_COL].to_numpy(float), 0.0), Y_COL] = 1.0

    tri = pd.read_csv(args.trigrams)
    tri.columns = tri.columns.astype(str)
    tri_id = tri.columns[0]
    tri[tri_id] = tri[tri_id].astype(str).str.strip()
    tri = tri.rename(columns={tri_id: ID_COL})

    merged = pd.merge(df, tri, on=ID_COL, how="left")

    missing_tri = [c for c in trigram_cols if c not in merged.columns]
    if missing_tri:
        raise ValueError(f"Trigram CSV missing {len(missing_tri)} required columns (first 10): {missing_tri[:10]}")

    merged[trigram_cols] = merged[trigram_cols].apply(pd.to_numeric, errors="coerce")
    merged = merged.dropna(subset=trigram_cols + [GLU_COL, SUB_COL]).copy()

    X = np.hstack([
        merged[trigram_cols].to_numpy(float),
        merged[[GLU_COL, SUB_COL]].to_numpy(float),
    ])

    pred = anchored_pred(model.predict(X), merged[GLU_COL].to_numpy(float))
    merged["Predicted_Activity"] = pred

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    merged[[ID_COL, GLU_COL, SUB_COL] + ([Y_COL] if has_truth else []) + ["Predicted_Activity"]].to_csv(
        args.out, index=False
    )
    print(f"[DONE] Wrote predictions: {args.out}")

    # Optional plots per enzyme
    if args.plot_dir is not None:
        os.makedirs(args.plot_dir, exist_ok=True)
        for eid, dfe in merged.groupby(ID_COL):
            dfe = dfe.sort_values(GLU_COL)
            g = dfe[GLU_COL].to_numpy(float)
            y_pred = dfe["Predicted_Activity"].to_numpy(float)
            y_true = dfe[Y_COL].to_numpy(float) if (has_truth and dfe[Y_COL].notna().all()) else None
            out_stub = os.path.join(args.plot_dir, f"{eid}_trigrams_gbr")
            plot_curve(eid, g, y_true, y_pred, out_stub)

        print(f"[DONE] Wrote plots to: {args.plot_dir}")


if __name__ == "__main__":
    main()
