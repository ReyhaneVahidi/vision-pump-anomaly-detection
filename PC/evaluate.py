"""
evaluate.py
===========
Preprocess and evaluate all trained anomaly detection models on unseen data.
Run AFTER train.py. Never retrains or adjusts thresholds — except for
EllipticEnvelope whose threshold is set here from the normal eval set
(LOO was skipped in train.py due to MCD numerical instability).

Workflow
--------
  1. Load preprocessing artefacts (scaler, clip bounds, feature names)
  2. Scale eval CSVs using training-time clip -> scale (never refit)
  3. Patch EllipticEnvelope threshold from normal eval set (95th percentile)
  4. Score each model on normal-eval and anomalous samples
  5. Compute FPR, detection rate, precision, recall, F1 per model
  6. Save plots and result CSVs to evaluation/

Inputs
------
  data/training/feature_names_selected.joblib   feature list
  data/training/clip_bounds.joblib              per-feature percentile bounds
  data/training/robust_scaler.joblib            fitted RobustScaler
  data/test/test_normal.csv                     raw features, unseen normal videos
  data/test/test_abnormal.csv                   raw features, anomalous videos (optional)
  models/<model>.joblib                         one per entry in MODELS

Outputs (evaluation/)
---------------------
  <model>_score_dist.png     score histogram with threshold line
  <model>_pca.png            PCA 2-D projection coloured by score
  evaluation_results.csv     one row per sample, score_* and flag_* per model
  evaluation_summary.json    metrics dict per model (consumed by compare.py)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.decomposition import PCA
# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAINING_DIR    = Path("data/training") # "data/training"
MODELS_DIR      = Path("models") # "models"
OUT_DIR         = Path("evaluation") # "evaluation"

NORMAL_EVAL_CSV = Path("data/test/test_normal.csv")
ABNORMAL_CSV    = Path("data/test/test_abnormal.csv")      # optional

EE_PERCENTILE   = 95   # percentile of normal eval scores used as EE threshold

MODELS = {
    "mahalanobis":       "mahalanobis.joblib",
    "lof":               "lof.joblib",
    "isoforest":         "isoforest.joblib",
    "elliptic_envelope": "elliptic_envelope.joblib",
    "ocsvm":             "ocsvm.joblib",
}
# ─────────────────────────────────────────────────────────────────────────────


# ── Preprocessing ─────────────────────────────────────────────────────────────

def scale_eval_csv(csv_path: Path, feature_names: list,
                   clip_bounds: dict, scaler) -> pd.DataFrame | None:
    """
    Load a raw feature CSV and apply training-time clip -> scale.
    Returns None if the file does not exist.
    """
    if not csv_path.exists():
        print(f"  [SKIP] not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: missing features {missing}")

    X = df[feature_names].copy()
    n_nan = X.isna().sum().sum()
    if n_nan:
        print(f"  [WARN] {n_nan} NaN values in {csv_path.name} — filling with median")
        X = X.fillna(X.median())

    for col in feature_names:
        lo, hi = clip_bounds[col]
        X[col] = X[col].clip(lo, hi)

    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names, index=df.index)
    if "file" in df.columns:
        X_scaled.insert(0, "file", df["file"].values)

    print(f"  Scaled {len(df):>4d} samples  ({csv_path.name})")
    return X_scaled


# ── EllipticEnvelope threshold patch ─────────────────────────────────────────

def patch_ee_threshold(art: dict, df_normal: pd.DataFrame,
                       ee_path: Path, percentile: int) -> float:
    """
    Set EllipticEnvelope threshold from the normal eval set and persist it.

    Uses the percentile-th percentile of normal eval scores — the same logic
    as LOO but on the held-out eval set instead of cross-validation folds.
    This is necessary because MCD is numerically unstable during LOO at the
    available sample-to-feature ratio (see train.py).

    The patched threshold is saved back to the .joblib file so subsequent
    runs and the Pi deployment load the correct value automatically.
    """
    features  = art["features"]
    X_normal  = df_normal[features].values
    ee_scores = -art["model"].score_samples(X_normal)
    threshold = float(np.percentile(ee_scores, percentile))

    art["threshold"] = threshold
    joblib.dump(art, ee_path)

    print(f"  EllipticEnvelope threshold set from normal eval set "
          f"({percentile}th pct) = {threshold:.4f}")
    print(f"  Eval scores: mean={ee_scores.mean():.3f}  "
          f"std={ee_scores.std():.3f}  "
          f"max={ee_scores.max():.3f}")
    return threshold


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_scores(art: dict, X: np.ndarray) -> np.ndarray:
    """Higher score = more anomalous, consistent with train.py convention."""
    if "mean" in art:
        return np.array([mahalanobis(x, art["mean"], art["cov_inv"]) for x in X])
    return -art["model"].score_samples(X)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(normal_sc: np.ndarray, abn_sc: np.ndarray | None,
                    threshold: float) -> dict:
    n_n = len(normal_sc)
    fp  = int((normal_sc > threshold).sum())
    m: dict = {
        "n_normal":            n_n,
        "false_positives":     fp,
        "false_positive_rate": round(fp / n_n, 4),
        "threshold":           round(float(threshold), 6),
    }
    if abn_sc is not None:
        n_a  = len(abn_sc)
        tp   = int((abn_sc > threshold).sum())
        dr   = tp / n_a
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1   = (2 * prec * dr / (prec + dr)) if (prec + dr) > 0 else 0.0
        m.update({
            "n_anomalous":    n_a,
            "tp":             tp,
            "detection_rate": round(dr,   4),
            "precision":      round(prec, 4),
            "recall":         round(dr,   4),
            "f1":             round(f1,   4),
        })
    return m


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_score_dist(name, normal_sc, threshold, abn_sc, out_dir):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(normal_sc, bins=30, color="steelblue", alpha=0.85,
            edgecolor="white", label=f"Normal eval (n={len(normal_sc)})")
    if abn_sc is not None:
        ax.hist(abn_sc, bins=max(8, len(abn_sc) // 2), color="tomato",
                alpha=0.85, edgecolor="white", label=f"Anomalous (n={len(abn_sc)})")
    ax.axvline(threshold, color="black", ls="--", lw=1.6,
               label=f"Threshold = {threshold:.4f}")
    ax.set_xlabel("Anomaly score  (higher = more anomalous)")
    ax.set_ylabel("Count")
    ax.set_title(f"{name} — Score Distribution")
    ax.legend()
    fig.tight_layout()
    p = out_dir / f"{name}_score_dist.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> {p.name}")


def plot_pca(name, df_normal, normal_sc, threshold, features,
             df_abn, abn_sc, out_dir):
    pca   = PCA(n_components=2, random_state=42)
    Xn_2d = pca.fit_transform(df_normal[features].values)
    ev    = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(Xn_2d[:, 0], Xn_2d[:, 1], c=normal_sc, cmap="Blues",
                    s=55, edgecolors="grey", linewidths=0.4, label="Normal", zorder=2)
    plt.colorbar(sc, ax=ax, label="Anomaly score")

    fp_mask = normal_sc > threshold
    if fp_mask.any():
        ax.scatter(Xn_2d[fp_mask, 0], Xn_2d[fp_mask, 1],
                   s=120, facecolors="none", edgecolors="orange", linewidths=1.6,
                   label=f"False positive ({fp_mask.sum()})", zorder=3)

    if df_abn is not None and abn_sc is not None:
        Xa_2d = pca.transform(df_abn[features].values)
        ax.scatter(Xa_2d[:, 0], Xa_2d[:, 1], c="tomato", marker="X",
                   s=120, edgecolors="darkred", linewidths=0.6,
                   label="Anomalous", zorder=4)
        missed = abn_sc <= threshold
        if missed.any():
            ax.scatter(Xa_2d[missed, 0], Xa_2d[missed, 1],
                       s=220, facecolors="none", edgecolors="purple", linewidths=1.6,
                       label=f"Missed ({missed.sum()})", zorder=5)

    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}% var)")
    ax.set_title(f"{name} — PCA Projection")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / f"{name}_pca.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"   -> {p.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load preprocessing artefacts
    feature_names = joblib.load(TRAINING_DIR / "feature_names_selected.joblib")
    clip_bounds   = joblib.load(TRAINING_DIR / "clip_bounds.joblib")
    scaler        = joblib.load(TRAINING_DIR / "robust_scaler.joblib")
    print(f"Features: {len(feature_names)}\n")

    # Scale eval data — never refit
    print("Scaling eval data...")
    df_normal = scale_eval_csv(NORMAL_EVAL_CSV, feature_names, clip_bounds, scaler)
    df_abn    = scale_eval_csv(ABNORMAL_CSV,    feature_names, clip_bounds, scaler)

    if df_normal is None:
        raise SystemExit(
            f"\n[ERROR] Normal eval CSV not found: {NORMAL_EVAL_CSV}\n"
            "  Run pump_extractor.py on held-out normal videos first."
        )

    # Build results table
    normal_ids = (df_normal["file"].tolist() if "file" in df_normal.columns
                  else [f"normal_{i}" for i in range(len(df_normal))])
    abn_ids = (df_abn["file"].tolist() if df_abn is not None and "file" in df_abn.columns
               else ([f"abn_{i}" for i in range(len(df_abn))] if df_abn is not None else []))

    results_df = pd.DataFrame({
        "file":  normal_ids + abn_ids,
        "label": ["normal"] * len(normal_ids) + ["anomalous"] * len(abn_ids),
    })

    all_metrics: dict = {}

    for name, fname in MODELS.items():
        model_path = MODELS_DIR / fname
        if not model_path.exists():
            print(f"\n  [SKIP] {name}: model not found")
            continue

        print(f"\n── {name} ──────────────────────────────────────")
        art      = joblib.load(model_path)
        features = art["features"]

        # Patch EllipticEnvelope threshold from normal eval set
        if name == "elliptic_envelope" and art["threshold"] is None:
            threshold = patch_ee_threshold(art, df_normal, model_path, EE_PERCENTILE)
        else:
            threshold = float(art["threshold"])

        normal_sc = compute_scores(art, df_normal[features].values)
        abn_sc    = (compute_scores(art, df_abn[features].values)
                     if df_abn is not None else None)

        m = compute_metrics(normal_sc, abn_sc, threshold)
        all_metrics[name] = m

        all_sc = (np.concatenate([normal_sc, abn_sc])
                  if abn_sc is not None else normal_sc)
        results_df[f"score_{name}"] = all_sc
        results_df[f"flag_{name}"]  = (all_sc > threshold).astype(int)

        print(f"  threshold : {threshold:.5f}")
        print(f"  FPR       : {m['false_positives']}/{m['n_normal']}"
              f"  ({m['false_positive_rate']*100:.1f}%)")
        if abn_sc is not None:
            print(f"  Detected  : {m['tp']}/{m['n_anomalous']}"
                  f"  ({m['detection_rate']*100:.1f}%)")
            print(f"  Precision : {m['precision']:.3f}  "
                  f"Recall : {m['recall']:.3f}  F1 : {m['f1']:.3f}")
            id_col = "file" if "file" in df_abn.columns else None
            for i, sc in enumerate(abn_sc):
                lbl = df_abn[id_col].iloc[i] if id_col else f"sample_{i}"
                tag = "DETECTED" if sc > threshold else "missed  "
                print(f"    {tag}  {str(lbl):<45}  score={sc:.5f}")

        plot_score_dist(name, normal_sc, threshold, abn_sc, OUT_DIR)
        plot_pca(name, df_normal, normal_sc, threshold, features,
                 df_abn, abn_sc, OUT_DIR)

    # Save artefacts
    results_df.to_csv(OUT_DIR / "evaluation_results.csv", index=False)
    with open(OUT_DIR / "evaluation_summary.json", "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\n   -> evaluation_results.csv")
    print(f"   -> evaluation_summary.json")

    # Summary table
    print("\n" + "=" * 68)
    print(f"  {'Model':<22} {'DR':>7} {'Prec':>7} {'Recall':>8} {'F1':>7} {'FPR':>7}")
    print("-" * 68)
    for name, m in all_metrics.items():
        print(f"  {name:<22}"
              f"  {m.get('detection_rate', 0)*100:>5.1f}%"
              f"  {m.get('precision',      0)*100:>5.1f}%"
              f"  {m.get('recall',         0)*100:>6.1f}%"
              f"  {m.get('f1',             0)*100:>5.1f}%"
              f"  {m['false_positive_rate']*100:>5.1f}%")
    print("=" * 68)
    print("\nDone. Run compare.py for comparison figures.")


if __name__ == "__main__":
    run_evaluation()