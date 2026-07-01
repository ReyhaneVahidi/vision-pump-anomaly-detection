"""
feature_selection.py
====================
Data leakage prevention:
    Feature importance, feature selection, clipping bounds, and scaling
    parameters are estimated using training data only.

Select the most informative features from the scaled training data using
permutation importance across multiple one-class models.

Why permutation importance:
    For each feature, we shuffle its values across all training samples and
    measure how much the anomaly score distribution changes. Features that
    cause a large change when shuffled are important — the model relies on
    them. Features that cause little change are not contributing and may
    even hurt performance by adding noise dimensions (curse of dimensionality).

    This is model-driven selection: the features we keep are the ones our
    actual deployed models rely on, not just the ones with high variance.

Workflow:
    1. Load scaled training data + feature names
    2. Fit IsolationForest and Mahalanobis on all 18 features
    3. For each feature: shuffle it N_PERMUTATIONS times, measure mean score
       change → importance score
    4. Plot importance ranking for both models
    5. Select features above IMPORTANCE_THRESHOLD (or top N_KEEP)
    6. Save feature_names_selected.joblib → used by train.py

Run AFTER preprocess_training.py, BEFORE train.py.

Inputs:
    data/training/features_scaled.csv
    data/training/feature_names.joblib

Outputs:
    data/training/feature_names_selected.joblib   subset of features
    evaluation/feature_importance.png              importance ranking plot
"""

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data/training")
OUT_DIR       = Path("evaluation")

N_PERMUTATIONS   = 30    # shuffles per feature — more = more stable estimate
RANDOM_SEED      = 42
N_KEEP           = None  # if set, keep top N features regardless of threshold
IMPORTANCE_THRESHOLD = 0.05  # drop features below this mean score change

# Models used for importance estimation.
# IsolationForest represents a tree-based approach and Mahalanobis
# represents a distance-based approach. Using both gives a more robust importance estimate.
USE_MODELS = ["isoforest", "mahalanobis"]

# Manual overrides for correlated pairs where importance scores are within
# measurement uncertainty (error bars overlap). Domain knowledge used as
# tiebreaker — the more physically interpretable feature is kept.
# Set to None to let importance decide automatically.
CORR_PAIR_OVERRIDES: dict[str, str] = {
    "dominant_freq_hz":  "keep",   # preferred over flow_dom_freq (more interpretable)
    "autocorr_at_period": "keep",  # preferred over phase_portrait_eccentricity
}
# ─────────────────────────────────────────────────────────────────────────────

rng = np.random.default_rng(RANDOM_SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fit_models(X: np.ndarray, feature_names: list) -> dict:
    """Fit the models used for importance estimation."""
    models = {}

    if "isoforest" in USE_MODELS:
        iso = IsolationForest(n_estimators=200, contamination="auto", random_state=RANDOM_SEED)
        iso.fit(X)
        models["isoforest"] = iso
        print(f"  Fitted IsolationForest")

    if "mahalanobis" in USE_MODELS:
        mean_v  = X.mean(axis=0)
        cov     = np.cov(X.T) + np.eye(X.shape[1]) * 1e-6
        cov_inv = np.linalg.pinv(cov)
        models["mahalanobis"] = {"mean": mean_v, "cov_inv": cov_inv}
        print(f"  Fitted Mahalanobis")

    if "lof" in USE_MODELS:
        k   = max(20, int(np.sqrt(len(X))))
        lof = LocalOutlierFactor(n_neighbors=k, novelty=True, contamination=0.1)
        lof.fit(X)
        models["lof"] = lof
        print(f"  Fitted LOF (k={k})")

    return models


def score_samples(models: dict, X: np.ndarray) -> np.ndarray:
    """
    Return mean anomaly score across all models (higher = more anomalous).
    Scores from each model are normalised to [0,1] before averaging so
    different score scales don't dominate.
    """
    all_scores = []

    for name, model in models.items():
        if name == "mahalanobis":
            sc = np.array([mahalanobis(x, model["mean"], model["cov_inv"]) for x in X])
        else:
            sc = -model.score_samples(X)

        # Normalise to [0, 1]
        sc_min, sc_max = sc.min(), sc.max()
        if sc_max > sc_min:
            sc = (sc - sc_min) / (sc_max - sc_min)

        all_scores.append(sc)

    return np.mean(all_scores, axis=0)


def permutation_importance(
    models: dict,
    X: np.ndarray,
    feature_names: list,
    n_permutations: int,
) -> pd.DataFrame:
    """
    Compute permutation importance for each feature.

    Importance = mean absolute change in anomaly score when the feature
    is randomly shuffled across all training samples.

    A high importance means the model relies on this feature.
    A near-zero importance means the feature is not contributing.
    """
    baseline = score_samples(models, X)
    baseline_mean = float(baseline.mean())

    records = []
    for j, feat in enumerate(feature_names):
        deltas = []
        for _ in range(n_permutations):
            X_perm = X.copy()
            perm_idx = rng.permutation(len(X_perm))
            X_perm[:, j] = X_perm[perm_idx, j]

            perm_scores = score_samples(models, X_perm)
            # How much did the mean score change?
            delta = float(np.mean(np.abs(perm_scores - baseline)))
            deltas.append(delta)

        records.append({
            "feature":    feat,
            "importance": float(np.mean(deltas)),
            "importance_std": float(np.std(deltas)),
        })
        print(f"  [{j+1:>2}/{len(feature_names)}] {feat:<30}  "
              f"importance={records[-1]['importance']:.4f} "
              f"(+/-{records[-1]['importance_std']:.4f})")

    df = pd.DataFrame(records).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def plot_importance(df: pd.DataFrame, threshold: float, out_dir: Path,
                    corr_drops: list | None = None) -> None:
    """Horizontal bar chart of feature importances with threshold line.
    Blue = selected, grey = below threshold, orange = dropped due to correlation.
    """
    corr_drops = set(corr_drops or [])
    fig, ax = plt.subplots(figsize=(8, max(5, len(df) * 0.38)))

    colours = []
    for _, row in df.iterrows():
        if row["feature"] in corr_drops:
            colours.append("#FF9800")   # orange = corr-dropped
        elif row["importance"] >= threshold:
            colours.append("#2196F3")   # blue = selected
        else:
            colours.append("#B0BEC5")   # grey = below threshold

    bars = ax.barh(df["feature"][::-1], df["importance"][::-1],
                   xerr=df["importance_std"][::-1],
                   color=colours[::-1], alpha=0.85, edgecolor="white",
                   capsize=3, error_kw=dict(lw=1, capthick=1))

    ax.axvline(threshold, color="red", ls="--", lw=1.4,
               label=f"Threshold = {threshold:.3f}")
    ax.set_xlabel("Mean absolute score change on permutation")
    ax.set_title(f"Permutation Feature Importance\n"
                 f"(blue = selected, grey = below threshold, orange = corr-pair dropped  "
                 f"models: {', '.join(USE_MODELS)})")
    ax.legend(fontsize=9)
    fig.tight_layout()

    p = out_dir / "feature_importance.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n   -> {p.name}")


def resolve_correlated_pairs(
    importance_df: pd.DataFrame,
    X: np.ndarray,
    feature_names: list,
    corr_threshold: float = 0.95,
) -> list[str]:
    """
    For any pair of features with |r| > corr_threshold, drop the lower-importance one.
    This prevents double-counting correlated dimensions in distance-based models.
    Returns the list of features to REMOVE.
    """
    feat_idx = {f: i for i, f in enumerate(feature_names)}
    imp_map  = dict(zip(importance_df["feature"], importance_df["importance"]))

    corr_matrix = np.corrcoef(X.T)
    to_drop = set()

    checked = set()
    for i, fi in enumerate(feature_names):
        for j, fj in enumerate(feature_names):
            if i >= j:
                continue
            pair = (fi, fj)
            if pair in checked:
                continue
            checked.add(pair)

            if abs(corr_matrix[i, j]) >= corr_threshold:
                # Check domain overrides first
                if CORR_PAIR_OVERRIDES.get(fi) == "keep":
                    winner, loser = fi, fj
                    reason = f"domain override keeps {fi}"
                elif CORR_PAIR_OVERRIDES.get(fj) == "keep":
                    winner, loser = fj, fi
                    reason = f"domain override keeps {fj}"
                else:
                    imp_i = imp_map.get(fi, 0)
                    imp_j = imp_map.get(fj, 0)
                    winner = fi if imp_i >= imp_j else fj
                    loser  = fj if imp_i >= imp_j else fi
                    reason = f"imp {imp_map.get(winner,0):.4f} > {imp_map.get(loser,0):.4f}"
                to_drop.add(loser)
                print(f"  Correlated pair (r={corr_matrix[i,j]:.4f}): "
                      f"{fi} vs {fj} -> keep {winner} ({reason}), drop {loser}")

    return list(to_drop)


def select_features(df: pd.DataFrame, n_keep: int | None,
                    threshold: float) -> list[str]:
    """Return selected feature names based on N_KEEP or IMPORTANCE_THRESHOLD."""
    if n_keep is not None:
        selected = df["feature"].head(n_keep).tolist()
        print(f"\nKeeping top {n_keep} features (N_KEEP override)")
    else:
        selected = df[df["importance"] >= threshold]["feature"].tolist()
        dropped  = df[df["importance"] <  threshold]["feature"].tolist()
        print(f"\nFeatures above threshold ({threshold}): {len(selected)}")
        if dropped:
            print(f"Dropped ({len(dropped)}): {dropped}")

    return selected


# ── Main ──────────────────────────────────────────────────────────────────────

def run() -> None:
    # Load data
    feature_names = joblib.load(PROCESSED_DIR / "feature_names.joblib")
    df = pd.read_csv(PROCESSED_DIR / "features_scaled.csv")
    X  = df[feature_names].values

    print(f"Training set: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"Permutations per feature: {N_PERMUTATIONS}\n")

    # Fit models
    print("Fitting models for importance estimation...")
    models = fit_models(X, feature_names)

    # Compute importance
    print(f"\nComputing permutation importance ({N_PERMUTATIONS} shuffles per feature)...")
    importance_df = permutation_importance(models, X, feature_names, N_PERMUTATIONS)

    print("\nImportance ranking:")
    print(importance_df.to_string(index=False))

    # Plot
    # Step 1: resolve correlated pairs using importance to decide which to keep
    print("\nResolving correlated pairs (keeping higher-importance feature)...")
    corr_drops = resolve_correlated_pairs(importance_df, X, feature_names)
    if not corr_drops:
        print("  No correlated pairs above threshold")

    # Step 2: apply threshold / N_KEEP selection on remaining features
    importance_filtered = importance_df[~importance_df["feature"].isin(corr_drops)].copy()
    selected = select_features(importance_filtered, N_KEEP, IMPORTANCE_THRESHOLD)

    # Also mark corr-dropped features in the plot
    importance_df["dropped_corr"] = importance_df["feature"].isin(corr_drops)
    plot_importance(importance_df, IMPORTANCE_THRESHOLD, OUT_DIR, corr_drops)

    if len(selected) < 3:
        print(f"\n[WARN] Only {len(selected)} features selected — threshold may be too high.")
        print("  Lower IMPORTANCE_THRESHOLD or set N_KEEP explicitly.")
        return

    # Save feature names
    out_path = PROCESSED_DIR / "feature_names_selected.joblib"
    joblib.dump(selected, out_path)
    print(f"\nSelected features ({len(selected)}): {selected}")
    print(f"Saved -> {out_path}")

    # Refit scaler and clip bounds on selected features only.
    # Use features_raw.csv (clipped but unscaled) so the saved artefacts
    # are in raw-feature units and evaluate.py can apply them correctly
    # to unseen raw eval features.
    print("\nRefitting scaler on selected features...")
    from sklearn.preprocessing import RobustScaler

    df_raw = pd.read_csv(PROCESSED_DIR / "features_raw.csv")
    X_sel = df_raw[selected].values

    clip_bounds_sel = {}
    X_clipped = X_sel.copy()
    for j, col in enumerate(selected):
        lo = float(np.percentile(X_sel[:, j], 1))
        hi = float(np.percentile(X_sel[:, j], 99))
        clip_bounds_sel[col] = (lo, hi)
        X_clipped[:, j] = np.clip(X_sel[:, j], lo, hi)

    scaler_sel = RobustScaler()
    scaler_sel.fit(pd.DataFrame(X_clipped, columns=selected))

    joblib.dump(scaler_sel,      PROCESSED_DIR / "robust_scaler.joblib")
    joblib.dump(clip_bounds_sel, PROCESSED_DIR / "clip_bounds.joblib")
    print(f"Scaler and clip bounds refitted on {len(selected)} features -> saved")
    print("\nRun train.py next — it will use feature_names_selected.joblib automatically.")


if __name__ == "__main__":
    run()