"""
train.py
========
Fit 5 one-class anomaly detection models on preprocessed normal pump data.
Run ONCE after preprocess_training.py and feature_selection.py.
Saves all model artefacts to models/.

Threshold method
----------------
Thresholds are set using leave-one-out cross-validation (LOO-CV). Each sample
is scored by a model retrained without it, giving 110 unbiased out-of-fold
(OOF) scores that simulate the score distribution on unseen normal data.
The threshold is the 95th percentile of these OOF scores, guaranteeing
approximately 5% FPR on unseen normal cycles by construction.

Exception — EllipticEnvelope:
    The MCD estimator becomes numerically unstable when trained on 109 samples
    with 14 features (ratio 6.8, near the stability limit of ~5x). LOO is
    therefore skipped for this model. Its threshold is set in evaluate.py
    using the held-out normal evaluation set instead.

Models
------
  Mahalanobis       — ellipsoidal boundary, most interpretable
  LOF               — local density, catches localised clusters
  Isolation Forest  — random tree splits, robust to irrelevant features
  Elliptic Envelope — robust Mahalanobis (MCD); threshold set from eval set
  OC-SVM            — non-linear RBF boundary; included for comparison only.
                      High FPR makes it unsuitable for deployment.

Inputs
------
    data/training/features_scaled.csv
    data/training/feature_names_selected.joblib  (or feature_names.joblib fallback)

Outputs
-------
    models/mahalanobis.joblib
    models/lof.joblib
    models/isoforest.joblib
    models/elliptic_envelope.joblib   (threshold=None, patched by evaluate.py)
    models/ocsvm.joblib
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

try:
    from pyod.models.ecod import ECOD
    HAS_PYOD = True
except ImportError:
    ECOD = None  # type: ignore[assignment,misc]
    HAS_PYOD = False

# ── CONFIG ────────────────────────────────────────────────────────────────────
TRAINING_DIR     = Path("data/training")
RAW_CSV          = TRAINING_DIR / "features_raw.csv"
MODELS_DIR       = "models" #"models"

LOO_PERCENTILE   = 99    # ~5% FPR on unseen normal data by construction
PCA_N_COMPONENTS = 0.95  # explained-variance ratio; sklearn picks n_components automatically
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(MODELS_DIR, exist_ok=True)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_CSV)

# Use selected feature subset if feature_selection.py has been run,
# otherwise fall back to full feature set from preprocess_training.py.
feat_sel_pkl = TRAINING_DIR / "feature_names_selected.joblib"
feat_all_pkl = TRAINING_DIR / "feature_names.joblib"
if feat_sel_pkl.exists():
    feature_names = joblib.load(feat_sel_pkl)
    print(f"Using selected features ({len(feature_names)}) "
          f"from feature_names_selected.joblib")
else:
    feature_names = joblib.load(feat_all_pkl)
    print(f"Using all {len(feature_names)} features "
          f"(run feature_selection.py to reduce)")

# features_raw.csv is already clipped by preprocess.py — apply scale only.
# (evaluate.py clips then scales raw eval features, arriving at the same space.)
scaler = joblib.load(TRAINING_DIR / "robust_scaler.joblib")
X = scaler.transform(df[feature_names])
print(f"Training set:  {X.shape[0]} samples x {X.shape[1]} features")
print(f"LOO threshold: {LOO_PERCENTILE}th percentile of OOF scores "
      f"(~{100 - LOO_PERCENTILE}% FPR on unseen normals)\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(name: str, bundle: dict) -> None:
    path = os.path.join(MODELS_DIR, f"{name}.joblib")
    joblib.dump(bundle, path)
    print(f"  saved -> {path}")


def _report(name: str, oof_scores: np.ndarray, threshold: float) -> None:
    print(f"  LOO threshold ({LOO_PERCENTILE}th pct) = {threshold:.4f}")
    print(f"  OOF scores: mean={oof_scores.mean():.3f}  "
          f"std={oof_scores.std():.3f}  "
          f"max={oof_scores.max():.3f}")


def compute_loo(model_name: str, X: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Leave-one-out threshold estimation.

    Retrains the model on n-1 samples and scores the held-out sample,
    repeated for all n samples. Returns the LOO_PERCENTILE of the resulting
    out-of-fold scores as the deployment threshold.
    """
    n = len(X)
    oof = np.zeros(n, dtype=np.float64)
    print(f"  LOO-CV ({n} folds)...", end="", flush=True)

    for i in range(n):
        X_tr = np.delete(X, i, axis=0)
        x_te = X[i: i + 1]

        if model_name == "mahalanobis":
            mu      = X_tr.mean(axis=0)
            cov_inv = np.linalg.pinv(
                np.cov(X_tr.T) + np.eye(X_tr.shape[1]) * 1e-6
            )
            oof[i] = float(mahalanobis(x_te[0], mu, cov_inv))

        elif model_name == "lof":
            k = max(20, int(np.sqrt(len(X_tr))))
            m = LocalOutlierFactor(n_neighbors=k, novelty=True, contamination=0.1)
            m.fit(X_tr)
            oof[i] = float(-m.score_samples(x_te)[0])

        elif model_name == "isoforest":
            m = IsolationForest(n_estimators=200, contamination="auto",
                                random_state=42)
            m.fit(X_tr)
            oof[i] = float(-m.score_samples(x_te)[0])

        elif model_name == "ocsvm":
            m = OneClassSVM(kernel="rbf", nu=0.01, gamma="scale")
            m.fit(X_tr)
            oof[i] = float(-m.score_samples(x_te)[0])

        elif model_name == "pca":
            m = PCA(n_components=PCA_N_COMPONENTS, random_state=42)
            m.fit(X_tr)
            x_rec = m.inverse_transform(m.transform(x_te))
            oof[i] = float(np.sum((x_te - x_rec) ** 2))

        elif model_name == "ecod":
            assert ECOD is not None
            m = ECOD()
            m.fit(X_tr)
            oof[i] = float(m.decision_function(x_te)[0])

        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f" {i+1}/{n}", end="", flush=True)

    print()
    return float(np.percentile(oof, LOO_PERCENTILE)), oof


# ── 2. MAHALANOBIS ────────────────────────────────────────────────────────────
print("Fitting Mahalanobis...")
mean_v  = X.mean(axis=0)
cov     = np.cov(X.T) + np.eye(X.shape[1]) * 1e-6
cov_inv = np.linalg.pinv(cov)

mah_thresh, mah_oof = compute_loo("mahalanobis", X)

_save("mahalanobis", {
    "mean":       mean_v,
    "cov_inv":    cov_inv,
    "threshold":  mah_thresh,
    "oof_scores": mah_oof,
    "features":   feature_names,
})
_report("mahalanobis", mah_oof, mah_thresh)

# ── 3. LOF ────────────────────────────────────────────────────────────────────
print("\nFitting LOF...")
k   = max(20, int(np.sqrt(len(X))))
lof = LocalOutlierFactor(n_neighbors=k, novelty=True, contamination=0.1)
lof.fit(X)

lof_thresh, lof_oof = compute_loo("lof", X)

_save("lof", {
    "model":      lof,
    "threshold":  lof_thresh,
    "oof_scores": lof_oof,
    "features":   feature_names,
})
print(f"  k = {k}")
_report("lof", lof_oof, lof_thresh)

# ── 4. ISOLATION FOREST ───────────────────────────────────────────────────────
print("\nFitting Isolation Forest...")
iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
iso.fit(X)

iso_thresh, iso_oof = compute_loo("isoforest", X)

_save("isoforest", {
    "model":      iso,
    "threshold":  iso_thresh,
    "oof_scores": iso_oof,
    "features":   feature_names,
})
_report("isoforest", iso_oof, iso_thresh)

# ── 5. ELLIPTIC ENVELOPE ──────────────────────────────────────────────────────
# LOO skipped: MCD numerically unstable at n=109, p=16 (ratio 6.8).
# threshold=None is patched by evaluate.py using the held-out normal eval set
# (95th percentile of eval scores), which is still unseen normal data.
print("\nFitting Elliptic Envelope...")
ee = EllipticEnvelope(contamination=0.1, support_fraction=0.9, random_state=42)
ee.fit(X)

print("  [NOTE] LOO skipped — MCD unstable at n=109, p=16. "
      "Threshold will be set from eval set in evaluate.py.")

_save("elliptic_envelope", {
    "model":      ee,
    "threshold":  None,   # patched by evaluate.py
    "oof_scores": None,
    "features":   feature_names,
})

# ── 6. OC-SVM ─────────────────────────────────────────────────────────────────
# Included for comparison only — high FPR excludes it from deployment ensemble.
print("\nFitting OC-SVM...")
svm = OneClassSVM(kernel="rbf", nu=0.01, gamma="scale")
svm.fit(X)

svm_thresh, svm_oof = compute_loo("ocsvm", X)

_save("ocsvm", {
    "model":      svm,
    "threshold":  svm_thresh,
    "oof_scores": svm_oof,
    "features":   feature_names,
})
print(f"  kernel=rbf  nu=0.01  gamma=scale")
_report("ocsvm", svm_oof, svm_thresh)

# ── 7. PCA ────────────────────────────────────────────────────────────────────
print("\nFitting PCA (reconstruction error)...")
pca = PCA(n_components=PCA_N_COMPONENTS, random_state=42)
pca.fit(X)
print(f"  n_components = {pca.n_components_}  "
      f"(explains {pca.explained_variance_ratio_.sum()*100:.1f}% variance)")

pca_thresh, pca_oof = compute_loo("pca", X)

_save("pca", {
    "model":      pca,
    "scorer":     "pca_reconstruction",
    "threshold":  pca_thresh,
    "oof_scores": pca_oof,
    "features":   feature_names,
})
_report("pca", pca_oof, pca_thresh)

# ── 8. ECOD ───────────────────────────────────────────────────────────────────
ecod_thresh: float | None = None
ecod_oof:    np.ndarray | None = None
if HAS_PYOD:
    assert ECOD is not None
    print("\nFitting ECOD...")
    ecod = ECOD()
    ecod.fit(X)
    ecod_scores = ecod.decision_scores_   # in-sample scores, computed during fit

    ecod_thresh, ecod_oof = compute_loo("ecod", X)

    _save("ecod", {
        "model":        ecod,
        "scorer":       "decision_function",
        "threshold":    ecod_thresh,
        "oof_scores":   ecod_oof,
        "train_scores": ecod_scores,
        "features":     feature_names,
    })
    _report("ecod", ecod_oof, ecod_thresh)
else:
    print("\n[SKIP] ECOD — install pyod first:  pip install pyod")

# ── OOF DISTRIBUTION PLOT ────────────────────────────────────────────────────

def plot_oof_distributions(oof_data: dict, out_dir: str) -> None:
    """
    Plot OOF score distributions for all LOO-calibrated models.
    Each subplot shows the histogram of 110 OOF scores with the
    95th-percentile threshold marked. EllipticEnvelope is omitted
    (LOO not available). Saved as loo_oof_distributions.pdf for thesis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    n = len(oof_data)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.8, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (name, (oof, thresh)) in zip(axes, oof_data.items()):
        ax.hist(oof, bins=25, color="steelblue", alpha=0.85, edgecolor="white")
        ax.axvline(thresh, color="black", ls="--", lw=1.6,
                   label=f"$\delta_{{\mathrm{{LOO}}}}$ = {thresh:.3f}")
        ax.set_title(name.replace("_", " "), fontsize=9)
        ax.set_xlabel("OOF anomaly score", fontsize=8)
        ax.set_ylabel("Count" if ax == axes[0] else "", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "LOO-CV Out-of-Fold Score Distributions\n"
        "(dashed line = 99th-percentile deployment threshold)",
        fontsize=10, y=1.02
    )
    fig.tight_layout()

    for fmt in ("pdf", "png"):
        p = Path(out_dir) / f"loo_oof_distributions.{fmt}"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        print(f"  -> {p}")
    plt.close(fig)


# collect OOF data for models with LOO (EllipticEnvelope excluded)
oof_data = {
    "mahalanobis": (mah_oof, mah_thresh),
    "lof":         (lof_oof, lof_thresh),
    "isoforest":   (iso_oof, iso_thresh),
    "ocsvm":       (svm_oof, svm_thresh),
    "pca":         (pca_oof, pca_thresh),
}
if ecod_oof is not None and ecod_thresh is not None:
    oof_data["ecod"] = (ecod_oof, ecod_thresh)

plot_oof_distributions(oof_data, os.path.join(os.path.dirname(MODELS_DIR), "evaluation"))

# ── DONE ──────────────────────────────────────────────────────────────────────
n_saved = 6 + (1 if HAS_PYOD else 0)
print(f"\nAll {n_saved} models saved -> {MODELS_DIR}/")
print("\nReminder: run evaluate.py next to patch EllipticEnvelope threshold "
      "and generate evaluation results.")