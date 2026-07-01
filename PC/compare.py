"""
compare.py
==========
Aggregate evaluation artefacts and produce all comparison figures.
Never re-runs model inference — reads only from evaluation/.

Run AFTER evaluate.py.

Inputs  (evaluation/)
---------------------
  evaluation_results.csv     one row per sample, score_* and flag_* columns
  evaluation_summary.json    metrics dict per model

Outputs  (evaluation/)
----------------------
  comparison_boxplot.png/pdf      score distributions: normal vs anomalous per model
  comparison_roc.png/pdf          ROC curves with AUC + marked operating point
  comparison_heatmap.png/pdf      per-sample detection matrix (anomalous samples)
  comparison_vote.png/pdf         majority vote confusion matrix
  comparison_score_dist.png/pdf   per-model score histograms (normal + anomaly)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

# ── CONFIG ────────────────────────────────────────────────────────────────────
OUT_DIR      = Path("evaluation")
RESULTS_CSV  = OUT_DIR / "evaluation_results.csv"
SUMMARY_JSON = OUT_DIR / "evaluation_summary.json"

# Models included in the majority vote
VOTE_MODELS    = ["mahalanobis", "lof", "isoforest", "elliptic_envelope", "ocsvm"]
VOTE_THRESHOLD = 3  # flag if >= this many models agree (majority of 5)

DPI     = 150
FORMATS = ["png", "pdf"]   # save every figure in both formats
# ─────────────────────────────────────────────────────────────────────────────


def _save(fig: plt.Figure, stem: str) -> None:
    """Save figure in all configured formats."""
    for fmt in FORMATS:
        p = OUT_DIR / f"{stem}.{fmt}"
        fig.savefig(p, dpi=DPI, bbox_inches="tight")
        print(f"   -> {p.name}")
    plt.close(fig)


# ── Majority vote ─────────────────────────────────────────────────────────────

def add_majority_vote(results_df: pd.DataFrame) -> pd.DataFrame:
    available = [m for m in VOTE_MODELS if f"flag_{m}" in results_df.columns]
    if not available:
        return results_df
    results_df["vote_count"] = results_df[[f"flag_{m}" for m in available]].sum(axis=1)
    results_df["vote_flag"]  = (results_df["vote_count"] >= VOTE_THRESHOLD).astype(int)
    return results_df


# ── 1. Score boxplot ──────────────────────────────────────────────────────────

def plot_score_boxplot(results_df: pd.DataFrame, model_names: list,
                       all_metrics: dict) -> None:
    """Score distributions per model — normal vs anomalous with threshold line."""
    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.8, 5), sharey=False)
    if n == 1:
        axes = [axes]

    df_n = results_df[results_df["label"] == "normal"]
    df_a = results_df[results_df["label"] == "anomalous"]

    for ax, name in zip(axes, model_names):
        col = f"score_{name}"
        if col not in results_df.columns:
            ax.set_visible(False)
            continue

        normal_sc = df_n[col].dropna().values
        data, labels, colours = [normal_sc], ["Normal"], ["steelblue"]

        abn_sc = df_a[col].dropna().values if not df_a.empty else np.array([])
        if abn_sc.size > 0:
            data.append(abn_sc)
            labels.append("Anomalous")
            colours.append("tomato")

        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", lw=2))
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.7)

        thr = all_metrics[name]["threshold"]
        ax.axhline(thr, color="black", ls="--", lw=1.2, label=f"thr={thr:.3f}")
        ax.set_title(name.replace("_", " "), fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.legend(fontsize=6)

    fig.suptitle("Score Distributions per Model", fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, "comparison_boxplot")


# ── 2. ROC curves ─────────────────────────────────────────────────────────────

def plot_roc_curves(results_df: pd.DataFrame, model_names: list,
                    all_metrics: dict) -> None:
    """ROC curves with AUC. Operating point marked as dot."""
    if not (results_df["label"] == "anomalous").any():
        print("  [SKIP] ROC curves: no anomalous samples")
        return

    y_true  = (results_df["label"] == "anomalous").astype(int).values
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(7, 6))

    for i, name in enumerate(model_names):
        col = f"score_{name}"
        if col not in results_df.columns:
            continue
        scores      = results_df[col].fillna(0).values
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc     = auc(fpr, tpr)
        colour      = colours[i % len(colours)]

        ax.plot(fpr, tpr, lw=1.8, color=colour,
                label=f"{name}  (AUC={roc_auc:.3f})")

        op_fpr = all_metrics[name]["false_positive_rate"]
        op_tpr = all_metrics[name].get("detection_rate", 0.0)
        ax.scatter([op_fpr], [op_tpr], s=80, color=colour, zorder=5,
                   edgecolors="black", linewidths=0.8)

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves  (dots = LOO threshold operating point)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    _save(fig, "comparison_roc")


# ── 3. Per-model score histograms ─────────────────────────────────────────────

def plot_score_distributions(results_df: pd.DataFrame, model_names: list,
                             all_metrics: dict) -> None:
    """
    Per-model score histograms showing normal eval and anomalous samples.
    Threshold marked. Good for thesis results section.
    """
    if not (results_df["label"] == "anomalous").any():
        print("  [SKIP] score dist: no anomalous samples")
        return

    n = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(n * 3.2, 4), sharey=False)
    if n == 1:
        axes = [axes]

    df_n = results_df[results_df["label"] == "normal"]
    df_a = results_df[results_df["label"] == "anomalous"]

    for ax, name in zip(axes, model_names):
        col = f"score_{name}"
        if col not in results_df.columns:
            ax.set_visible(False)
            continue

        normal_sc = df_n[col].dropna().values
        abn_sc    = df_a[col].dropna().values

        ax.hist(normal_sc, bins=20, color="steelblue", alpha=0.75,
                edgecolor="white", label=f"Normal (n={len(normal_sc)})")
        if abn_sc.size > 0:
            ax.hist(abn_sc, bins=max(5, len(abn_sc) // 2), color="tomato",
                    alpha=0.85, edgecolor="white", label=f"Anomalous (n={len(abn_sc)})")

        thr = all_metrics[name]["threshold"]
        ax.axvline(thr, color="black", ls="--", lw=1.4, label=f"thr={thr:.3f}")
        ax.set_title(name.replace("_", " "), fontsize=8)
        ax.set_xlabel("Anomaly score", fontsize=7)
        ax.set_ylabel("Count" if ax == axes[0] else "", fontsize=7)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)

    fig.suptitle("Score Distributions — Normal vs Anomalous", fontsize=11)
    fig.tight_layout()
    _save(fig, "comparison_score_dist")


# ── 4. Detection heatmap ──────────────────────────────────────────────────────

def plot_detection_heatmap(results_df: pd.DataFrame, model_names: list) -> None:
    """Per-sample detection matrix for anomalous samples including vote row."""
    df_abn = results_df[results_df["label"] == "anomalous"]
    if df_abn.empty:
        print("  [SKIP] heatmap: no anomalous samples")
        return

    display_cols = [f"flag_{m}" for m in model_names if f"flag_{m}" in df_abn.columns]
    if "vote_flag" in df_abn.columns:
        display_cols.append("vote_flag")

    mat    = df_abn[display_cols].values.T
    files  = (df_abn["file"].tolist() if "file" in df_abn.columns
              else [f"s{i}" for i in range(len(df_abn))])
    # shorten filenames for display
    files  = [f.replace(".mp4", "").split("_")[-1] if "_" in f else f for f in files]
    ylabels = [c.replace("flag_", "").replace("_", " ") for c in display_cols]

    fig, ax = plt.subplots(figsize=(max(7, len(files) * 0.7),
                                    len(display_cols) * 0.65 + 1.8))
    ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
              interpolation="nearest")

    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            ax.text(c, r, "\u2713" if mat[r, c] else "\u2717",
                    ha="center", va="center", fontsize=10,
                    color="black" if mat[r, c] else "darkred")

    ax.set_xticks(range(len(files)))
    ax.set_xticklabels(files, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_title("Per-Sample Detection Matrix  (anomalous samples only)", fontsize=10)

    if "vote_flag" in df_abn.columns:
        ax.axhline(len(display_cols) - 1.5, color="black", lw=1.5, ls="--")

    fig.tight_layout()
    _save(fig, "comparison_heatmap")


# ── 5. Majority vote confusion matrix ────────────────────────────────────────

def plot_vote_confusion(results_df: pd.DataFrame) -> None:
    """2x2 confusion matrix for the majority vote decision."""
    if "vote_flag" not in results_df.columns:
        print("  [SKIP] vote confusion: vote_flag column missing")
        return
    if not (results_df["label"] == "anomalous").any():
        print("  [SKIP] vote confusion: no anomalous samples")
        return

    y_true = (results_df["label"] == "anomalous").astype(int).values
    y_pred = results_df["vote_flag"].values

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    mat     = np.array([[tp, fn], [fp, tn]])
    labels  = [["TP", "FN"], ["FP", "TN"]]
    colours = np.array([[0.55, 0.92], [0.92, 0.55]])

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(colours, cmap="RdYlGn", vmin=0, vmax=1)

    for r in range(2):
        for c in range(2):
            ax.text(c, r, f"{labels[r][c]}\n{mat[r, c]}",
                    ha="center", va="center", fontsize=15, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted\nAnomalous", "Predicted\nNormal"], fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Actually\nAnomalous", "Actually\nNormal"], fontsize=9)
    ax.set_title(f"Majority Vote Confusion Matrix\n"
                 f"(\u2265{VOTE_THRESHOLD}/{len(VOTE_MODELS)} models: "
                 f"{', '.join(m.replace('_',' ') for m in VOTE_MODELS)})",
                 fontsize=8)
    fig.tight_layout()
    _save(fig, "comparison_vote")


# ── 6. Console summary table ──────────────────────────────────────────────────

def print_summary_table(all_metrics: dict, results_df: pd.DataFrame) -> None:
    """Print thesis-ready metrics table to console."""
    print("\n" + "=" * 74)
    print(f"  {'Model':<22} {'AUC':>6} {'DR':>7} {'Prec':>7} {'Recall':>8} "
          f"{'F1':>7} {'FPR':>7}")
    print("-" * 74)

    y_true = (results_df["label"] == "anomalous").astype(int).values

    for name, m in all_metrics.items():
        col = f"score_{name}"
        if col in results_df.columns and y_true.sum() > 0:
            scores = results_df[col].fillna(0).values
            fpr_c, tpr_c, _ = roc_curve(y_true, scores)
            roc_auc = f"{auc(fpr_c, tpr_c):.3f}"
        else:
            roc_auc = "  n/a"

        print(f"  {name:<22}"
              f"  {roc_auc:>6}"
              f"  {m.get('detection_rate', 0)*100:>5.1f}%"
              f"  {m.get('precision',      0)*100:>5.1f}%"
              f"  {m.get('recall',         0)*100:>6.1f}%"
              f"  {m.get('f1',             0)*100:>5.1f}%"
              f"  {m['false_positive_rate']*100:>5.1f}%")

    # Majority vote row
    if "vote_flag" in results_df.columns and y_true.sum() > 0:
        vp   = results_df["vote_flag"].values
        tp   = int(((vp == 1) & (y_true == 1)).sum())
        fp   = int(((vp == 1) & (y_true == 0)).sum())
        n_a  = int(y_true.sum())
        n_n  = int((y_true == 0).sum())
        dr   = tp / n_a if n_a > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1   = 2 * prec * dr / (prec + dr) if (prec + dr) > 0 else 0
        fpr  = fp / n_n if n_n > 0 else 0
        print("-" * 74)
        print(f"  {'majority vote (>=3/5)':<22}"
              f"  {'  n/a':>6}"
              f"  {dr*100:>5.1f}%"
              f"  {prec*100:>5.1f}%"
              f"  {dr*100:>6.1f}%"
              f"  {f1*100:>5.1f}%"
              f"  {fpr*100:>5.1f}%")

    print("=" * 74)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_comparison() -> None:
    if not RESULTS_CSV.exists() or not SUMMARY_JSON.exists():
        raise SystemExit(
            "\n[ERROR] Evaluation artefacts not found.\n"
            f"  Expected:\n    {RESULTS_CSV}\n    {SUMMARY_JSON}\n"
            "  Run evaluate.py first."
        )

    results_df = pd.read_csv(RESULTS_CSV)
    with open(SUMMARY_JSON) as f:
        all_metrics = json.load(f)

    results_df  = add_majority_vote(results_df)
    model_names = list(all_metrics.keys())

    print(f"Loaded {len(results_df)} samples, {len(model_names)} models.\n")
    print("── Generating figures ──────────────────────────────")

    plot_score_boxplot(results_df, model_names, all_metrics)
    plot_roc_curves(results_df, model_names, all_metrics)
    plot_score_distributions(results_df, model_names, all_metrics)
    plot_detection_heatmap(results_df, model_names)
    plot_vote_confusion(results_df)
    print_summary_table(all_metrics, results_df)

    print("\nDone.")


if __name__ == "__main__":
    run_comparison()