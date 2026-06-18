"""
evaluation.py
-------------
Comparative evaluation of all three clustering models.

Metrics computed
----------------
- Silhouette Score          (higher = better separated clusters)
- Davies-Bouldin Index      (lower = better)
- Calinski-Harabasz Score   (higher = better)
- Noise percentage          (DBSCAN only)
- Cluster size distribution

Usage
-----
    python src/evaluation.py
    from src.evaluation import compare_models, print_summary_table
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

log = logging.getLogger(__name__)
FIGURES_DIR = Path("reports/figures")


# ── Metric Computation ─────────────────────────────────────────────────────────

def compute_metrics(X: np.ndarray, labels: np.ndarray, name: str) -> dict:
    """
    Compute all internal validity metrics for a given label assignment.
    Handles DBSCAN noise points (label=-1) by excluding them from metric calc.
    """
    mask = labels >= 0
    X_valid, labels_valid = X[mask], labels[mask]
    n_clusters = len(set(labels_valid))
    noise_pct = (~mask).mean() * 100

    if n_clusters < 2:
        log.warning("%s: fewer than 2 clusters found — metrics skipped.", name)
        return {"model": name, "n_clusters": n_clusters, "noise_pct": noise_pct,
                "silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}

    sil  = silhouette_score(X_valid, labels_valid)
    db   = davies_bouldin_score(X_valid, labels_valid)
    ch   = calinski_harabasz_score(X_valid, labels_valid)

    log.info(
        "%-18s | clusters=%d | noise=%.1f%% | sil=%.3f | db=%.3f | ch=%.1f",
        name, n_clusters, noise_pct, sil, db, ch,
    )
    return {
        "model": name,
        "n_clusters": n_clusters,
        "noise_pct": round(noise_pct, 2),
        "silhouette": round(sil, 4),
        "davies_bouldin": round(db, 4),
        "calinski_harabasz": round(ch, 2),
    }


def compare_models(X: np.ndarray, results: dict) -> pd.DataFrame:
    """
    Run metrics for all models and return a comparison DataFrame.

    Parameters
    ----------
    X       : Scaled feature matrix
    results : Output of segmentation.run_all()

    Returns
    -------
    DataFrame with one row per model
    """
    rows = []
    model_map = {
        "K-Means": results["kmeans"]["labels"],
        "DBSCAN": results["dbscan"]["labels"],
        "Hierarchical (Ward)": results["hierarchical"]["labels"],
    }
    for name, labels in model_map.items():
        rows.append(compute_metrics(X, labels, name))
    df = pd.DataFrame(rows).set_index("model")
    return df


def print_summary_table(metrics_df: pd.DataFrame) -> None:
    """Pretty-print the comparison table to stdout."""
    print("\n" + "═" * 72)
    print("  MODEL COMPARISON — INTERNAL CLUSTERING VALIDITY METRICS")
    print("═" * 72)
    print(metrics_df.to_string())
    print("═" * 72)
    print("  ↑ silhouette & calinski_harabasz: higher is better")
    print("  ↓ davies_bouldin: lower is better\n")


# ── Cluster Profile ────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "recency_days", "frequency", "monetary_value", "session_count",
    "avg_session_min", "categories_browsed", "discount_usage_pct",
    "mobile_pct", "email_open_rate", "cart_abandon_rate",
]

PERSONA_NAMES = {
    0: "High-Value Loyalists",
    1: "Deal Seekers",
    2: "Dormant Potentials",
    3: "New Explorers",
}


def cluster_profile(df: pd.DataFrame, label_col: str = "segment_kmeans") -> pd.DataFrame:
    """Compute per-cluster mean for each feature. Returns a styled summary."""
    profile = df.groupby(label_col)[FEATURE_COLS].mean().round(2)
    profile.index = [PERSONA_NAMES.get(i, f"Cluster {i}") for i in profile.index]
    profile["count"] = df.groupby(label_col).size().values
    profile["% of base"] = (profile["count"] / len(df) * 100).round(1)
    return profile


# ── Visualizations ─────────────────────────────────────────────────────────────

def plot_metrics_comparison(metrics_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Bar chart comparing silhouette and Davies-Bouldin across models."""
    models = metrics_df.index.tolist()
    sil = metrics_df["silhouette"].tolist()
    db  = metrics_df["davies_bouldin"].tolist()

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, sil, width, label="Silhouette ↑", color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + width / 2, db,  width, label="Davies-Bouldin ↓", color="#DD8452", alpha=0.85)

    ax.bar_label(bars1, fmt="%.3f", fontsize=9, padding=3)
    ax.bar_label(bars2, fmt="%.3f", fontsize=9, padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison — Clustering Validity Metrics", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(max(sil), max(db)) * 1.25)
    fig.tight_layout()

    if save:
        _save_fig(fig, "model_comparison_metrics.png")
    return fig


def plot_cluster_heatmap(
    df: pd.DataFrame,
    label_col: str = "segment_kmeans",
    save: bool = True,
) -> plt.Figure:
    """Heatmap of normalized feature means per cluster — key for stakeholder slides."""
    import seaborn as sns

    profile = df.groupby(label_col)[FEATURE_COLS].mean()
    profile_norm = (profile - profile.min()) / (profile.max() - profile.min())
    profile_norm.index = [PERSONA_NAMES.get(i, f"Cluster {i}") for i in profile_norm.index]

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(
        profile_norm,
        annot=profile.values.round(1),
        fmt="g",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Normalized Mean (0=min, 1=max)"},
    )
    ax.set_title("Customer Segment Profiles — Feature Heatmap", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=30, ha="right", fontsize=9)
    fig.tight_layout()

    if save:
        _save_fig(fig, "cluster_heatmap.png")
    return fig


def plot_cluster_distribution(
    df: pd.DataFrame,
    label_col: str = "segment_kmeans",
    save: bool = True,
) -> plt.Figure:
    """Pie + bar chart showing segment size distribution."""
    counts = df[label_col].value_counts().sort_index()
    names = [PERSONA_NAMES.get(i, f"Cluster {i}") for i in counts.index]
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.pie(counts, labels=names, autopct="%1.1f%%", colors=palette,
            startangle=140, textprops={"fontsize": 9})
    ax1.set_title("Segment Size Distribution", fontsize=12, fontweight="bold")

    bars = ax2.barh(names, counts, color=palette, alpha=0.85)
    ax2.bar_label(bars, labels=[f"{v:,}" for v in counts], padding=5, fontsize=9)
    ax2.set_xlabel("Number of Customers", fontsize=10)
    ax2.set_title("Customers per Segment", fontsize=12, fontweight="bold")
    ax2.invert_yaxis()

    fig.tight_layout()
    if save:
        _save_fig(fig, "cluster_distribution.png")
    return fig


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info("Figure saved → %s", path)
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    from src.preprocessing import run_pipeline
    from src.segmentation import run_all

    X, df_clean, scaler = run_pipeline("data/processed/customers.csv")
    results = run_all(X)

    metrics = compare_models(X, results)
    print_summary_table(metrics)
    plot_metrics_comparison(metrics)

    # Attach best model labels and plot profiles
    df_clean["segment_kmeans"] = results["kmeans"]["labels"]
    df_clean["segment_dbscan"] = results["dbscan"]["labels"]
    df_clean["segment_hierarchical"] = results["hierarchical"]["labels"]

    print("\n── K-Means Cluster Profiles ──")
    print(cluster_profile(df_clean).to_string())

    plot_cluster_heatmap(df_clean)
    plot_cluster_distribution(df_clean)

    print("\n✅ Evaluation complete. Figures saved to reports/figures/")
