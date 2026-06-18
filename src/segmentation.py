"""
segmentation.py
---------------
Clustering engine: K-Means, DBSCAN, and Hierarchical Agglomerative Clustering.

Each model is wrapped in a consistent interface:
    fit_<model>(X, **kwargs) → labels: np.ndarray

The `run_all` function trains all three, logs metrics, and returns a results dict.

Usage
-----
    python src/segmentation.py
    # or import:
    from src.segmentation import run_all
    results = run_all(X_scaled)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

log = logging.getLogger(__name__)
FIGURES_DIR = Path("reports/figures")


# ── K-Means ────────────────────────────────────────────────────────────────────

def find_optimal_k(
    X: np.ndarray,
    k_range: range = range(2, 11),
    random_state: int = 42,
    save_plot: bool = True,
) -> int:
    """
    Elbow method + Silhouette analysis to choose optimal k.
    Saves combined diagnostic plot and returns recommended k.
    """
    inertias, sil_scores = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))
        log.info("  k=%d | inertia=%.1f | silhouette=%.3f", k, km.inertia_, sil_scores[-1])

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    log.info("✅ Optimal k (by silhouette): %d", best_k)

    if save_plot:
        _plot_elbow_silhouette(k_range, inertias, sil_scores, best_k)

    return best_k


def fit_kmeans(
    X: np.ndarray,
    k: int = 4,
    random_state: int = 42,
    n_init: int = 10,
) -> np.ndarray:
    """Fit K-Means and return cluster labels."""
    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    log.info("K-Means (k=%d) → silhouette=%.3f | davies_bouldin=%.3f", k, sil, db)
    return labels, km


# ── DBSCAN ─────────────────────────────────────────────────────────────────────

def tune_dbscan(
    X: np.ndarray,
    eps_values: list[float] | None = None,
    min_samples_values: list[int] | None = None,
) -> tuple[float, int]:
    """
    Grid search over eps and min_samples.
    Returns best (eps, min_samples) by silhouette score (ignoring noise).
    """
    eps_values = eps_values or [0.3, 0.5, 0.7, 1.0, 1.5]
    min_samples_values = min_samples_values or [3, 5, 8, 10]

    best_score, best_eps, best_min = -1, 0.5, 5

    log.info("DBSCAN grid search:")
    for eps in eps_values:
        for ms in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=ms)
            labels = db.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_pct = (labels == -1).mean() * 100
            if n_clusters < 2:
                continue
            try:
                sil = silhouette_score(X[labels != -1], labels[labels != -1])
            except Exception:
                continue
            log.info("  eps=%.1f min_samples=%d → clusters=%d noise=%.1f%% sil=%.3f",
                     eps, ms, n_clusters, noise_pct, sil)
            if sil > best_score:
                best_score, best_eps, best_min = sil, eps, ms

    log.info("✅ Best DBSCAN params: eps=%.2f, min_samples=%d (sil=%.3f)",
             best_eps, best_min, best_score)
    return best_eps, best_min


def fit_dbscan(X: np.ndarray, eps: float = 0.7, min_samples: int = 5) -> np.ndarray:
    """Fit DBSCAN and return cluster labels (-1 = noise)."""
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean() * 100
    log.info("DBSCAN → %d clusters | %.1f%% noise points", n_clusters, noise_pct)

    non_noise = labels != -1
    if non_noise.sum() > 1 and len(set(labels[non_noise])) > 1:
        sil = silhouette_score(X[non_noise], labels[non_noise])
        db_score = davies_bouldin_score(X[non_noise], labels[non_noise])
        log.info("  silhouette=%.3f | davies_bouldin=%.3f", sil, db_score)

    return labels, db


# ── Hierarchical ───────────────────────────────────────────────────────────────

def fit_hierarchical(
    X: np.ndarray,
    n_clusters: int = 4,
    linkage: str = "ward",
) -> np.ndarray:
    """Fit Agglomerative Hierarchical Clustering and return labels."""
    hac = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = hac.fit_predict(X)
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    log.info("Hierarchical (k=%d, linkage=%s) → silhouette=%.3f | davies_bouldin=%.3f",
             n_clusters, linkage, sil, db)
    return labels, hac


def plot_dendrogram(X: np.ndarray, max_display: int = 50, save: bool = True) -> plt.Figure:
    """Truncated dendrogram using scipy (Ward linkage)."""
    from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage

    linked = scipy_linkage(X[:200], method="ward")  # subsample for speed
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(linked, truncate_mode="level", p=5, ax=ax,
               color_threshold=0.7 * max(linked[:, 2]))
    ax.set_title("Hierarchical Clustering Dendrogram (Ward Linkage)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sample Index / Cluster Size", fontsize=10)
    ax.set_ylabel("Distance", fontsize=10)
    fig.tight_layout()

    if save:
        _save_fig(fig, "dendrogram.png")
    return fig


# ── Run All ────────────────────────────────────────────────────────────────────

def run_all(X: np.ndarray, auto_tune: bool = False) -> dict:
    """
    Train K-Means, DBSCAN, and Hierarchical; return results dict with labels and models.

    Parameters
    ----------
    X         : Scaled feature matrix
    auto_tune : If True, run elbow/grid-search tuning (slower)

    Returns
    -------
    dict with keys: 'kmeans', 'dbscan', 'hierarchical'
    Each value: {'labels': np.ndarray, 'model': fitted model}
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log.info("═══ Segmentation Pipeline ═══")

    results = {}

    # K-Means
    log.info("\n── K-Means ──")
    k = find_optimal_k(X) if auto_tune else 4
    km_labels, km_model = fit_kmeans(X, k=k)
    results["kmeans"] = {"labels": km_labels, "model": km_model, "k": k}

    # DBSCAN
    log.info("\n── DBSCAN ──")
    eps, ms = tune_dbscan(X) if auto_tune else (0.7, 5)
    db_labels, db_model = fit_dbscan(X, eps=eps, min_samples=ms)
    results["dbscan"] = {"labels": db_labels, "model": db_model, "eps": eps, "min_samples": ms}

    # Hierarchical
    log.info("\n── Hierarchical ──")
    hac_labels, hac_model = fit_hierarchical(X, n_clusters=4)
    results["hierarchical"] = {"labels": hac_labels, "model": hac_model}

    log.info("\n✅ All models trained. See evaluation.py for comparative metrics.")
    return results


# ── Helpers ────────────────────────────────────────────────────────────────────

def _plot_elbow_silhouette(
    k_range: range,
    inertias: list,
    sil_scores: list,
    best_k: int,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ks = list(k_range)
    ax1.plot(ks, inertias, "o-", color="#4C72B0", lw=2)
    ax1.axvline(best_k, color="red", linestyle="--", lw=1.2, alpha=0.7, label=f"Optimal k={best_k}")
    ax1.set_title("Elbow Method — Inertia vs k", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.legend()

    ax2.plot(ks, sil_scores, "o-", color="#DD8452", lw=2)
    ax2.axvline(best_k, color="red", linestyle="--", lw=1.2, alpha=0.7, label=f"Best k={best_k}")
    ax2.set_title("Silhouette Score vs k", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.legend()

    fig.tight_layout()
    _save_fig(fig, "kmeans_tuning.png")


def _save_fig(fig: plt.Figure, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info("Figure saved → %s", path)
    plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pickle
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.preprocessing import run_pipeline

    input_csv = Path("data/processed/customers.csv")
    if not input_csv.exists():
        log.error("Dataset not found. Run `python data/generate_synthetic.py` first.")
        sys.exit(1)

    X, df_clean, scaler = run_pipeline(input_csv)
    results = run_all(X, auto_tune=False)

    # Attach labels back to the DataFrame
    df_clean["segment_kmeans"] = results["kmeans"]["labels"]
    df_clean["segment_dbscan"] = results["dbscan"]["labels"]
    df_clean["segment_hierarchical"] = results["hierarchical"]["labels"]
    out = Path("data/processed/customers_segmented.csv")
    df_clean.to_csv(out, index=False)
    print(f"\n✅ Segmented data saved → {out}")
