"""
pca_analysis.py
---------------
Principal Component Analysis for dimensionality reduction and visualization.

Responsibilities
----------------
- Determine optimal number of components via explained variance
- Reduce feature space for clustering efficiency
- Generate 2D / 3D projections for stakeholder-facing visualizations
- Produce biplot of feature loadings

Usage
-----
    from src.pca_analysis import run_pca, plot_explained_variance, plot_pca_2d
    X_pca, pca = run_pca(X_scaled, n_components=2)
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

log = logging.getLogger(__name__)

FIGURES_DIR = Path("reports/figures")
FEATURE_COLS = [
    "recency_days", "frequency", "monetary_value", "session_count",
    "avg_session_min", "categories_browsed", "discount_usage_pct",
    "mobile_pct", "email_open_rate", "cart_abandon_rate",
]


# ── Core PCA ───────────────────────────────────────────────────────────────────

def run_pca(
    X: np.ndarray,
    n_components: int | float = 0.90,
    random_state: int = 42,
) -> tuple[np.ndarray, PCA]:
    """
    Fit PCA and return transformed data plus the fitted model.

    Parameters
    ----------
    X            : Scaled feature matrix
    n_components : int → exact components; float (0–1) → variance threshold
    random_state : For reproducibility

    Returns
    -------
    X_pca : Transformed array
    pca   : Fitted PCA object (access .explained_variance_ratio_, .components_)
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X)

    if isinstance(n_components, float):
        log.info(
            "PCA: retained %d components explaining %.1f%% variance",
            pca.n_components_,
            100 * pca.explained_variance_ratio_.sum(),
        )
    else:
        log.info(
            "PCA: %d components explain %.1f%% variance",
            n_components,
            100 * pca.explained_variance_ratio_.sum(),
        )

    return X_pca, pca


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_explained_variance(pca_full: PCA, save: bool = True) -> plt.Figure:
    """
    Scree plot: individual and cumulative explained variance.
    Annotate the 'elbow' and the 90% threshold.
    """
    evr = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(evr)
    n = len(evr)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(range(1, n + 1), evr * 100, alpha=0.6, color="#4C72B0", label="Individual")
    ax.plot(range(1, n + 1), cumulative * 100, "o-", color="#DD8452", lw=2, label="Cumulative")
    ax.axhline(90, color="gray", linestyle="--", lw=1, alpha=0.7, label="90% threshold")

    # Mark where cumulative crosses 90%
    cross90 = np.argmax(cumulative >= 0.90) + 1
    ax.axvline(cross90, color="gray", linestyle=":", lw=1)
    ax.annotate(
        f"PC{cross90}\n({cumulative[cross90-1]*100:.1f}%)",
        xy=(cross90, 90),
        xytext=(cross90 + 0.4, 82),
        fontsize=9,
        color="gray",
    )

    ax.set_xlabel("Principal Component", fontsize=11)
    ax.set_ylabel("Explained Variance (%)", fontsize=11)
    ax.set_title("PCA — Explained Variance by Component", fontsize=13, fontweight="bold")
    ax.set_xticks(range(1, n + 1))
    ax.legend()
    fig.tight_layout()

    if save:
        _save(fig, "pca_explained_variance.png")
    return fig


def plot_pca_2d(
    X_pca: np.ndarray,
    labels: np.ndarray | None = None,
    label_names: dict | None = None,
    save: bool = True,
) -> plt.Figure:
    """
    2D scatter of PC1 vs PC2, colored by cluster label.
    Used in stakeholder presentations to show segment separation.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

    if labels is not None:
        unique = np.unique(labels[labels >= 0])
        for i, uid in enumerate(unique):
            mask = labels == uid
            name = (label_names or {}).get(uid, f"Cluster {uid}")
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=30, alpha=0.65,
                       color=palette[i % len(palette)], label=name, edgecolors="none")
        # Noise points (DBSCAN label = -1)
        if -1 in labels:
            noise = labels == -1
            ax.scatter(X_pca[noise, 0], X_pca[noise, 1], s=15, alpha=0.3,
                       color="lightgray", label="Noise (DBSCAN)", edgecolors="none")
        ax.legend(title="Segment", fontsize=9)
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], s=25, alpha=0.5, color="#4C72B0", edgecolors="none")

    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.set_title("Customer Segments — PCA 2D Projection", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save:
        _save(fig, "pca_2d_clusters.png")
    return fig


def plot_biplot(
    X_pca: np.ndarray,
    pca: PCA,
    labels: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    save: bool = True,
) -> plt.Figure:
    """
    PCA biplot: scatter of observations + feature loading arrows.
    Communicates which features drive each principal component.
    """
    feature_names = feature_names or FEATURE_COLS
    loadings = pca.components_.T  # shape (n_features, n_components)

    fig, ax = plt.subplots(figsize=(10, 8))
    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    # Observations
    if labels is not None:
        for i, uid in enumerate(np.unique(labels[labels >= 0])):
            mask = labels == uid
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=20, alpha=0.35,
                       color=palette[i % len(palette)], edgecolors="none")
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], s=15, alpha=0.3, color="#4C72B0", edgecolors="none")

    # Loading arrows (scaled)
    scale = np.max(np.abs(X_pca[:, :2])) / np.max(np.abs(loadings[:, :2])) * 0.7
    for j, feat in enumerate(feature_names):
        dx, dy = loadings[j, 0] * scale, loadings[j, 1] * scale
        ax.annotate(
            "", xy=(dx, dy), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="darkred", lw=1.5),
        )
        ax.text(dx * 1.12, dy * 1.12, feat, fontsize=8, color="darkred", ha="center")

    ax.axhline(0, color="gray", lw=0.5, alpha=0.5)
    ax.axvline(0, color="gray", lw=0.5, alpha=0.5)
    ax.set_xlabel("PC 1", fontsize=11)
    ax.set_ylabel("PC 2", fontsize=11)
    ax.set_title("PCA Biplot — Feature Loadings & Observations", fontsize=13, fontweight="bold")
    fig.tight_layout()

    if save:
        _save(fig, "pca_biplot.png")
    return fig


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, filename: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info("Figure saved → %s", path)
