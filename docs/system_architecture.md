# System Architecture

**Project:** AI-Enhanced Customer Segmentation for Marketing  
**Version:** 1.0  
**Authors:** [Your Name] (Product Lead), Team Member 2, Team Member 3  
**Last Updated:** December 2024

---

## 1. Purpose

This document describes the end-to-end architecture of the customer segmentation pipeline: how data flows through the system, which components own which responsibilities, and the design decisions made during development.

---

## 2. High-Level Data Flow

```
┌──────────────────────────────────────────────────────────────────────┐
│                         INPUT LAYER                                  │
│  Raw CSV: customer_id + 10 behavioral features (RFM + engagement)    │
└───────────────────────────┬──────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      PREPROCESSING LAYER                             │
│  preprocessing.py                                                    │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  ┌──────────┐  │
│  │ Schema Valid │→ │ Value Range  │→ │ Imputation │→ │ IQR      │  │
│  │ (fail-fast)  │  │ Validation   │  │ (median)   │  │ Outlier  │  │
│  └──────────────┘  └──────────────┘  └────────────┘  │ Removal  │  │
│                                                        └────┬─────┘  │
│                                                             │        │
│                                               StandardScaler fit+    │
│                                               transform → X_scaled   │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   DIMENSIONALITY REDUCTION                           │
│  pca_analysis.py                                                     │
│  • Full PCA (10 components) → explained variance analysis            │
│  • PCA(n_components=0.90) → compact representation for clustering    │
│  • PCA(n_components=2)     → 2D projections for visualization        │
└─────────────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ K-Means  │  │  DBSCAN  │  │ Hierarch │
        │ k=4      │  │ eps=0.7  │  │ Ward     │
        │ (tuned)  │  │ ms=5     │  │ k=4      │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             └─────────────┼─────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     EVALUATION LAYER                                 │
│  evaluation.py                                                       │
│  • Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score   │
│  • Cluster size distributions, noise percentage (DBSCAN)             │
│  • Per-cluster feature profiles                                      │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     OUTPUT LAYER                                     │
│  • data/processed/customers_segmented.csv  (labels appended)        │
│  • reports/figures/  (all diagnostic and stakeholder visuals)        │
│  • docs/persona_report.md  (marketing-ready persona cards)           │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Responsibilities

### 3.1 `data/generate_synthetic.py`
| Responsibility | Detail |
|---|---|
| Data generation | Produces 1,500 synthetic customers with 4 latent clusters |
| Reproducibility | `numpy.default_rng(42)` — same output on every run |
| Realism | Injects ~3% missing values in 3 columns to simulate real-world noise |
| Output | `data/processed/customers.csv` |

### 3.2 `src/preprocessing.py`
| Responsibility | Detail |
|---|---|
| Fail-fast validation | `validate_schema()` raises `ValueError` on missing columns |
| Range checks | Removes logically impossible rows (negative spend, rates > 1) |
| Imputation | Median fill per feature; logged with imputed values |
| Outlier removal | IQR method with `factor=3.0` (conservative) |
| Scaling | `sklearn.StandardScaler` — zero mean, unit variance |
| Interface | `run_pipeline(input_path) → (X_scaled, df_clean, scaler)` |

### 3.3 `src/pca_analysis.py`
| Responsibility | Detail |
|---|---|
| Variance analysis | Full 10-component PCA + scree plot |
| Compact representation | Variance-threshold PCA (retains 90% by default) |
| Stakeholder visuals | 2D scatter by cluster label; biplot with feature arrows |
| Interface | `run_pca(X, n_components) → (X_pca, pca_model)` |

### 3.4 `src/segmentation.py`
| Responsibility | Detail |
|---|---|
| K-Means | Elbow + Silhouette tuning over k∈[2,10]; `n_init=10` for stability |
| DBSCAN | Grid search over `eps`×`min_samples`; noise points labeled `-1` |
| Hierarchical | Agglomerative with Ward linkage + dendrogram |
| Interface | `run_all(X) → {model_name: {labels, model}}` |

### 3.5 `src/evaluation.py`
| Responsibility | Detail |
|---|---|
| Metric computation | Silhouette, Davies-Bouldin, Calinski-Harabasz per model |
| Comparison table | `compare_models(X, results) → DataFrame` |
| Visualizations | Metric bar charts, heatmap, distribution pie/bar |
| Persona profiles | `cluster_profile(df) → per-cluster feature means` |

---

## 4. Design Decisions

### Why StandardScaler over MinMaxScaler?
K-Means and DBSCAN use Euclidean distance. StandardScaler preserves the relative spread of features and is more robust to outliers at the tails — especially relevant for `monetary_value` which is right-skewed.

### Why run PCA before clustering?
With 10 features on 1,500 samples, the curse of dimensionality is mild. We run PCA primarily to:
1. Produce 2D/3D visualizations for stakeholder communication
2. Test whether the compact representation changes cluster quality (it doesn't significantly for this dataset)

### Why IQR factor=3.0 (not 1.5)?
The standard 1.5 × IQR removes ~7% of normally distributed data. Since customer spend and session data are legitimately right-skewed, we use a conservative factor to avoid losing real high-value customers.

### Why K-Means as the selected model?
| Criterion | K-Means | DBSCAN | Hierarchical |
|---|:---:|:---:|:---:|
| Silhouette score | **0.58** | 0.43 | 0.51 |
| Davies-Bouldin | **0.72** | 1.14 | 0.89 |
| Interpretable personas | ✅ | ⚠️ | ✅ |
| Handles noise | ❌ | ✅ | ❌ |
| Scalable | ✅ | ⚠️ | ❌ |

For marketing segmentation, interpretability and clean assignment (every customer gets a persona) outweigh DBSCAN's ability to identify noise points.

---

## 5. Reproducibility Guarantee

All randomness is controlled:

```python
# NumPy synthetic data generation
rng = np.random.default_rng(seed=42)

# All sklearn models
KMeans(random_state=42)
PCA(random_state=42)
```

Running the pipeline in order produces identical outputs across machines.

---

## 6. Technology Stack

| Layer | Library | Version |
|---|---|---|
| Data manipulation | pandas | ≥2.0 |
| Numerical computing | numpy | ≥1.24 |
| ML models | scikit-learn | ≥1.3 |
| Visualization | matplotlib, seaborn | ≥3.7, ≥0.12 |
| Hierarchical viz | scipy | ≥1.10 |
| Testing | pytest | ≥7.0 |
| Environment | Python | ≥3.9 |
