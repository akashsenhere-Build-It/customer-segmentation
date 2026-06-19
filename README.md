# AI-Enhanced Customer Segmentation for Marketing

> **Capstone Project · Aug 2024 – Dec 2024**  
> Unsupervised machine learning pipeline that translates raw e-commerce behavior data into actionable marketing personas — from raw data to stakeholder-ready insights.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Team & Roles](#team--roles)
3. [Architecture](#architecture)
4. [Models Implemented](#models-implemented)
5. [Key Results](#key-results)
6. [Repository Structure](#repository-structure)
7. [Getting Started](#getting-started)
8. [Documentation](#documentation)
9. [Reproducibility](#reproducibility)

---

## Project Overview

Marketing teams struggle to personalize campaigns at scale because they rely on broad demographic buckets rather than actual behavior. This project delivers an end-to-end AI pipeline that:

- **Ingests** raw clickstream / purchase behavior data
- **Preprocesses** and validates it for ML readiness
- **Clusters** customers into distinct personas using K-Means, DBSCAN, and Hierarchical Clustering
- **Reduces dimensionality** with PCA for both modeling efficiency and human-interpretable visualization
- **Exports** per-cluster personas and campaign recommendations

The framework was built with reproducibility and non-technical stakeholder communication as first-class requirements.

---

## Team & Roles

| Member | Role |
|--------|------|
| **Akash Sen** | Product Lead — scope definition, milestone planning, dev task allocation, architecture docs, final presentation |
| Team Member 2 |  — model implementation, hyperparameter tuning, comparative analysis |
| Team Member 3 |  — preprocessing pipeline, validation suite, dataset management |

---

## Architecture

```
Raw Data (CSV / synthetic)
        │
        ▼
┌─────────────────────┐
│  Data Preprocessing  │  ← preprocessing.py
│  - Missing values    │
│  - Encoding          │
│  - Scaling           │
│  - Validation checks │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Dimensionality     │  ← pca_analysis.py
│   Reduction (PCA)    │
└────────┬────────────┘
         │
         ▼
┌──────────────────────────────────────┐
│         Clustering Engine             │
│  ┌──────────┐ ┌────────┐ ┌────────┐  │  ← segmentation.py
│  │  K-Means │ │ DBSCAN │ │  HAC   │  │
│  └──────────┘ └────────┘ └────────┘  │
└────────────────┬─────────────────────┘
                 │
                 ▼
┌─────────────────────┐
│  Evaluation &        │  ← evaluation.py
│  Comparative         │
│  Analysis            │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Persona Reports &   │  ← visualizations.py
│  Visualizations      │
└─────────────────────┘
```

Full system architecture doc: [`docs/system_architecture.md`](docs/system_architecture.md)

---

## Models Implemented

| Model | Purpose | Key Hyperparameters |
|-------|---------|-------------------|
| **K-Means** | Baseline hard clustering, fast persona discovery | `k` tuned via Elbow + Silhouette |
| **DBSCAN** | Density-based; captures irregular shapes & noise | `eps`, `min_samples` via grid search |
| **Hierarchical (Agglomerative)** | Dendrogram-driven; no `k` required upfront | `linkage`: ward / complete / average |

Comparative analysis of all three methods is in [`notebooks/03_model_comparison.ipynb`](notebooks/03_model_comparison.ipynb).

---

## Key Results

| Metric | K-Means (k=4) | DBSCAN | Hierarchical |
|--------|:---:|:---:|:---:|
| Silhouette Score | **0.58** | 0.43 | 0.51 |
| Davies-Bouldin Index | **0.72** | 1.14 | 0.89 |
| Noise Points | 0 | ~8% | 0 |
| Interpretability | High | Medium | High |

**Selected model:** K-Means (k=4) — best balance of separation quality and persona interpretability for marketing use.

### Discovered Personas
| Cluster | Label | Characteristics |
|---------|-------|----------------|
| 0 | **High-Value Loyalists** | High spend, high frequency, broad category range |
| 1 | **Deal Seekers** | High sessions, low conversion, discount-driven |
| 2 | **Dormant Potentials** | Low recency, previously high spend |
| 3 | **New Explorers** | Recent sign-ups, narrow categories, mobile-heavy |

---

## Repository Structure

```
customer-segmentation/
├── README.md                   ← You are here
├── requirements.txt
├── data/
│   ├── raw/                    ← Source datasets (gitignored if real)
│   ├── processed/              ← Cleaned, encoded, scaled outputs
│   └── generate_synthetic.py   ← Reproducible synthetic data generator
├── src/
│   ├── preprocessing.py        ← Full preprocessing & validation pipeline
│   ├── pca_analysis.py         ← PCA dimensionality reduction module
│   ├── segmentation.py         ← K-Means, DBSCAN, Hierarchical clustering
│   ├── evaluation.py           ← Silhouette, Davies-Bouldin, cluster stats
│   └── visualizations.py       ← All plots, persona cards, PCA biplots
├── notebooks/
│   ├── 01_EDA_and_Preprocessing.ipynb
│   ├── 02_Clustering_Models.ipynb
│   └── 03_model_comparison.ipynb
├── docs/
│   ├── system_architecture.md
│   ├── workflow_specification.md
│   └── persona_report.md
├── reports/
│   └── figures/                ← All generated charts (PNG/SVG)
├── tests/
│   └── test_preprocessing.py
└── .gitignore
```

---

## Getting Started

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

```bash
git clone https://github.com/akashsenhere-build-it/customer-segmentation.git
cd customer-segmentation
pip install -r requirements.txt
```

### Generate Synthetic Data

```bash
python data/generate_synthetic.py
```

### Run the Full Pipeline

```bash
# Option 1: Python script
python src/segmentation.py

# Option 2: Step through notebooks
jupyter lab notebooks/
```

### Run Tests

```bash
pytest tests/ -v
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/system_architecture.md`](docs/system_architecture.md) | End-to-end data flow, component responsibilities, design decisions |
| [`docs/workflow_specification.md`](docs/workflow_specification.md) | Sprint milestones, task allocation, definition of done |
| [`docs/persona_report.md`](docs/persona_report.md) | Final cluster personas with marketing recommendations |

---

## Reproducibility

All randomness is seeded (`random_state=42`). The synthetic data generator produces the same dataset on every run. To fully reproduce results:

```bash
python data/generate_synthetic.py
python src/preprocessing.py
python src/segmentation.py
python src/evaluation.py
```

Figures are saved to `reports/figures/`.

---

## Technologies

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![pandas](https://img.shields.io/badge/pandas-data-green)
![matplotlib](https://img.shields.io/badge/matplotlib-viz-red)
![seaborn](https://img.shields.io/badge/seaborn-viz-lightblue)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-orange)

---

*Capstone Project — [Your University] · [Your Program] · 2024*
