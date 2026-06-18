# Workflow Specification

**Project:** AI-Enhanced Customer Segmentation for Marketing  
**Type:** Capstone Project  
**Duration:** Aug 2024 – Dec 2024  
**Team Size:** 3

---

## 1. Project Scope

### Problem Statement
Marketing campaigns using demographic-only targeting (age, gender, location) have low conversion rates because they ignore how customers actually behave. This project builds an ML-powered segmentation framework that clusters customers by behavioral patterns — frequency of purchase, session depth, channel preference, price sensitivity — and maps each cluster to an actionable marketing persona.

### In Scope
- Behavioral feature engineering from raw e-commerce event data
- Unsupervised clustering (K-Means, DBSCAN, Hierarchical)
- PCA-based visualization for stakeholder presentation
- Cluster persona documentation and marketing recommendations
- End-to-end reproducible Python pipeline with tests

### Out of Scope
- Real-time scoring / production deployment
- A/B testing of marketing campaigns against segments
- Integration with CRM or marketing automation platforms
- Supervised learning (no ground-truth labels exist)

---

## 2. Team Structure & Task Allocation

| Member | Role | Primary Responsibilities |
|--------|------|--------------------------|
| **[Your Name]** | Product Lead | Scope definition, milestone planning, dev task allocation, architecture docs, final presentation |
| Member 2 | ML Engineer | Model implementation, hyperparameter tuning, comparative analysis, evaluation metrics |
| Member 3 | Data Engineer | Preprocessing pipeline, validation suite, data quality, workflow documentation |

---

## 3. Milestones

### Sprint 1 — Foundation (Aug – Sep 2024)
**Goal:** Environment setup, data strategy, architecture design

| Task | Owner | Deliverable |
|------|-------|-------------|
| Define project scope and success criteria | Product Lead | `docs/workflow_specification.md` |
| Design system architecture | Product Lead | `docs/system_architecture.md` |
| Set up GitHub repo, branch strategy, and CI | All | Repository structure |
| Identify data sources; design synthetic generator | Data Engineer | `data/generate_synthetic.py` |
| Literature review: K-Means vs DBSCAN vs HAC | ML Engineer | Notes in `notebooks/01_EDA` |

**Definition of Done:** Repo live, synthetic data generating, architecture document reviewed by team.

---

### Sprint 2 — Data Pipeline (Sep – Oct 2024)
**Goal:** Production-quality preprocessing pipeline with full validation

| Task | Owner | Deliverable |
|------|-------|-------------|
| Implement schema validation | Data Engineer | `preprocessing.validate_schema()` |
| Implement value-range checks | Data Engineer | `preprocessing.validate_value_ranges()` |
| Implement median imputation | Data Engineer | `preprocessing.handle_missing()` |
| Implement IQR outlier removal | Data Engineer | `preprocessing.remove_outliers_iqr()` |
| Implement StandardScaler integration | Data Engineer | `preprocessing.scale_features()` |
| Write unit tests (20+ test cases) | Data Engineer + ML Engineer | `tests/test_preprocessing.py` |
| Exploratory data analysis notebook | All | `notebooks/01_EDA_and_Preprocessing.ipynb` |

**Definition of Done:** `pytest tests/ -v` passes 100%; EDA notebook reviewed and signed off.

---

### Sprint 3 — Modeling (Oct – Nov 2024)
**Goal:** All three clustering models implemented, tuned, and evaluated

| Task | Owner | Deliverable |
|------|-------|-------------|
| K-Means: elbow + silhouette tuning | ML Engineer | `segmentation.find_optimal_k()` |
| DBSCAN: eps × min_samples grid search | ML Engineer | `segmentation.tune_dbscan()` |
| Hierarchical: Ward linkage + dendrogram | ML Engineer | `segmentation.fit_hierarchical()` |
| PCA module: variance analysis + biplot | ML Engineer | `pca_analysis.py` |
| Evaluation metrics framework | ML Engineer | `evaluation.compare_models()` |
| Comparative analysis notebook | ML Engineer | `notebooks/03_model_comparison.ipynb` |
| Model selection decision (documented) | Product Lead | `docs/system_architecture.md §4` |

**Definition of Done:** All three models produce valid cluster assignments; comparison table generated; model selection rationale documented.

---

### Sprint 4 — Insights & Delivery (Nov – Dec 2024)
**Goal:** Stakeholder-ready output: personas, visuals, final presentation

| Task | Owner | Deliverable |
|------|-------|-------------|
| Persona naming and characterization | Product Lead + ML Engineer | `docs/persona_report.md` |
| Cluster feature heatmap | ML Engineer | `reports/figures/cluster_heatmap.png` |
| PCA 2D scatter (colored by segment) | ML Engineer | `reports/figures/pca_2d_clusters.png` |
| Segment distribution charts | Data Engineer | `reports/figures/cluster_distribution.png` |
| Final presentation deck | Product Lead | Slides (7 slides, non-technical audience) |
| README completion | Product Lead | `README.md` |
| Code cleanup, docstrings, linting | All | Clean codebase |
| Academic submission | All | Final deliverable |

**Definition of Done:** Pipeline runs end-to-end (`generate_synthetic → preprocessing → segmentation → evaluation`); all figures generated; presentation delivered to evaluators.

---

## 4. Git Workflow

```
main          ← stable, reviewed code only
  └── dev     ← integration branch
        ├── feature/preprocessing-pipeline
        ├── feature/kmeans-model
        ├── feature/dbscan-model
        ├── feature/hierarchical-model
        ├── feature/pca-visualization
        └── feature/evaluation-metrics
```

**Merge policy:** PR required to merge into `dev`; at least one reviewer approval. Merge to `main` only at sprint boundaries.

---

## 5. Success Criteria

| Criterion | Target | Met? |
|-----------|--------|------|
| Pipeline runs end-to-end without errors | 100% | ✅ |
| Tests pass | ≥20 test cases, 100% pass rate | ✅ |
| K-Means silhouette score | ≥0.45 | ✅ (0.58) |
| Cluster personas are interpretable | Validated by non-technical evaluators | ✅ |
| Documentation covers all components | Architecture + workflow + persona docs | ✅ |
| Presentation delivered to evaluators | 15-min final presentation | ✅ |

---

## 6. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Real dataset unavailable | High | High | Designed synthetic generator from the start |
| DBSCAN sensitive to scale | Medium | Medium | StandardScaler applied before all models |
| Poor cluster separation | Low | High | Multiple models + metric comparison to fall back on best |
| Team capacity constraints | Medium | Medium | Modular code allows parallel development |
