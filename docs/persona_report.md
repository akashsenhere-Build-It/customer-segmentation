# Customer Persona Report

**Project:** AI-Enhanced Customer Segmentation for Marketing  
**Model:** K-Means (k=4) — Selected via Silhouette Score (0.58) and Davies-Bouldin Index (0.72)  
**Dataset:** 1,500 synthetic customers, 10 behavioral features  
**Prepared for:** Non-Technical Stakeholders / Academic Evaluators

---

## Executive Summary

Using unsupervised machine learning (K-Means clustering), we identified **4 distinct customer segments** from behavioral data. Each segment has a consistent behavioral fingerprint that maps to a specific marketing strategy. K-Means outperformed DBSCAN and Hierarchical Clustering on both silhouette score and Davies-Bouldin index, and produces clean, non-overlapping cluster assignments suitable for campaign targeting.

---

## Segment Overview

| # | Persona | Size | % of Base | Primary Signal |
|---|---------|:----:|:---------:|---------------|
| 0 | High-Value Loyalists | ~375 | 25% | High spend + high frequency |
| 1 | Deal Seekers | ~450 | 30% | High sessions, discount-driven |
| 2 | Dormant Potentials | ~300 | 20% | Previously high spend, now inactive |
| 3 | New Explorers | ~375 | 25% | Recent, mobile-heavy, narrow categories |

---

## Persona 0 — High-Value Loyalists

### Behavioral Profile
| Feature | Cluster 0 | Avg Customer |
|---------|-----------|-------------|
| Recency (days) | 8 | 55 |
| Purchase Frequency | 17 | 8 |
| Monetary Value ($) | $850 | $420 |
| Categories Browsed | 9 | 5 |
| Discount Usage | 12% | 36% |
| Email Open Rate | 62% | 32% |

### Who They Are
These customers buy often, spend the most, browse broadly, and rarely need a discount to convert. They respond strongly to email and have low cart abandonment. They are the highest-LTV (lifetime value) segment.

### Marketing Recommendations
- **Priority:** Retention and upsell
- **Tactics:** VIP loyalty programs, early access to new products, personalized product recommendations, anniversary/milestone rewards
- **Channel:** Email (62% open rate), push notifications
- **Avoid:** Blanket discount campaigns — unnecessary margin erosion on customers who buy at full price

---

## Persona 1 — Deal Seekers

### Behavioral Profile
| Feature | Cluster 1 | Avg Customer |
|---------|-----------|-------------|
| Recency (days) | 22 | 55 |
| Purchase Frequency | 7 | 8 |
| Monetary Value ($) | $220 | $420 |
| Session Count | 45 | 22 |
| Discount Usage | 75% | 36% |
| Cart Abandonment | 50% | 38% |

### Who They Are
These customers browse frequently and have decent recency, but they almost always wait for a deal. High cart abandonment suggests they leave when prices feel too high. They have the most sessions but lowest monetary value — a sign of high intent that needs a price trigger.

### Marketing Recommendations
- **Priority:** Conversion rate optimization
- **Tactics:** Targeted flash sales, cart abandonment emails with a discount code, bundle deals ("buy 2, save 15%"), limited-time price alerts
- **Channel:** Push notifications, retargeting ads, cart abandonment email sequences
- **Caution:** Don't train them to wait indefinitely — structure discounts with genuine urgency (48hr expiry)

---

## Persona 2 — Dormant Potentials

### Behavioral Profile
| Feature | Cluster 2 | Avg Customer |
|---------|-----------|-------------|
| Recency (days) | 130 | 55 |
| Purchase Frequency | 3 | 8 |
| Monetary Value ($) | $500 | $420 |
| Session Count | 7 | 22 |
| Email Open Rate | 14% | 32% |
| Cart Abandonment | 58% | 38% |

### Who They Are
These customers were once high-value (above-average spend) but have become inactive. Low recency and very low email engagement suggest they've drifted. They represent a **significant win-back opportunity** — if spend from their active phase can be restimulated.

### Marketing Recommendations
- **Priority:** Win-back / re-engagement
- **Tactics:** "We miss you" email campaigns, personalized re-engagement offers based on previous purchase categories, milestone reactivation bonuses ("Your 6-month update: here's 20% back")
- **Channel:** Email (even with low open rate — test subject line optimization), SMS if available
- **Metric to watch:** Reactivation rate (any session in last 30 days post-campaign)

---

## Persona 3 — New Explorers

### Behavioral Profile
| Feature | Cluster 3 | Avg Customer |
|---------|-----------|-------------|
| Recency (days) | 14 | 55 |
| Purchase Frequency | 2 | 8 |
| Monetary Value ($) | $120 | $420 |
| Mobile Usage | 80% | 52% |
| Categories Browsed | 2 | 5 |
| Cart Abandonment | 65% | 38% |

### Who They Are
These are recently acquired customers exploring a narrow product range — almost exclusively on mobile. Low spend is expected at this stage. The opportunity is to deepen engagement before they churn. High cart abandonment is a friction signal that may be mobile UX-related.

### Marketing Recommendations
- **Priority:** Onboarding and category expansion
- **Tactics:** Welcome sequences educating across categories, mobile-first UX improvements, first-purchase incentives to lock in the habit, "customers like you also bought" cross-category recommendations
- **Channel:** In-app notifications, push, SMS — their email engagement is lowest; mobile is the primary channel
- **KPI:** 2nd purchase rate within 60 days; category breadth expansion

---

## Model Comparison Summary

| Model | Silhouette ↑ | Davies-Bouldin ↓ | Clusters | Noise |
|-------|:---:|:---:|:---:|:---:|
| **K-Means (k=4)** | **0.58** | **0.72** | 4 | 0% |
| Hierarchical (Ward, k=4) | 0.51 | 0.89 | 4 | 0% |
| DBSCAN (eps=0.7, ms=5) | 0.43 | 1.14 | 3–5 | ~8% |

K-Means was selected as the production model. Full comparative analysis: `notebooks/03_model_comparison.ipynb`.

---

## How to Use This Report

1. **Marketing team:** Use the persona descriptions and channel recommendations directly to configure campaign targeting rules in your CRM.
2. **Data team:** The `data/processed/customers_segmented.csv` file contains a `segment_kmeans` column (0–3) for each customer — join on `customer_id`.
3. **Stakeholders:** See `reports/figures/` for all visuals used in the final presentation.
4. **Technical reviewers:** See `docs/system_architecture.md` for model selection rationale and `notebooks/03_model_comparison.ipynb` for full metric analysis.
