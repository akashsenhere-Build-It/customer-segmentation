"""
generate_synthetic.py
---------------------
Generates a reproducible synthetic e-commerce customer behavior dataset
for use in the AI-Enhanced Customer Segmentation pipeline.

Produces: data/processed/customers.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
N_CUSTOMERS = 1500
OUTPUT_DIR = Path(__file__).parent / "processed"


def generate_dataset(n: int = N_CUSTOMERS, seed: int = SEED) -> pd.DataFrame:
    """
    Generate a synthetic customer behavior dataset with 4 latent clusters.

    Features
    --------
    - recency_days       : Days since last purchase (lower = more recent)
    - frequency          : Number of purchases in last 6 months
    - monetary_value     : Total spend ($) in last 6 months
    - session_count      : Website sessions in last 6 months
    - avg_session_min    : Average session duration (minutes)
    - categories_browsed : Distinct product categories viewed
    - discount_usage_pct : % of orders that used a discount code
    - mobile_pct         : % of sessions from mobile device
    - email_open_rate    : Marketing email open rate (0–1)
    - cart_abandon_rate  : Shopping cart abandonment rate (0–1)
    """
    rng = np.random.default_rng(seed)

    # --- Cluster 0: High-Value Loyalists ---
    n0 = int(n * 0.25)
    c0 = pd.DataFrame({
        "recency_days":       rng.integers(1, 20, n0),
        "frequency":          rng.integers(10, 25, n0),
        "monetary_value":     rng.normal(850, 120, n0).clip(400),
        "session_count":      rng.integers(20, 50, n0),
        "avg_session_min":    rng.normal(12, 3, n0).clip(3),
        "categories_browsed": rng.integers(6, 12, n0),
        "discount_usage_pct": rng.uniform(0.05, 0.20, n0),
        "mobile_pct":         rng.uniform(0.30, 0.60, n0),
        "email_open_rate":    rng.uniform(0.45, 0.80, n0),
        "cart_abandon_rate":  rng.uniform(0.05, 0.25, n0),
        "true_cluster":       0,
    })

    # --- Cluster 1: Deal Seekers ---
    n1 = int(n * 0.30)
    c1 = pd.DataFrame({
        "recency_days":       rng.integers(5, 45, n1),
        "frequency":          rng.integers(4, 12, n1),
        "monetary_value":     rng.normal(220, 80, n1).clip(50),
        "session_count":      rng.integers(25, 70, n1),
        "avg_session_min":    rng.normal(8, 2, n1).clip(2),
        "categories_browsed": rng.integers(4, 9, n1),
        "discount_usage_pct": rng.uniform(0.55, 0.95, n1),
        "mobile_pct":         rng.uniform(0.40, 0.75, n1),
        "email_open_rate":    rng.uniform(0.30, 0.55, n1),
        "cart_abandon_rate":  rng.uniform(0.35, 0.65, n1),
        "true_cluster":       1,
    })

    # --- Cluster 2: Dormant Potentials ---
    n2 = int(n * 0.20)
    c2 = pd.DataFrame({
        "recency_days":       rng.integers(90, 180, n2),
        "frequency":          rng.integers(1, 5, n2),
        "monetary_value":     rng.normal(500, 150, n2).clip(100),
        "session_count":      rng.integers(2, 12, n2),
        "avg_session_min":    rng.normal(5, 2, n2).clip(1),
        "categories_browsed": rng.integers(2, 6, n2),
        "discount_usage_pct": rng.uniform(0.10, 0.40, n2),
        "mobile_pct":         rng.uniform(0.20, 0.55, n2),
        "email_open_rate":    rng.uniform(0.05, 0.25, n2),
        "cart_abandon_rate":  rng.uniform(0.40, 0.75, n2),
        "true_cluster":       2,
    })

    # --- Cluster 3: New Explorers ---
    n3 = n - n0 - n1 - n2
    c3 = pd.DataFrame({
        "recency_days":       rng.integers(1, 30, n3),
        "frequency":          rng.integers(1, 4, n3),
        "monetary_value":     rng.normal(120, 60, n3).clip(10),
        "session_count":      rng.integers(3, 15, n3),
        "avg_session_min":    rng.normal(6, 2, n3).clip(1),
        "categories_browsed": rng.integers(1, 4, n3),
        "discount_usage_pct": rng.uniform(0.10, 0.35, n3),
        "mobile_pct":         rng.uniform(0.60, 0.95, n3),
        "email_open_rate":    rng.uniform(0.10, 0.35, n3),
        "cart_abandon_rate":  rng.uniform(0.50, 0.80, n3),
        "true_cluster":       3,
    })

    df = pd.concat([c0, c1, c2, c3], ignore_index=True)

    # Add customer IDs and shuffle
    df.insert(0, "customer_id", [f"CUST_{i:05d}" for i in range(len(df))])
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Inject ~3% missing values to simulate real-world data quality issues
    for col in ["avg_session_min", "email_open_rate", "categories_browsed"]:
        mask = rng.random(len(df)) < 0.03
        df.loc[mask, col] = np.nan

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    out_path = OUTPUT_DIR / "customers.csv"
    df.to_csv(out_path, index=False)
    print(f"✅ Synthetic dataset saved → {out_path}")
    print(f"   Shape : {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n   Cluster distribution (true labels, training reference only):")
    print(df["true_cluster"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
