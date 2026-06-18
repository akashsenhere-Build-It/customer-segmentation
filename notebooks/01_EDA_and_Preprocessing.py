# %% [markdown]
# # Notebook 01 — Exploratory Data Analysis & Preprocessing
#
# **Project:** AI-Enhanced Customer Segmentation for Marketing
# **Author:** [Your Name] & Team
#
# This notebook walks through:
# 1. Loading and inspecting the synthetic customer dataset
# 2. Univariate and bivariate EDA
# 3. Running the preprocessing pipeline
# 4. Validating outputs before modeling

# %% [markdown]
# ## 1. Setup

# %%
import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.preprocessing import run_pipeline, FEATURE_COLS

plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.float_format', '{:.2f}'.format)
FIGURES = Path('../reports/figures')
FIGURES.mkdir(parents=True, exist_ok=True)

print("Imports OK")

# %% [markdown]
# ## 2. Load Data

# %%
# If data doesn't exist yet, generate it
import subprocess
data_path = Path('../data/processed/customers.csv')
if not data_path.exists():
    subprocess.run(['python', '../data/generate_synthetic.py'], check=True)

df_raw = pd.read_csv(data_path)
print(f"Shape: {df_raw.shape}")
df_raw.head()

# %% [markdown]
# ## 3. Data Overview

# %%
print("=== Data Types ===")
print(df_raw.dtypes)
print("\n=== Missing Values ===")
print(df_raw.isnull().sum())
print("\n=== Descriptive Statistics ===")
df_raw[FEATURE_COLS].describe().round(2)

# %% [markdown]
# ## 4. Univariate Distributions

# %%
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(FEATURE_COLS):
    axes[i].hist(df_raw[col].dropna(), bins=30, color='#4C72B0', alpha=0.8, edgecolor='white')
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel('')

fig.suptitle('Feature Distributions (Raw Data)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(FIGURES / 'eda_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 5. Correlation Heatmap

# %%
fig, ax = plt.subplots(figsize=(11, 9))
corr = df_raw[FEATURE_COLS].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, ax=ax, linewidths=0.5)
ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES / 'eda_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 6. Missing Value Analysis

# %%
missing = df_raw[FEATURE_COLS].isnull().mean() * 100
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing):
    fig, ax = plt.subplots(figsize=(8, 4))
    missing.plot(kind='barh', color='#DD8452', ax=ax)
    ax.set_xlabel('Missing (%)')
    ax.set_title('Missing Value Rate by Feature', fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES / 'eda_missing.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(missing)
else:
    print("No missing values in feature columns (they may already be in non-feature cols)")

# %% [markdown]
# ## 7. Run Preprocessing Pipeline

# %%
X_scaled, df_clean, scaler = run_pipeline(data_path)

print(f"Original rows: {len(df_raw):,}")
print(f"After preprocessing: {len(df_clean):,}")
print(f"Rows removed: {len(df_raw) - len(df_clean):,} ({(len(df_raw)-len(df_clean))/len(df_raw)*100:.1f}%)")
print(f"\nScaled feature matrix shape: {X_scaled.shape}")

# %% [markdown]
# ## 8. Validate Preprocessing Outputs

# %%
# Confirm zero nulls
assert df_clean[FEATURE_COLS].isnull().sum().sum() == 0, "Nulls remain!"
print("✅ No null values remain")

# Confirm scaling: mean ≈ 0, std ≈ 1
means = X_scaled.mean(axis=0)
stds = X_scaled.std(axis=0)
print(f"✅ Feature means (should be ~0): {means.round(3)}")
print(f"✅ Feature stds  (should be ~1): {stds.round(3)}")

# %% [markdown]
# ## 9. Post-Scaling Distribution Check

# %%
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()

for i, col in enumerate(FEATURE_COLS):
    axes[i].hist(X_scaled[:, i], bins=30, color='#55A868', alpha=0.8, edgecolor='white')
    axes[i].set_title(col, fontsize=10)
    axes[i].axvline(0, color='red', linestyle='--', lw=1, alpha=0.5)

fig.suptitle('Feature Distributions (After Standardization)', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(FIGURES / 'eda_scaled_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## 10. Summary
#
# | Step | Input Rows | Output Rows | Notes |
# |------|-----------|------------|-------|
# | Raw data | 1,500 | 1,500 | 3% synthetic missing values injected |
# | Value range validation | 1,500 | ~1,500 | Removes logically impossible values |
# | Median imputation | — | — | Fills nulls in 3 features |
# | IQR outlier removal | ~1,500 | ~1,450 | Conservative factor=3.0 |
# | Standardization | — | — | Zero mean, unit variance |
#
# **Preprocessed data is ready for PCA and clustering.**
# Proceed to: `02_Clustering_Models.ipynb`
