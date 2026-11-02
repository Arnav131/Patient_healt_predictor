"""
Exploratory Data Analysis (EDA) Engine for Cardiovascular Dataset
Generates correlation heatmaps, categorical association matrices, and biomarker distributions.
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

try:
    from .config import FIGURES_DIR
    from .data_loader import load_cardio_data
    from .feature_engineering import CardioFeatureEngineer
except ImportError:
    from src.config import FIGURES_DIR
    from src.data_loader import load_cardio_data
    from src.feature_engineering import CardioFeatureEngineer


def calculate_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Computes bias-corrected Cramer's V for two categorical variables."""
    cm = pd.crosstab(x, y)
    chi2 = chi2_contingency(cm)[0]
    n = cm.sum().sum()
    if n == 0:
        return 0.0
    phi2 = chi2 / n
    r, k = cm.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2corr / denom))


def run_cardio_eda(output_dir: Path = FIGURES_DIR):
    """Executes comprehensive EDA and saves diagnostic figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    df_raw = load_cardio_data()

    fe = CardioFeatureEngineer()
    df = fe.transform(df_raw)

    # 1. Pearson Correlation Heatmap for Continuous Biomarkers
    num_cols = ["age_years", "height", "weight", "ap_hi", "ap_lo", "bmi", "pulse_pressure", "map", "cardio"]
    corr_p = df[num_cols].corr(method="pearson")

    plt.figure(figsize=(11, 9))
    sns.heatmap(corr_p, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True)
    plt.title("Pearson Correlation Heatmap (Cardiovascular Biomarkers)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    p_path = output_dir / "pearson_correlation_heatmap.png"
    plt.savefig(p_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved Pearson heatmap to {p_path}")

    # 2. Spearman Rank Correlation Heatmap
    corr_s = df[num_cols].corr(method="spearman")
    plt.figure(figsize=(11, 9))
    sns.heatmap(corr_s, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True)
    plt.title("Spearman Rank Correlation Heatmap (Cardiovascular Biomarkers)", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    s_path = output_dir / "spearman_correlation_heatmap.png"
    plt.savefig(s_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved Spearman heatmap to {s_path}")

    # 3. Cramer's V Categorical Association Heatmap
    cat_cols = ["gender", "cholesterol", "gluc", "smoke", "alco", "active", "aha_bp_stage", "bmi_category", "cardio"]
    cramers_df = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)
    for c1 in cat_cols:
        for c2 in cat_cols:
            cramers_df.loc[c1, c2] = calculate_cramers_v(df[c1], df[c2])

    plt.figure(figsize=(10, 8))
    sns.heatmap(cramers_df, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=0.6, square=True)
    plt.title("Cramer's V Categorical Association Matrix", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    cv_path = output_dir / "cramers_v_heatmap.png"
    plt.savefig(cv_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved Cramer's V heatmap to {cv_path}")

    # 4. Target & Physiological Risk Distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # A: Systolic BP by Cardio
    sns.boxplot(data=df, x="cardio", y="ap_hi", ax=axes[0, 0], palette=["#4575b4", "#d73027"])
    axes[0, 0].set_title("Systolic Blood Pressure (ap_hi) by CVD Status", fontsize=12, fontweight="bold")
    axes[0, 0].set_xticklabels(["Healthy (0)", "Cardiovascular Disease (1)"])

    # B: BMI by Cardio
    sns.boxplot(data=df, x="cardio", y="bmi", ax=axes[0, 1], palette=["#4575b4", "#d73027"])
    axes[0, 1].set_title("Body Mass Index (BMI) by CVD Status", fontsize=12, fontweight="bold")
    axes[0, 1].set_xticklabels(["Healthy (0)", "Cardiovascular Disease (1)"])

    # C: Cholesterol Prevalence by Cardio
    chol_c = pd.crosstab(df["cholesterol"], df["cardio"], normalize="index") * 100
    chol_c.plot(kind="bar", stacked=True, ax=axes[1, 0], color=["#4575b4", "#d73027"])
    axes[1, 0].set_title("CVD Rate across Cholesterol Tiers (1:Normal, 2:Above, 3:High)", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Cholesterol Tier")
    axes[1, 0].set_ylabel("Percentage (%)")
    axes[1, 0].legend(["Healthy", "CVD"])

    # D: AHA Blood Pressure Stages
    bp_c = pd.crosstab(df["aha_bp_stage"], df["cardio"], normalize="index") * 100
    bp_order = ["Normal", "Elevated", "Stage_1", "Stage_2", "Crisis"]
    bp_c = bp_c.reindex(bp_order).dropna()
    bp_c.plot(kind="bar", stacked=True, ax=axes[1, 1], color=["#4575b4", "#d73027"])
    axes[1, 1].set_title("CVD Rate across AHA Blood Pressure Stages", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("AHA Blood Pressure Stage")
    axes[1, 1].set_ylabel("Percentage (%)")
    axes[1, 1].legend(["Healthy", "CVD"])

    plt.tight_layout()
    dist_path = output_dir / "target_and_feature_distributions.png"
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"[EDA] Saved distribution plots to {dist_path}")


if __name__ == "__main__":
    run_cardio_eda()
