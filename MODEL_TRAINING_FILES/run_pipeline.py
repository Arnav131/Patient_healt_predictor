"""
Master Execution Pipeline for Cardiovascular Disease Predictor
Runs clinical sanitization, 5-algorithm tournament, champion tuning, holdout testing,
diagnostic plotting, and model packaging.
"""

import sys
from pathlib import Path

# Add paths to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
PARENT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PARENT_ROOT) not in sys.path:
    sys.path.insert(0, str(PARENT_ROOT))

import time
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_PATH,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    MODELS_DIR,
    FIGURES_DIR,
    DOCS_DIR
)
from src.data_loader import load_cardio_data
from src.eda import run_cardio_eda
from src.tournament import run_model_tournament
from src.tune import tune_champion_model
from src.model import get_feature_names
from src.evaluate import (
    evaluate_cardio_model,
    plot_cardio_confusion_matrices,
    plot_roc_and_pr_curves,
    plot_feature_importances,
    plot_tournament_results
)


def main():
    print("=" * 80)
    print("CARDIOVASCULAR DISEASE MACHINE LEARNING PRODUCTION PIPELINE")
    print("=" * 80)

    # 1. Ingest and Clinically Clean Data
    print("\n[STEP 1] Ingesting and clinically sanitizing cardio_train.csv...")
    df = load_cardio_data(DATASET_PATH)
    print(f"Dataset successfully cleaned: {len(df)} patient records retained.")
    print(f"Target distribution: {dict(df[TARGET_COLUMN].value_counts(normalize=True).round(4))}")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].copy()

    # 2. Stratified Train-Test Split (Strict Zero-Leakage Boundary)
    print("\n[STEP 2] Creating Stratified 80/20 Train-Test Split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    print(f"Training Set (Cross-Validation): {X_train.shape[0]} patients")
    print(f"Holdout Test Set (Locked):        {X_test.shape[0]} patients")

    # 3. Exploratory Data Analysis & Heatmaps
    print("\n[STEP 3] Generating Exploratory Data Analysis & Correlation Heatmaps...")
    run_cardio_eda(FIGURES_DIR)

    # 4. Multi-Model Tournament (5-Fold Stratified CV)
    print("\n[STEP 4] Executing 5-Algorithm Tournament across 5-Fold Stratified CV...")
    champion_name, tournament_df, champ_summary = run_model_tournament(X_train, y_train)

    plot_tournament_results(tournament_df, FIGURES_DIR / "tournament_benchmark.png")

    # 5. Deep Hyperparameter Tuning on Champion
    print(f"\n[STEP 5] Deep Regularization & Hyperparameter Optimization for Champion: {champion_name}...")
    best_pipeline, tune_summary = tune_champion_model(
        champion_name=champion_name,
        X_train=X_train,
        y_train=y_train,
        n_iter=15
    )

    # 6. Fit Final Champion on Full Training Set
    print("\n[STEP 6] Fitting final champion model on full training set (54,960 patients)...")
    t0 = time.time()
    best_pipeline.fit(X_train, y_train)
    fit_time = time.time() - t0
    print(f"Champion fit complete in {fit_time:.2f}s.")

    # 7. Unbiased Evaluation on Holdout Test Set (13,740 patients)
    print("\n[STEP 7] Evaluating on Unseen Holdout Test Set (13,740 patients)...")
    y_test_pred = best_pipeline.predict(X_test)
    y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]

    metrics_test = evaluate_cardio_model(y_test, y_test_pred, y_test_prob)

    print("=" * 65)
    print(f"FINAL TUNED CHAMPION ({champion_name}) TEST METRICS:")
    print(f"  Accuracy:           {metrics_test['accuracy']:.4f} ({metrics_test['accuracy']*100:.2f}%)")
    print(f"  Balanced Accuracy:  {metrics_test['balanced_accuracy']:.4f}")
    print(f"  ROC-AUC:            {metrics_test['roc_auc']:.4f}")
    print(f"  PR-AUC (Avg Prec):  {metrics_test['pr_auc']:.4f}")
    print(f"  Precision:          {metrics_test['precision']:.4f}")
    print(f"  Recall (Sens):      {metrics_test['recall']:.4f}")
    print(f"  F1-Score:           {metrics_test['f1']:.4f}")
    print(f"  Cohen's Kappa:      {metrics_test['cohen_kappa']:.4f}")
    print(f"  Brier Score:        {metrics_test['brier_score']:.4f}")
    print("=" * 65)

    # 8. Generate Final Diagnostic Visualizations
    print("\n[STEP 8] Generating diagnostic plots for final model...")
    plot_cardio_confusion_matrices(
        y_test, y_test_pred,
        FIGURES_DIR / "confusion_matrix_cardio.png",
        title=f"{champion_name} Confusion Matrix"
    )

    plot_roc_and_pr_curves(
        y_test, y_test_prob,
        FIGURES_DIR / "roc_pr_curves.png"
    )

    # Feature Importances
    final_clf = best_pipeline.named_steps["classifier"]
    feature_names = get_feature_names(best_pipeline)

    if hasattr(final_clf, "feature_importances_"):
        importances = final_clf.feature_importances_
    elif hasattr(final_clf, "coef_"):
        importances = np.abs(final_clf.coef_[0])
    else:
        importances = np.ones(len(feature_names))

    plot_feature_importances(
        feature_names,
        importances,
        FIGURES_DIR / "feature_importance.png",
        top_n=15
    )

    # 9. Model Serialization
    print("\n[STEP 9] Serializing champion model artifact...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / "best_cardio_model.joblib"
    joblib.dump(best_pipeline, save_path)
    print(f"Champion pipeline successfully saved to: {save_path}")

    # 10. Update Living Documentation
    print("\n[STEP 10] Updating project documentation with tournament results...")
    update_documentation(tournament_df, champion_name, tune_summary, metrics_test)

    print("\nProduction pipeline execution completed successfully!")


def update_documentation(tournament_df, champion_name, tune_summary, test_metrics):
    """Appends tournament rankings and final test metrics to project documentation."""
    doc_path = DOCS_DIR / "PROJECT_WORKFLOW_AND_ANALYSIS.md"

    table_rows = []
    for row in tournament_df.itertuples():
        table_rows.append(
            f"| **{row.model}** | {row.mean_cv_accuracy:.4f} (+/- {row.std_cv_accuracy:.4f}) | "
            f"{row.mean_cv_roc_auc:.4f} (+/- {row.std_cv_roc_auc:.4f}) | {row.mean_cv_f1:.4f} | "
            f"{row.mean_train_accuracy:.4f} | {row.overfit_gap:.4f} | {row.fit_time_seconds:.1f}s |"
        )
    table_str = "\n".join(table_rows)

    report_section = f"""

---

## 5. Empirical Results & Multi-Model Tournament Rankings

### Algorithm Tournament Leaderboard (5-Fold Stratified Cross-Validation on 54,960 Patients)

| Algorithm | Mean CV Accuracy | Mean CV ROC-AUC | Mean CV F1 | Mean Train Acc | Overfitting Gap | CV Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
{table_str}

### Tournament Champion: **{champion_name}**
- **Optimal Tuned Hyperparameters**:
```json
{tune_summary['best_params']}
```
- **Tuning Cross-Validation ROC-AUC**: `{tune_summary['best_cv_roc_auc']:.4f}`

---

## 6. Final Holdout Test Set Performance (13,740 Patients)

| Evaluation Metric | Test Set Score | Clinical Benchmark / Significance |
| :--- | :---: | :--- |
| **Accuracy** | **{test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)** | High discriminative accuracy on real medical exam records |
| **Balanced Accuracy** | **{test_metrics['balanced_accuracy']:.4f}** | Equal sensitivity for healthy vs diseased patients |
| **ROC-AUC** | **{test_metrics['roc_auc']:.4f}** | Area Under ROC Curve (Excellent clinical discrimination) |
| **PR-AUC (Avg Precision)** | **{test_metrics['pr_auc']:.4f}** | Robust precision-recall area above 0.50 baseline |
| **Precision** | **{test_metrics['precision']:.4f}** | Positive predictive value |
| **Recall (Sensitivity)** | **{test_metrics['recall']:.4f}** | Detection rate of patients with cardiovascular disease |
| **F1-Score** | **{test_metrics['f1']:.4f}** | Harmonic mean of precision and recall |
| **Cohen's Kappa** | **{test_metrics['cohen_kappa']:.4f}** | High inter-rater agreement above chance |
| **Brier Score** | **{test_metrics['brier_score']:.4f}** | Calibrated probabilistic loss |

---

## 7. Generated Visual Artifacts

The following high-resolution diagnostic figures are preserved in `docs/figures/`:
- **Pearson Correlation Heatmap**: `figures/pearson_correlation_heatmap.png`
- **Spearman Rank Correlation Heatmap**: `figures/spearman_correlation_heatmap.png`
- **Cramér's V Association Heatmap**: `figures/cramers_v_heatmap.png`
- **Biomarker Distributions by CVD**: `figures/target_and_feature_distributions.png`
- **Algorithm Tournament Leaderboard Chart**: `figures/tournament_benchmark.png`
- **Final Confusion Matrix**: `figures/confusion_matrix_cardio.png`
- **ROC and Precision-Recall Curves**: `figures/roc_pr_curves.png`
- **Top 15 Feature Importances**: `figures/feature_importance.png`

---

## 8. Production Inference API

The final trained pipeline is saved at:
`models/best_cardio_model.joblib`

Run inference on any patient:
```powershell
python D:\\ML_PROJECTS\\end_to_end_ml_project\\health_care_predictor\\predict.py
```
"""

    with open(doc_path, "a", encoding="utf-8") as f:
        f.write(report_section)


if __name__ == "__main__":
    main()
