"""
Comprehensive Evaluation Engine and Clinical Visualizer
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    brier_score_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    classification_report
)


def evaluate_cardio_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """
    Computes statistical and clinical evaluation metrics for binary CVD prediction.
    """
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)
    kappa = cohen_kappa_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)

    report = classification_report(y_true, y_pred, target_names=["Healthy (0)", "CVD (1)"], output_dict=True)

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(avg_prec),
        "cohen_kappa": float(kappa),
        "brier_score": float(brier),
        "classification_report": report
    }


def plot_cardio_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Cardiovascular Disease Confusion Matrix"
):
    """Plots side-by-side counts and normalized confusion matrix."""
    labels = ["Healthy (0)", "CVD (1)"]
    cm_raw = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.heatmap(cm_raw, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[0], cbar=False)
    axes[0].set_title(f"{title} (Patient Counts)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Predicted Diagnosis")
    axes[0].set_ylabel("True Diagnosis")

    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=axes[1], cbar=True)
    axes[1].set_title(f"{title} (Normalized Sensitivity/Specificity)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Diagnosis")
    axes[1].set_ylabel("True Diagnosis")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EVAL] Saved confusion matrix to {output_path}")


def plot_roc_and_pr_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Path
):
    """Plots ROC Curve and Precision-Recall Curve side-by-side."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC Curve
    axes[0].plot(fpr, tpr, color="#d73027", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Guess (AUC = 0.500)")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall Curve
    axes[1].plot(rec, prec, color="#4575b4", lw=2.5, label=f"PR Curve (AUC = {pr_auc:.4f})")
    baseline_pr = float(np.mean(y_true))
    axes[1].axhline(y=baseline_pr, color="k", linestyle="--", lw=1.5, label=f"Baseline Pretest ({baseline_pr:.2f})")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11)
    axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    axes[1].set_title("Precision-Recall (PR) Curve", fontsize=13, fontweight="bold")
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EVAL] Saved ROC & PR curves to {output_path}")


def plot_feature_importances(
    feature_names: List[str],
    importances: np.ndarray,
    output_path: Path,
    top_n: int = 15
):
    """Plots top N features by model importance."""
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 7))
    sns.barplot(data=df_imp, x="importance", y="feature", palette="magma")
    plt.title(f"Top {top_n} Cardiovascular Feature Importances", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Relative Feature Importance", fontsize=11)
    plt.ylabel("Clinical Predictor", fontsize=11)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EVAL] Saved feature importances to {output_path}")


def plot_tournament_results(
    results_df: pd.DataFrame,
    output_path: Path
):
    """Plots comparison bar chart across all tournament candidate algorithms."""
    plt.figure(figsize=(10, 6))
    df_sorted = results_df.sort_values(by="mean_cv_accuracy", ascending=False)
    
    sns.barplot(data=df_sorted, x="model", y="mean_cv_accuracy", palette="viridis")
    plt.title("Multi-Model Tournament: 5-Fold Cross-Validation Accuracy", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Algorithm", fontsize=11)
    plt.ylabel("Mean 5-Fold CV Accuracy", fontsize=11)
    plt.ylim([0.65, 0.85])
    
    for idx, row in enumerate(df_sorted.itertuples()):
        plt.text(idx, row.mean_cv_accuracy + 0.005, f"{row.mean_cv_accuracy*100:.2f}%\n(AUC {row.mean_cv_roc_auc:.3f})",
                 ha="center", fontsize=9, fontweight="bold")
                 
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[EVAL] Saved tournament chart to {output_path}")
