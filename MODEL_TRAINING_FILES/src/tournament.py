"""
Multi-Model Tournament Engine
Benchmarks Random Forest, LightGBM, XGBoost, Extra Trees, and Logistic Regression
across identical 5-Fold Stratified Cross-Validation folds.
"""

import time
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

try:
    from .config import RANDOM_STATE, CV_FOLDS
    from .model import get_candidate_models, build_cardio_pipeline
except ImportError:
    from src.config import RANDOM_STATE, CV_FOLDS
    from src.model import get_candidate_models, build_cardio_pipeline


def run_model_tournament(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
    """
    Executes 5-fold cross-validation on all contender algorithms, ranks them,
    and returns the tournament champion.
    """
    candidates = get_candidate_models()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy": "accuracy",
        "roc_auc": "roc_auc",
        "f1": "f1"
    }

    results = []

    print(f"\n{'='*75}")
    print(f"STARTING 5-MODEL TOURNAMENT ACROSS {CV_FOLDS}-FOLD STRATIFIED CV ({len(X_train)} SAMPLES)")
    print(f"{'='*75}")

    for name, model in candidates.items():
        print(f"\n[TOURNAMENT] Evaluating {name}...")
        pipeline = build_cardio_pipeline(model)
        
        t0 = time.time()
        cv_res = cross_validate(
            pipeline, X_train, y_train,
            cv=cv, scoring=scoring,
            n_jobs=-1, return_train_score=True
        )
        t_elapsed = time.time() - t0

        mean_acc = float(np.mean(cv_res["test_accuracy"]))
        std_acc = float(np.std(cv_res["test_accuracy"]))
        mean_auc = float(np.mean(cv_res["test_roc_auc"]))
        std_auc = float(np.std(cv_res["test_roc_auc"]))
        mean_f1 = float(np.mean(cv_res["test_f1"]))
        train_acc = float(np.mean(cv_res["train_accuracy"]))

        print(f"  --> {name} | CV Acc: {mean_acc:.4f} (+/- {std_acc:.4f}) | ROC-AUC: {mean_auc:.4f} | F1: {mean_f1:.4f} | Time: {t_elapsed:.1f}s")

        results.append({
            "model": name,
            "mean_cv_accuracy": mean_acc,
            "std_cv_accuracy": std_acc,
            "mean_cv_roc_auc": mean_auc,
            "std_cv_roc_auc": std_auc,
            "mean_cv_f1": mean_f1,
            "mean_train_accuracy": train_acc,
            "overfit_gap": train_acc - mean_acc,
            "fit_time_seconds": t_elapsed
        })

    results_df = pd.DataFrame(results).sort_values(by="mean_cv_roc_auc", ascending=False).reset_index(drop=True)

    champion_name = str(results_df.iloc[0]["model"])
    print(f"\n{'='*75}")
    print(f"TOURNAMENT CHAMPION: {champion_name} (CV ROC-AUC: {results_df.iloc[0]['mean_cv_roc_auc']:.4f}, Accuracy: {results_df.iloc[0]['mean_cv_accuracy']:.4f})")
    print(f"{'='*75}")

    champion_summary = {
        "champion": champion_name,
        "results_table": results_df.to_dict(orient="records")
    }

    return champion_name, results_df, champion_summary
