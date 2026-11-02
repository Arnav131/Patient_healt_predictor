"""
Deep Hyperparameter Tuning Engine for Tournament Champion
"""

import time
from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from .config import RANDOM_STATE, CV_FOLDS
    from .model import build_cardio_pipeline
except ImportError:
    from src.config import RANDOM_STATE, CV_FOLDS
    from src.model import build_cardio_pipeline


def get_champion_param_space(model_name: str) -> Dict[str, Any]:
    """Returns tailored hyperparameter search spaces for the champion algorithm."""
    if model_name == "LightGBM":
        return {
            "classifier__n_estimators": [100, 150, 200, 250],
            "classifier__max_depth": [4, 5, 6, 8],
            "classifier__num_leaves": [15, 31, 63],
            "classifier__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "classifier__subsample": [0.7, 0.85, 1.0],
            "classifier__colsample_bytree": [0.7, 0.85, 1.0],
            "classifier__reg_alpha": [0.0, 0.1, 1.0],
            "classifier__reg_lambda": [0.0, 1.0, 5.0]
        }
    elif model_name == "XGBoost":
        return {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__max_depth": [3, 4, 5, 6],
            "classifier__learning_rate": [0.03, 0.05, 0.08, 0.1],
            "classifier__subsample": [0.7, 0.85, 1.0],
            "classifier__colsample_bytree": [0.7, 0.85, 1.0],
            "classifier__gamma": [0.0, 0.1, 0.5],
            "classifier__reg_alpha": [0.0, 0.5, 2.0]
        }
    else:  # RandomForest or other tree ensemble
        return {
            "classifier__n_estimators": [100, 150, 200],
            "classifier__max_depth": [10, 14, 18, 22],
            "classifier__min_samples_split": [5, 10, 20],
            "classifier__min_samples_leaf": [2, 4, 8],
            "classifier__max_features": ["sqrt", "log2", 0.4],
            "classifier__criterion": ["gini", "entropy"]
        }


def tune_champion_model(
    champion_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 15
) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Executes Stratified 5-Fold RandomizedSearchCV optimization on the champion algorithm.
    """
    param_space = get_champion_param_space(champion_name)

    if champion_name == "LightGBM":
        base_clf = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    elif champion_name == "XGBoost":
        base_clf = XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
    else:
        base_clf = RandomForestClassifier(bootstrap=True, random_state=RANDOM_STATE, n_jobs=-1)

    pipeline = build_cardio_pipeline(base_clf)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print(f"\n[TUNE] Starting {n_iter} optimization iterations for {champion_name} across {CV_FOLDS} folds...")
    start_t = time.time()

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_space,
        n_iter=n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    search.fit(X_train, y_train)
    elapsed = time.time() - start_t

    best_pipeline = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_

    print(f"[TUNE] Completed tuning in {elapsed:.2f}s.")
    print(f"[TUNE] Best Mean CV ROC-AUC: {best_score:.4f}")
    print(f"[TUNE] Best Hyperparameters: {best_params}")

    tuning_summary = {
        "champion_name": champion_name,
        "best_params": best_params,
        "best_cv_roc_auc": float(best_score),
        "tuning_time_seconds": float(elapsed)
    }

    return best_pipeline, tuning_summary
