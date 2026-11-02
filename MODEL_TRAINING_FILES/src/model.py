"""
Cardiovascular Model Factory and Pipeline Assembler
"""

from typing import Dict, Any, List
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

try:
    from .feature_engineering import CardioFeatureEngineer
    from .preprocessor import build_cardio_preprocessor
    from .config import RANDOM_STATE
except ImportError:
    from src.feature_engineering import CardioFeatureEngineer
    from src.preprocessor import build_cardio_preprocessor
    from src.config import RANDOM_STATE


def get_candidate_models() -> Dict[str, Any]:
    """
    Returns candidate algorithm instances with established baseline parameters.
    """
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="sqrt",
            bootstrap=True,
            oob_score=True,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            num_leaves=31,
            learning_rate=0.08,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features="sqrt",
            bootstrap=True,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    }


def build_cardio_pipeline(classifier) -> Pipeline:
    """
    Constructs an end-to-end scikit-learn Pipeline with:
    1. Clinical Feature Engineering
    2. Column Scaling & Encoding
    3. Selected Classifier
    """
    return Pipeline(steps=[
        ("feature_engineering", CardioFeatureEngineer()),
        ("preprocessor", build_cardio_preprocessor()),
        ("classifier", classifier)
    ])


def get_feature_names(fitted_pipeline: Pipeline) -> List[str]:
    """
    Extracts transformed feature names from the fitted preprocessor step.
    """
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    return list(preprocessor.get_feature_names_out())
