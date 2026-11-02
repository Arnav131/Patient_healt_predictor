"""
Unit and Integration Tests for Cardiovascular Disease Predictor Pipeline
"""

import sys
from pathlib import Path

# Add project root and training files to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_ROOT = PROJECT_ROOT / "MODEL_TRAINING_FILES"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import pytest
import numpy as np
import pandas as pd
import tempfile
import joblib

from src.config import DATASET_PATH, TARGET_COLUMN
from MODEL_TRAINING_FILES.src.data_loader import load_cardio_data
from src.feature_engineering import CardioFeatureEngineer
from MODEL_TRAINING_FILES.src.model import get_candidate_models, build_cardio_pipeline
from predict import predict_cardio_risk


def test_cardio_data_loader():
    """Validates data ingestion, transposition correction, and outlier removal."""
    df = load_cardio_data(DATASET_PATH)
    assert len(df) > 65000, f"Expected >65,000 clean rows, got {len(df)}"
    assert TARGET_COLUMN in df.columns, f"Target {TARGET_COLUMN} missing"
    assert set(df[TARGET_COLUMN].unique()) == {0, 1}, "Target must be binary 0 and 1"
    assert (df["ap_hi"] >= df["ap_lo"]).all(), "Physiological violation: ap_hi must be >= ap_lo"
    assert (df["ap_hi"] >= 70).all() and (df["ap_hi"] <= 240).all(), "ap_hi out of bounds"
    assert (df["ap_lo"] >= 40).all() and (df["ap_lo"] <= 160).all(), "ap_lo out of bounds"
    assert "id" not in df.columns, "Identifier column 'id' must be dropped"


def test_cardio_feature_engineering_integrity():
    """Validates clinical feature extraction logic and value ranges."""
    df = load_cardio_data(DATASET_PATH).head(100)
    fe = CardioFeatureEngineer()
    df_transformed = fe.transform(df)

    expected_cols = [
        "age_years",
        "bmi",
        "pulse_pressure",
        "map",
        "aha_bp_stage",
        "bmi_category",
        "metabolic_score",
        "lifestyle_risk",
        "age_bp_interaction"
    ]
    for col in expected_cols:
        assert col in df_transformed.columns, f"Engineered column {col} missing"

    # Clinical sanity checks
    assert (df_transformed["pulse_pressure"] > 0).all(), "Pulse pressure must be positive"
    assert (df_transformed["bmi"] > 10).all() and (df_transformed["bmi"] < 100).all(), "BMI outside human limits"
    assert set(df_transformed["aha_bp_stage"].unique()).issubset({"Normal", "Elevated", "Stage_1", "Stage_2", "Crisis"})
    assert set(df_transformed["bmi_category"].unique()).issubset({"Underweight", "Normal", "Overweight", "Obese"})
    assert (df_transformed["metabolic_score"] >= 0).all() and (df_transformed["metabolic_score"] <= 4).all()


def test_zero_data_leakage_in_feature_engineer():
    """Confirms transformer executes pure row-wise mapping without learning global test parameters."""
    fe = CardioFeatureEngineer()
    dummy_data = pd.DataFrame({
        "age": [18000, 20000],
        "height": [170, 160],
        "weight": [70.0, 80.0],
        "ap_hi": [120, 140],
        "ap_lo": [80, 90],
        "cholesterol": [1, 2],
        "gluc": [1, 1],
        "smoke": [0, 1],
        "alco": [0, 0],
        "active": [1, 0]
    })
    fe.fit(dummy_data)
    out = fe.transform(dummy_data)
    assert len(out) == 2
    # Verify no state was dynamically mutated
    assert len(fe.__dict__) == 0


def test_candidate_models_initialization():
    """Confirms all 5 tournament models are present and valid."""
    candidates = get_candidate_models()
    expected_models = {"RandomForest", "LightGBM", "XGBoost", "ExtraTrees", "LogisticRegression"}
    assert set(candidates.keys()) == expected_models


def test_pipeline_fit_predict():
    """Validates end-to-end pipeline training, prediction shapes, and probabilities."""
    df = load_cardio_data(DATASET_PATH).head(200)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    candidates = get_candidate_models()
    rf = candidates["RandomForest"]
    rf.n_estimators = 10
    rf.max_depth = 5

    pipeline = build_cardio_pipeline(rf)
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    probs = pipeline.predict_proba(X)

    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_cardio_inference_api():
    """Validates inference helper with sample patient inputs."""
    df = load_cardio_data(DATASET_PATH).head(150)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    candidates = get_candidate_models()
    rf = candidates["RandomForest"]
    rf.n_estimators = 10
    rf.max_depth = 4
    pipeline = build_cardio_pipeline(rf)
    pipeline.fit(X, y)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        joblib.dump(pipeline, tmp.name)
        sample_patient = {
            "age": 52,
            "gender": 1,
            "height": 165,
            "weight": 68.0,
            "ap_hi": 130,
            "ap_lo": 85,
            "cholesterol": 1,
            "gluc": 1,
            "smoke": 0,
            "alco": 0,
            "active": 1
        }
        res = predict_cardio_risk(sample_patient, model_path=Path(tmp.name))
        assert res["cvd_prediction"] in [0, 1]
        assert 0.0 <= res["cvd_probability"] <= 1.0
        assert res["risk_tier"] in ["Low Risk", "Moderate Risk", "High Risk", "Critical Risk"]
