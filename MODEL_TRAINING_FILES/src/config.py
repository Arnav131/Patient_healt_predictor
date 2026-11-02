"""
Configuration Settings for Cardiovascular Disease Predictor Pipeline
"""

from pathlib import Path

# Paths (Dynamically resolved relative to this file for cross-platform & cloud compatibility)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = BASE_DIR / "MODEL_TRAINING_FILES" / "dataset" / "cardio_train.csv"
DOCS_DIR = BASE_DIR / "docs"
FIGURES_DIR = DOCS_DIR / "figures"
MODELS_DIR = BASE_DIR / "models"

# Target & Identifier Definitions
TARGET_COLUMN = "cardio"
IDENTIFIER_COLUMNS = ["id"]

# Physiological Outlier Boundaries
MIN_HEIGHT = 120.0
MAX_HEIGHT = 220.0
MIN_WEIGHT = 35.0
MAX_WEIGHT = 200.0
MIN_AP_HI = 70.0
MAX_AP_HI = 240.0
MIN_AP_LO = 40.0
MAX_AP_LO = 160.0

# Feature Groups
RAW_NUMERICAL_FEATURES = [
    "age_years",
    "height",
    "weight",
    "ap_hi",
    "ap_lo"
]

ENGINEERED_NUMERICAL_FEATURES = [
    "bmi",
    "pulse_pressure",
    "map",
    "age_bp_interaction",
    "metabolic_score"
]

RAW_CATEGORICAL_FEATURES = [
    "gender",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active"
]

ENGINEERED_CATEGORICAL_FEATURES = [
    "aha_bp_stage",
    "bmi_category",
    "lifestyle_risk"
]

# Partitioning and CV Settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
