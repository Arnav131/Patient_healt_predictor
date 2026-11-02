"""
Preprocessor and Column Transformer Module for Cardiovascular Predictor
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer


def build_cardio_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that scales continuous features with RobustScaler
    and one-hot encodes nominal/ordinal categories.
    """
    numeric_features = [
        "age_years",
        "height",
        "weight",
        "ap_hi",
        "ap_lo",
        "bmi",
        "pulse_pressure",
        "map",
        "age_bp_interaction",
        "metabolic_score"
    ]

    categorical_features = [
        "gender",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        "aha_bp_stage",
        "bmi_category",
        "lifestyle_risk"
    ]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        remainder="drop"
    )

    return preprocessor
