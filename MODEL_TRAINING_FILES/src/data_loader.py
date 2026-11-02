"""
Cardiovascular Dataset Loader and Clinical Sanitizer Module
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

try:
    from .config import (
        DATASET_PATH,
        TARGET_COLUMN,
        IDENTIFIER_COLUMNS,
        MIN_HEIGHT,
        MAX_HEIGHT,
        MIN_WEIGHT,
        MAX_WEIGHT,
        MIN_AP_HI,
        MAX_AP_HI,
        MIN_AP_LO,
        MAX_AP_LO
    )
except ImportError:
    from src.config import (
        DATASET_PATH,
        TARGET_COLUMN,
        IDENTIFIER_COLUMNS,
        MIN_HEIGHT,
        MAX_HEIGHT,
        MIN_WEIGHT,
        MAX_WEIGHT,
        MIN_AP_HI,
        MAX_AP_HI,
        MIN_AP_LO,
        MAX_AP_LO
    )


def load_cardio_data(filepath: Path = DATASET_PATH) -> pd.DataFrame:
    """
    Loads raw cardiovascular records from CSV, applies automated transposition fixes
    for inverted blood pressures, and filters biologically impossible extremes.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset not found at {filepath}")

    # Semicolon delimited
    df = pd.read_csv(filepath, sep=";")

    # Drop pure duplicates
    df = df.drop_duplicates()

    # Drop patient identifier column (prevents index leakage)
    for col in IDENTIFIER_COLUMNS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 1. Correct inverted blood pressure recordings
    hi = np.maximum(df["ap_hi"], df["ap_lo"])
    lo = np.minimum(df["ap_hi"], df["ap_lo"])
    df["ap_hi"] = hi
    df["ap_lo"] = lo

    # 2. Filter physiological measurement outliers / typos
    valid_mask = (
        (df["ap_hi"] >= MIN_AP_HI) & (df["ap_hi"] <= MAX_AP_HI) &
        (df["ap_lo"] >= MIN_AP_LO) & (df["ap_lo"] <= MAX_AP_LO) &
        (df["height"] >= MIN_HEIGHT) & (df["height"] <= MAX_HEIGHT) &
        (df["weight"] >= MIN_WEIGHT) & (df["weight"] <= MAX_WEIGHT) &
        (df["ap_hi"] >= df["ap_lo"] + 5)
    )

    df_clean = df[valid_mask].copy().reset_index(drop=True)

    # Convert age from days to years
    df_clean["age_years"] = df_clean["age"] / 365.25

    return df_clean


def get_train_test_split_data(filepath: Path = DATASET_PATH) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits cleaned dataset into feature matrix X and target vector y.
    """
    df = load_cardio_data(filepath)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].copy()
    return X, y
