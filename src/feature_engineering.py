"""
Cardiovascular Feature Engineering Module
Implements leak-free sklearn-compatible transformer.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CardioFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transforms raw physiological indicators into hemodynamic, anthropometric,
    and metabolic risk features.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        # 1. Age in years
        if "age_years" not in X_out.columns and "age" in X_out.columns:
            X_out["age_years"] = X_out["age"] / 365.25

        # 2. Body Mass Index (BMI)
        height_m = X_out["height"] / 100.0
        bmi = X_out["weight"] / (height_m ** 2)
        X_out["bmi"] = bmi.round(2)

        # 3. Pulse Pressure (PP = SBP - DBP)
        X_out["pulse_pressure"] = X_out["ap_hi"] - X_out["ap_lo"]

        # 4. Mean Arterial Pressure (MAP = DBP + 1/3 PP)
        X_out["map"] = (X_out["ap_lo"] + (X_out["pulse_pressure"] / 3.0)).round(2)

        # 5. AHA Blood Pressure Staging
        def classify_aha_bp(row) -> str:
            sbp = row["ap_hi"]
            dbp = row["ap_lo"]
            if sbp > 180 or dbp > 120:
                return "Crisis"
            elif sbp >= 140 or dbp >= 90:
                return "Stage_2"
            elif (130 <= sbp <= 139) or (80 <= dbp <= 89):
                return "Stage_1"
            elif (120 <= sbp <= 129) and (dbp < 80):
                return "Elevated"
            else:
                return "Normal"

        X_out["aha_bp_stage"] = X_out.apply(classify_aha_bp, axis=1)

        # 6. WHO BMI Categories
        def classify_bmi(bmi_val: float) -> str:
            if bmi_val < 18.5:
                return "Underweight"
            elif bmi_val < 25.0:
                return "Normal"
            elif bmi_val < 30.0:
                return "Overweight"
            else:
                return "Obese"

        X_out["bmi_category"] = X_out["bmi"].apply(classify_bmi)

        # 7. Metabolic Syndrome Composite Score (0 to 4)
        is_obese = (X_out["bmi"] >= 30.0).astype(int)
        high_chol = (X_out["cholesterol"] >= 2).astype(int)
        high_gluc = (X_out["gluc"] >= 2).astype(int)
        high_bp = ((X_out["ap_hi"] >= 140) | (X_out["ap_lo"] >= 90)).astype(int)
        X_out["metabolic_score"] = is_obese + high_chol + high_gluc + high_bp

        # 8. Lifestyle Risk Score (0 to 3)
        inactive = (1 - X_out["active"]).astype(int)
        X_out["lifestyle_risk"] = X_out["smoke"].astype(int) + X_out["alco"].astype(int) + inactive

        # 9. Age and Systolic Blood Pressure Interaction
        X_out["age_bp_interaction"] = (X_out["age_years"] * X_out["ap_hi"]).round(2)

        # Drop raw age in days to prevent redundancy
        if "age" in X_out.columns:
            X_out = X_out.drop(columns=["age"])

        return X_out
