"""
Production Inference Engine for Cardiovascular Disease Predictor
"""

import sys
from pathlib import Path
from typing import Dict, Any, Union
import joblib
import pandas as pd
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR


def predict_cardio_risk(
    patient: Union[Dict[str, Any], pd.DataFrame],
    model_path: Path = None
) -> Dict[str, Any]:
    """
    Evaluates patient examination data and returns predicted CVD diagnosis,
    risk probability, and clinical risk stratification.
    """
    if model_path is None:
        model_path = MODELS_DIR / "best_cardio_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}. Run run_pipeline.py first.")

    model = joblib.load(model_path)

    if isinstance(patient, dict):
        df_patient = pd.DataFrame([patient])
    else:
        df_patient = patient.copy()

    # If age is given in years (e.g. 55), convert to days if needed
    if "age" in df_patient.columns and df_patient["age"].max() < 120:
        df_patient["age"] = df_patient["age"] * 365.25

    # Automatic correction for swapped blood pressures in incoming inputs
    if "ap_hi" in df_patient.columns and "ap_lo" in df_patient.columns:
        hi = np.maximum(df_patient["ap_hi"], df_patient["ap_lo"])
        lo = np.minimum(df_patient["ap_hi"], df_patient["ap_lo"])
        df_patient["ap_hi"] = hi
        df_patient["ap_lo"] = lo

    pred = int(model.predict(df_patient)[0])
    prob_cvd = float(model.predict_proba(df_patient)[0, 1])

    # Clinical risk stratification
    if prob_cvd < 0.25:
        risk_tier = "Low Risk"
        recommendation = "Maintain regular physical activity and annual health screening."
    elif prob_cvd < 0.50:
        risk_tier = "Moderate Risk"
        recommendation = "Lifestyle modifications recommended. Monitor blood pressure and lipid panels."
    elif prob_cvd < 0.75:
        risk_tier = "High Risk"
        recommendation = "Clinical evaluation advised. Cardiologist consultation and therapeutic intervention recommended."
    else:
        risk_tier = "Critical Risk"
        recommendation = "Immediate cardiovascular clinical workup required. High likelihood of arterial disease."

    return {
        "cvd_prediction": pred,
        "diagnosis": "Cardiovascular Disease Present" if pred == 1 else "Healthy (Absence of CVD)",
        "cvd_probability": prob_cvd,
        "risk_percentage": round(prob_cvd * 100, 2),
        "risk_tier": risk_tier,
        "clinical_recommendation": recommendation
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cardiovascular Disease Risk Prediction Engine")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt to input patient metrics")
    parser.add_argument("--age", type=float, help="Age in years (e.g. 52)")
    parser.add_argument("--gender", type=int, choices=[1, 2], help="1: Female, 2: Male")
    parser.add_argument("--height", type=float, help="Height in cm (e.g. 170)")
    parser.add_argument("--weight", type=float, help="Weight in kg (e.g. 78.5)")
    parser.add_argument("--ap_hi", type=float, help="Systolic blood pressure (e.g. 135)")
    parser.add_argument("--ap_lo", type=float, help="Diastolic blood pressure (e.g. 85)")
    parser.add_argument("--cholesterol", type=int, choices=[1, 2, 3], help="1: Normal, 2: Above Normal, 3: Well Above")
    parser.add_argument("--gluc", type=int, choices=[1, 2, 3], default=1, help="1: Normal, 2: Above Normal, 3: Well Above")
    parser.add_argument("--smoke", type=int, choices=[0, 1], default=0, help="0: Non-smoker, 1: Smoker")
    parser.add_argument("--alco", type=int, choices=[0, 1], default=0, help="0: Non-drinker, 1: Drinker")
    parser.add_argument("--active", type=int, choices=[0, 1], default=1, help="0: Inactive, 1: Active")

    args = parser.parse_args()

    if args.interactive:
        print("\n" + "=" * 60)
        print("INTERACTIVE PATIENT CARDIOVASCULAR RISK EVALUATION")
        print("=" * 60)
        try:
            age = float(input("Enter Age (years): "))
            gender = int(input("Enter Gender (1 for Female, 2 for Male): "))
            height = float(input("Enter Height (cm): "))
            weight = float(input("Enter Weight (kg): "))
            ap_hi = float(input("Enter Systolic BP (mmHg, e.g. 120): "))
            ap_lo = float(input("Enter Diastolic BP (mmHg, e.g. 80): "))
            chol = int(input("Enter Cholesterol (1: Normal, 2: Above Normal, 3: High): "))
            gluc = int(input("Enter Glucose (1: Normal, 2: Above Normal, 3: High) [Default 1]: ") or 1)
            smoke = int(input("Smoker? (0: No, 1: Yes) [Default 0]: ") or 0)
            alco = int(input("Alcohol intake? (0: No, 1: Yes) [Default 0]: ") or 0)
            active = int(input("Physically active? (0: No, 1: Yes) [Default 1]: ") or 1)

            patient_data = {
                "age": age, "gender": gender, "height": height, "weight": weight,
                "ap_hi": ap_hi, "ap_lo": ap_lo, "cholesterol": chol, "gluc": gluc,
                "smoke": smoke, "alco": alco, "active": active
            }
            res = predict_cardio_risk(patient_data)
            print("\n" + "-" * 50)
            print(f"DIAGNOSIS:       {res['diagnosis']}")
            print(f"CVD RISK:        {res['risk_percentage']}% ({res['risk_tier']})")
            print(f"RECOMMENDATION:  {res['clinical_recommendation']}")
            print("-" * 50)
        except Exception as e:
            print(f"Error during interactive input: {e}")

    elif args.age is not None and args.ap_hi is not None:
        patient_data = {
            "age": args.age,
            "gender": args.gender or 1,
            "height": args.height or 165.0,
            "weight": args.weight or 70.0,
            "ap_hi": args.ap_hi,
            "ap_lo": args.ap_lo or 80.0,
            "cholesterol": args.cholesterol or 1,
            "gluc": args.gluc,
            "smoke": args.smoke,
            "alco": args.alco,
            "active": args.active
        }
        res = predict_cardio_risk(patient_data)
        print("\n" + "=" * 60)
        print("CUSTOM PATIENT RISK ASSESSMENT")
        print("=" * 60)
        print(f"Diagnosis:        {res['diagnosis']}")
        print(f"CVD Probability:  {res['risk_percentage']}% ({res['risk_tier']})")
        print(f"Recommendation:   {res['clinical_recommendation']}")

    else:
        # Default side-by-side demonstration
        print("\n" + "=" * 65)
        print("CARDIOVASCULAR PREDICTOR: DEMONSTRATION INFERENCE")
        print("=" * 65)

        healthy_patient = {
            "age": 32, "gender": 1, "height": 165, "weight": 58.0,
            "ap_hi": 115, "ap_lo": 75, "cholesterol": 1, "gluc": 1,
            "smoke": 0, "alco": 0, "active": 1
        }
        high_risk_patient = {
            "age": 58, "gender": 2, "height": 175, "weight": 92.0,
            "ap_hi": 150, "ap_lo": 95, "cholesterol": 3, "gluc": 2,
            "smoke": 1, "alco": 1, "active": 0
        }

        res_healthy = predict_cardio_risk(healthy_patient)
        res_risk = predict_cardio_risk(high_risk_patient)

        print("\n[Patient A: 32yo, Normal BP 115/75, Normal Chol, Active]")
        print(f"  Result:      {res_healthy['diagnosis']}")
        print(f"  Risk Score:  {res_healthy['risk_percentage']}% -> {res_healthy['risk_tier']}")
        print(f"  Action:      {res_healthy['clinical_recommendation']}")

        print("\n[Patient B: 58yo, High BP 150/95, High Chol, Smoker, Sedentary]")
        print(f"  Result:      {res_risk['diagnosis']}")
        print(f"  Risk Score:  {res_risk['risk_percentage']}% -> {res_risk['risk_tier']}")
        print(f"  Action:      {res_risk['clinical_recommendation']}")
        print("=" * 65)
