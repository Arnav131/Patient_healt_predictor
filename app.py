"""
CardioPulse AI - FastAPI Production Web Application Backend
Serves interactive clinical web dashboard, REST API inference,
and medical lab report extraction.
"""

import sys
from pathlib import Path
from typing import Optional, Dict, Any
import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predict import predict_cardio_risk
from src.report_parser import MedicalReportParser

app = FastAPI(
    title="CardioPulse AI - Clinical Cardiovascular Predictor",
    description="Production-grade clinical decision support web application.",
    version="2.0.0"
)

# Static & Templates setup
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))


class PatientInput(BaseModel):
    age: float = Field(..., ge=10, le=120, description="Age in years")
    gender: int = Field(..., ge=1, le=2, description="1: Female, 2: Male")
    height: float = Field(..., ge=100, le=250, description="Height in cm")
    weight: float = Field(..., ge=30, le=250, description="Weight in kg")
    ap_hi: float = Field(..., ge=60, le=260, description="Systolic blood pressure (mmHg)")
    ap_lo: float = Field(..., ge=30, le=180, description="Diastolic blood pressure (mmHg)")
    cholesterol: int = Field(1, ge=1, le=3, description="1: Normal, 2: Above Normal, 3: High")
    gluc: int = Field(1, ge=1, le=3, description="1: Normal, 2: Above Normal, 3: High")
    smoke: int = Field(0, ge=0, le=1, description="0: Non-smoker, 1: Smoker")
    alco: int = Field(0, ge=0, le=1, description="0: Non-drinker, 1: Drinker")
    active: int = Field(1, ge=0, le=1, description="0: Inactive, 1: Active")


@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    """Renders the main CardioPulse AI web application."""
    return templates.TemplateResponse(request, "index.html")


@app.post("/api/predict")
async def api_predict(patient: PatientInput):
    """
    Evaluates patient cardiovascular risk and returns probability,
    classification, hemodynamic biomarkers, and clinical recommendations.
    """
    try:
        p_dict = patient.model_dump()
        res = predict_cardio_risk(p_dict)

        # Calculate hemodynamic metrics for UI display
        height_m = patient.height / 100.0
        bmi = round(patient.weight / (height_m ** 2), 2)
        if bmi < 18.5:
            bmi_cat = "Underweight"
        elif bmi < 25.0:
            bmi_cat = "Normal Weight"
        elif bmi < 30.0:
            bmi_cat = "Overweight"
        elif bmi < 35.0:
            bmi_cat = "Obese (Class 1)"
        else:
            bmi_cat = "Severely Obese (Class 2+)"

        hi = max(patient.ap_hi, patient.ap_lo)
        lo = min(patient.ap_hi, patient.ap_lo)
        pulse_pressure = int(hi - lo)
        map_val = round(lo + (pulse_pressure / 3.0), 1)

        # AHA BP Staging
        if hi > 180 or lo > 120:
            aha_stage = "Hypertensive Crisis"
            aha_desc = "Emergency medical evaluation indicated"
        elif hi >= 140 or lo >= 90:
            aha_stage = "Hypertension Stage 2"
            aha_desc = "High arterial pressure requiring therapeutic intervention"
        elif (130 <= hi <= 139) or (80 <= lo <= 89):
            aha_stage = "Hypertension Stage 1"
            aha_desc = "Mildly elevated pressure; lifestyle and dietary management advised"
        elif (120 <= hi <= 129) and (lo < 80):
            aha_stage = "Elevated Blood Pressure"
            aha_desc = "Prehypertensive trend"
        else:
            aha_stage = "Normal Blood Pressure"
            aha_desc = "Optimal cardiovascular hemodynamics"

        # Metabolic score
        metabolic_score = int((bmi >= 30.0) + (patient.cholesterol >= 2) + (patient.gluc >= 2) + (hi >= 140 or lo >= 90))

        return {
            "status": "success",
            "prediction": res["cvd_prediction"],
            "diagnosis": res["diagnosis"],
            "probability": res["cvd_probability"],
            "risk_percentage": res["risk_percentage"],
            "risk_tier": res["risk_tier"],
            "recommendation": res["clinical_recommendation"],
            "biomarkers": {
                "bmi": bmi,
                "bmi_category": bmi_cat,
                "pulse_pressure": pulse_pressure,
                "mean_arterial_pressure": map_val,
                "aha_bp_stage": aha_stage,
                "aha_description": aha_desc,
                "metabolic_score": metabolic_score
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/api/extract-report")
async def extract_medical_report(file: UploadFile = File(...)):
    """
    Ingests an uploaded laboratory report (PDF or text/image) and parses
    cardiovascular biomarkers to auto-fill the diagnostic console.
    """
    try:
        file_bytes = await file.read()
        raw_text = MedicalReportParser.extract_text_from_file(file_bytes, file.filename)
        parse_res = MedicalReportParser.parse_biomarkers(raw_text)

        return {
            "status": "success",
            "filename": file.filename,
            "detected_count": parse_res["detected_count"],
            "extracted_data": parse_res["biomarkers"],
            "evidence": parse_res["evidence_snippets"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse report: {str(e)}")


@app.get("/api/presets")
async def get_clinical_presets():
    """Returns archetypal clinical profiles for quick evaluation."""
    return {
        "healthy_young": {
            "name": "Healthy Young Adult",
            "description": "Age 28, Active, Normal BP (116/76), Normal Lipids",
            "data": {
                "age": 28, "gender": 1, "height": 165, "weight": 58,
                "ap_hi": 116, "ap_lo": 76, "cholesterol": 1, "gluc": 1,
                "smoke": 0, "alco": 0, "active": 1
            }
        },
        "prehypertensive_worker": {
            "name": "Prehypertensive Professional",
            "description": "Age 45, Sedentary, Overweight, Borderline BP (134/86)",
            "data": {
                "age": 45, "gender": 2, "height": 174, "weight": 82,
                "ap_hi": 134, "ap_lo": 86, "cholesterol": 2, "gluc": 1,
                "smoke": 0, "alco": 1, "active": 0
            }
        },
        "critical_metabolic": {
            "name": "Critical Metabolic Syndrome Patient",
            "description": "Age 58, Stage 2 Hypertension (152/96), Obese, High Cholesterol",
            "data": {
                "age": 58, "gender": 2, "height": 172, "weight": 96,
                "ap_hi": 152, "ap_lo": 96, "cholesterol": 3, "gluc": 2,
                "smoke": 1, "alco": 0, "active": 0
            }
        }
    }


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "service": "CardioPulse AI Engine"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
