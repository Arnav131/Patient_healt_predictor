"""
Integration Tests for CardioPulse AI Web Application & Endpoints
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_home_page_endpoint():
    """Confirms homepage renders successfully with editorial title."""
    response = client.get("/")
    assert response.status_code == 200
    assert "CardioPulse" in response.text
    assert "Direct Patient Biometrics" in response.text


def test_api_health_endpoint():
    """Confirms service health check."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_api_presets_endpoint():
    """Confirms clinical presets are served."""
    response = client.get("/api/presets")
    assert response.status_code == 200
    data = response.json()
    assert "healthy_young" in data
    assert "critical_metabolic" in data


def test_api_predict_healthy_patient():
    """Tests risk prediction on a healthy clinical profile."""
    payload = {
        "age": 30.0,
        "gender": 1,
        "height": 165.0,
        "weight": 58.0,
        "ap_hi": 115.0,
        "ap_lo": 75.0,
        "cholesterol": 1,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["prediction"] == 0
    assert data["risk_tier"] == "Low Risk"
    assert "biomarkers" in data
    assert data["biomarkers"]["aha_bp_stage"] == "Normal Blood Pressure"


def test_api_predict_critical_patient():
    """Tests risk prediction on a high-risk clinical profile."""
    payload = {
        "age": 58.0,
        "gender": 2,
        "height": 172.0,
        "weight": 96.0,
        "ap_hi": 155.0,
        "ap_lo": 98.0,
        "cholesterol": 3,
        "gluc": 2,
        "smoke": 1,
        "alco": 1,
        "active": 0
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["prediction"] == 1
    assert data["risk_tier"] in ["High Risk", "Critical Risk"]
    assert data["biomarkers"]["metabolic_score"] >= 3


def test_api_extract_report():
    """Tests medical report extraction endpoint with synthetic lab report text."""
    report_text = """
    CENTRAL PATHOLOGY REPORT
    Age: 52 Years
    Gender: Male
    BP: 144/92 mmHg
    Total Cholesterol: 235 mg/dL
    Glucose: 118 mg/dL
    Weight: 84 kg
    Height: 175 cm
    Non-smoker
    """
    files = {"file": ("lab_report.txt", report_text.encode("utf-8"), "text/plain")}
    response = client.post("/api/extract-report", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["detected_count"] >= 5
    assert data["extracted_data"]["ap_hi"] == 144
    assert data["extracted_data"]["ap_lo"] == 92
    assert data["extracted_data"]["cholesterol"] == 2
