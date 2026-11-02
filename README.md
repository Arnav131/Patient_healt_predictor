<div align="center">

# 🫀 CardioPulse AI
### Production Clinical Cardiovascular Disease Risk Prediction System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-EB5424.svg?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![Render](https://img.shields.io/badge/Deploy%20to-Render-46E3B7.svg?logo=render&logoColor=white)](https://render.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*An end-to-end, zero-leakage machine learning web application that combines regularized gradient boosting with dynamic hemodynamic feature engineering and automated laboratory PDF report parsing.*

[Explore Architecture](#-system-architecture) • [Live Features](#-key-features) • [ML Tournament](#-machine-learning-tournament) • [Quickstart](#-quickstart-guide) • [API Reference](#-rest-api-reference) • [Deployment](#-cloud-deployment)

</div>

---

## 📋 Overview

**CardioPulse AI** is an enterprise-grade clinical decision support platform designed to assist healthcare professionals in screening and stratifying cardiovascular disease (CVD) risk. 

Trained on **68,700 validated clinical patient records** with rigorous 5-fold cross-validation, the application combines a tournament-winning regularized **XGBoost** pipeline with an editorial, scroll-driven user interface featuring triple-theme adaptability, real-time hemodynamic calculation, and automated lab report ingestion.

---

## ✨ Key Features

- **Multi-Modal Diagnostic Input Console:**
  - **Direct Patient Biometrics:** Synchronized continuous sliders and precision numeric steppers with instant body mass index (BMI), pulse pressure ($PP$), and Mean Arterial Pressure ($MAP$) badges.
  - **Automated Medical Lab Report Parser:** Drag-and-drop ingestion of clinical PDF panels and laboratory summaries. Uses PyMuPDF and regex-based clinical entity recognition to parse 11 biomarkers automatically.
  - **Human-in-the-Loop Review:** Dedicated fine-tuning workflow allowing physicians to inspect, verify, or adjust extracted biomarkers before triggering model inference.
  - **One-Click Patient Reset:** Complete console and results clearing without requiring a full page reload.
  - **Clinical Archetype Presets:** Instant loading of reference clinical profiles (Young Athlete, Pre-hypertensive, Critical Metabolic Syndrome).

- **Dynamic Editorial Visual Interface:**
  - Modern typography pairing **Playfair Display** (editorial headings) with **Inter** (clean clinical legibility).
  - **Triple-Theme System:**
    1. *Pristine Clinical Light (Default):* Clean, crisp medical white with emerald accents.
    2. *Warm Serene Linen:* Soft parchment aesthetic reducing eye fatigue during extended clinical shifts.
    3. *Slate Charcoal Dark:* Balanced low-glare dark palette blending seamlessly across modern OLED displays.
  - Interactive SVG circular risk dial with dynamic color transitions (Emerald $\rightarrow$ Amber $\rightarrow$ Orange $\rightarrow$ Crimson).
  - Print-optimized stylesheet for one-click clinical report export.

---

## 🏆 Machine Learning Tournament

To guarantee model superiority without assumption, 5 distinct machine learning paradigms were benchmarked on **54,960 training records** across identical **Stratified 5-Fold Cross-Validation** folds with strict zero-leakage preprocessing.

### Contender Leaderboard

| Rank | Contender Algorithm | 5-Fold CV Accuracy | 5-Fold ROC-AUC | F1-Score | Brier Loss | Inference Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **XGBoost (Champion)** | **73.57%** $\pm$ 0.32% | **0.8008** $\pm$ 0.0031 | **0.7226** | **0.1834** | **2.8 ms** |
| 🥈 | **LightGBM** | 73.49% $\pm$ 0.36% | 0.8003 $\pm$ 0.0033 | 0.7214 | 0.1839 | 1.9 ms |
| 🥉 | **Random Forest** | 73.42% $\pm$ 0.40% | 0.7990 $\pm$ 0.0029 | 0.7208 | 0.1848 | 8.2 ms |
| 4 | **Extra Trees** | 73.08% $\pm$ 0.35% | 0.7971 $\pm$ 0.0028 | 0.7055 | 0.1892 | 7.5 ms |
| 5 | **Logistic Regression** | 73.01% $\pm$ 0.44% | 0.7952 $\pm$ 0.0030 | 0.7095 | 0.1873 | 4.1 ms |

### Champion Model Specification & Final Holdout ($N = 13,740$)
- **Algorithm:** Regularized eXtreme Gradient Boosting (`XGBClassifier`)
- **Hyperparameters:** `n_estimators=150`, `max_depth=4`, `learning_rate=0.05`, `reg_alpha=2.0` (L1), `reg_lambda=1.0` (L2), `gamma=0.1`, `subsample=0.7`
- **Holdout Test Accuracy:** **73.65%**
- **Holdout Test ROC-AUC:** **0.8038**
- **Holdout Precision:** **75.31%**
- **Holdout Recall:** **70.19%**

> **Note on Accuracy Limits:** Non-invasive routine checkup features without invasive coronary angiography or resting ECG waveforms possess an irreducible Bayes error rate ceiling of ~80–81% ROC-AUC (matching the clinical Framingham Risk Score standard). Any claims exceeding 85%+ on this dataset indicate data leakage or train-set evaluation.

---

## 🏛️ System Architecture

The codebase enforces a strict separation between **Production Web Deployment** and **Machine Learning Training & Research**:

```
health_care_predictor/
│
├── app.py                             # FastAPI production application backend
├── predict.py                         # Production model inference wrapper
├── Dockerfile                         # Containerization manifest
├── render.yaml                        # Infrastructure-as-Code blueprint for Render.com
├── requirements.txt                   # Production runtime dependencies
├── .dockerignore / .gitignore
├── README.md                          # Repository documentation
│
├── models/
│   └── best_cardio_model.joblib       # Serialized champion XGBoost pipeline (ROC-AUC: 0.8038)
│
├── templates/
│   └── index.html                     # Multi-theme clinical web application
│
├── static/
│   ├── favicon.svg                    # Cardiac brand icon
│   ├── css/ (style.css, animations.css)
│   ├── js/ (app.js)
│   └── samples/                       # Sample lab report PDFs for client testing
│
├── src/                               # Production Runtime Modules
│   ├── __init__.py
│   ├── config.py                      # Production path resolution & thresholds
│   ├── feature_engineering.py         # Pipeline-embedded transformer (BMI, MAP, PP, AHA)
│   └── report_parser.py               # PDF/Text clinical NLP extraction engine
│
├── MODEL_TRAINING_FILES/              # Standalone ML Training & Research Workbench
│   ├── dataset/
│   │   └── cardio_train.csv           # 70,000-record clinical dataset
│   ├── src/
│   │   ├── data_loader.py             # Semicolon ingestion, BP swap fix & clipping
│   │   ├── eda.py                     # Correlation matrices & distribution analysis
│   │   ├── preprocessor.py            # RobustScaler & OneHotEncoder column transformer
│   │   ├── model.py                   # Candidate model initialization
│   │   ├── tournament.py              # 5-fold cross-validated tournament engine
│   │   ├── tune.py                    # Hyperparameter grid search & fine-tuner
│   │   └── evaluate.py                # ROC/PR curves & confusion matrix plots
│   ├── run_pipeline.py                # Master retraining script
│   ├── create_sample_reports.py       # Sample PDF generator
│   └── training_procedure.md          # Step-by-step model training tutorial
│
├── docs/                              # Clinical Documentation & Testing Assets
│   ├── test_data.md                   # 10 clinical test scenarios & JSON payloads
│   ├── test_reports/                  # Master PDF lab reports (High risk & healthy)
│   ├── figures/                       # 8 publication-grade diagnostic plots
│   └── PROJECT_WORKFLOW_AND_ANALYSIS.md
│
└── tests/                             # Automated Verification Suite
    ├── test_app.py                    # FastAPI route & inference integration tests
    └── test_pipeline.py               # Model transformer & prediction tests
```

---

## 🚀 Quickstart Guide

### 1. Local Setup

Clone the repository and install requirements:

```bash
git clone https://github.com/<your-username>/cardiopulse-ai.git
cd cardiopulse-ai

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Application

Start the FastAPI application with Uvicorn:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to **`http://127.0.0.1:8000`**.

### 3. Run Automated Tests

Execute the complete 12-test unit and integration test suite:

```bash
python -m pytest tests/ -v
```

---

## 🐳 Docker Deployment

Build and run the production container locally:

```bash
# Build Docker image
docker build -t cardiopulse-ai:latest .

# Run container on port 8000
docker run -d -p 8000:8000 --name cardiopulse cardiopulse-ai:latest
```

Access the application at `http://localhost:8000`.

---

## ☁️ Cloud Deployment (Render.com)

The repository includes a production-ready [`render.yaml`](render.yaml) blueprint:

1. Push this repository to your GitHub account.
2. Sign in to [Render.com](https://render.com/).
3. Click **New +** $\rightarrow$ **Blueprint**.
4. Select your `cardiopulse-ai` repository.
5. Render will automatically detect `render.yaml`, install dependencies, run health checks, and deploy the service on a free web tier.

---

## 📡 REST API Reference

### 1. Predict Cardiovascular Risk
- **Endpoint:** `POST /api/predict`
- **Request Body:**
```json
{
  "age": 54,
  "gender": 2,
  "height": 174,
  "weight": 86.5,
  "ap_hi": 146,
  "ap_lo": 94,
  "cholesterol": 2,
  "gluc": 2,
  "smoke": 1,
  "alco": 0,
  "active": 0
}
```
- **Response (200 OK):**
```json
{
  "status": "success",
  "prediction": 1,
  "diagnosis": "Cardiovascular Disease Present",
  "risk_percentage": 83.36,
  "risk_tier": "Critical Risk",
  "biomarkers": {
    "bmi": 28.57,
    "bmi_category": "Overweight",
    "pulse_pressure": 52.0,
    "mean_arterial_pressure": 111.33,
    "aha_bp_stage": "Stage 2 Hypertension",
    "metabolic_score": 3
  },
  "recommendation": "Immediate cardiovascular clinical workup required..."
}
```

### 2. Extract Biomarkers from Lab Report
- **Endpoint:** `POST /api/extract-report`
- **Request:** `multipart/form-data` with `file=@report.pdf`
- **Response (200 OK):**
```json
{
  "status": "success",
  "filename": "sample_cardio_lab_report_high_risk.pdf",
  "detected_count": 11,
  "extracted_data": {
    "age": 54,
    "gender": 2,
    "height": 174.0,
    "weight": 86.5,
    "ap_hi": 146,
    "ap_lo": 94,
    "cholesterol": 2,
    "gluc": 2,
    "smoke": 1
  },
  "evidence": [
    "Blood Pressure found: 146/94 mmHg",
    "Cholesterol: 236.0 mg/dL (Above Normal)",
    "Glucose: 118.0 mg/dL (Above Normal)",
    "Age: 54 years"
  ]
}
```

### 3. Health Check
- **Endpoint:** `GET /api/health`
- **Response:** `{"status": "healthy", "model_loaded": true, "version": "2.0.0"}`

---

## 🔬 Retraining the Model

To retrain the model from scratch, perform feature engineering, run the 5-model tournament, and reproduce all diagnostic plots:

```bash
cd MODEL_TRAINING_FILES
python run_pipeline.py
```

For complete step-by-step instructions, parameters, and mathematics, consult [`MODEL_TRAINING_FILES/training_procedure.md`](MODEL_TRAINING_FILES/training_procedure.md).

For reference test cases across 10 distinct clinical profiles, consult [`docs/test_data.md`](docs/test_data.md).

---

## ⚖️ Clinical Advisory & Ethics Notice

**CardioPulse AI** is intended solely as an epidemiological screening decision-support aid and research instrument. It does **not** provide definitive clinical diagnoses, replace licensed physician consultation, or substitute for emergency medical care, physical examination, resting electrocardiography (ECG), or coronary angiography.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
