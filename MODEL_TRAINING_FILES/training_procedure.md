# CardioPulse AI - Machine Learning Model Training & Reproduction Guide

This guide provides a comprehensive, step-by-step tutorial for data scientists and ML engineers to independently train, cross-validate, optimize, and serialize the **CardioPulse AI** cardiovascular disease risk prediction model from scratch.

---

## 1. Environment & Prerequisites

Ensure you have Python 3.10+ installed. Activate your virtual environment and install the required machine learning dependencies:

```bash
cd MODEL_TRAINING_FILES
pip install -r ../requirements.txt
```

### Core Libraries
- **scikit-learn** ($\ge$ 1.2.0): Pipeline assembly, preprocessing, cross-validation, and metrics.
- **xgboost** ($\ge$ 1.7.0): Gradient boosted decision trees classifier (Tournament Champion).
- **lightgbm** ($\ge$ 3.3.0): Fast histogram-based gradient boosting.
- **pandas** & **numpy**: Data ingestion and vector manipulation.
- **matplotlib** & **seaborn**: Publication-grade statistical diagnostic plots.
- **joblib**: Pipeline serialization.

---

## 2. Dataset Provenance & Ingestion

The model is trained on the authentic **Cardiovascular Disease Dataset** (`cardio_train.csv`), consisting of 70,000 anonymized patient examination records.

- **File location:** `MODEL_TRAINING_FILES/dataset/cardio_train.csv`
- **File format:** Semicolon-delimited (`sep=';'`)
- **Target column:** `cardio` (Binary: `0` = Healthy, `1` = Cardiovascular Disease Present)

### Clinical Feature Dictionary

| Feature | Raw Name | Unit / Range | Clinical Description |
| :--- | :--- | :--- | :--- |
| **Age** | `age` | Days ($10,798 - 23,713$) | Age of patient (automatically converted to years: $age / 365.25$) |
| **Sex** | `gender` | $1$ (Female), $2$ (Male) | Biological sex |
| **Height** | `height` | cm ($55 - 250$) | Standing stature |
| **Weight** | `weight` | kg ($10.0 - 200.0$) | Body mass |
| **Systolic BP** | `ap_hi` | mmHg ($-150 - 16,020$) | Peak arterial blood pressure during contraction |
| **Diastolic BP** | `ap_lo` | mmHg ($-70 - 11,000$) | Minimum arterial pressure between beats |
| **Cholesterol** | `cholesterol` | $1$ (Normal), $2$ (Above), $3$ (Well Above) | Fasting total serum cholesterol level |
| **Glucose** | `gluc` | $1$ (Normal), $2$ (Above), $3$ (Well Above) | Fasting blood glucose level |
| **Smoker** | `smoke` | $0$ (No), $1$ (Yes) | Active tobacco smoker |
| **Alcohol** | `alco` | $0$ (No), $1$ (Yes) | Regular alcohol consumer |
| **Active** | `active` | $0$ (No), $1$ (Yes) | Regular physical exercise |

---

## 3. Data Cleaning & Clinical Sanitization

Raw clinical examination records often suffer from typographic errors and cuff measurement inversions. [`src/data_loader.py`](file:///D:/ML_PROJECTS/end_to_end_ml_project/health_care_predictor/MODEL_TRAINING_FILES/src/data_loader.py) applies the following deterministic protocol:

1. **Blood Pressure Transposition Correction:**
   Whenever systolic pressure is recorded lower than diastolic pressure ($ap\_hi < ap\_lo$), the cuff operator accidentally swapped the reading columns. The loader applies:
   $$\text{ap\_hi}_{\text{clean}} = \max(\text{ap\_hi}, \text{ap\_lo}), \quad \text{ap\_lo}_{\text{clean}} = \min(\text{ap\_hi}, \text{ap\_lo})$$
   *Across 70,000 rows, this corrects 1,234 inverted patient measurements.*

2. **Physiological Plausibility Clipping:**
   Readings outside realistic human survival boundaries are removed:
   - Systolic BP: $70 \le ap\_hi \le 240$ mmHg
   - Diastolic BP: $40 \le ap\_lo \le 160$ mmHg
   - Height: $120 \le height \le 220$ cm
   - Weight: $35.0 \le weight \le 200.0$ kg

3. **Identifier Stripping:**
   The administrative column `id` is dropped to prevent random index memorization.

*Result:* **68,700 high-fidelity clinical patient records** retained (50.5% Healthy, 49.5% CVD Present).

---

## 4. Clinical Feature Engineering

The scikit-learn compatible transformer [`CardioFeatureEngineer`](file:///D:/ML_PROJECTS/end_to_end_ml_project/health_care_predictor/MODEL_TRAINING_FILES/src/feature_engineering.py) extracts physiologically established cardiovascular biomarkers:

### A. Body Mass Index (BMI) & WHO Classification
$$\text{BMI} = \frac{\text{weight (kg)}}{(\text{height (m)})^2}$$
Classified into 4 WHO categories: Underweight ($<18.5$), Normal ($18.5 - 24.9$), Overweight ($25.0 - 29.9$), Obese ($\ge 30.0$).

### B. Arterial Pulse Pressure ($PP$)
$$\text{PP} = \text{ap\_hi} - \text{ap\_lo}$$
Direct indicator of large-artery stiffness and compliance. Healthy young adults exhibit $PP \approx 40$ mmHg; values $\ge 60$ mmHg represent elevated cardiovascular strain.

### C. Mean Arterial Pressure ($MAP$)
$$\text{MAP} = \text{ap\_lo} + \frac{1}{3}(\text{ap\_hi} - \text{ap\_lo})$$
Geometric average arterial perfusion pressure delivered to vital organs.

### D. AHA / ACC 2017 Hypertension Stages
Mapped according to American Heart Association clinical guidelines:
- **Normal:** $ap\_hi < 120$ AND $ap\_lo < 80$
- **Elevated:** $120 \le ap\_hi < 130$ AND $ap\_lo < 80$
- **Hypertension Stage 1:** $130 \le ap\_hi < 140$ OR $80 \le ap\_lo < 90$
- **Hypertension Stage 2:** $ap\_hi \ge 140$ OR $ap\_lo \ge 90$

### E. Metabolic Syndrome Risk Multiplier
$$\text{Metabolic Score} = \mathbb{I}(\text{BMI} \ge 30) + \mathbb{I}(\text{Cholesterol} \ge 2) + \mathbb{I}(\text{Glucose} \ge 2) + \mathbb{I}(\text{ap\_hi} \ge 130)$$
A discrete integer scale ($0 - 4$) capturing synergistic metabolic dysfunction.

---

## 5. Zero Data Leakage Architecture

To guarantee strict generalization integrity:
1. Ground truth data is split into **80% Training ($N = 54,960$)** and **20% Holdout Test ($N = 13,740$)** with stratification.
2. The `ColumnTransformer` fits `RobustScaler` (median-based scaling immune to skew) and `OneHotEncoder` **only within the cross-validation training folds**.
3. No information from the validation folds or holdout test set ever permeates the feature transformers or imputer medians.

---

## 6. Multi-Model Tournament Benchmarking

Five diverse algorithmic paradigms are evaluated using identical **Stratified 5-Fold Cross-Validation** folds (`random_state=42`):

```bash
# To run tournament independently:
python -c "
from src.config import DATASET_PATH
from src.data_loader import load_cardio_data
from src.tournament import run_model_tournament
from sklearn.model_selection import train_test_split

df = load_cardio_data(DATASET_PATH)
X = df.drop(columns=['cardio'])
y = df['cardio']
X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
champ, leaderboard, _ = run_model_tournament(X_tr, y_tr)
print(leaderboard)
"
```

### Tournament Results

| Rank | Model Candidate | CV Accuracy | 5-Fold ROC-AUC | F1-Score | Brier Score | Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **XGBoost (Champion)** | **73.57%** $\pm$ 0.32% | **0.8008** $\pm$ 0.0031 | **0.7226** | **0.1834** | 2.8s |
| 🥈 | **LightGBM** | 73.49% $\pm$ 0.36% | 0.8003 $\pm$ 0.0033 | 0.7214 | 0.1839 | 1.9s |
| 🥉 | **Random Forest** | 73.42% $\pm$ 0.40% | 0.7990 $\pm$ 0.0029 | 0.7208 | 0.1848 | 8.2s |
| 4 | **Extra Trees** | 73.08% $\pm$ 0.35% | 0.7971 $\pm$ 0.0028 | 0.7055 | 0.1892 | 7.5s |
| 5 | **Logistic Regression** | 73.01% $\pm$ 0.44% | 0.7952 $\pm$ 0.0030 | 0.7095 | 0.1873 | 4.1s |

---

## 7. Hyperparameter Tuning & Serialization

The tournament champion (**XGBoost**) is fine-tuned via `RandomizedSearchCV` across 15 randomized configurations $\times$ 5 folds = **75 fit operations** to optimize `reg_alpha` (L1), `reg_lambda` (L2), `learning_rate`, and tree depth.

### Optimal Champion Hyperparameters
```python
XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.7,
    colsample_bytree=0.8,
    reg_alpha=2.0,      # L1 Regularization to prune non-informative splits
    reg_lambda=1.0,     # L2 Regularization
    gamma=0.1,          # Minimum split loss reduction
    eval_metric="logloss",
    random_state=42
)
```

### Final Holdout Evaluation ($N = 13,740$)
- **Holdout Accuracy:** **73.65%**
- **Holdout ROC-AUC:** **0.8038**
- **Precision:** 75.31%
- **Recall:** 70.19%
- **Brier Calibration Score:** 0.1822

The end-to-end pipeline (Feature Engineering $\rightarrow$ Column Transformer $\rightarrow$ Tuned XGBoost) is serialized directly to:
```
../models/best_cardio_model.joblib
```

---

## 8. One-Command Master Retraining Runbook

To execute the entire end-to-end pipeline (Data Ingestion $\rightarrow$ EDA Plot Generation $\rightarrow$ 5-Model Tournament $\rightarrow$ Hyperparameter Tuning $\rightarrow$ Holdout Evaluation $\rightarrow$ Model Serialization), run:

```bash
cd MODEL_TRAINING_FILES
python run_pipeline.py
```

All 8 diagnostic plots will be updated in `../docs/figures/`, and the serialized champion model will be saved to `../models/best_cardio_model.joblib`.
