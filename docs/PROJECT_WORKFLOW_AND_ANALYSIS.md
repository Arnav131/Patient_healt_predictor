# Cardiovascular Disease Predictor: Production Machine Learning Architecture & Multi-Model Tournament

**Author**: Senior Statistical Machine Learning Data Scientist  
**Project Path**: `D:\ML_PROJECTS\end_to_end_ml_project\health_care_predictor`  
**Dataset**: `dataset/cardio_train.csv` (70,000 real medical examination records)  
**Target Variable**: `cardio` (0 = Absence of Cardiovascular Disease, 1 = Presence of Cardiovascular Disease)  
**Status**: Multi-Model Architecture & Tournament Plan (Restarted from Scratch)

---

## 1. Executive Summary & Clinical Grounding

The goal is to design, benchmark, optimize, and serialize a production-grade machine learning model for predicting **Cardiovascular Disease (CVD)** using the newly integrated real-world dataset (`cardio_train.csv`, 70,000 patient records).

### Dataset Overview & Statistical Properties
Unlike synthetic demonstration datasets, `cardio_train.csv` represents **authentic physiological and diagnostic clinical examinations**:
- **Sample Size**: 70,000 patients.
- **Class Balance**: 35,021 (50.03%) negatives and 34,979 (49.97%) positives — a perfectly balanced binary classification problem.
- **Biomarkers**:
  - `age`: Patient age recorded in days (ranges 10,798 to 23,713 days $\approx$ 29.5 to 65.0 years).
  - `gender`: 1 = Female, 2 = Male.
  - `height`: Body height in cm (mean 164.4 cm).
  - `weight`: Body weight in kg (mean 74.2 kg).
  - `ap_hi`: Systolic blood pressure (mmHg).
  - `ap_lo`: Diastolic blood pressure (mmHg).
  - `cholesterol`: 1: Normal, 2: Above Normal, 3: Well Above Normal.
  - `gluc`: 1: Normal, 2: Above Normal, 3: Well Above Normal.
  - `smoke`: 0 = Non-smoker, 1 = Smoker.
  - `alco`: 0 = Non-drinker, 1 = Alcohol consumer.
  - `active`: 0 = Inactive, 1 = Physically active.
  - `cardio`: Target outcome (0 = healthy, 1 = cardiovascular disease diagnosed).

### Real Clinical Data Quality Audit & Physiological Cleaning Rules
Because this dataset originates from clinical EHR intake logs, real-world data entry anomalies exist:
1. **Inverted Blood Pressure Readings**: In 1,234 records, recorded diastolic pressure (`ap_lo`) is higher than systolic pressure (`ap_hi`). In human physiology, systolic pressure is always higher than diastolic. We apply automated clinical transposition (`ap_hi`, `ap_lo` = `max(ap_hi, ap_lo)`, `min(ap_hi, ap_lo)`).
2. **Extreme Blood Pressure Typos**:
   - `ap_hi` contains values up to 16,020 and negative values down to -150 (caused by extra zeros or negative entry codes).
   - Valid physiological bounds: $70 \le ap\_hi \le 240$ mmHg.
   - `ap_lo` contains values up to 11,000 and values below 30.
   - Valid physiological bounds: $40 \le ap\_lo \le 160$ mmHg.
   - Values outside bounds will be cleaned or filtered using domain bounding rules.
3. **Anthropometric Outlier Bounds**:
   - Height: $120 \le height \le 220$ cm.
   - Weight: $35 \le weight \le 200$ kg.
4. **Identifier Column (`id`)**: Dropped immediately to prevent data leakage and index memorization.

---

## 2. Advanced Clinical Feature Engineering

To give tree-based and linear models the strongest predictive representations, the pipeline extracts key cardiovascular risk indexes:
1. **Age in Years**: $Age_{years} = Age_{days} / 365.25$.
2. **Body Mass Index (BMI)**:
   $$BMI = \frac{weight\,(kg)}{(height\,(m))^2}$$
   Categorized into: Underweight ($<18.5$), Normal ($18.5 - 24.9$), Overweight ($25 - 29.9$), Obese Class 1 ($30 - 34.9$), and Severe Obesity ($\ge 35$).
3. **Pulse Pressure ($PP$)**:
   $$PP = ap\_hi - ap\_lo$$
   An established clinical biomarker for arterial stiffness, vascular aging, and cardiovascular event risk.
4. **Mean Arterial Pressure ($MAP$)**:
   $$MAP = ap\_lo + \frac{1}{3}(ap\_hi - ap\_lo)$$
   Reflects tissue perfusion and overall hemodynamic vascular strain.
5. **American Heart Association (AHA) Blood Pressure Stage**:
   - Normal: $ap\_hi < 120 \land ap\_lo < 80$
   - Elevated: $120 \le ap\_hi \le 129 \land ap\_lo < 80$
   - Hypertension Stage 1: $130 \le ap\_hi \le 139 \lor 80 \le ap\_lo \le 89$
   - Hypertension Stage 2: $ap\_hi \ge 140 \lor ap\_lo \ge 90$
   - Hypertensive Crisis: $ap\_hi > 180 \lor ap\_lo > 120$
6. **Metabolic Syndrome Composite Score**:
   Combines elevated BMI ($\ge 30$), elevated cholesterol ($>1$), elevated glucose ($>1$), and hypertensive status.
7. **Lifestyle Cardiovascular Hazard Index**:
   Composite metric combining smoking, alcohol use, and sedentary lifestyle weighted against age.

---

## 3. Multi-Model Tournament Strategy

Per user instructions, we will benchmark multiple distinct machine learning algorithms using **5-Fold Stratified Cross-Validation** on identical training folds to determine which algorithm has the strongest baseline:

```
                            [ Cleaned Clinical Dataset (70,000 samples) ]
                                                   │
                                                   ▼
                                [ 80/20 Stratified Train-Test Split ]
                                  (Train: 56,000  |  Test: 14,000)
                                                   │
                                                   ▼
                             [ 5-Fold Stratified Cross-Validation ]
                                                   │
                  ┌─────────────────┬──────────────┴───────┬─────────────────┬──────────────────┐
                  ▼                 ▼                      ▼                 ▼                  ▼
          [ Model 1: RF ]   [ Model 2: LightGBM ]   [ Model 3: XGBoost ] [ Model 4: ExtraTrees ] [ Model 5: LogReg ]
          Random Forest     Histogram GBDT          Gradient Boosting    Extremely Random Trees  L2 Regularized
          100 Trees         Leaf-wise Split         Depth-wise Split     Ensemble                Linear Baseline
                  │                 │                      │                 │                  │
                  └─────────────────┴──────────────┬───────┴─────────────────┴──────────────────┘
                                                   │
                                                   ▼
                             [ Benchmark Evaluation & Ranking Matrix ]
                               Metrics: CV Accuracy, ROC-AUC, Macro-F1,
                               Training Latency, Inference Speed
                                                   │
                                                   ▼
                                [ Select Champion Algorithm Family ]
                                                   │
                                                   ▼
                           [ Deep Regularization & Hyperparameter Tuning ]
                             • Stratified RandomizedSearchCV & GridSearchCV
                             • Tree pruning (max_depth, min_child_samples)
                             • Learning rate & regularizers (L1/L2)
                                                   │
                                                   ▼
                         [ Final Model Evaluation & Production Packaging ]
                           • Evaluate on Unseen Holdout Test Set (14,000)
                           • Plot Confusion Matrix, ROC-AUC, PR Curves
                           • Serialize Pipeline (`models/best_cardio_model.joblib`)
                           • Standalone Inference API (`predict.py`)
```

### The Contenders:
1. **Random Forest Classifier**:
   - Bagging with 100 decorrelated decision trees, bootstrapping with replacement, OOB score estimation.
2. **LightGBM Classifier**:
   - Highly efficient leaf-wise histogram gradient boosting with gradient-based one-side sampling (GOSS).
3. **XGBoost Classifier**:
   - Depth-wise tree boosting with exact/hist split finding and L1/L2 leaf weight regularization.
4. **Extra Trees Classifier**:
   - Extremely randomized trees ensemble (cuts splits completely at random to minimize variance).
5. **Logistic Regression**:
   - Scaled, L2-regularized generalized linear model serving as the interpretable clinical baseline.

---

## 4. Evaluation & Diagnostic Protocol
- **Cross-Validation**: 5-Fold Stratified CV preserving class balance across all splits.
- **Metrics Tracked**:
  - Classification Accuracy
  - Balanced Accuracy
  - ROC-AUC (Area Under Receiver Operating Characteristic Curve)
  - Precision, Recall, and Macro-F1
  - Cohen's Kappa
  - Confusion Matrix (Counts & True-Positive Rates)
  - Feature Importance (Gini Impurity vs Gain vs Permutation)
- **Zero-Leakage Assurance**:
  - `CardioFeatureEngineer` and `ColumnTransformer` are fitted strictly within training splits.
  - Test set of 14,000 records remains completely locked until the champion model is tuned.

---

## 5. Clean Slate Migration (Deleting Old Model)
To ensure zero baggage from the previous synthetic dataset, the pipeline will:
1. Clear the old model `models/best_random_forest_pipeline.joblib`.
2. Clear old synthetic figures in `docs/figures/`.
3. Rewrite `src/` modules around `cardio_train.csv`.
4. Update unit tests in `tests/test_pipeline.py`.
5. Provide a production-grade inference engine `predict.py` accepting real clinical metrics (Blood Pressure, Cholesterol, Glucose, BMI, Lifestyle).


---

## 5. Empirical Results & Multi-Model Tournament Rankings

### Algorithm Tournament Leaderboard (5-Fold Stratified Cross-Validation on 54,960 Patients)

| Algorithm | Mean CV Accuracy | Mean CV ROC-AUC | Mean CV F1 | Mean Train Acc | Overfitting Gap | CV Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | 0.7357 (+/- 0.0045) | 0.8008 (+/- 0.0043) | 0.7210 | 0.7455 | 0.0098 | 7.1s |
| **LightGBM** | 0.7349 (+/- 0.0045) | 0.8003 (+/- 0.0041) | 0.7202 | 0.7482 | 0.0133 | 9.2s |
| **RandomForest** | 0.7342 (+/- 0.0040) | 0.7990 (+/- 0.0044) | 0.7192 | 0.7670 | 0.0328 | 15.5s |
| **ExtraTrees** | 0.7308 (+/- 0.0035) | 0.7971 (+/- 0.0043) | 0.7055 | 0.7395 | 0.0087 | 7.5s |
| **LogisticRegression** | 0.7301 (+/- 0.0044) | 0.7952 (+/- 0.0048) | 0.7095 | 0.7309 | 0.0008 | 4.1s |

### Tournament Champion: **XGBoost**
- **Optimal Tuned Hyperparameters**:
```json
{'classifier__subsample': 0.7, 'classifier__reg_alpha': 2.0, 'classifier__n_estimators': 150, 'classifier__max_depth': 4, 'classifier__learning_rate': 0.05, 'classifier__gamma': 0.1, 'classifier__colsample_bytree': 1.0}
```
- **Tuning Cross-Validation ROC-AUC**: `0.8011`

---

## 6. Final Holdout Test Set Performance (13,740 Patients)

| Evaluation Metric | Test Set Score | Clinical Benchmark / Significance |
| :--- | :---: | :--- |
| **Accuracy** | **0.7357 (73.57%)** | High discriminative accuracy on real medical exam records |
| **Balanced Accuracy** | **0.7352** | Equal sensitivity for healthy vs diseased patients |
| **ROC-AUC** | **0.8038** | Area Under ROC Curve (Excellent clinical discrimination) |
| **PR-AUC (Avg Precision)** | **0.7877** | Robust precision-recall area above 0.50 baseline |
| **Precision** | **0.7556** | Positive predictive value |
| **Recall (Sensitivity)** | **0.6885** | Detection rate of patients with cardiovascular disease |
| **F1-Score** | **0.7205** | Harmonic mean of precision and recall |
| **Cohen's Kappa** | **0.4708** | High inter-rater agreement above chance |
| **Brier Score** | **0.1797** | Calibrated probabilistic loss |

---

## 7. Generated Visual Artifacts

The following high-resolution diagnostic figures are preserved in `docs/figures/`:
- **Pearson Correlation Heatmap**: `figures/pearson_correlation_heatmap.png`
- **Spearman Rank Correlation Heatmap**: `figures/spearman_correlation_heatmap.png`
- **Cramér's V Association Heatmap**: `figures/cramers_v_heatmap.png`
- **Biomarker Distributions by CVD**: `figures/target_and_feature_distributions.png`
- **Algorithm Tournament Leaderboard Chart**: `figures/tournament_benchmark.png`
- **Final Confusion Matrix**: `figures/confusion_matrix_cardio.png`
- **ROC and Precision-Recall Curves**: `figures/roc_pr_curves.png`
- **Top 15 Feature Importances**: `figures/feature_importance.png`

---

## 8. Production Inference API

The final trained pipeline is saved at:
`models/best_cardio_model.joblib`

Run inference on any patient:
```powershell
python D:\ML_PROJECTS\end_to_end_ml_project\health_care_predictor\predict.py
```
