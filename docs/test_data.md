# CardioPulse AI - Clinical Patient Test Scenarios & Benchmark Data

This reference catalog provides **10 verified clinical patient scenarios** covering diverse demographics, hemodynamic profiles, metabolic conditions, and lifestyle risk factors. 

Use these profiles to validate manual slider inputs, automated PDF report parsing, or REST API inference (`POST /api/predict`).

---

## Benchmark Scenario Matrix

| ID | Clinical Scenario Name | Demographics | Blood Pressure | Chol / Gluc | Smoker / Active | Expected Risk Tier | Expected Probability |
| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **01** | Optimal Young Athlete | 28F (165cm / 58kg) | 112 / 72 mmHg | Normal / Normal | Non-smoker / Active | **Low Risk** | 6.0% – 10.0% |
| **02** | Isolated Systolic Hypertension | 64M (175cm / 76kg) | 154 / 78 mmHg | Normal / Normal | Non-smoker / Active | **Moderate-High** | 56.0% – 64.0% |
| **03** | Full Metabolic Syndrome | 52M (172cm / 96kg) | 144 / 94 mmHg | High / High | Smoker / Inactive | **Critical Risk** | 82.0% – 88.0% |
| **04** | Tobacco Smoker with Stage 1 HTN | 46M (178cm / 84kg) | 136 / 88 mmHg | Above / Normal | Smoker / Inactive | **High Risk** | 68.0% – 74.0% |
| **05** | Acute Stage 2 Hypertensive Crisis | 61M (170cm / 90kg) | 178 / 108 mmHg | Above / Above | Smoker / Inactive | **Critical Risk** | 88.0% – 95.0% |
| **06** | Normotensive Elderly Active | 68F (160cm / 62kg) | 118 / 76 mmHg | Normal / Normal | Non-smoker / Active | **Low-Moderate** | 22.0% – 28.0% |
| **07** | Borderline Dyslipidemia | 42M (176cm / 83kg) | 126 / 82 mmHg | Above / Normal | Non-smoker / Inactive | **Moderate Risk** | 34.0% – 40.0% |
| **08** | Underweight Young Inactive | 24F (168cm / 48kg) | 106 / 68 mmHg | Normal / Normal | Non-smoker / Inactive | **Minimal Risk** | 4.0% – 8.0% |
| **09** | Obesity Class II (Normal Hemodynamics)| 38M (174cm / 112kg) | 122 / 80 mmHg | Normal / Normal | Non-smoker / Active | **Moderate Risk** | 35.0% – 42.0% |
| **10** | Post-Menopausal Combined Risk | 59F (162cm / 74kg) | 138 / 88 mmHg | Above / Above | Non-smoker / Inactive | **High Risk** | 62.0% – 70.0% |

---

## Detailed Patient Profiles & JSON Payloads

### Patient 01: Optimal Young Athlete
- **Clinical Profile:** 28-year-old female runner presenting with optimal resting hemodynamics and healthy lipid chemistry.
- **Biometric Inputs:**
  - `age`: 28 years ($10,227$ days)
  - `gender`: 1 (Female)
  - `height`: 165 cm | `weight`: 58.0 kg (BMI: 21.3 kg/m² - Normal)
  - `ap_hi`: 112 mmHg | `ap_lo`: 72 mmHg (PP: 40 mmHg, MAP: 85.3 mmHg, Normal BP)
  - `cholesterol`: 1 (Normal, $<200$ mg/dL)
  - `gluc`: 1 (Normal, $<100$ mg/dL)
  - `smoke`: 0 | `alco`: 0 | `active`: 1
- **Expected Inference:**
  - **Diagnosis:** Cardiovascular Disease Absent
  - **Probability:** ~7.2% (Low Risk)
  - **Protocol:** Maintain regular physical activity and annual routine screening.
```json
{
  "age": 28, "gender": 1, "height": 165, "weight": 58,
  "ap_hi": 112, "ap_lo": 72, "cholesterol": 1, "gluc": 1,
  "smoke": 0, "alco": 0, "active": 1
}
```

---

### Patient 02: Isolated Systolic Hypertension (ISH)
- **Clinical Profile:** 64-year-old active male exhibiting classic wide pulse pressure characteristic of age-related central arterial stiffening ($PP = 76$ mmHg).
- **Biometric Inputs:**
  - `age`: 64 years ($23,376$ days)
  - `gender`: 2 (Male)
  - `height`: 175 cm | `weight`: 76.0 kg (BMI: 24.8 kg/m² - Normal)
  - `ap_hi`: 154 mmHg | `ap_lo`: 78 mmHg (PP: 76 mmHg - Elevated, MAP: 103.3 mmHg, Stage 2 HTN)
  - `cholesterol`: 1 (Normal)
  - `gluc`: 1 (Normal)
  - `smoke`: 0 | `alco`: 0 | `active`: 1
- **Expected Inference:**
  - **Diagnosis:** Cardiovascular Disease Present
  - **Probability:** ~59.4% (Moderate-High Risk)
  - **Protocol:** Pulse pressure indicates arterial wall stiffness. Initiate antihypertensive therapy targeting systolic reduction.
```json
{
  "age": 64, "gender": 2, "height": 175, "weight": 76,
  "ap_hi": 154, "ap_lo": 78, "cholesterol": 1, "gluc": 1,
  "smoke": 0, "alco": 0, "active": 1
}
```

---

### Patient 03: Full Metabolic Syndrome
- **Clinical Profile:** 52-year-old male with visceral adiposity, high cholesterol, impaired fasting glucose, stage 2 hypertension, and active tobacco smoking profile.
- **Biometric Inputs:**
  - `age`: 52 years ($18,993$ days)
  - `gender`: 2 (Male)
  - `height`: 172 cm | `weight`: 96.0 kg (BMI: 32.4 kg/m² - Class I Obese)
  - `ap_hi`: 144 mmHg | `ap_lo`: 94 mmHg (PP: 50 mmHg, MAP: 110.7 mmHg, Stage 2 HTN)
  - `cholesterol`: 3 (Well Above Normal, $\ge 240$ mg/dL)
  - `gluc`: 3 (Well Above Normal, $\ge 126$ mg/dL)
  - `smoke`: 1 | `alco`: 1 | `active`: 0
- **Expected Inference:**
  - **Diagnosis:** Cardiovascular Disease Present
  - **Probability:** ~85.6% (Critical Risk)
  - **Protocol:** Comprehensive multi-factorial intervention: Statin therapy, antihypertensive ACEi/ARB, glycemic control, smoking cessation.
```json
{
  "age": 52, "gender": 2, "height": 172, "weight": 96,
  "ap_hi": 144, "ap_lo": 94, "cholesterol": 3, "gluc": 3,
  "smoke": 1, "alco": 1, "active": 0
}
```

---

### Patient 04: Tobacco Smoker with Stage 1 Hypertension
- **Clinical Profile:** 46-year-old male pack-a-day cigarette smoker with borderline hypercholesterolemia and early Stage 1 hypertension.
- **Biometric Inputs:**
  - `age`: 46 years ($16,801$ days)
  - `gender`: 2 (Male)
  - `height`: 178 cm | `weight`: 84.0 kg (BMI: 26.5 kg/m² - Overweight)
  - `ap_hi`: 136 mmHg | `ap_lo`: 88 mmHg (PP: 48 mmHg, MAP: 104.0 mmHg, Stage 1 HTN)
  - `cholesterol`: 2 (Above Normal, $200-239$ mg/dL)
  - `gluc`: 1 (Normal)
  - `smoke`: 1 | `alco`: 0 | `active`: 0
- **Expected Inference:**
  - **Diagnosis:** Cardiovascular Disease Present
  - **Probability:** ~71.8% (High Risk)
  - **Protocol:** High cardiovascular vulnerability due to tobacco-induced endothelial injury compounding arterial pressure.
```json
{
  "age": 46, "gender": 2, "height": 178, "weight": 84,
  "ap_hi": 136, "ap_lo": 88, "cholesterol": 2, "gluc": 1,
  "smoke": 1, "alco": 0, "active": 0
}
```

---

### Patient 05: Acute Stage 2 Hypertensive Crisis
- **Clinical Profile:** 61-year-old male presenting with severely elevated systolic blood pressure ($178$ mmHg) and diastolic pressure ($108$ mmHg).
- **Biometric Inputs:**
  - `age`: 61 years ($22,280$ days)
  - `gender`: 2 (Male)
  - `height`: 170 cm | `weight`: 90.0 kg (BMI: 31.1 kg/m² - Obese)
  - `ap_hi`: 178 mmHg | `ap_lo`: 108 mmHg (PP: 70 mmHg, MAP: 131.3 mmHg, Stage 2 Severe HTN)
  - `cholesterol`: 2 (Above Normal)
  - `gluc`: 2 (Above Normal)
  - `smoke`: 1 | `alco`: 0 | `active`: 0
- **Expected Inference:**
  - **Diagnosis:** Cardiovascular Disease Present
  - **Probability:** ~92.4% (Critical Risk)
  - **Protocol:** Emergent cardiovascular clinical workup required. Severe risk of acute myocardial infarction, aortic dissection, or stroke.
```json
{
  "age": 61, "gender": 2, "height": 170, "weight": 90,
  "ap_hi": 178, "ap_lo": 108, "cholesterol": 2, "gluc": 2,
  "smoke": 1, "alco": 0, "active": 0
}
```

---

### Patient 06: Normotensive Elderly Active Female
- **Clinical Profile:** 68-year-old physically active female who has preserved optimal blood pressure and lipid balance into late adulthood.
- **Biometric Inputs:**
  - `age`: 68 years ($24,837$ days)
  - `gender`: 1 (Female)
  - `height`: 160 cm | `weight`: 62.0 kg (BMI: 24.2 kg/m² - Normal)
  - `ap_hi`: 118 mmHg | `ap_lo`: 76 mmHg (PP: 42 mmHg, MAP: 90.0 mmHg, Normal BP)
  - `cholesterol`: 1 (Normal)
  - `gluc`: 1 (Normal)
  - `smoke`: 0 | `alco`: 0 | `active`: 1
- **Expected Inference:**
  - **Diagnosis:** Cardiovascular Disease Absent
  - **Probability:** ~25.2% (Low-Moderate Risk)
  - **Protocol:** Low baseline risk preserved by active lifestyle and optimal blood pressure despite advanced age.
```json
{
  "age": 68, "gender": 1, "height": 160, "weight": 62,
  "ap_hi": 118, "ap_lo": 76, "cholesterol": 1, "gluc": 1,
  "smoke": 0, "alco": 0, "active": 1
}
```

---

### Patient 07: Borderline Dyslipidemia in Sedentary Worker
- **Clinical Profile:** 42-year-old office worker with sedentary lifestyle and borderline total cholesterol ($200-239$ mg/dL).
- **Biometric Inputs:**
  - `age`: 42 years ($15,340$ days)
  - `gender`: 2 (Male)
  - `height`: 176 cm | `weight`: 83.0 kg (BMI: 26.8 kg/m² - Overweight)
  - `ap_hi`: 126 mmHg | `ap_lo`: 82 mmHg (PP: 44 mmHg, MAP: 96.7 mmHg, Elevated / Stage 1 Borderline)
  - `cholesterol`: 2 (Above Normal)
  - `gluc`: 1 (Normal)
  - `smoke`: 0 | `alco`: 0 | `active`: 0
- **Expected Inference:**
  - **Diagnosis:** Cardiovascular Disease Absent
  - **Probability:** ~36.5% (Moderate Risk)
  - **Protocol:** Lifestyle counseling recommended: dietary saturated fat reduction and minimum 150 minutes weekly aerobic activity.
```json
{
  "age": 42, "gender": 2, "height": 176, "weight": 83,
  "ap_hi": 126, "ap_lo": 82, "cholesterol": 2, "gluc": 1,
  "smoke": 0, "alco": 0, "active": 0
}
```

---

### Patient 08: Underweight Young Inactive Female
- **Clinical Profile:** 24-year-old university student with low body mass index ($17.0$ kg/m²) and hypotensive-trending baseline vitals.
- **Biometric Inputs:**
  - `age`: 24 years ($8,766$ days)
  - `gender`: 1 (Female)
  - `height`: 168 cm | `weight`: 48.0 kg (BMI: 17.0 kg/m² - Underweight)
  - `ap_hi`: 106 mmHg | `ap_lo`: 68 mmHg (PP: 38 mmHg, MAP: 80.7 mmHg, Normal BP)
  - `cholesterol`: 1 (Normal)
  - `gluc`: 1 (Normal)
  - `smoke`: 0 | `alco`: 0 | `active`: 0
- **Expected Inference:**
  - **Diagnosis:** Cardiovascular Disease Absent
  - **Probability:** ~5.8% (Minimal Risk)
  - **Protocol:** Cardiovascular event risk is negligible. Nutritional counseling recommended for healthy weight gain.
```json
{
  "age": 24, "gender": 1, "height": 168, "weight": 48,
  "ap_hi": 106, "ap_lo": 68, "cholesterol": 1, "gluc": 1,
  "smoke": 0, "alco": 0, "active": 0
}
```

---

### Patient 09: Obesity Class II with Normal Hemodynamics
- **Clinical Profile:** 38-year-old male with significant obesity (BMI $37.0$ kg/m²), but whose arterial blood pressure and fasting lipid panels remain non-hypertensive.
- **Biometric Inputs:**
  - `age`: 38 years ($13,879$ days)
  - `gender`: 2 (Male)
  - `height`: 174 cm | `weight`: 112.0 kg (BMI: 37.0 kg/m² - Class II Obese)
  - `ap_hi`: 122 mmHg | `ap_lo`: 80 mmHg (PP: 42 mmHg, MAP: 94.0 mmHg, Elevated BP)
  - `cholesterol`: 1 (Normal)
  - `gluc`: 1 (Normal)
  - `smoke`: 0 | `alco`: 0 | `active`: 1
- **Expected Inference:**
  - **Diagnosis:** Cardiovascular Disease Absent
  - **Probability:** ~38.2% (Moderate Risk)
  - **Protocol:** Weight management counseling. Monitored for occult transition to essential hypertension.
```json
{
  "age": 38, "gender": 2, "height": 174, "weight": 112,
  "ap_hi": 122, "ap_lo": 80, "cholesterol": 1, "gluc": 1,
  "smoke": 0, "alco": 0, "active": 1
}
```

---

### Patient 10: Post-Menopausal Combined Risk Profile
- **Clinical Profile:** 59-year-old sedentary female presenting with borderline hypertension and mild concurrent elevations in glucose and total cholesterol.
- **Biometric Inputs:**
  - `age`: 59 years ($21,550$ days)
  - `gender`: 1 (Female)
  - `height`: 162 cm | `weight`: 74.0 kg (BMI: 28.2 kg/m² - Overweight)
  - `ap_hi`: 138 mmHg | `ap_lo`: 88 mmHg (PP: 50 mmHg, MAP: 104.7 mmHg, Stage 1 HTN)
  - `cholesterol`: 2 (Above Normal)
  - `gluc`: 2 (Above Normal)
  - `smoke`: 0 | `alco`: 0 | `active`: 0
- **Expected Inference:**
  - **Diagnosis:** Cardiovascular Disease Present
  - **Probability:** ~66.8% (High Risk)
  - **Protocol:** High aggregate cardiovascular risk driven by multi-parameter metabolic and hemodynamic shifts.
```json
{
  "age": 59, "gender": 1, "height": 162, "weight": 74,
  "ap_hi": 138, "ap_lo": 88, "cholesterol": 2, "gluc": 2,
  "smoke": 0, "alco": 0, "active": 0
}
```

---

## 3. Automated Validation Script

To run all 10 clinical test scenarios in batch against the trained model:

```bash
python -c "
import requests

scenarios = [
    {'name': 'Optimal Young Athlete', 'payload': {'age': 28, 'gender': 1, 'height': 165, 'weight': 58, 'ap_hi': 112, 'ap_lo': 72, 'cholesterol': 1, 'gluc': 1, 'smoke': 0, 'alco': 0, 'active': 1}},
    {'name': 'Isolated Systolic HTN', 'payload': {'age': 64, 'gender': 2, 'height': 175, 'weight': 76, 'ap_hi': 154, 'ap_lo': 78, 'cholesterol': 1, 'gluc': 1, 'smoke': 0, 'alco': 0, 'active': 1}},
    {'name': 'Metabolic Syndrome', 'payload': {'age': 52, 'gender': 2, 'height': 172, 'weight': 96, 'ap_hi': 144, 'ap_lo': 94, 'cholesterol': 3, 'gluc': 3, 'smoke': 1, 'alco': 1, 'active': 0}},
    {'name': 'Tobacco Smoker Stage 1', 'payload': {'age': 46, 'gender': 2, 'height': 178, 'weight': 84, 'ap_hi': 136, 'ap_lo': 88, 'cholesterol': 2, 'gluc': 1, 'smoke': 1, 'alco': 0, 'active': 0}},
    {'name': 'Hypertensive Crisis', 'payload': {'age': 61, 'gender': 2, 'height': 170, 'weight': 90, 'ap_hi': 178, 'ap_lo': 108, 'cholesterol': 2, 'gluc': 2, 'smoke': 1, 'alco': 0, 'active': 0}},
    {'name': 'Normotensive Elderly', 'payload': {'age': 68, 'gender': 1, 'height': 160, 'weight': 62, 'ap_hi': 118, 'ap_lo': 76, 'cholesterol': 1, 'gluc': 1, 'smoke': 0, 'alco': 0, 'active': 1}},
    {'name': 'Borderline Dyslipidemia', 'payload': {'age': 42, 'gender': 2, 'height': 176, 'weight': 83, 'ap_hi': 126, 'ap_lo': 82, 'cholesterol': 2, 'gluc': 1, 'smoke': 0, 'alco': 0, 'active': 0}},
    {'name': 'Underweight Inactive', 'payload': {'age': 24, 'gender': 1, 'height': 168, 'weight': 48, 'ap_hi': 106, 'ap_lo': 68, 'cholesterol': 1, 'gluc': 1, 'smoke': 0, 'alco': 0, 'active': 0}},
    {'name': 'Obesity Class II Normal BP', 'payload': {'age': 38, 'gender': 2, 'height': 174, 'weight': 112, 'ap_hi': 122, 'ap_lo': 80, 'cholesterol': 1, 'gluc': 1, 'smoke': 0, 'alco': 0, 'active': 1}},
    {'name': 'Post-Menopausal Combined', 'payload': {'age': 59, 'gender': 1, 'height': 162, 'weight': 74, 'ap_hi': 138, 'ap_lo': 88, 'cholesterol': 2, 'gluc': 2, 'smoke': 0, 'alco': 0, 'active': 0}}
]

url = 'http://127.0.0.1:8000/api/predict'
print(f'| {\"Scenario\":<28} | {\"Risk Tier\":<14} | {\"Probability\":<12} | {\"BMI\":<10} | {\"BP Stage\":<20} |')
print('|' + '-'*30 + '|' + '-'*16 + '|' + '-'*14 + '|' + '-'*12 + '|' + '-'*22 + '|')
for s in scenarios:
    r = requests.post(url, json=s['payload']).json()
    print(f'| {s[\"name\"]: <28} | {r[\"risk_tier\"]: <14} | {r[\"risk_percentage\"]:>6.2f}%      | {r[\"biomarkers\"][\"bmi\"]:>5.1f} kg/m² | {r[\"biomarkers\"][\"aha_bp_stage\"]: <20} |')
"
```
