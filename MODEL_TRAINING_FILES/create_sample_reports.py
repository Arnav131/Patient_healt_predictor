"""
Generates realistic sample clinical lab report PDFs for testing CardioPulse AI.
"""

import os
import fitz

OUT_DIR = r"D:\ML_PROJECTS\end_to_end_ml_project\health_care_predictor"


def create_report(filename, patient_info, vitals, labs, notes):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # Standard A4

    # Header banner
    page.draw_rect(fitz.Rect(40, 40, 555, 102), color=(0.01, 0.52, 0.78), fill=(0.01, 0.52, 0.78))
    page.insert_text((55, 70), "METROPOLITAN DIAGNOSTIC LABORATORIES", fontsize=15, fontname="helv", color=(1, 1, 1))
    page.insert_text((55, 88), "COMPREHENSIVE METABOLIC & CARDIOVASCULAR RISK PANEL", fontsize=9, fontname="helv", color=(0.88, 0.94, 1))

    # Patient info box
    page.draw_rect(fitz.Rect(40, 115, 555, 215), color=(0.85, 0.88, 0.92), fill=(0.97, 0.98, 0.99))
    page.insert_text((55, 134), "PATIENT DEMOGRAPHICS & CLINICAL INTAKE", fontsize=9.5, fontname="helv", color=(0.1, 0.2, 0.3))

    y = 156
    page.insert_text((55, y), f"Patient Name: {patient_info['name']}", fontsize=10, fontname="helv")
    page.insert_text((300, y), f"Accession ID: {patient_info['id']}", fontsize=10, fontname="helv")
    y += 18
    page.insert_text((55, y), f"Age: {patient_info['age']} Years", fontsize=10, fontname="helv")
    page.insert_text((300, y), f"Biological Sex: {patient_info['gender']}", fontsize=10, fontname="helv")
    y += 18
    page.insert_text((55, y), f"Height: {patient_info['height']} cm", fontsize=10, fontname="helv")
    page.insert_text((300, y), f"Weight: {patient_info['weight']} kg", fontsize=10, fontname="helv")

    # Vitals Section
    y = 235
    page.draw_line(fitz.Point(40, y), fitz.Point(555, y), color=(0.85, 0.88, 0.92), width=1)
    y += 20
    page.insert_text((45, y), "HEMODYNAMIC VITAL SIGNS", fontsize=11, fontname="helv", color=(0.01, 0.52, 0.78))
    y += 20
    for k, v in vitals.items():
        page.insert_text((55, y), f"{k}: {v}", fontsize=10, fontname="helv")
        y += 18

    # Lab Panels Section
    y += 10
    page.draw_line(fitz.Point(40, y), fitz.Point(555, y), color=(0.85, 0.88, 0.92), width=1)
    y += 20
    page.insert_text((45, y), "SERUM BIOCHEMICAL & LIPID PROFILE", fontsize=11, fontname="helv", color=(0.01, 0.52, 0.78))
    y += 22

    # Table header
    page.draw_rect(fitz.Rect(45, y - 14, 550, y + 6), color=(0.92, 0.94, 0.96), fill=(0.92, 0.94, 0.96))
    page.insert_text((55, y), "Test Description", fontsize=9, fontname="helv", color=(0.2, 0.25, 0.3))
    page.insert_text((260, y), "Result", fontsize=9, fontname="helv", color=(0.2, 0.25, 0.3))
    page.insert_text((370, y), "Units", fontsize=9, fontname="helv", color=(0.2, 0.25, 0.3))
    page.insert_text((450, y), "Reference Range", fontsize=9, fontname="helv", color=(0.2, 0.25, 0.3))
    y += 20

    for item in labs:
        is_abnormal = item[4]
        res_color = (0.85, 0.1, 0.15) if is_abnormal else (0.1, 0.1, 0.1)
        page.insert_text((55, y), item[0], fontsize=10, fontname="helv")
        page.insert_text((260, y), item[1], fontsize=10, fontname="helv", color=res_color)
        page.insert_text((370, y), item[2], fontsize=10, fontname="helv", color=(0.4, 0.4, 0.4))
        page.insert_text((450, y), item[3], fontsize=9, fontname="helv", color=(0.4, 0.4, 0.4))
        y += 18

    # Clinical Notes & History
    y += 15
    page.draw_line(fitz.Point(40, y), fitz.Point(555, y), color=(0.85, 0.88, 0.92), width=1)
    y += 20
    page.insert_text((45, y), "PHYSICIAN CLINICAL ASSESSMENT & SOCIAL HISTORY", fontsize=11, fontname="helv", color=(0.01, 0.52, 0.78))
    y += 20
    for line in notes:
        page.insert_text((55, y), line, fontsize=9.5, fontname="helv", color=(0.25, 0.25, 0.25))
        y += 16

    # Footer
    page.draw_line(fitz.Point(40, 780), fitz.Point(555, 780), color=(0.85, 0.88, 0.92), width=1)
    page.insert_text((45, 796), "Electronic Verification: Certified Diagnostic Pathology Report - ISO 15189 Accredited", fontsize=8, fontname="helv", color=(0.5, 0.5, 0.5))

    filepath = os.path.join(OUT_DIR, filename)
    doc.save(filepath)
    doc.close()
    print(f"Generated: {filepath}")


if __name__ == "__main__":
    # 1. High Risk Patient
    create_report(
        "sample_cardio_lab_report_high_risk.pdf",
        {"name": "Robert Jenkins", "id": "ACC-2026-8841", "age": 54, "gender": "Male", "height": 174, "weight": 86.5},
        {
            "Blood Pressure (NIBT)": "146/94 mmHg",
            "Resting Radial Pulse": "82 bpm (Regular rhythm)"
        },
        [
            ["Total Serum Cholesterol", "236", "mg/dL", "< 200 mg/dL", True],
            ["HDL Cholesterol", "39", "mg/dL", "> 40 mg/dL", True],
            ["LDL Cholesterol (Calculated)", "156", "mg/dL", "< 100 mg/dL", True],
            ["Serum Triglycerides", "205", "mg/dL", "< 150 mg/dL", True],
            ["Fasting Blood Glucose", "118", "mg/dL", "70 - 99 mg/dL", True],
            ["Glycated Hemoglobin (HbA1c)", "6.1", "%", "< 5.7 %", True]
        ],
        [
            "Social History: Active tobacco smoker (15 cigarettes/day for 18 years).",
            "Physical Activity: Sedentary occupational profile; limited aerobic exercise.",
            "Impression: Combined dyslipidemia and essential hypertension with impaired fasting glucose.",
            "Recommendation: Formal cardiovascular risk stratification and therapeutic management."
        ]
    )

    # 2. Healthy Low Risk Patient
    create_report(
        "sample_cardio_lab_report_healthy.pdf",
        {"name": "Elena Rostova", "id": "ACC-2026-1294", "age": 31, "gender": "Female", "height": 166, "weight": 58.5},
        {
            "Blood Pressure (NIBT)": "114/74 mmHg",
            "Resting Radial Pulse": "68 bpm (Sinus rhythm)"
        },
        [
            ["Total Serum Cholesterol", "168", "mg/dL", "< 200 mg/dL", False],
            ["HDL Cholesterol", "58", "mg/dL", "> 50 mg/dL", False],
            ["LDL Cholesterol (Calculated)", "94", "mg/dL", "< 100 mg/dL", False],
            ["Serum Triglycerides", "82", "mg/dL", "< 150 mg/dL", False],
            ["Fasting Blood Glucose", "88", "mg/dL", "70 - 99 mg/dL", False],
            ["Glycated Hemoglobin (HbA1c)", "5.1", "%", "< 5.7 %", False]
        ],
        [
            "Social History: Non-smoker. Denies regular tobacco or alcohol consumption.",
            "Physical Activity: Regularly active (4x weekly cardio and resistance training).",
            "Impression: Normal metabolic, glycemic, and lipid profiles within optimal physiological limits.",
            "Recommendation: Maintain healthy diet and continue routine annual wellness screening."
        ]
    )
