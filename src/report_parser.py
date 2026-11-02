"""
Medical Lab Report and Prescription Document Parser
Extracts clinical cardiovascular biomarkers from PDF documents and images
using PyMuPDF (fitz) and Clinical Regular Expression Entity Recognition.
"""

import re
from typing import Dict, Any, Optional
from pathlib import Path
import fitz  # PyMuPDF


class MedicalReportParser:
    """
    Parses unstructured and semi-structured laboratory and medical reports
    to extract cardiovascular physiological biomarkers.
    """

    @staticmethod
    def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
        """
        Extracts raw text content from uploaded PDF or image files.
        """
        ext = filename.lower().split(".")[-1]
        text_content = ""

        if ext == "pdf":
            try:
                doc = fitz.open(stream=file_bytes, filetype="pdf")
                for page in doc:
                    text_content += page.get_text() + "\n"
                doc.close()
            except Exception as e:
                text_content = f"Error reading PDF: {e}"
        else:
            # Plain text fallback or basic string decoding
            try:
                text_content = file_bytes.decode("utf-8", errors="ignore")
            except Exception:
                text_content = ""

        return text_content

    @classmethod
    def parse_biomarkers(cls, text: str) -> Dict[str, Any]:
        """
        Extracts clinical indicators from text using medical regex patterns.
        """
        extracted = {}
        snippets = []

        # 1. Blood Pressure: e.g. "BP: 135/85", "140/90 mmHg", "Blood Pressure (NIBT): 146/94"
        bp_match = re.search(
            r"(?:b\.?p\.?|blood\s*pressure|nibt|arterial\s*pressure|vitals?)[^0-9\n]{0,35}(\d{2,3})\s*[/xX]\s*(\d{2,3})",
            text,
            re.IGNORECASE
        )
        if not bp_match:
            # Fallback: find any valid standalone systolic/diastolic reading like "140/90 mmHg"
            fallback_matches = re.finditer(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b(?:\s*mm\s*hg)?", text, re.IGNORECASE)
            for m in fallback_matches:
                s, d = float(m.group(1)), float(m.group(2))
                if 70 <= s <= 240 and 40 <= d <= 160 and s > d:
                    bp_match = m
                    break

        if bp_match:
            sys_val = float(bp_match.group(1))
            dia_val = float(bp_match.group(2))
            if 60 <= sys_val <= 250 and 30 <= dia_val <= 180:
                extracted["ap_hi"] = int(max(sys_val, dia_val))
                extracted["ap_lo"] = int(min(sys_val, dia_val))
                snippets.append(f"Blood Pressure found: {extracted['ap_hi']}/{extracted['ap_lo']} mmHg")

        # 2. Total Cholesterol: e.g. "Total Cholesterol: 215 mg/dL"
        chol_match = re.search(
            r"(?:total\s*cholesterol|serum\s*cholesterol|cholesterol|t\.?\s*chol)\s*[:=-]?\s*(\d{2,3}(?:\.\d+)?)",
            text,
            re.IGNORECASE
        )
        if chol_match:
            val = float(chol_match.group(1))
            if 80 <= val <= 500:
                if val < 200:
                    tier = 1  # Normal
                    desc = "Normal (<200 mg/dL)"
                elif val < 240:
                    tier = 2  # Above Normal
                    desc = "Above Normal (200-239 mg/dL)"
                else:
                    tier = 3  # Well Above Normal
                    desc = "High (>=240 mg/dL)"
                extracted["cholesterol"] = tier
                extracted["cholesterol_raw"] = val
                snippets.append(f"Cholesterol: {val} mg/dL ({desc})")

        # 3. Fasting Glucose / Blood Sugar: e.g. "Glucose: 115 mg/dL", "FBS: 104"
        gluc_match = re.search(
            r"(?:fasting\s*blood\s*sugar|fbs|fasting\s*glucose|blood\s*glucose|glucose)\s*[:=-]?\s*(\d{2,3}(?:\.\d+)?)",
            text,
            re.IGNORECASE
        )
        if gluc_match:
            val = float(gluc_match.group(1))
            if 50 <= val <= 400:
                if val < 100:
                    tier = 1  # Normal
                    desc = "Normal (<100 mg/dL)"
                elif val < 126:
                    tier = 2  # Above Normal (Prediabetes)
                    desc = "Above Normal (100-125 mg/dL)"
                else:
                    tier = 3  # High (Diabetic range)
                    desc = "High (>=126 mg/dL)"
                extracted["gluc"] = tier
                extracted["gluc_raw"] = val
                snippets.append(f"Glucose: {val} mg/dL ({desc})")

        # 4. Age: e.g. "Age: 52", "52 Y/O", "52 yrs"
        age_match = re.search(r"(?:age|years\s*old)\s*[:=-]?\s*(\d{1,3})", text, re.IGNORECASE)
        if age_match:
            val = int(age_match.group(1))
            if 18 <= val <= 100:
                extracted["age"] = val
                snippets.append(f"Age: {val} years")

        # 5. Weight: e.g. "Weight: 82 kg", "Wt: 75.5 kgs"
        wt_match = re.search(r"(?:weight|wt\.?)\s*[:=-]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:kg|kgs)?", text, re.IGNORECASE)
        if wt_match:
            val = float(wt_match.group(1))
            if 35 <= val <= 220:
                extracted["weight"] = round(val, 1)
                snippets.append(f"Weight: {extracted['weight']} kg")

        # 6. Height: e.g. "Height: 172 cm", "Ht: 168"
        ht_match = re.search(r"(?:height|ht\.?)\s*[:=-]?\s*(\d{2,3}(?:\.\d+)?)\s*(?:cm)?", text, re.IGNORECASE)
        if ht_match:
            val = float(ht_match.group(1))
            if 120 <= val <= 220:
                extracted["height"] = round(val, 1)
                snippets.append(f"Height: {extracted['height']} cm")

        # 7. Gender: e.g. "Sex: Male", "Gender: F"
        gender_match = re.search(r"(?:gender|sex)\s*[:=-]?\s*(male|female|m|f)\b", text, re.IGNORECASE)
        if gender_match:
            g_str = gender_match.group(1).lower()
            if g_str in ["female", "f"]:
                extracted["gender"] = 1
                snippets.append("Gender: Female (1)")
            else:
                extracted["gender"] = 2
                snippets.append("Gender: Male (2)")

        # 8. Smoking Status
        if re.search(r"\b(?:non-?smoker|smoking\s*[:=-]?\s*(?:no|none|never)|denies\s*tobacco|no\s*smoking)\b", text, re.IGNORECASE):
            extracted["smoke"] = 0
            snippets.append("Smoker: No (0)")
        elif re.search(r"\b(?:smoker|smoking\s*[:=-]?\s*yes|tobacco\s*user?|active\s*smoker)\b", text, re.IGNORECASE):
            extracted["smoke"] = 1
            snippets.append("Smoker: Yes (1)")

        return {
            "biomarkers": extracted,
            "detected_count": len(extracted),
            "evidence_snippets": snippets,
            "raw_text_length": len(text)
        }
