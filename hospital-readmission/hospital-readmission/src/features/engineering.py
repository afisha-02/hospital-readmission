"""
Clinical feature engineering module for Hospital Readmission Risk Intelligence Platform.
Derives domain-inspired utilization, severity, and diagnostic features.
"""

import sys
import re
from typing import Dict, Any, List
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent if "Path" in globals() else None
from pathlib import Path
if PROJECT_ROOT is None:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger("feature_engineering")

# 24 Diabetes medication columns in UCI dataset
DIABETES_MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"
]

FEATURE_DICTIONARY = [
    {
        "original_feature": "number_outpatient, number_emergency, number_inpatient",
        "derived_feature": "total_encounters_prior",
        "clinical_reason": "Aggregates past 12-month acute and ambulatory healthcare system contact.",
        "leakage_risk": "None: Represents historical visits prior to the current admission index."
    },
    {
        "original_feature": "number_emergency, number_inpatient, time_in_hospital",
        "derived_feature": "utilization_intensity",
        "clinical_reason": "Weights high-acuity inpatient/ER visits normalized by length of stay.",
        "leakage_risk": "None: Inpatient and emergency counts are prior history; stay duration is pre-discharge."
    },
    {
        "original_feature": "number_inpatient, number_emergency",
        "derived_feature": "high_utilizer_flag",
        "clinical_reason": "Identifies frequent flyer patients (>= 2 prior acute admissions).",
        "leakage_risk": "None: Historical utilization prior to current admission."
    },
    {
        "original_feature": "num_lab_procedures, time_in_hospital",
        "derived_feature": "lab_per_day",
        "clinical_reason": "Measures acute monitoring intensity and diagnostic complexity during stay.",
        "leakage_risk": "None: Completed during current inpatient stay prior to discharge."
    },
    {
        "original_feature": "num_medications, time_in_hospital",
        "derived_feature": "med_per_day",
        "clinical_reason": "Measures therapeutic complexity per inpatient day.",
        "leakage_risk": "None: Administered/prescribed during hospital stay."
    },
    {
        "original_feature": "23 specific diabetes drug columns",
        "derived_feature": "diabetes_med_count",
        "clinical_reason": "Polypharmacy count of distinct active glycemic control agents.",
        "leakage_risk": "None: Active discharge orders."
    },
    {
        "original_feature": "diag_1",
        "derived_feature": "primary_diagnosis_group",
        "clinical_reason": "Collapses granular ICD-9 codes into 9 major clinical organ system categories.",
        "leakage_risk": "None: Primary admission/discharge diagnosis on encounter record."
    },
    {
        "original_feature": "diag_2",
        "derived_feature": "secondary_diagnosis_group",
        "clinical_reason": "Identifies secondary chronic comorbidity burden (e.g. renal, cardiovascular).",
        "leakage_risk": "None: Documented during current inpatient stay."
    }
]

def map_icd9_to_category(code) -> str:
    """
    Maps an ICD-9 diagnosis code into standard clinical categories:
    Circulatory, Respiratory, Digestive, Diabetes, Injury, Musculoskeletal,
    Genitourinary, Neoplasms, or Other.
    """
    if pd.isna(code) or str(code).strip() in ["", "?", "nan", "None"]:
        return "Missing/Unknown"
        
    code_str = str(code).strip()
    
    # Check for Diabetes (250.xx)
    if code_str.startswith("250"):
        return "Diabetes"
    # Check for V or E supplementary codes
    if code_str.startswith("V") or code_str.startswith("E"):
        return "Other"
        
    # Extract numerical part
    try:
        num = float(code_str.split(".")[0])
    except ValueError:
        return "Other"
        
    if (390 <= num <= 459) or num == 785:
        return "Circulatory"
    elif (460 <= num <= 519) or num == 786:
        return "Respiratory"
    elif (520 <= num <= 579) or num == 787:
        return "Digestive"
    elif (580 <= num <= 629) or num == 788:
        return "Genitourinary"
    elif 140 <= num <= 239:
        return "Neoplasms"
    elif 710 <= num <= 739:
        return "Musculoskeletal"
    elif 800 <= num <= 999:
        return "Injury/Poisoning"
    else:
        return "Other"

class FeatureEngineer:
    """Clinical feature engineering transformer for hospital encounters."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies clinical feature engineering transformations."""
        df_feat = df.copy()
        
        # 1. Total prior utilization encounters
        outpatient = pd.to_numeric(df_feat.get("number_outpatient", 0), errors="coerce").fillna(0)
        emergency = pd.to_numeric(df_feat.get("number_emergency", 0), errors="coerce").fillna(0)
        inpatient = pd.to_numeric(df_feat.get("number_inpatient", 0), errors="coerce").fillna(0)
        stay_time = pd.to_numeric(df_feat.get("time_in_hospital", 1), errors="coerce").fillna(1)
        labs = pd.to_numeric(df_feat.get("num_lab_procedures", 0), errors="coerce").fillna(0)
        meds = pd.to_numeric(df_feat.get("num_medications", 0), errors="coerce").fillna(0)
        
        df_feat["total_encounters_prior"] = outpatient + emergency + inpatient
        
        # 2. Utilization intensity (weighted by acuity and normalized by stay)
        df_feat["utilization_intensity"] = (
            (inpatient * 3.0 + emergency * 2.0 + outpatient * 1.0) / (stay_time + 1.0)
        ).round(3)
        
        # 3. High utilizer flag
        df_feat["high_utilizer_flag"] = (
            ((inpatient >= 2) | (emergency >= 2)).astype(int)
        ).map({1: "High", 0: "Normal"})
        
        # 4. Stay efficiency / acute intensity ratios
        df_feat["lab_per_day"] = (labs / (stay_time + 0.1)).round(2)
        df_feat["med_per_day"] = (meds / (stay_time + 0.1)).round(2)
        
        # 5. Diabetes active medication count
        active_med_count = pd.Series(0, index=df_feat.index)
        for med in DIABETES_MEDICATIONS:
            if med in df_feat.columns:
                active_med_count += df_feat[med].isin(["Steady", "Up", "Down"]).astype(int)
        df_feat["diabetes_med_count"] = active_med_count
        
        # 6. Diagnosis groups (ICD-9 mapping)
        if "diag_1" in df_feat.columns:
            df_feat["primary_diagnosis_group"] = df_feat["diag_1"].apply(map_icd9_to_category)
        else:
            df_feat["primary_diagnosis_group"] = "Other"
            
        if "diag_2" in df_feat.columns:
            df_feat["secondary_diagnosis_group"] = df_feat["diag_2"].apply(map_icd9_to_category)
        else:
            df_feat["secondary_diagnosis_group"] = "Other"
            
        # 7. Clean Age group
        if "age" in df_feat.columns:
            df_feat["age"] = df_feat["age"].fillna("[50-60)")
            
        # 8. Clean Gender
        if "gender" in df_feat.columns:
            df_feat["gender"] = df_feat["gender"].replace({"Unknown/Invalid": "Unknown", np.nan: "Unknown"})
            
        # 9. Clean Race
        if "race" in df_feat.columns:
            df_feat["race"] = df_feat["race"].fillna("Missing")

        # 10. Clean lab and A1C results
        if "max_glu_serum" in df_feat.columns:
            df_feat["max_glu_serum"] = df_feat["max_glu_serum"].fillna("None")
        if "A1Cresult" in df_feat.columns:
            df_feat["A1Cresult"] = df_feat["A1Cresult"].fillna("None")

        return df_feat

    @staticmethod
    def get_feature_dictionary() -> List[Dict[str, str]]:
        return FEATURE_DICTIONARY

if __name__ == "__main__":
    from src.data.loader import load_mapped_data
    from src.validation.leakage_audit import LeakageAuditor
    
    df_raw = load_mapped_data()
    auditor = LeakageAuditor()
    df_clean, _ = auditor.audit_and_cleanse(df_raw)
    
    engineer = FeatureEngineer()
    df_feat = engineer.transform(df_clean)
    
    print("\n--- FEATURE ENGINEERING SUMMARY ---")
    print(f"Features created successfully. New shape: {df_feat.shape}")
    print("\nSample Engineered Features:")
    sample_cols = [
        "total_encounters_prior", "utilization_intensity", "high_utilizer_flag",
        "lab_per_day", "med_per_day", "diabetes_med_count", "primary_diagnosis_group"
    ]
    print(df_feat[sample_cols].head())
