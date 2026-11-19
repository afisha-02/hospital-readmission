"""
Unit tests for clinical feature engineering and ICD-9 mapping.
"""

import pytest
import pandas as pd
import numpy as np
from src.features.engineering import FeatureEngineer, map_icd9_to_category

def test_icd9_mapping():
    """Verifies ICD-9 clinical organ system categorizations."""
    assert map_icd9_to_category("250.01") == "Diabetes"
    assert map_icd9_to_category("250.83") == "Diabetes"
    assert map_icd9_to_category("410.1") == "Circulatory"   # AMI
    assert map_icd9_to_category("428") == "Circulatory"     # Heart failure
    assert map_icd9_to_category("493") == "Respiratory"     # Asthma
    assert map_icd9_to_category("571") == "Digestive"       # Cirrhosis
    assert map_icd9_to_category("174") == "Neoplasms"       # Breast neoplasm
    assert map_icd9_to_category("715") == "Musculoskeletal" # Osteoarthrosis
    assert map_icd9_to_category("820") == "Injury/Poisoning"# Femur fracture
    assert map_icd9_to_category("V45") == "Other"
    assert map_icd9_to_category(None) == "Missing/Unknown"

def test_feature_engineering_utilization_intensity():
    """Verifies correct calculation of utilization aggregates and intensity ratios."""
    sample_df = pd.DataFrame([{
        "time_in_hospital": 3,
        "number_outpatient": 2,
        "number_emergency": 1,
        "number_inpatient": 2,
        "num_lab_procedures": 30,
        "num_medications": 12,
        "diag_1": "428",
        "diag_2": "250.00",
        "metformin": "Steady",
        "insulin": "Up"
    }])
    
    engineer = FeatureEngineer()
    df_transformed = engineer.transform(sample_df)
    
    # total_encounters = 2 + 1 + 2 = 5
    assert df_transformed.loc[0, "total_encounters_prior"] == 5
    
    # utilization_intensity = (2*3 + 1*2 + 2*1) / (3 + 1) = 10 / 4 = 2.5
    assert df_transformed.loc[0, "utilization_intensity"] == 2.5
    
    # high utilizer flag = 'High' because inpatient >= 2
    assert df_transformed.loc[0, "high_utilizer_flag"] == "High"
    
    # diabetes_med_count = 2 (metformin + insulin)
    assert df_transformed.loc[0, "diabetes_med_count"] == 2
    
    # ICD-9 mappings
    assert df_transformed.loc[0, "primary_diagnosis_group"] == "Circulatory"
    assert df_transformed.loc[0, "secondary_diagnosis_group"] == "Diabetes"
