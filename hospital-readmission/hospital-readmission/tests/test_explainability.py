"""
Unit tests for SHAP TreeExplainer and Counterfactual Analyzer.
"""

import pytest
import numpy as np
import pandas as pd
from src.utils.config import load_config, resolve_path
from src.explainability.shap_explainer import ClinicalShapExplainer
from src.explainability.counterfactual import CounterfactualAnalyzer
from src.preprocessing.pipeline import ClinicalDataPreprocessor
import joblib

def test_shap_local_explanation():
    """Verifies that patient-level SHAP explanation returns positive and negative contributors."""
    config = load_config()
    preprocessor = ClinicalDataPreprocessor.load_artifacts()
    explainer = ClinicalShapExplainer.load_artifact()
    
    test_path = resolve_path(config["paths"]["processed_test"])
    df_test = pd.read_parquet(test_path).head(1)
    X_proc = preprocessor.transform(df_test)
    
    explanation = explainer.explain_patient(X_proc[0], top_k=6)
    
    assert "top_factors" in explanation
    assert len(explanation["top_factors"]) <= 6
    assert "disclaimer" in explanation
    assert "These factors contributed" in explanation["disclaimer"]
    
    # Check top factor structure
    factor = explanation["top_factors"][0]
    assert "feature" in factor
    assert "shap_value" in factor
    assert "direction" in factor
    assert factor["direction"] in ["INCREASED_RISK", "DECREASED_RISK"]

def test_counterfactual_analyzer():
    """Verifies counterfactual analysis on a simulated high-risk patient."""
    config = load_config()
    preprocessor = ClinicalDataPreprocessor.load_artifacts()
    cal_model = joblib.load(resolve_path(config["paths"]["model"]))
    
    analyzer = CounterfactualAnalyzer(cal_model, preprocessor)
    high_risk_dict = {
        "time_in_hospital": 8,
        "num_lab_procedures": 60,
        "num_procedures": 2,
        "num_medications": 24,
        "number_outpatient": 1,
        "number_emergency": 2,
        "number_inpatient": 3,
        "number_diagnoses": 9,
        "race": "Caucasian",
        "gender": "Female",
        "age": "[70-80)",
        "admission_type_desc": "Emergency",
        "discharge_disposition_desc": "SNF/Nursing",
        "admission_source_desc": "Emergency Room",
        "max_glu_serum": "None",
        "A1Cresult": ">8",
        "primary_diagnosis_group": "Circulatory",
        "secondary_diagnosis_group": "Diabetes",
        "change": "Ch",
        "diabetesMed": "Yes",
        "insulin": "Up",
        "metformin": "No",
        "total_encounters_prior": 6,
        "utilization_intensity": 3.0,
        "high_utilizer_flag": "High",
        "lab_per_day": 7.5,
        "med_per_day": 3.0,
        "diabetes_med_count": 1
    }
    
    cf_res = analyzer.analyze(high_risk_dict, threshold=0.15)
    assert cf_res["status"] in ["COUNTERFACTUALS_GENERATED", "ALREADY_BELOW_THRESHOLD"]
    assert "disclaimer" in cf_res
    assert "MATHEMATICAL COUNTERFACTUAL" in cf_res["disclaimer"]
