"""
FastAPI REST service for Hospital Readmission Risk Intelligence Platform.
Endpoints for single prediction, batch prediction, SHAP explanation, metrics, drift audit, and health checks.
"""

import sys
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.logger import setup_logger
from src.preprocessing.pipeline import ClinicalDataPreprocessor
from src.features.engineering import FeatureEngineer
from src.explainability.shap_explainer import ClinicalShapExplainer
from src.explainability.counterfactual import CounterfactualAnalyzer
from src.monitoring.drift_detector import ClinicalDriftDetector

logger = setup_logger("api")

# Pydantic Schemas
class PatientEncounterInput(BaseModel):
    time_in_hospital: int = Field(default=3, ge=1, le=14, description="Inpatient stay duration in days")
    num_lab_procedures: int = Field(default=40, ge=1, le=150, description="Count of laboratory procedures")
    num_procedures: int = Field(default=1, ge=0, le=10, description="Count of non-lab inpatient procedures")
    num_medications: int = Field(default=12, ge=1, le=100, description="Count of distinct medications administered")
    number_outpatient: int = Field(default=0, ge=0, le=50, description="Outpatient encounters in past 12 months")
    number_emergency: int = Field(default=0, ge=0, le=50, description="Emergency room visits in past 12 months")
    number_inpatient: int = Field(default=0, ge=0, le=50, description="Inpatient hospitalizations in past 12 months")
    number_diagnoses: int = Field(default=7, ge=1, le=20, description="Documented diagnoses count")
    race: str = Field(default="Caucasian", description="Reported race / demographic background")
    gender: str = Field(default="Female", description="Biological sex")
    age: str = Field(default="[60-70)", description="Age decile bracket")
    admission_type_desc: str = Field(default="Emergency", description="Admission urgency: Emergency, Urgent, Elective, Other")
    discharge_disposition_desc: str = Field(default="Home", description="Discharge destination: Home, SNF/Nursing, Rehabilitation, Other")
    admission_source_desc: str = Field(default="Emergency Room", description="Referral channel")
    max_glu_serum: str = Field(default="None", description="Serum glucose test: None, Norm, >200, >300")
    A1Cresult: str = Field(default="None", description="HbA1c test: None, Norm, >7, >8")
    primary_diagnosis_group: str = Field(default="Circulatory", description="Primary ICD-9 diagnosis organ system")
    secondary_diagnosis_group: str = Field(default="Diabetes", description="Secondary ICD-9 diagnosis category")
    change: str = Field(default="No", description="Change in diabetic medication: No, Ch")
    diabetesMed: str = Field(default="Yes", description="Diabetic medication prescribed: No, Yes")
    insulin: str = Field(default="No", description="Insulin dosage status: No, Steady, Up, Down")
    metformin: str = Field(default="Steady", description="Metformin status: No, Steady, Up, Down")
    custom_threshold: Optional[float] = Field(default=None, ge=0.01, le=0.99, description="Optional custom operating threshold")

class BatchPredictionResponseItem(BaseModel):
    index: int
    probability: float
    prediction: int
    risk_category: str
    threshold: float

class PredictionResponse(BaseModel):
    probability: float
    prediction: int
    risk_category: str
    threshold: float
    model_version: str
    feature_version: str
    decision_summary: str
    disclaimer: str

# Global artifacts cache
STATE = {
    "config": None,
    "model": None,
    "uncalibrated_model": None,
    "preprocessor": None,
    "explainer": None,
    "counterfactual_analyzer": None,
    "drift_detector": None,
    "threshold": 0.125
}

def load_application_state():
    """Loads model artifacts and preprocessors once during API startup."""
    try:
        config = load_config()
        STATE["config"] = config

        
        preprocessor = ClinicalDataPreprocessor.load_artifacts()
        STATE["preprocessor"] = preprocessor
        
        model_path = resolve_path(config["paths"]["model"])
        if model_path.exists():
            STATE["model"] = joblib.load(model_path)
            logger.info("Loaded calibrated XGBoost model artifact.")
            
        uncal_path = resolve_path(config["paths"]["uncalibrated_model"])
        if uncal_path.exists():
            STATE["uncalibrated_model"] = joblib.load(uncal_path)["model"]
            
        explainer_path = resolve_path(config["paths"]["explainer"])
        if explainer_path.exists():
            STATE["explainer"] = ClinicalShapExplainer.load_artifact(str(explainer_path))
            logger.info("Loaded SHAP explainer artifact.")
            
        if STATE["model"] and STATE["preprocessor"]:
            STATE["counterfactual_analyzer"] = CounterfactualAnalyzer(STATE["model"], STATE["preprocessor"])
            
        drift_baseline_path = resolve_path(config["paths"]["drift_baseline"])
        if drift_baseline_path.exists():
            baseline_df = pd.read_parquet(drift_baseline_path)
            STATE["drift_detector"] = ClinicalDriftDetector(baseline_df, config)
            
        # Optimal threshold from report if exists
        thresh_path = resolve_path(config["paths"]["threshold_report"])
        if thresh_path.exists():
            with open(thresh_path, "r", encoding="utf-8") as f:
                t_data = json.load(f)
                STATE["threshold"] = t_data.get("optimal_thresholds", {}).get("f1_optimal", {}).get("threshold", 0.125)
                
    except Exception as e:
        logger.error(f"Startup initialization warning: {e}")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    load_application_state()
    yield

# FastAPI Application
app = FastAPI(
    title="Hospital Readmission Risk Intelligence API",
    description="Explainable machine learning REST API for proactive 30-day hospital readmission risk assessment.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")

def health_check():
    """Health check endpoint indicating model readiness."""
    is_ready = STATE["model"] is not None and STATE["preprocessor"] is not None
    return {
        "status": "HEALTHY" if is_ready else "INITIALIZING",
        "model_loaded": STATE["model"] is not None,
        "explainer_loaded": STATE["explainer"] is not None,
        "preprocessor_loaded": STATE["preprocessor"] is not None,
        "version": "1.0.0"
    }

@app.get("/model-info")
def get_model_info():
    """Metadata regarding current production candidate model and parameters."""
    return {
        "model_name": "Calibrated XGBoost Classifier",
        "model_version": "1.0.0",
        "feature_version": "1.2.0",
        "operating_threshold": STATE["threshold"],
        "target_definition": "30-day unplanned hospital readmission (<30 days)",
        "calibration_method": "Sigmoid (Platt Scaling)",
        "intended_use": "Analytical risk estimation and explainability for research and clinical quality review.",
        "disclaimer": "RESEARCH / PORTFOLIO PROTOTYPE. Not a clinical decision-making or diagnostic system."
    }

def categorize_risk(prob: float, threshold: float) -> str:
    if prob >= 0.25:
        return "HIGH"
    elif prob >= 0.10:
        return "MODERATE"
    else:
        return "LOW"

@app.post("/predict", response_model=PredictionResponse)
def predict_readmission(patient: PatientEncounterInput):
    """Calculates readmission probability, binary classification, and risk tier."""
    if STATE["model"] is None or STATE["preprocessor"] is None:
        raise HTTPException(status_code=503, detail="Model artifact unavailable.")
        
    try:
        # Convert Pydantic payload to DataFrame
        data_dict = patient.model_dump()
        threshold = patient.custom_threshold if patient.custom_threshold is not None else STATE["threshold"]
        
        # Apply clinical feature engineering
        engineer = FeatureEngineer(STATE["config"])
        df_in = pd.DataFrame([data_dict])
        df_feat = engineer.transform(df_in)
        
        # Transform through preprocessor
        x_proc = STATE["preprocessor"].transform(df_feat)
        
        # Predict probability from calibrated model
        prob = float(STATE["model"].predict_proba(x_proc)[0, 1])
        prediction = int(prob >= threshold)
        risk_category = categorize_risk(prob, threshold)
        
        return PredictionResponse(
            probability=round(prob, 4),
            prediction=prediction,
            risk_category=risk_category,
            threshold=round(threshold, 4),
            model_version="1.0.0",
            feature_version="1.2.0",
            decision_summary=(
                f"Model estimates a {prob*100:.1f}% readmission probability ({risk_category} risk). "
                f"Prediction is {'POSITIVE' if prediction == 1 else 'NEGATIVE'} at threshold {threshold:.3f}."
            ),
            disclaimer="Research prototype. Not for clinical decision-making."
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

@app.post("/explain")
def explain_prediction(patient: PatientEncounterInput):
    """Provides SHAP factor contributions and mathematical counterfactual analysis."""
    if STATE["explainer"] is None or STATE["preprocessor"] is None:
        raise HTTPException(status_code=503, detail="SHAP Explainer unavailable.")
        
    try:
        data_dict = patient.model_dump()
        threshold = patient.custom_threshold if patient.custom_threshold is not None else STATE["threshold"]
        
        engineer = FeatureEngineer(STATE["config"])
        df_in = pd.DataFrame([data_dict])
        df_feat = engineer.transform(df_in)
        x_proc = STATE["preprocessor"].transform(df_feat)
        
        # Local SHAP explanation
        shap_explanation = STATE["explainer"].explain_patient(x_proc[0], top_k=8)
        
        # Counterfactual analysis
        counterfactuals = None
        if STATE["counterfactual_analyzer"]:
            counterfactuals = STATE["counterfactual_analyzer"].analyze(df_feat.iloc[0].to_dict(), threshold=threshold)
            
        prob = float(STATE["model"].predict_proba(x_proc)[0, 1])
        
        return {
            "probability": round(prob, 4),
            "threshold": round(threshold, 4),
            "shap_explanation": shap_explanation,
            "counterfactual_analysis": counterfactuals,
            "clinical_disclaimer": "These factors contributed mathematically to the model prediction. They do NOT establish medical causality."
        }
    except Exception as e:
        logger.error(f"Explainability error: {e}")
        raise HTTPException(status_code=400, detail=f"Explainability error: {str(e)}")

@app.post("/predict/batch")
def predict_batch(file: UploadFile = File(...)):
    """Processes uploaded CSV file of patient encounters and returns risk scores."""
    if STATE["model"] is None or STATE["preprocessor"] is None:
        raise HTTPException(status_code=503, detail="Model artifact unavailable.")
        
    try:
        df_uploaded = pd.read_csv(file.file, na_values=["?", "None", "NULL"], low_memory=False)
        engineer = FeatureEngineer(STATE["config"])
        df_feat = engineer.transform(df_uploaded)
        x_proc = STATE["preprocessor"].transform(df_feat)
        
        probs = STATE["model"].predict_proba(x_proc)[:, 1]
        threshold = STATE["threshold"]
        
        records = []
        for i, p in enumerate(probs):
            records.append({
                "patient_index": i + 1,
                "readmission_probability": round(float(p), 4),
                "predicted_readmission": int(p >= threshold),
                "risk_tier": categorize_risk(float(p), threshold)
            })
            
        return {
            "total_encounters_scored": len(records),
            "operating_threshold": threshold,
            "high_risk_count": sum(1 for r in records if r["risk_tier"] == "HIGH"),
            "moderate_risk_count": sum(1 for r in records if r["risk_tier"] == "MODERATE"),
            "low_risk_count": sum(1 for r in records if r["risk_tier"] == "LOW"),
            "predictions_sample": records[:50]
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Batch processing error: {str(e)}")

@app.get("/metrics")
def get_metrics():
    """Returns genuine benchmark evaluation comparison across all evaluated models."""
    metrics_path = resolve_path(STATE["config"]["paths"]["metrics_report"])
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="Metrics report not found.")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/drift")
def get_drift_status():
    """Runs drift audit on held-out test distribution vs training baseline."""
    if STATE["drift_detector"] is None:
        raise HTTPException(status_code=503, detail="Drift detector unavailable.")
        
    test_path = resolve_path(STATE["config"]["paths"]["processed_test"])
    if not test_path.exists():
        raise HTTPException(status_code=404, detail="Test evaluation dataset not found.")
        
    test_df = pd.read_parquet(test_path).sample(min(800, len(pd.read_parquet(test_path))), random_state=42)
    report = STATE["drift_detector"].evaluate_batch(test_df)
    return report

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)
