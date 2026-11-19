"""
Integration tests for FastAPI REST service endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.api import app, load_application_state

@pytest.fixture(scope="module")
def client():
    load_application_state()
    return TestClient(app)

def test_api_health(client):
    """Verifies GET /health endpoint status."""
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "HEALTHY"
    assert data["model_loaded"] is True
    assert data["explainer_loaded"] is True

def test_api_model_info(client):
    """Verifies GET /model-info endpoint metadata."""
    r = client.get("/model-info")
    assert r.status_code == 200
    data = r.json()
    assert "Calibrated XGBoost" in data["model_name"]
    assert "disclaimer" in data

def test_api_predict(client):
    """Verifies POST /predict endpoint inference."""
    payload = {
        "time_in_hospital": 3,
        "num_lab_procedures": 35,
        "num_procedures": 1,
        "num_medications": 10,
        "number_outpatient": 0,
        "number_emergency": 0,
        "number_inpatient": 1,
        "number_diagnoses": 6,
        "race": "Caucasian",
        "gender": "Female",
        "age": "[60-70)",
        "admission_type_desc": "Emergency",
        "discharge_disposition_desc": "Home",
        "admission_source_desc": "Emergency Room",
        "max_glu_serum": "None",
        "A1Cresult": "Norm",
        "primary_diagnosis_group": "Circulatory",
        "secondary_diagnosis_group": "Diabetes",
        "change": "No",
        "diabetesMed": "Yes",
        "insulin": "Steady",
        "metformin": "No",
        "custom_threshold": 0.125
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert data["prediction"] in [0, 1]
    assert data["risk_category"] in ["LOW", "MODERATE", "HIGH"]
    assert data["threshold"] == 0.125

def test_api_explain(client):
    """Verifies POST /explain endpoint returns factors and counterfactuals."""
    payload = {
        "time_in_hospital": 4,
        "num_lab_procedures": 40,
        "num_procedures": 1,
        "num_medications": 12,
        "number_outpatient": 0,
        "number_emergency": 1,
        "number_inpatient": 2,
        "number_diagnoses": 7,
        "race": "Caucasian",
        "gender": "Male",
        "age": "[70-80)",
        "admission_type_desc": "Emergency",
        "discharge_disposition_desc": "SNF/Nursing",
        "admission_source_desc": "Emergency Room",
        "max_glu_serum": "None",
        "A1Cresult": ">7",
        "primary_diagnosis_group": "Circulatory",
        "secondary_diagnosis_group": "Diabetes",
        "change": "Ch",
        "diabetesMed": "Yes",
        "insulin": "Up",
        "metformin": "No"
    }
    r = client.post("/explain", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "shap_explanation" in data
    assert "top_factors" in data["shap_explanation"]
    assert len(data["shap_explanation"]["top_factors"]) > 0

def test_api_metrics(client):
    """Verifies GET /metrics endpoint returns model benchmark table."""
    r = client.get("/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "comparison_table" in data
    assert len(data["comparison_table"]) >= 5

def test_api_drift(client):
    """Verifies GET /drift endpoint returns drift audit report."""
    r = client.get("/drift")
    assert r.status_code == 200
    data = r.json()
    assert "overall_status" in data
    assert "feature_audit" in data
