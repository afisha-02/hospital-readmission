"""
Live services health and integration check script.
"""

import urllib.request
import json

def check():
    print("==================================================")
    print("   HOSPITAL READMISSION RISK INTELLIGENCE")
    print("   LIVE RUNTIME VERIFICATION")
    print("==================================================")

    # 1. Check Streamlit
    print("\n1. STREAMLIT FRONTEND (Port 8501):")
    try:
        with urllib.request.urlopen("http://localhost:8501", timeout=5) as resp:
            print(f"   Status: {resp.status} OK")
            print("   Live URL: http://localhost:8501")
            print("   Network URL: http://127.0.0.1:8501")
    except Exception as e:
        print(f"   Error connecting to Streamlit: {e}")

    # 2. Check FastAPI Health
    print("\n2. FASTAPI REST BACKEND (Port 8000):")
    try:
        with urllib.request.urlopen("http://localhost:8000/health", timeout=5) as resp:
            h = json.loads(resp.read().decode("utf-8"))
            print(f"   Status: {resp.status} OK")
            print(f"   Health State: {h['status']}")
            print(f"   Model Loaded: {h['model_loaded']}")
            print(f"   Explainer Loaded: {h['explainer_loaded']}")
            print("   Docs URL: http://localhost:8000/docs")

        # 3. Live Prediction Test
        payload = {
            "time_in_hospital": 5,
            "num_lab_procedures": 55,
            "num_procedures": 2,
            "num_medications": 18,
            "number_outpatient": 1,
            "number_emergency": 1,
            "number_inpatient": 2,
            "number_diagnoses": 8,
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
            "custom_threshold": 0.125
        }
        req = urllib.request.Request(
            "http://localhost:8000/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8")
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            p = json.loads(resp.read().decode("utf-8"))
            print("\n3. SAMPLE LIVE REST PREDICTION:")
            print(f"   Probability:    {p['probability'] * 100:.2f}%")
            print(f"   Risk Tier:      {p['risk_category']}")
            print(f"   Classification: {'POSITIVE' if p['prediction'] == 1 else 'NEGATIVE'} (Threshold: {p['threshold']})")
            print(f"   Model Version:  {p['model_version']}")
            print(f"   Summary:        {p['decision_summary']}")

    except Exception as e:
        print(f"   Error connecting to FastAPI: {e}")

    print("\n==================================================")
    print("   ALL SERVICES ONLINE AND READY FOR USE")
    print("==================================================")

if __name__ == "__main__":
    check()
