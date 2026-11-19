"""
Counterfactual explanation module for Hospital Readmission Risk Intelligence.
Identifies minimal model input shifts required to reduce estimated risk below decision threshold.
Strictly framed as mathematical sensitivity analysis, NOT medical guidance.
"""

import sys
import copy
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent if "Path" in globals() else None
from pathlib import Path
if PROJECT_ROOT is None:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger

logger = setup_logger("counterfactual")

DISCLAIMER_TEXT = (
    "MATHEMATICAL COUNTERFACTUAL ANALYSIS ONLY: "
    "These counterfactual adjustments represent mathematical sensitivities of the statistical model. "
    "They do NOT constitute medical directives, clinical care plans, or actionable patient recommendations. "
    "Clinical treatment decisions must always be made by licensed healthcare professionals."
)

class CounterfactualAnalyzer:
    """Analyzes what feature perturbations move the model's predicted risk below threshold."""
    
    def __init__(self, model: Any, preprocessor: Any):
        self.model = model
        self.preprocessor = preprocessor

    def analyze(
        self,
        patient_row_dict: Dict[str, Any],
        threshold: float = 0.35,
        max_perturbations: int = 4
    ) -> Dict[str, Any]:
        """
        Evaluates potential model-sensitive perturbations to transition prediction below threshold.
        """
        # Baseline probability
        df_base = pd.DataFrame([patient_row_dict])
        x_base = self.preprocessor.transform(df_base)
        prob_base = float(self.model.predict_proba(x_base)[0, 1])
        
        is_above_threshold = prob_base >= threshold
        
        if not is_above_threshold:
            return {
                "status": "ALREADY_BELOW_THRESHOLD",
                "current_probability": round(prob_base, 4),
                "threshold": float(threshold),
                "message": f"Patient's predicted risk ({prob_base*100:.1f}%) is already below the operating threshold ({threshold*100:.1f}%).",
                "counterfactuals": [],
                "disclaimer": DISCLAIMER_TEXT
            }
            
        counterfactuals = []
        
        # Test 1: Discharge to Home with home health support
        if patient_row_dict.get("discharge_disposition_desc") != "Home":
            df_mod = pd.DataFrame([patient_row_dict])
            df_mod["discharge_disposition_desc"] = "Home"
            x_mod = self.preprocessor.transform(df_mod)
            p_mod = float(self.model.predict_proba(x_mod)[0, 1])
            reduction = prob_base - p_mod
            if reduction > 0.005:
                counterfactuals.append({
                    "perturbed_feature": "discharge_disposition_desc",
                    "original_value": patient_row_dict.get("discharge_disposition_desc", "Unknown"),
                    "counterfactual_value": "Home",
                    "new_probability": round(p_mod, 4),
                    "risk_reduction_pct": round(reduction * 100, 2),
                    "achieves_target": bool(p_mod < threshold),
                    "rationale": "Model associates routine home discharge with lower acute 30-day readmission likelihood."
                })
                
        # Test 2: Active Diabetes Medication Reconciliation
        curr_meds = int(patient_row_dict.get("num_medications", 10))
        if curr_meds > 12:
            df_mod = pd.DataFrame([patient_row_dict])
            df_mod["num_medications"] = max(curr_meds - 4, 6)
            x_mod = self.preprocessor.transform(df_mod)
            p_mod = float(self.model.predict_proba(x_mod)[0, 1])
            reduction = prob_base - p_mod
            if reduction > 0.005:
                counterfactuals.append({
                    "perturbed_feature": "num_medications",
                    "original_value": curr_meds,
                    "counterfactual_value": max(curr_meds - 4, 6),
                    "new_probability": round(p_mod, 4),
                    "risk_reduction_pct": round(reduction * 100, 2),
                    "achieves_target": bool(p_mod < threshold),
                    "rationale": "Model weights high polypharmacy burden as a correlate of complex multimorbidity."
                })
                
        # Test 3: Glycemic medication stabilization (Change -> No)
        if patient_row_dict.get("change") == "Ch":
            df_mod = pd.DataFrame([patient_row_dict])
            df_mod["change"] = "No"
            x_mod = self.preprocessor.transform(df_mod)
            p_mod = float(self.model.predict_proba(x_mod)[0, 1])
            reduction = prob_base - p_mod
            if reduction > 0.002:
                counterfactuals.append({
                    "perturbed_feature": "change",
                    "original_value": "Ch (Medication changed during encounter)",
                    "counterfactual_value": "No (Regimen stabilized prior to discharge)",
                    "new_probability": round(p_mod, 4),
                    "risk_reduction_pct": round(reduction * 100, 2),
                    "achieves_target": bool(p_mod < threshold),
                    "rationale": "Acute medication adjustment indicates ongoing glycemic titration."
                })

        # Test 4: Length of stay optimization
        curr_stay = int(patient_row_dict.get("time_in_hospital", 4))
        if curr_stay > 4:
            df_mod = pd.DataFrame([patient_row_dict])
            df_mod["time_in_hospital"] = max(curr_stay - 2, 2)
            x_mod = self.preprocessor.transform(df_mod)
            p_mod = float(self.model.predict_proba(x_mod)[0, 1])
            reduction = prob_base - p_mod
            if reduction > 0.005:
                counterfactuals.append({
                    "perturbed_feature": "time_in_hospital",
                    "original_value": f"{curr_stay} days",
                    "counterfactual_value": f"{max(curr_stay - 2, 2)} days",
                    "new_probability": round(p_mod, 4),
                    "risk_reduction_pct": round(reduction * 100, 2),
                    "achieves_target": bool(p_mod < threshold),
                    "rationale": "Prolonged hospital stay is captured by the model as a proxy for clinical severity."
                })

        # Sort by risk reduction
        counterfactuals.sort(key=lambda x: x["risk_reduction_pct"], reverse=True)
        
        return {
            "status": "COUNTERFACTUALS_GENERATED",
            "current_probability": round(prob_base, 4),
            "threshold": float(threshold),
            "is_above_threshold": is_above_threshold,
            "counterfactuals": counterfactuals[:max_perturbations],
            "disclaimer": DISCLAIMER_TEXT
        }
