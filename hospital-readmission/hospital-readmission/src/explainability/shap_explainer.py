"""
SHAP Explainability module for Hospital Readmission Risk Intelligence.
Computes TreeExplainer SHAP values, global beeswarm and bar charts,
and patient-level waterfall explanations with non-causal language.
"""

import sys
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.logger import setup_logger
from src.preprocessing.pipeline import ClinicalDataPreprocessor

logger = setup_logger("shap_explainer")

class ClinicalShapExplainer:
    """Manages TreeSHAP explanations for global and local patient predictions."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: List[str] = []
        self.background_data: Optional[np.ndarray] = None
        self.global_importance_: Optional[pd.DataFrame] = None

    def fit_or_load_explainer(self, base_model: Any, background_sample: np.ndarray, feature_names: List[str]):
        """Initializes TreeExplainer on background sample."""
        self.feature_names = feature_names
        self.background_data = background_sample
        logger.info(f"Initializing SHAP TreeExplainer with {len(feature_names)} features...")
        self.explainer = shap.TreeExplainer(base_model)
        return self.explainer

    def compute_global_explanations(self, X_sample: np.ndarray, output_dir: Path) -> pd.DataFrame:
        """
        Computes SHAP values on representative sample, plots summary beeswarm and bar charts,
        and returns mean absolute SHAP feature rankings.
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized.")
            
        logger.info(f"Computing global SHAP values for sample of {len(X_sample)} encounters...")
        shap_values = self.explainer(X_sample)
        
        # Calculate mean absolute SHAP
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        df_importance = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)
        self.global_importance_ = df_importance
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. SHAP Feature Importance Bar Plot
        fig, ax = plt.subplots(figsize=(10, 6.5))
        top_15 = df_importance.head(15).iloc[::-1]
        bars = ax.barh(top_15["feature"], top_15["mean_abs_shap"], color="#0284c7", edgecolor="#000000", linewidth=0.6)
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", fontsize=8.5)
            
        ax.set_title("Global SHAP Feature Importance (Mean |SHAP Value|)", fontweight="bold", pad=12)
        ax.set_xlabel("Mean Absolute SHAP Value (Impact on Log-Odds Prediction)")
        ax.set_xlim(0, max(top_15["mean_abs_shap"]) * 1.18)
        fig.savefig(output_dir / "11_shap_importance_bar.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        
        # 2. SHAP Summary Beeswarm Plot
        fig, ax = plt.subplots(figsize=(11, 7))
        # Pass features and names explicitly
        shap.summary_plot(
            shap_values.values, X_sample,
            feature_names=self.feature_names,
            max_display=15, show=False
        )
        plt.title("SHAP Beeswarm: Directional Impact on Readmission Log-Odds", fontweight="bold", pad=12)
        plt.savefig(output_dir / "12_shap_summary_beeswarm.png", dpi=200, bbox_inches="tight")
        plt.close()
        
        # Save rankings to JSON
        top_records = df_importance.head(20).to_dict(orient="records")
        with open(resolve_path("reports/shap_global_importance.json"), "w", encoding="utf-8") as f:
            json.dump(top_records, f, indent=2)
            
        logger.info(f"Global SHAP analysis complete. Figures saved to {output_dir}")
        return df_importance

    def explain_patient(self, x_patient: np.ndarray, top_k: int = 8) -> Dict[str, Any]:
        """
        Computes patient-level local explanation.
        Strictly frames factors as:
        'contributed to the model prediction', NOT 'caused readmission'.
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized.")
            
        if x_patient.ndim == 1:
            x_patient = x_patient.reshape(1, -1)
            
        explanation = self.explainer(x_patient)
        shap_vals = explanation.values[0]
        base_val = float(explanation.base_values[0])
        feature_vals = x_patient[0]
        
        items = []
        for name, s_val, f_val in zip(self.feature_names, shap_vals, feature_vals):
            items.append({
                "feature": name,
                "shap_value": round(float(s_val), 4),
                "feature_value": round(float(f_val), 4),
                "direction": "INCREASED_RISK" if s_val > 0 else "DECREASED_RISK",
                "abs_impact": abs(float(s_val))
            })
            
        # Sort by absolute impact
        items_sorted = sorted(items, key=lambda x: x["abs_impact"], reverse=True)
        top_factors = items_sorted[:top_k]
        
        risk_increasing = [item for item in top_factors if item["direction"] == "INCREASED_RISK"]
        risk_decreasing = [item for item in top_factors if item["direction"] == "DECREASED_RISK"]
        
        return {
            "base_value_log_odds": round(base_val, 4),
            "patient_log_odds": round(float(base_val + sum(shap_vals)), 4),
            "top_factors": top_factors,
            "risk_increasing_factors": risk_increasing,
            "risk_decreasing_factors": risk_decreasing,
            "disclaimer": "These factors contributed to the mathematical model output. They do NOT represent clinical causality."
        }

    def save_artifact(self, output_path: str = None):
        """Saves explainer and feature names to disk."""
        if output_path is None:
            output_path = resolve_path(self.config["paths"]["explainer"])
        else:
            output_path = Path(output_path)
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "explainer": self.explainer,
            "feature_names": self.feature_names,
            "global_importance": self.global_importance_
        }, output_path)
        logger.info(f"SHAP explainer artifact saved to {output_path}")

    @classmethod
    def load_artifact(cls, artifact_path: str = None) -> "ClinicalShapExplainer":
        """Loads fitted explainer from disk."""
        config = load_config()
        if artifact_path is None:
            artifact_path = resolve_path(config["paths"]["explainer"])
        else:
            artifact_path = Path(artifact_path)
            
        data = joblib.load(artifact_path)
        instance = cls(config)
        instance.explainer = data["explainer"]
        instance.feature_names = data["feature_names"]
        instance.global_importance_ = data["global_importance"]
        return instance

if __name__ == "__main__":
    config = load_config()
    preprocessor = ClinicalDataPreprocessor.load_artifacts()
    
    val_df = pd.read_parquet(resolve_path(config["paths"]["processed_val"]))
    X_val_proc = preprocessor.transform(val_df)
    
    uncal_data = joblib.load(resolve_path(config["paths"]["uncalibrated_model"]))
    base_model = uncal_data["model"]
    
    # Take representative background sample of 500 rows for global SHAP
    np.random.seed(42)
    sample_indices = np.random.choice(len(X_val_proc), size=min(600, len(X_val_proc)), replace=False)
    X_sample = X_val_proc[sample_indices]
    
    explainer = ClinicalShapExplainer(config)
    explainer.fit_or_load_explainer(base_model, X_sample, preprocessor.feature_names_)
    df_imp = explainer.compute_global_explanations(X_sample, resolve_path("reports/figures"))
    explainer.save_artifact()
    
    # Test on a single high-risk patient
    patient_idx = 0
    local_exp = explainer.explain_patient(X_sample[patient_idx])
    
    print("\n--- SHAP EXPLAINABILITY COMPLETE ---")
    print("Top 5 Global Predictive Features:")
    for _, row in df_imp.head(5).iterrows():
        print(f"  {row['feature']:<35}: {row['mean_abs_shap']:.4f}")
    print("\nSample Patient Local Explanation:")
    print("  Factors increasing predicted risk:")
    for f in local_exp["risk_increasing_factors"][:3]:
        print(f"    + {f['feature']} (SHAP: {f['shap_value']:+.4f})")
    print("  Factors decreasing predicted risk:")
    for f in local_exp["risk_decreasing_factors"][:3]:
        print(f"    - {f['feature']} (SHAP: {f['shap_value']:+.4f})")
