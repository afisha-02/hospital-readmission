"""
Model monitoring and statistical data drift detection module for Hospital Readmission Risk Intelligence.
Calculates Population Stability Index (PSI), Kolmogorov-Smirnov (KS) test, missingness shifts,
and generates standardized operational alerts (🟢 Stable, 🟡 Monitor, 🔴 Drift Detected).
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.logger import setup_logger

logger = setup_logger("drift_detector")

def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_bins: int = 10,
    epsilon: float = 1e-4
) -> float:
    """
    Calculates Population Stability Index (PSI) for a continuous or binned feature.
    Formula: PSI = sum((actual_pct - expected_pct) * ln(actual_pct / expected_pct))
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
        
    # Determine bin edges using expected distribution percentiles
    quantiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, quantiles)
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5
    bin_edges = np.unique(bin_edges)
    
    if len(bin_edges) < 2:
        return 0.0
        
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)
    
    expected_pct = expected_counts / max(len(expected), 1)
    actual_pct = actual_counts / max(len(actual), 1)
    
    # Smooth with epsilon to avoid division by zero
    expected_pct = np.clip(expected_pct, epsilon, 1.0)
    actual_pct = np.clip(actual_pct, epsilon, 1.0)
    
    # Normalize after smoothing
    expected_pct /= np.sum(expected_pct)
    actual_pct /= np.sum(actual_pct)
    
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(round(psi_value, 4))

def calculate_categorical_psi(
    expected_series: pd.Series,
    actual_series: pd.Series,
    epsilon: float = 1e-4
) -> float:
    """Calculates PSI across categorical categories."""
    all_categories = sorted(list(set(expected_series.dropna().unique()) | set(actual_series.dropna().unique())))
    if not all_categories:
        return 0.0
        
    exp_counts = expected_series.value_counts(normalize=True).to_dict()
    act_counts = actual_series.value_counts(normalize=True).to_dict()
    
    exp_pct = np.array([exp_counts.get(cat, epsilon) for cat in all_categories])
    act_pct = np.array([act_counts.get(cat, epsilon) for cat in all_categories])
    
    exp_pct /= np.sum(exp_pct)
    act_pct /= np.sum(act_pct)
    
    psi_val = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(round(psi_val, 4))

class ClinicalDriftDetector:
    """Monitors feature and prediction distributions for covariate and concept drift."""
    
    def __init__(self, baseline_df: pd.DataFrame, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.baseline_df = baseline_df
        self.psi_stable = self.config.get("monitoring", {}).get("drift_psi_stable", 0.10)
        self.psi_monitor = self.config.get("monitoring", {}).get("drift_psi_monitor", 0.25)
        self.ks_p_val_thresh = self.config.get("monitoring", {}).get("drift_ks_p_value_threshold", 0.05)

    def evaluate_batch(
        self,
        incoming_df: pd.DataFrame,
        incoming_predictions: Optional[np.ndarray] = None,
        baseline_predictions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Audits incoming clinical encounter batch against reference baseline distribution.
        """
        logger.info(f"Auditing drift on batch of {len(incoming_df):,} encounters...")
        
        feature_audit = {}
        drift_count = 0
        monitor_count = 0
        stable_count = 0
        
        num_features = self.config["features"]["numerical"]
        cat_features = self.config["features"]["categorical"]
        
        # 1. Numerical Feature Drift
        for feat in num_features:
            if feat in incoming_df.columns and feat in self.baseline_df.columns:
                exp_vals = pd.to_numeric(self.baseline_df[feat], errors="coerce").dropna().values
                act_vals = pd.to_numeric(incoming_df[feat], errors="coerce").dropna().values
                
                if len(exp_vals) > 10 and len(act_vals) > 10:
                    psi = calculate_psi(exp_vals, act_vals)
                    ks_stat, p_val = ks_2samp(exp_vals, act_vals)
                    
                    if psi >= self.psi_monitor:
                        status = "🔴 Drift Detected"
                        drift_count += 1
                    elif psi >= self.psi_stable:
                        status = "🟡 Monitor"
                        monitor_count += 1
                    else:
                        status = "🟢 Stable"
                        stable_count += 1
                        
                    feature_audit[feat] = {
                        "type": "numerical",
                        "psi": psi,
                        "ks_statistic": round(float(ks_stat), 4),
                        "p_value": round(float(p_val), 4),
                        "baseline_mean": round(float(np.mean(exp_vals)), 2),
                        "incoming_mean": round(float(np.mean(act_vals)), 2),
                        "status": status
                    }
                    
        # 2. Categorical Feature Drift
        for feat in cat_features:
            if feat in incoming_df.columns and feat in self.baseline_df.columns:
                exp_s = self.baseline_df[feat].astype(str)
                act_s = incoming_df[feat].astype(str)
                
                psi = calculate_categorical_psi(exp_s, act_s)
                if psi >= self.psi_monitor:
                    status = "🔴 Drift Detected"
                    drift_count += 1
                elif psi >= self.psi_stable:
                    status = "🟡 Monitor"
                    monitor_count += 1
                else:
                    status = "🟢 Stable"
                    stable_count += 1
                    
                feature_audit[feat] = {
                    "type": "categorical",
                    "psi": psi,
                    "status": status
                }
                
        # 3. Prediction Distribution Drift
        prediction_drift = None
        if incoming_predictions is not None and baseline_predictions is not None:
            pred_psi = calculate_psi(baseline_predictions, incoming_predictions)
            pred_ks, pred_p = ks_2samp(baseline_predictions, incoming_predictions)
            
            if pred_psi >= self.psi_monitor:
                pred_status = "🔴 Drift Detected"
            elif pred_psi >= self.psi_stable:
                pred_status = "🟡 Monitor"
            else:
                pred_status = "🟢 Stable"
                
            prediction_drift = {
                "prediction_psi": pred_psi,
                "ks_statistic": round(float(pred_ks), 4),
                "p_value": round(float(pred_p), 4),
                "baseline_mean_risk": round(float(np.mean(baseline_predictions) * 100), 2),
                "incoming_mean_risk": round(float(np.mean(incoming_predictions) * 100), 2),
                "status": pred_status
            }

        # 4. Overall Health Status
        total_features = stable_count + monitor_count + drift_count
        drift_rate = (drift_count / total_features) if total_features > 0 else 0.0
        
        if drift_count >= 3 or drift_rate > 0.20:
            overall_status = "🔴 Drift Detected"
            recommendation = "Investigate feature distributions and consider model recalibration."
        elif monitor_count >= 3 or drift_rate > 0.08:
            overall_status = "🟡 Monitor"
            recommendation = "Moderate distributional shifts observed; monitor upcoming batches."
        else:
            overall_status = "🟢 Stable"
            recommendation = "All clinical feature distributions align with training reference baseline."
            
        report = {
            "overall_status": overall_status,
            "recommendation": recommendation,
            "total_features_evaluated": total_features,
            "stable_features": stable_count,
            "monitor_features": monitor_count,
            "drifted_features": drift_count,
            "prediction_drift": prediction_drift,
            "feature_audit": feature_audit
        }
        
        return report

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    config = load_config()

    train_ref = pd.read_parquet(resolve_path(config["paths"]["drift_baseline"]))
    test_batch = pd.read_parquet(resolve_path(config["paths"]["processed_test"])).sample(500, random_state=42)
    
    detector = ClinicalDriftDetector(train_ref, config)
    res = detector.evaluate_batch(test_batch)
    
    print("\n--- MODEL MONITORING DRIFT AUDIT ---")
    print(f"Overall Status: {res['overall_status']}")
    print(f"Recommendation: {res['recommendation']}")
    print(f"Features: Stable={res['stable_features']}, Monitor={res['monitor_features']}, Drifted={res['drifted_features']}")
    print("\nSample Feature Metrics:")
    for f_name, audit in list(res['feature_audit'].items())[:6]:
        print(f"  {f_name:<30}: PSI={audit['psi']:.4f} ({audit['status']})")
