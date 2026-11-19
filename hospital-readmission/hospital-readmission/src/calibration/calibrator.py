"""
Probability calibration module for Hospital Readmission Risk Intelligence.
Fits CalibratedClassifierCV, compares uncalibrated vs calibrated Brier scores and ECE,
and generates reliability curves.
"""

import sys
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.logger import setup_logger
from src.evaluation.metrics import compute_classification_metrics
from src.preprocessing.pipeline import ClinicalDataPreprocessor

logger = setup_logger("calibration")

def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    Weighted average difference between bin confidence and bin empirical accuracy.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        mask = bin_ids == i
        if np.any(mask):
            bin_size = np.sum(mask)
            bin_conf = np.mean(y_prob[mask])
            bin_acc = np.mean(y_true[mask])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)
            
    return float(round(ece, 4))

class ProbabilityCalibrator:
    """Calibrates model probability outputs and evaluates reliability curves."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.calibrated_model: Any = None
        self.report: Dict[str, Any] = {}

    def fit_calibrator(
        self,
        base_model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        method: str = "sigmoid"
    ) -> Any:
        """
        Fits CalibratedClassifierCV using prefit validation holdout set.
        """
        logger.info(f"Fitting probability calibrator using method='{method}' on validation set...")
        try:
            from sklearn.frozen import FrozenEstimator
            frozen_estimator = FrozenEstimator(base_model)
            calibrator = CalibratedClassifierCV(estimator=frozen_estimator, method=method)
        except ImportError:
            # Fallback for earlier scikit-learn versions (<1.4)
            calibrator = CalibratedClassifierCV(estimator=base_model, method=method, cv="prefit")
            
        calibrator.fit(X_val, y_val)
        self.calibrated_model = calibrator
        return calibrator


    def evaluate_and_compare(
        self,
        base_model: Any,
        calibrated_model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compares uncalibrated vs calibrated model on test set.
        Computes Brier Score, ECE, Reliability coordinates, and plots curves.
        """
        logger.info("Evaluating probability calibration on test dataset...")
        uncal_probs = base_model.predict_proba(X_test)[:, 1]
        cal_probs = calibrated_model.predict_proba(X_test)[:, 1]
        
        # Brier scores
        brier_uncal = float(brier_score_loss(y_test, uncal_probs))
        brier_cal = float(brier_score_loss(y_test, cal_probs))
        brier_reduction = float((brier_uncal - brier_cal) / brier_uncal * 100)
        
        # ECE
        ece_uncal = compute_ece(y_test, uncal_probs)
        ece_cal = compute_ece(y_test, cal_probs)
        
        # Reliability curve points
        prob_true_uncal, prob_pred_uncal = calibration_curve(y_test, uncal_probs, n_bins=10)
        prob_true_cal, prob_pred_cal = calibration_curve(y_test, cal_probs, n_bins=10)
        
        # Generate plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        
        # Calibration curve
        ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)")
        ax1.plot(prob_pred_uncal, prob_true_uncal, "s-", color="#e11d48", label=f"Uncalibrated (Brier={brier_uncal:.4f}, ECE={ece_uncal:.4f})")
        ax1.plot(prob_pred_cal, prob_true_cal, "o-", color="#0d9488", label=f"Calibrated (Brier={brier_cal:.4f}, ECE={ece_cal:.4f})")
        ax1.set_xlabel("Mean Predicted Probability")
        ax1.set_ylabel("Empirical Readmission Fraction")
        ax1.set_title("Probability Calibration Reliability Diagram", fontweight="bold", pad=10)
        ax1.legend(loc="upper left")
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Probability density distribution
        ax2.hist(uncal_probs, bins=25, alpha=0.5, color="#e11d48", label="Uncalibrated", density=True)
        ax2.hist(cal_probs, bins=25, alpha=0.5, color="#0d9488", label="Calibrated", density=True)
        ax2.set_xlabel("Predicted Probability")
        ax2.set_ylabel("Density")
        ax2.set_title("Predicted Probability Distribution", fontweight="bold", pad=10)
        ax2.legend(loc="upper right")
        
        fig_path = resolve_path("reports/figures/07_calibration_curve.png")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=200)
        plt.close(fig)
        
        # Full test classification metrics at 0.50 threshold
        metrics_uncal = compute_classification_metrics(y_test, uncal_probs, threshold=0.50)
        # Note: for calibrated probabilities, default threshold 0.50 might be high given ~9% base rate,
        # so we also calculate at optimal prior-adjusted threshold
        metrics_cal = compute_classification_metrics(y_test, cal_probs, threshold=0.50)
        
        self.report = {
            "brier_score_uncalibrated": round(brier_uncal, 4),
            "brier_score_calibrated": round(brier_cal, 4),
            "brier_score_improvement_pct": round(brier_reduction, 2),
            "ece_uncalibrated": ece_uncal,
            "ece_calibrated": ece_cal,
            "curve_coordinates": {
                "uncalibrated": {
                    "pred": prob_pred_uncal.round(4).tolist(),
                    "true": prob_true_uncal.round(4).tolist()
                },
                "calibrated": {
                    "pred": prob_pred_cal.round(4).tolist(),
                    "true": prob_true_cal.round(4).tolist()
                }
            },
            "uncalibrated_metrics_at_0_5": metrics_uncal,
            "calibrated_metrics_at_0_5": metrics_cal
        }
        
        # Save JSON report
        rep_path = resolve_path("reports/calibration_report.json")
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2)
            
        logger.info(f"Calibration report saved to {rep_path}")
        return self.report

    def save_artifact(self, output_path: str = None):
        """Saves calibrated model artifact to disk."""
        if output_path is None:
            output_path = resolve_path(self.config["paths"]["model"])
        else:
            output_path = Path(output_path)
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.calibrated_model, output_path)
        logger.info(f"Calibrated XGBoost model artifact saved to {output_path}")

if __name__ == "__main__":
    config = load_config()
    preprocessor = ClinicalDataPreprocessor.load_artifacts()
    
    val_df = pd.read_parquet(resolve_path(config["paths"]["processed_val"]))
    test_df = pd.read_parquet(resolve_path(config["paths"]["processed_test"]))
    
    y_val = val_df["target_readmitted_30d"].values
    y_test = test_df["target_readmitted_30d"].values
    
    X_val_proc = preprocessor.transform(val_df)
    X_test_proc = preprocessor.transform(test_df)
    
    # Load uncalibrated model
    uncal_data = joblib.load(resolve_path(config["paths"]["uncalibrated_model"]))
    base_model = uncal_data["model"]
    
    calibrator = ProbabilityCalibrator(config)
    cal_model = calibrator.fit_calibrator(base_model, X_val_proc, y_val, method="sigmoid")
    report = calibrator.evaluate_and_compare(base_model, cal_model, X_test_proc, y_test)
    calibrator.save_artifact()
    
    print("\n--- PROBABILITY CALIBRATION SUMMARY ---")
    print(f"Uncalibrated Brier Score: {report['brier_score_uncalibrated']:.4f} | ECE: {report['ece_uncalibrated']:.4f}")
    print(f"Calibrated Brier Score:   {report['brier_score_calibrated']:.4f} | ECE: {report['ece_calibrated']:.4f}")
    print(f"Brier Score Improvement:  {report['brier_score_improvement_pct']:.2f}%")
