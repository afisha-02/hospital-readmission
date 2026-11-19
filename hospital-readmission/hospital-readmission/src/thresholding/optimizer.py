"""
Threshold optimization and sensitivity analysis across operating cutoffs.
Sweeps operating thresholds, calculates clinical confusion metrics, and identifies optimal operating points.
"""

import sys
import json
import joblib
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.logger import setup_logger
from src.evaluation.metrics import compute_classification_metrics
from src.thresholding.cost_matrix import CostMatrix
from src.preprocessing.pipeline import ClinicalDataPreprocessor

logger = setup_logger("threshold_optimizer")

class ThresholdOptimizer:
    """Evaluates classification performance and cost curves across decision thresholds."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        cfg_costs = self.config.get("cost_matrix", {})
        self.cost_matrix = CostMatrix(
            cost_fn=cfg_costs.get("cost_fn", 5000.0),
            cost_fp=cfg_costs.get("cost_fp", 500.0),
            cost_tp=cfg_costs.get("cost_tp", 750.0),
            cost_tn=cfg_costs.get("cost_tn", 0.0)
        )

    def evaluate_grid(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: List[float] = None
    ) -> List[Dict[str, Any]]:
        """Sweeps thresholds and computes precision, recall, F1, specificity, FPR, FNR, and expected cost."""
        if thresholds is None:
            thresholds = [round(t, 3) for t in np.arange(0.05, 0.96, 0.025)]
            
        results = []
        for t in thresholds:
            m = compute_classification_metrics(y_true, y_prob, threshold=t)
            cm = m["confusion_matrix"]
            cost_eval = self.cost_matrix.calculate_cost(cm["tn"], cm["fp"], cm["fn"], cm["tp"])
            
            row = {
                "threshold": round(float(t), 3),
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "sensitivity": m["recall"],
                "specificity": m["specificity"],
                "f1": m["f1"],
                "fpr": m["false_positive_rate"],
                "fnr": m["false_negative_rate"],
                "tp": cm["tp"],
                "fp": cm["fp"],
                "fn": cm["fn"],
                "tn": cm["tn"],
                "total_cost": cost_eval["total_cost"],
                "cost_per_patient": cost_eval["per_patient_cost"],
                "youden_j": round(float(m["recall"] + m["specificity"] - 1.0), 4)
            }
            results.append(row)
            
        return results

    def find_optimal_thresholds(self, grid_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identifies F1-optimal, Youden-optimal, and Cost-optimal operating thresholds."""
        df_grid = pd.DataFrame(grid_results)
        
        # Max F1
        f1_opt = df_grid.loc[df_grid["f1"].idxmax()].to_dict()
        
        # Max Youden's J Index (Sensitivity + Specificity - 1)
        youden_opt = df_grid.loc[df_grid["youden_j"].idxmax()].to_dict()
        
        # Min Expected Cost
        cost_opt = df_grid.loc[df_grid["cost_per_patient"].idxmin()].to_dict()
        
        # Baseline at 0.50
        baseline_idx = (df_grid["threshold"] - 0.50).abs().idxmin()
        default_05 = df_grid.loc[baseline_idx].to_dict()
        
        cost_savings = default_05["total_cost"] - cost_opt["total_cost"]
        cost_savings_pct = (cost_savings / default_05["total_cost"] * 100) if default_05["total_cost"] > 0 else 0
        
        return {
            "f1_optimal": f1_opt,
            "youden_balanced_optimal": youden_opt,
            "cost_optimal": cost_opt,
            "default_0_50": default_05,
            "cost_savings_at_optimal": round(float(cost_savings), 2),
            "cost_savings_pct": round(float(cost_savings_pct), 2)
        }

    def generate_plots(self, grid_results: List[Dict[str, Any]], output_dir: Path):
        """Generates visual threshold tradeoff curves and expected cost curves."""
        df_grid = pd.DataFrame(grid_results)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Trade-off Curves
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(df_grid["threshold"], df_grid["precision"], "b-", label="Precision", linewidth=2)
        ax.plot(df_grid["threshold"], df_grid["recall"], "g-", label="Recall (Sensitivity)", linewidth=2)
        ax.plot(df_grid["threshold"], df_grid["f1"], "m--", label="F1-Score", linewidth=2)
        ax.plot(df_grid["threshold"], df_grid["specificity"], "c:", label="Specificity", linewidth=1.8)
        
        # Mark max F1
        best_f1_row = df_grid.loc[df_grid["f1"].idxmax()]
        ax.scatter([best_f1_row["threshold"]], [best_f1_row["f1"]], color="red", s=80, zorder=5, label=f"Max F1 ({best_f1_row['threshold']:.2f})")
        
        ax.set_title("Performance Trade-offs Across Operating Thresholds", fontweight="bold", pad=12)
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Metric Value")
        ax.set_xlim([0.05, 0.95])
        ax.set_ylim([0.0, 1.05])
        ax.legend(loc="center right")
        fig.savefig(output_dir / "08_threshold_tradeoffs.png", dpi=200)
        plt.close(fig)
        
        # Plot 2: Expected Cost Curve
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(df_grid["threshold"], df_grid["cost_per_patient"], color="#e11d48", linewidth=2.2, label="Expected Cost / Patient ($)")
        
        best_cost_row = df_grid.loc[df_grid["cost_per_patient"].idxmin()]
        ax.scatter([best_cost_row["threshold"]], [best_cost_row["cost_per_patient"]], color="#0d9488", s=100, zorder=5,
                   label=f"Cost Optimal ({best_cost_row['threshold']:.2f}: ${best_cost_row['cost_per_patient']:.0f})")
        
        ax.set_title("Illustrative Expected Clinical Cost Curve by Decision Threshold", fontweight="bold", pad=12)
        ax.set_xlabel("Decision Threshold")
        ax.set_ylabel("Expected Cost Per Patient ($)")
        ax.legend(loc="upper right")
        fig.savefig(output_dir / "09_expected_cost_curve.png", dpi=200)
        plt.close(fig)

if __name__ == "__main__":
    config = load_config()
    preprocessor = ClinicalDataPreprocessor.load_artifacts()
    test_df = pd.read_parquet(resolve_path(config["paths"]["processed_test"]))
    y_test = test_df["target_readmitted_30d"].values
    X_test_proc = preprocessor.transform(test_df)
    
    cal_model = joblib.load(resolve_path(config["paths"]["model"]))
    cal_probs = cal_model.predict_proba(X_test_proc)[:, 1]
    
    optimizer = ThresholdOptimizer(config)
    grid = optimizer.evaluate_grid(y_test, cal_probs)
    optimal_pts = optimizer.find_optimal_thresholds(grid)
    optimizer.generate_plots(grid, resolve_path("reports/figures"))
    
    out_json = resolve_path("reports/threshold_analysis.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "optimal_thresholds": optimal_pts,
            "grid_evaluation": grid
        }, f, indent=2)
        
    print("\n--- THRESHOLD OPTIMIZATION SUMMARY ---")
    print(f"F1-Optimal Threshold:       {optimal_pts['f1_optimal']['threshold']} (F1={optimal_pts['f1_optimal']['f1']:.4f}, Recall={optimal_pts['f1_optimal']['recall']:.4f})")
    print(f"Youden Balanced Threshold:  {optimal_pts['youden_balanced_optimal']['threshold']} (Sensitivity={optimal_pts['youden_balanced_optimal']['recall']:.4f}, Specificity={optimal_pts['youden_balanced_optimal']['specificity']:.4f})")
    print(f"Cost-Optimal Threshold:     {optimal_pts['cost_optimal']['threshold']} (Cost/patient: ${optimal_pts['cost_optimal']['cost_per_patient']:.2f})")
    print(f"Cost Savings vs 0.50 Naive: ${optimal_pts['cost_savings_at_optimal']:,} ({optimal_pts['cost_savings_pct']:.1f}%)")
