"""
Model comparison and comprehensive evaluation benchmark generator.
Evaluates Dummy, Logistic Regression, Random Forest, XGBoost (Raw), and XGBoost (Calibrated)
on the exact held-out test split and writes reports/model_comparison.json.
"""

import sys
import json
import joblib
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.logger import setup_logger
from src.evaluation.metrics import compute_classification_metrics, compute_curves
from src.preprocessing.pipeline import ClinicalDataPreprocessor

logger = setup_logger("model_comparison")

def generate_comparison_suite():
    config = load_config()
    preprocessor = ClinicalDataPreprocessor.load_artifacts()
    
    test_df = pd.read_parquet(resolve_path(config["paths"]["processed_test"]))
    y_test = test_df["target_readmitted_30d"].values
    X_test_proc = preprocessor.transform(test_df)
    
    # 1. Load baseline models
    baselines_artifact = joblib.load(resolve_path(config["paths"]["baseline_models"]))
    dummy_model = baselines_artifact["models"]["Dummy Classifier"]
    logreg_model = baselines_artifact["models"]["Logistic Regression"]
    rf_model = baselines_artifact["models"]["Random Forest"]
    
    # 2. Load uncalibrated and calibrated XGBoost
    uncal_artifact = joblib.load(resolve_path(config["paths"]["uncalibrated_model"]))
    xgb_raw = uncal_artifact["model"]
    
    cal_model = joblib.load(resolve_path(config["paths"]["model"]))
    
    # Operating threshold for comparison:
    # 0.50 for raw models (which use scale_pos_weight or class_weight='balanced')
    # 0.125 for calibrated model (F1-optimal probability threshold aligned with 9% base rate)
    models_to_evaluate = {
        "Dummy Classifier": (dummy_model, 0.50, False),
        "Logistic Regression": (logreg_model, 0.50, False),
        "Random Forest": (rf_model, 0.50, False),
        "XGBoost (Uncalibrated)": (xgb_raw, 0.50, False),
        "XGBoost (Calibrated, 0.125 Opt Threshold)": (cal_model, 0.125, True),
        "XGBoost (Calibrated, 0.50 Threshold)": (cal_model, 0.50, True)
    }
    
    comparison_table = []
    detailed_results = {}
    
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5.5))
    ax_roc.plot([0, 1], [0, 1], "k--", label="Chance (AUC=0.50)")
    ax_pr.axhline(float(np.mean(y_test)), color="k", linestyle="--", label=f"Prevalence ({np.mean(y_test)*100:.1f}%)")
    
    colors = {
        "Dummy Classifier": "#94a3b8",
        "Logistic Regression": "#3b82f6",
        "Random Forest": "#8b5cf6",
        "XGBoost (Uncalibrated)": "#f97316",
        "XGBoost (Calibrated, 0.125 Opt Threshold)": "#0d9488"
    }
    
    for name, (model, thresh, is_cal) in models_to_evaluate.items():
        probs = model.predict_proba(X_test_proc)[:, 1]
        metrics = compute_classification_metrics(y_test, probs, threshold=thresh)
        curves = compute_curves(y_test, probs)
        
        row = {
            "model": name,
            "threshold": thresh,
            "pr_auc": metrics["pr_auc"],
            "roc_auc": metrics["roc_auc"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "f1": metrics["f1"],
            "specificity": metrics["specificity"],
            "brier_score": metrics["brier_score"]
        }
        comparison_table.append(row)
        detailed_results[name] = {
            "metrics": metrics,
            "curves": curves
        }
        
        # Plot curves for primary variants
        if name in colors:
            color = colors[name]
            # ROC
            fpr = curves["roc_curve"]["fpr"]
            tpr = curves["roc_curve"]["tpr"]
            ax_roc.plot(fpr, tpr, label=f"{name} (AUC={metrics['roc_auc']:.3f})", color=color, linewidth=1.8)
            
            # PR
            rec = curves["pr_curve"]["recall"]
            prec = curves["pr_curve"]["precision"]
            ax_pr.plot(rec, prec, label=f"{name} (PR-AUC={metrics['pr_auc']:.3f})", color=color, linewidth=1.8)

    ax_roc.set_title("ROC Curves Comparison (Test Set)", fontweight="bold", pad=10)
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)")
    ax_roc.set_ylabel("True Positive Rate (Recall)")
    ax_roc.legend(loc="lower right", fontsize=8.5)
    
    ax_pr.set_title("Precision-Recall Curves Comparison (Test Set)", fontweight="bold", pad=10)
    ax_pr.set_xlabel("Recall (Sensitivity)")
    ax_pr.set_ylabel("Precision (Positive Predictive Value)")
    ax_pr.legend(loc="upper right", fontsize=8.5)
    
    fig_path = resolve_path("reports/figures/10_model_comparison_curves.png")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    
    # Save comparison report JSON
    output_report = {
        "comparison_table": comparison_table,
        "detailed_results": detailed_results,
        "prevalence": float(np.mean(y_test)),
        "test_records": len(y_test)
    }
    
    out_path = resolve_path(config["paths"]["metrics_report"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_report, f, indent=2)
        
    logger.info(f"Model comparison report saved to {out_path}")
    
    # Display table in terminal
    df_cmp = pd.DataFrame(comparison_table)
    print("\n" + "="*85)
    print("                    FLAGSHIP MODEL BENCHMARK COMPARISON TABLE")
    print("="*85)
    header = f"{'Model':<42} | {'PR-AUC':<7} | {'ROC-AUC':<7} | {'Recall':<7} | {'F1':<7} | {'Brier':<7}"
    print(header)
    print("-" * 85)
    for r in comparison_table:
        print(f"{r['model']:<42} | {r['pr_auc']:<7.4f} | {r['roc_auc']:<7.4f} | {r['recall']:<7.4f} | {r['f1']:<7.4f} | {r['brier_score']:<7.4f}")
    print("="*85)

if __name__ == "__main__":
    generate_comparison_suite()
