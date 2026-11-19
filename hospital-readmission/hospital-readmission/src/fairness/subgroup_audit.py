"""
Fairness and demographic subgroup performance audit module for Hospital Readmission Risk Intelligence.
Evaluates model calibration and error disparities across Age, Gender, and Race categories.
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
from src.preprocessing.pipeline import ClinicalDataPreprocessor

logger = setup_logger("subgroup_fairness")

class SubgroupFairnessAuditor:
    """Evaluates error rate disparities and performance across demographic strata."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()

    def audit_subgroups(
        self,
        df_test: pd.DataFrame,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.125
    ) -> Dict[str, Any]:
        """Audits model performance across age, gender, and race categories."""
        logger.info(f"Auditing demographic subgroup metrics at threshold={threshold}...")
        
        # Define subgroups
        df_eval = df_test.copy()
        df_eval["y_true"] = y_true
        df_eval["y_prob"] = y_prob
        df_eval["y_pred"] = (y_prob >= threshold).astype(int)
        
        # 1. Age simplification
        def bin_age(age_val):
            val_s = str(age_val)
            if any(dec in val_s for dec in ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)"]):
                return "< 50 years"
            elif any(dec in val_s for dec in ["[50-60)", "[60-70)"]):
                return "50 - 69 years"
            else:
                return "70+ years"
                
        df_eval["age_group"] = df_eval["age"].apply(bin_age)
        
        # 2. Gender
        df_eval["gender_clean"] = df_eval["gender"].replace({"Unknown/Invalid": "Unknown", np.nan: "Unknown"})
        df_eval["gender_clean"] = df_eval["gender_clean"].apply(lambda x: x if x in ["Female", "Male"] else "Other/Unknown")
        
        # 3. Race
        df_eval["race_clean"] = df_eval["race"].fillna("Missing")
        df_eval["race_clean"] = df_eval["race_clean"].apply(
            lambda x: x if x in ["Caucasian", "AfricanAmerican"] else "Other/Unreported"
        )
        
        attributes = {
            "Age Group": "age_group",
            "Gender": "gender_clean",
            "Race / Ethnicity": "race_clean"
        }
        
        subgroup_results = {}
        for attr_label, col in attributes.items():
            attr_report = {}
            for grp_name, grp_df in df_eval.groupby(col):
                if len(grp_df) < 30:
                    continue
                g_true = grp_df["y_true"].values
                g_prob = grp_df["y_prob"].values
                m = compute_classification_metrics(g_true, g_prob, threshold=threshold)
                
                attr_report[grp_name] = {
                    "sample_size": int(len(grp_df)),
                    "sample_pct": round(float(len(grp_df) / len(df_eval) * 100), 1),
                    "base_readmission_rate": round(float(np.mean(g_true) * 100), 2),
                    "selection_rate": round(float(np.mean(grp_df["y_pred"]) * 100), 2),
                    "recall": m["recall"],
                    "precision": m["precision"],
                    "f1": m["f1"],
                    "specificity": m["specificity"],
                    "fpr": m["false_positive_rate"],
                    "fnr": m["false_negative_rate"],
                    "roc_auc": m["roc_auc"],
                    "pr_auc": m["pr_auc"]
                }
            subgroup_results[attr_label] = attr_report
            
        # Disparity summary
        disparity_summary = {}
        for attr_label, grps in subgroup_results.items():
            recalls = [v["recall"] for v in grps.values() if v["recall"] > 0]
            sel_rates = [v["selection_rate"] for v in grps.values() if v["selection_rate"] > 0]
            if recalls:
                disparity_summary[f"{attr_label}_equal_opportunity_ratio"] = round(min(recalls) / max(recalls), 3)
            if sel_rates:
                disparity_summary[f"{attr_label}_disparate_impact_ratio"] = round(min(sel_rates) / max(sel_rates), 3)

        final_report = {
            "operating_threshold": threshold,
            "disclaimer": (
                "RESPONSIBLE AI / FAIRNESS AUDIT: "
                "Performance varies across demographic subgroups. "
                "This model is a research prototype trained on historical data and has not been clinically validated. "
                "Disparities reflect historical documentation practices, social determinants of health, and epidemiological variations."
            ),
            "disparity_summary": disparity_summary,
            "subgroups": subgroup_results
        }
        
        # Save JSON
        rep_path = resolve_path("reports/subgroup_fairness_report.json")
        rep_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rep_path, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)
            
        self._plot_subgroup_disparities(subgroup_results, resolve_path("reports/figures"))
        logger.info(f"Subgroup fairness report saved to {rep_path}")
        return final_report

    def _plot_subgroup_disparities(self, subgroup_results: Dict[str, Any], fig_dir: Path):
        """Generates multi-attribute subgroup comparison bar chart."""
        fig_dir.mkdir(parents=True, exist_ok=True)
        
        rows = []
        for attr, grps in subgroup_results.items():
            for g_name, metrics in grps.items():
                rows.append({
                    "Subgroup": f"{attr}: {g_name}",
                    "Recall": metrics["recall"],
                    "Precision": metrics["precision"],
                    "F1": metrics["f1"]
                })
        df_plot = pd.DataFrame(rows)
        
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(df_plot))
        width = 0.25
        
        ax.bar(x - width, df_plot["Recall"], width, label="Recall (Sensitivity)", color="#0284c7")
        ax.bar(x, df_plot["Precision"], width, label="Precision", color="#0d9488")
        ax.bar(x + width, df_plot["F1"], width, label="F1-Score", color="#f59e0b")
        
        ax.set_ylabel("Metric Value")
        ax.set_title("Subgroup Performance Disparity Audit (Operating Threshold)", fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(df_plot["Subgroup"], rotation=35, ha="right", fontsize=9)
        ax.set_ylim(0, 0.6)
        ax.legend(loc="upper right")
        
        fig_path = fig_dir / "13_subgroup_fairness.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    config = load_config()
    preprocessor = ClinicalDataPreprocessor.load_artifacts()
    test_df = pd.read_parquet(resolve_path(config["paths"]["processed_test"]))
    y_test = test_df["target_readmitted_30d"].values
    X_test_proc = preprocessor.transform(test_df)
    
    cal_model = joblib.load(resolve_path(config["paths"]["model"]))
    cal_probs = cal_model.predict_proba(X_test_proc)[:, 1]
    
    auditor = SubgroupFairnessAuditor(config)
    report = auditor.audit_subgroups(test_df, y_test, cal_probs, threshold=0.125)
    
    print("\n--- FAIRNESS / SUBGROUP AUDIT COMPLETE ---")
    for attr, grps in report["subgroups"].items():
        print(f"\n{attr}:")
        for g_name, m in grps.items():
            print(f"  {g_name:<18} (N={m['sample_size']:<5}, Base Rate={m['base_readmission_rate']:>5.2f}%): Recall={m['recall']:.4f}, Prec={m['precision']:.4f}, F1={m['f1']:.4f}")
