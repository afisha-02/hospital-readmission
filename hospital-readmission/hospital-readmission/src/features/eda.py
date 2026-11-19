"""
Exploratory Data Analysis (EDA) pipeline for Hospital Readmission Risk Intelligence.
Generates publication-quality figures and analytical summaries answering core clinical questions.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.logger import setup_logger

logger = setup_logger("eda")

# Healthcare visual palette (Slate, Navy, Teal, Amber, Crimson)
PRIMARY_COLOR = "#1e3a8a"    # Navy
SECONDARY_COLOR = "#0d9488"  # Teal
ACCENT_COLOR = "#e11d48"     # Crimson
BG_COLOR = "#f8fafc"         # Slate Light
TEXT_COLOR = "#0f172a"       # Slate Dark

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.family": "sans-serif",
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.autolayout": True
})

class ClinicalEDA:
    """Executes clinical exploratory data analysis and generates visual reports."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.fig_dir = resolve_path("reports/figures")
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    def run_analysis(self, df_clean: pd.DataFrame, df_raw: pd.DataFrame) -> Dict[str, Any]:
        """Runs the complete EDA suite and exports figures and summary data."""
        logger.info("Executing comprehensive Clinical EDA...")
        summary = {}

        # 1. Target Distribution Chart
        summary["target"] = self._plot_target_distribution(df_clean)

        # 2. Missing Value Analysis Chart (Raw dataset)
        summary["missingness"] = self._plot_missingness(df_raw)

        # 3. Readmission Rate by Prior Utilization
        summary["utilization"] = self._plot_readmission_by_utilization(df_clean)

        # 4. Readmission by Age & Length of Stay
        summary["demographics"] = self._plot_readmission_by_age_and_stay(df_clean)

        # 5. Readmission by Primary Diagnosis Group
        summary["diagnosis"] = self._plot_readmission_by_diagnosis(df_clean)

        # 6. Correlation Heatmap
        summary["correlations"] = self._plot_correlation_matrix(df_clean)

        # Save summary JSON
        summary_path = resolve_path("reports/eda_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"EDA suite complete. Figures saved to {self.fig_dir}")
        return summary

    def _plot_target_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Answers: What is the degree of class imbalance for 30-day readmissions?"""
        target_series = (df["readmitted"] == "<30").map({True: "Readmitted (<30d)", False: "No 30d Readmission"})
        counts = target_series.value_counts()
        pcts = target_series.value_counts(normalize=True) * 100

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(
            counts.index, counts.values,
            color=[PRIMARY_COLOR, ACCENT_COLOR],
            width=0.55, edgecolor="#000000", linewidth=0.8
        )
        for bar in bars:
            height = bar.get_height()
            pct = height / len(df) * 100
            ax.annotate(
                f"{height:,}\n({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

        ax.set_title("Target Distribution: 30-Day Hospital Readmission Rate", fontweight="bold", pad=12)
        ax.set_ylabel("Patient Encounter Count")
        ax.set_ylim(0, max(counts.values) * 1.18)
        fig.savefig(self.fig_dir / "01_target_distribution.png", dpi=200)
        plt.close(fig)

        return {
            "negative_count": int(counts.get("No 30d Readmission", 0)),
            "positive_count": int(counts.get("Readmitted (<30d)", 0)),
            "positive_rate_pct": round(float(pcts.get("Readmitted (<30d)", 0)), 2)
        }

    def _plot_missingness(self, df_raw: pd.DataFrame) -> Dict[str, Any]:
        """Answers: Which clinical features exhibit excessive missingness?"""
        missing_pct = (df_raw.isna().mean() * 100).sort_values(ascending=False)
        missing_top = missing_pct[missing_pct > 0.05].head(10)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(
            x=missing_top.values, y=missing_top.index,
            ax=ax, color=SECONDARY_COLOR, edgecolor="#000000", linewidth=0.8
        )
        ax.axvline(30, color=ACCENT_COLOR, linestyle="--", linewidth=1.5, label="30% Drop Threshold")
        for i, val in enumerate(missing_top.values):
            ax.text(val + 1, i, f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")

        ax.set_title("Missing Value Audit across Raw Clinical Features", fontweight="bold", pad=12)
        ax.set_xlabel("Missingness Percentage (%)")
        ax.set_xlim(0, 108)
        ax.legend(loc="lower right")
        fig.savefig(self.fig_dir / "02_missingness_audit.png", dpi=200)
        plt.close(fig)

        return missing_top.round(2).to_dict()

    def _plot_readmission_by_utilization(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Answers: How strongly does prior inpatient utilization escalate readmission risk?"""
        df_temp = df.copy()
        df_temp["inpatient_binned"] = df_temp["number_inpatient"].clip(upper=4).map({
            0: "0 visits", 1: "1 visit", 2: "2 visits", 3: "3 visits", 4: "4+ visits"
        })
        df_temp["is_readmitted"] = (df_temp["readmitted"] == "<30").astype(int)
        
        rates = df_temp.groupby("inpatient_binned")["is_readmitted"].agg(["count", "mean"])
        rates["readmission_pct"] = (rates["mean"] * 100).round(2)
        # Reindex to order
        order = ["0 visits", "1 visit", "2 visits", "3 visits", "4+ visits"]
        rates = rates.reindex([o for o in order if o in rates.index])

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(
            rates.index, rates["readmission_pct"],
            color="#0284c7", edgecolor="#000000", linewidth=0.8
        )
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )

        ax.set_title("30-Day Readmission Risk by Prior Inpatient Visits (Past 12 Months)", fontweight="bold", pad=12)
        ax.set_ylabel("Readmission Rate (%)")
        ax.set_ylim(0, max(rates["readmission_pct"]) * 1.25)
        fig.savefig(self.fig_dir / "03_readmission_by_prior_utilization.png", dpi=200)
        plt.close(fig)

        return rates["readmission_pct"].to_dict()

    def _plot_readmission_by_age_and_stay(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Answers: How do age demographics and length of hospital stay correlate with readmission?"""
        df_temp = df.copy()
        df_temp["is_readmitted"] = (df_temp["readmitted"] == "<30").astype(int)
        age_rates = df_temp.groupby("age")["is_readmitted"].mean() * 100

        fig, ax = plt.subplots(figsize=(9, 4.5))
        sns.barplot(
            x=age_rates.index, y=age_rates.values,
            ax=ax, color="#4f46e5", edgecolor="#000000", linewidth=0.8
        )
        for i, val in enumerate(age_rates.values):
            ax.text(i, val + 0.3, f"{val:.1f}%", ha="center", fontsize=8.5, fontweight="bold")

        ax.set_title("30-Day Readmission Rate across Age Deciles", fontweight="bold", pad=12)
        ax.set_ylabel("Readmission Rate (%)")
        ax.set_xlabel("Age Group")
        ax.set_ylim(0, max(age_rates.values) * 1.25)
        plt.xticks(rotation=30)
        fig.savefig(self.fig_dir / "04_readmission_by_age_and_stay.png", dpi=200)
        plt.close(fig)

        return age_rates.round(2).to_dict()

    def _plot_readmission_by_diagnosis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Answers: Which primary clinical diagnoses carry the highest readmission liability?"""
        df_temp = df.copy()
        df_temp["is_readmitted"] = (df_temp["readmitted"] == "<30").astype(int)
        diag_rates = (df_temp.groupby("primary_diagnosis_group")["is_readmitted"].mean() * 100).sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8.5, 5))
        sns.barplot(
            x=diag_rates.values, y=diag_rates.index,
            ax=ax, color="#059669", edgecolor="#000000", linewidth=0.8
        )
        for i, val in enumerate(diag_rates.values):
            ax.text(val + 0.2, i, f"{val:.1f}%", va="center", fontsize=9, fontweight="bold")

        ax.set_title("30-Day Readmission Rate by Primary Diagnostic Organ System", fontweight="bold", pad=12)
        ax.set_xlabel("Readmission Rate (%)")
        ax.set_xlim(0, max(diag_rates.values) * 1.2)
        fig.savefig(self.fig_dir / "05_readmission_by_diagnosis.png", dpi=200)
        plt.close(fig)

        return diag_rates.round(2).to_dict()

    def _plot_correlation_matrix(self, df: pd.DataFrame) -> Dict[str, float]:
        """Answers: What are the strongest numerical linear correlations with readmission?"""
        df_temp = df.copy()
        df_temp["is_readmitted"] = (df_temp["readmitted"] == "<30").astype(int)
        
        num_cols = [
            "time_in_hospital", "num_lab_procedures", "num_procedures",
            "num_medications", "number_outpatient", "number_emergency",
            "number_inpatient", "number_diagnoses", "total_encounters_prior",
            "utilization_intensity", "diabetes_med_count", "is_readmitted"
        ]
        available_cols = [c for c in num_cols if c in df_temp.columns]
        corr = df_temp[available_cols].corr()

        fig, ax = plt.subplots(figsize=(8.5, 7))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="Blues",
            ax=ax, cbar_kws={"label": "Pearson Correlation"},
            linewidths=0.5, annot_kws={"size": 8}
        )
        ax.set_title("Correlation Heatmap: Clinical & Utilization Features vs Readmission", fontweight="bold", pad=12)
        plt.xticks(rotation=45, ha="right")
        fig.savefig(self.fig_dir / "06_correlation_heatmap.png", dpi=200)
        plt.close(fig)

        readmit_corrs = corr["is_readmitted"].drop("is_readmitted").round(4).to_dict()
        return readmit_corrs

if __name__ == "__main__":
    from src.data.loader import load_raw_data, load_mapped_data
    from src.validation.leakage_audit import LeakageAuditor
    from src.features.engineering import FeatureEngineer

    df_raw = load_raw_data()
    df_mapped = load_mapped_data()
    df_clean, _ = LeakageAuditor().audit_and_cleanse(df_mapped)
    df_feat = FeatureEngineer().transform(df_clean)

    eda = ClinicalEDA()
    summary = eda.run_analysis(df_feat, df_raw)
    print("\n--- EDA RUN COMPLETE ---")
    print(f"Target readmission rate: {summary['target']['positive_rate_pct']}%")
    print(f"Top missing features: {list(summary['missingness'].keys())[:3]}")
    print("Correlations with readmission:")
    for k, v in sorted(summary['correlations'].items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        print(f"  {k}: {v:+.4f}")
