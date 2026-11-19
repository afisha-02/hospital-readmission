"""
Generates clean, structured Jupyter notebooks for documentation and exploration.
"""

import json
from pathlib import Path

def create_notebook(cells, filepath):
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.13"},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Created notebook: {filepath}")

def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}

def code_cell(code):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [code]}

def main():
    nb_dir = Path("notebooks")
    nb_dir.mkdir(parents=True, exist_ok=True)

    # 01_data_quality.ipynb
    create_notebook([
        md_cell("# 🏥 Hospital Readmission Risk Intelligence\n## Notebook 01: Data Quality & Formal Contract Validation\n\nValidates raw UCI Diabetes 130-US Hospitals dataset against formal schema constraints, audit missingness, and verify target integrity."),
        code_cell("import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()))\n\nfrom src.data.loader import load_raw_data\nfrom src.validation.data_contract import DataContractValidator\n\ndf_raw = load_raw_data()\nvalidator = DataContractValidator()\nis_valid, report = validator.validate(df_raw)\n\nprint('Validation Status:', report['validation_status'])\nprint('Total Encounters:', report['total_rows'])\nprint('Duplicate Rows:', report['duplicate_rows'])\nprint('Overall Missingness:', str(report['overall_missing_cells_pct']) + '%')\nprint('Target Distribution:', report['target_distribution'])")
    ], nb_dir / "01_data_quality.ipynb")

    # 02_eda.ipynb
    create_notebook([
        md_cell("# 🏥 Hospital Readmission Risk Intelligence\n## Notebook 02: Clinical Exploratory Data Analysis (EDA)\n\nAnalyzes target imbalance, length of stay, prior utilization patterns, diagnoses, and demographic correlations."),
        code_cell("from src.data.loader import load_mapped_data\nfrom src.validation.leakage_audit import LeakageAuditor\nfrom src.features.engineering import FeatureEngineer\nfrom src.features.eda import ClinicalEDA\n\ndf_raw = load_mapped_data()\ndf_clean, leakage_rep = LeakageAuditor().audit_and_cleanse(df_raw)\ndf_feat = FeatureEngineer().transform(df_clean)\n\neda = ClinicalEDA()\nsummary = eda.run_analysis(df_feat, df_raw)\n\nprint('Target Distribution:', summary['target'])\nprint('Top Correlations with Readmission:', summary['correlations'])")
    ], nb_dir / "02_eda.ipynb")

    # 03_feature_engineering.ipynb
    create_notebook([
        md_cell("# 🏥 Hospital Readmission Risk Intelligence\n## Notebook 03: Domain Feature Engineering & Leakage Prevention\n\nDerives acute utilization intensity ratios, ICD-9 organ system groupings, and diabetes polypharmacy metrics without target leakage."),
        code_cell("from src.features.engineering import FeatureEngineer\n\nengineer = FeatureEngineer()\ndf_feat = engineer.transform(df_clean)\n\nprint('Engineered shape:', df_feat.shape)\ndisplay_cols = ['total_encounters_prior', 'utilization_intensity', 'high_utilizer_flag', 'primary_diagnosis_group']\nprint(df_feat[display_cols].head(10))")
    ], nb_dir / "03_feature_engineering.ipynb")

    # 04_baselines.ipynb
    create_notebook([
        md_cell("# 🏥 Hospital Readmission Risk Intelligence\n## Notebook 04: Baseline Model Suite & Imbalance Benchmarking\n\nCompares Dummy, Logistic Regression, Random Forest, and default XGBoost, auditing class weighting vs SMOTE."),
        code_cell("from src.preprocessing.pipeline import ClinicalDataPreprocessor\nfrom src.models.baselines import BaselineModelSuite\nimport pandas as pd\n\npreprocessor = ClinicalDataPreprocessor.load_artifacts()\ntest_df = pd.read_parquet('data/processed/test.parquet')\nX_test = preprocessor.transform(test_df)\ny_test = test_df['target_readmitted_30d'].values\n\nprint('Baseline evaluation environment loaded.')")
    ], nb_dir / "04_baselines.ipynb")

    # 05_xgboost.ipynb
    create_notebook([
        md_cell("# 🏥 Hospital Readmission Risk Intelligence\n## Notebook 05: XGBoost Bayesian Optimization & Early Stopping\n\nTunes XGBoost hyperparameters using Optuna to maximize validation PR-AUC with MLflow experiment tracking."),
        code_cell("from src.models.train_xgboost import XGBoostTrainer\nprint('XGBoost Optuna Trainer initialized.')")
    ], nb_dir / "05_xgboost.ipynb")

    # 06_evaluation.ipynb
    create_notebook([
        md_cell("# 🏥 Hospital Readmission Risk Intelligence\n## Notebook 06: Model Evaluation, Calibration & Threshold Lab\n\nComputes comprehensive metrics, reliability diagrams, cost matrix analysis, and SHAP explainability."),
        code_cell("import json\nwith open('reports/model_comparison.json') as f:\n    res = json.load(f)\nfor r in res['comparison_table']:\n    print(f\"{r['model']:<38} | PR-AUC: {r['pr_auc']:.4f} | Brier: {r['brier_score']:.4f}\")")
    ], nb_dir / "06_evaluation.ipynb")

if __name__ == "__main__":
    main()
