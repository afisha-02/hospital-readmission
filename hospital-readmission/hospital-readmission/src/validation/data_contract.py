"""
Data contract and validation layer for the Hospital Readmission Risk Intelligence Platform.
Enforces schemas, value ranges, missingness bounds, and generates audit reports.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.logger import setup_logger

logger = setup_logger("data_contract")

REQUIRED_RAW_COLUMNS = [
    "encounter_id", "patient_nbr", "race", "gender", "age",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses", "max_glu_serum", "A1Cresult",
    "change", "diabetesMed", "readmitted"
]

ALLOWED_VALUES = {
    "gender": ["Female", "Male", "Unknown/Invalid", "Unknown"],
    "change": ["No", "Ch"],
    "diabetesMed": ["No", "Yes"],
    "readmitted": ["NO", ">30", "<30"]
}

NUMERICAL_BOUNDS = {
    "time_in_hospital": (1, 14),
    "num_lab_procedures": (1, 150),
    "num_procedures": (0, 10),
    "num_medications": (1, 100),
    "number_outpatient": (0, 100),
    "number_emergency": (0, 100),
    "number_inpatient": (0, 100),
    "number_diagnoses": (1, 20)
}

class DataContractValidator:
    """Validates raw and incoming clinical encounter datasets against formal contract."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []

    def validate(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """
        Runs comprehensive data contract validation suite.
        Returns (is_valid, validation_report).
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        n_rows, n_cols = df.shape
        duplicate_rows = int(df.duplicated().sum())
        duplicate_encounters = int(df["encounter_id"].duplicated().sum()) if "encounter_id" in df.columns else 0
        
        # 1. Required column validation
        missing_cols = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
        if missing_cols:
            self.validation_errors.append(f"Missing required columns: {missing_cols}")
            
        # 2. Duplicate check
        if duplicate_rows > 0:
            self.validation_warnings.append(f"Found {duplicate_rows} duplicate rows in dataset.")
        if duplicate_encounters > 0:
            self.validation_errors.append(f"Found {duplicate_encounters} duplicate encounter IDs.")
            
        # 3. Target integrity check
        target_col = self.config["data"]["target_column"]
        target_distribution = {}
        if target_col in df.columns:
            target_counts = df[target_col].value_counts(dropna=False).to_dict()
            target_distribution = {str(k): int(v) for k, v in target_counts.items()}
            invalid_targets = [v for v in df[target_col].dropna().unique() if v not in ALLOWED_VALUES["readmitted"]]
            if invalid_targets:
                self.validation_errors.append(f"Unexpected target values detected: {invalid_targets}")
        else:
            self.validation_warnings.append(f"Target column '{target_col}' not present (acceptable for unlabelled inference).")
            
        # 4. Value ranges and bounds validation
        out_of_bounds = {}
        for col, (min_v, max_v) in NUMERICAL_BOUNDS.items():
            if col in df.columns:
                num_series = pd.to_numeric(df[col], errors="coerce")
                violations = int(((num_series < min_v) | (num_series > max_v)).sum())
                if violations > 0:
                    out_of_bounds[col] = {
                        "violations": violations,
                        "min_observed": float(num_series.min()),
                        "max_observed": float(num_series.max()),
                        "expected_range": [min_v, max_v]
                    }
                    self.validation_warnings.append(
                        f"Column '{col}' has {violations} values outside expected bounds [{min_v}, {max_v}]."
                    )
                    
        # 5. Missingness audit
        missingness_counts = df.isna().sum().to_dict()
        missingness_pct = (df.isna().mean() * 100).round(2).to_dict()
        high_missing_cols = {col: pct for col, pct in missingness_pct.items() if pct > 30.0}
        
        # 6. Categorical domain checks
        unexpected_categories = {}
        for col, allowed in ALLOWED_VALUES.items():
            if col in df.columns and col != "readmitted":
                found = df[col].dropna().unique().tolist()
                unexpected = [x for x in found if x not in allowed]
                if unexpected:
                    unexpected_categories[col] = unexpected
                    self.validation_warnings.append(f"Column '{col}' contains unexpected categories: {unexpected}")

        is_valid = len(self.validation_errors) == 0
        
        report = {
            "validation_status": "PASSED" if is_valid else "FAILED",
            "total_rows": int(n_rows),
            "total_columns": int(n_cols),
            "duplicate_rows": duplicate_rows,
            "duplicate_encounters": duplicate_encounters,
            "overall_missing_cells_pct": round(float(df.isna().sum().sum()) / (n_rows * n_cols) * 100, 2),
            "target_distribution": target_distribution,
            "high_missingness_features": high_missing_cols,
            "out_of_bounds_features": out_of_bounds,
            "unexpected_categories": unexpected_categories,
            "errors": self.validation_errors,
            "warnings": self.validation_warnings
        }
        
        return is_valid, report

    def generate_and_save_report(self, df: pd.DataFrame, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Runs validation and writes the JSON audit report to disk."""
        if output_path is None:
            output_path = resolve_path(self.config["paths"]["quality_report"])
        else:
            output_path = Path(output_path)
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        is_valid, report = self.validate(df)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Data quality report saved to {output_path} (Status: {report['validation_status']})")
        return report

if __name__ == "__main__":
    from src.data.loader import load_raw_data
    df_raw = load_raw_data()
    validator = DataContractValidator()
    report = validator.generate_and_save_report(df_raw)
    print("\n--- DATA QUALITY REPORT SUMMARY ---")
    print(f"Status: {report['validation_status']}")
    print(f"Rows: {report['total_rows']:,}")
    print(f"Features: {report['total_columns']}")
    print(f"Duplicate rows: {report['duplicate_rows']}")
    print(f"Overall missingness: {report['overall_missing_cells_pct']}%")
    print("Target distribution:")
    for k, v in report['target_distribution'].items():
        pct = (v / report['total_rows']) * 100
        print(f"  {k}: {v:,} ({pct:.2f}%)")
