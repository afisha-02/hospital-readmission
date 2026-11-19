"""
Unit tests for data validation layer and contract enforcement.
"""

import pytest
import pandas as pd
import numpy as np
from src.validation.data_contract import DataContractValidator, REQUIRED_RAW_COLUMNS

def test_data_contract_valid_dataframe():
    """Tests that a properly formatted dataframe passes contract validation."""
    data = {col: [1] for col in REQUIRED_RAW_COLUMNS}
    data["gender"] = ["Female"]
    data["change"] = ["No"]
    data["diabetesMed"] = ["Yes"]
    data["readmitted"] = ["<30"]
    data["race"] = ["Caucasian"]
    data["age"] = ["[60-70)"]
    data["max_glu_serum"] = ["None"]
    data["A1Cresult"] = [">7"]
    
    df = pd.DataFrame(data)
    validator = DataContractValidator()
    is_valid, report = validator.validate(df)
    
    assert is_valid is True
    assert report["validation_status"] == "PASSED"
    assert report["total_rows"] == 1
    assert len(report["errors"]) == 0

def test_data_contract_missing_required_columns():
    """Tests that missing required columns triggers a validation failure."""
    df_incomplete = pd.DataFrame({"age": ["[50-60)"], "gender": ["Male"]})
    validator = DataContractValidator()
    is_valid, report = validator.validate(df_incomplete)
    
    assert is_valid is False
    assert report["validation_status"] == "FAILED"
    assert len(report["errors"]) > 0
    assert any("Missing required columns" in err for err in report["errors"])

def test_data_contract_invalid_target_values():
    """Tests detection of corrupt or unexpected target values."""
    data = {col: [1] for col in REQUIRED_RAW_COLUMNS}
    data["gender"] = ["Male"]
    data["change"] = ["No"]
    data["diabetesMed"] = ["Yes"]
    data["readmitted"] = ["INVALID_TARGET_CODE"]
    data["race"] = ["Caucasian"]
    data["age"] = ["[50-60)"]
    data["max_glu_serum"] = ["None"]
    data["A1Cresult"] = ["None"]
    
    df = pd.DataFrame(data)
    validator = DataContractValidator()
    is_valid, report = validator.validate(df)
    
    assert is_valid is False
    assert any("Unexpected target values" in err for err in report["errors"])
