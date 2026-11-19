"""
Unit tests for data preprocessing pipeline and ColumnTransformer.
"""

import pytest
import numpy as np
import pandas as pd
from src.preprocessing.pipeline import ClinicalDataPreprocessor
from src.utils.config import load_config, resolve_path

def test_preprocessor_load_and_transform():
    """Verifies that the fitted preprocessor loads correctly and transforms data without NaNs."""
    config = load_config()
    preprocessor = ClinicalDataPreprocessor.load_artifacts()
    
    assert preprocessor.preprocessor is not None
    assert len(preprocessor.feature_names_) == 88
    
    # Load 5 sample rows from test set
    test_path = resolve_path(config["paths"]["processed_test"])
    if test_path.exists():
        df_test = pd.read_parquet(test_path).head(10)
        X_proc = preprocessor.transform(df_test)
        
        assert X_proc.shape == (10, 88)
        assert not np.isnan(X_proc).any()
        assert not np.isinf(X_proc).any()

def test_target_preparation():
    """Verifies binary encoding of raw readmitted column."""
    preprocessor = ClinicalDataPreprocessor()
    raw_targets = pd.Series(["<30", ">30", "NO", "<30", "NO"])
    binary = preprocessor.prepare_target(raw_targets)
    
    np.testing.assert_array_equal(binary, np.array([1, 0, 0, 1, 0]))
