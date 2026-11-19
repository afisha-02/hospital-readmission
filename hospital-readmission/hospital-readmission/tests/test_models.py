"""
Unit tests for model loading, prediction bounds, and baseline models.
"""

import pytest
import joblib
import numpy as np
import pandas as pd
from src.utils.config import load_config, resolve_path
from src.preprocessing.pipeline import ClinicalDataPreprocessor

def test_calibrated_model_inference():
    """Verifies that the calibrated model produces valid probabilities in [0, 1]."""
    config = load_config()
    preprocessor = ClinicalDataPreprocessor.load_artifacts()
    
    model_path = resolve_path(config["paths"]["model"])
    assert model_path.exists(), f"Model artifact not found at {model_path}"
    
    model = joblib.load(model_path)
    
    test_path = resolve_path(config["paths"]["processed_test"])
    df_test = pd.read_parquet(test_path).head(20)
    X_proc = preprocessor.transform(df_test)
    
    probs = model.predict_proba(X_proc)[:, 1]
    
    assert len(probs) == 20
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert not np.isnan(probs).any()

def test_baselines_loading():
    """Verifies that the baseline suite contains Dummy, Logistic Regression, Random Forest, and XGBoost."""
    config = load_config()
    baselines_path = resolve_path(config["paths"]["baseline_models"])
    assert baselines_path.exists(), f"Baselines not found at {baselines_path}"
    
    data = joblib.load(baselines_path)
    models = data["models"]
    
    assert "Dummy Classifier" in models
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    assert "XGBoost (Default)" in models
