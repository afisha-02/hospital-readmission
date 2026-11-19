"""
Unit tests for probability calibration metrics, threshold optimization, and cost matrix.
"""

import pytest
import numpy as np
from src.calibration.calibrator import compute_ece
from src.thresholding.cost_matrix import CostMatrix
from src.thresholding.optimizer import ThresholdOptimizer
from src.evaluation.metrics import compute_classification_metrics

def test_ece_computation():
    """Verifies that Expected Calibration Error is 0 for perfectly calibrated predictions."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Perfect calibration: 5 predictions of 0.0, 5 predictions of 1.0
    y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    ece = compute_ece(y_true, y_prob, n_bins=5)
    assert ece == 0.0

def test_cost_matrix_calculation():
    """Verifies expected cost calculation formula."""
    matrix = CostMatrix(cost_fn=5000.0, cost_fp=500.0, cost_tp=750.0, cost_tn=0.0)
    # 10 TN, 2 FP, 1 FN, 3 TP => Total = 1*5000 + 2*500 + 3*750 + 10*0 = 5000 + 1000 + 2250 = 8250
    res = matrix.calculate_cost(tn=10, fp=2, fn=1, tp=3)
    
    assert res["total_cost"] == 8250.0
    assert res["per_patient_cost"] == round(8250.0 / 16, 2)
    assert res["fn_cost_burden"] == 5000.0
    assert res["fp_cost_burden"] == 1000.0

def test_threshold_metrics_consistency():
    """Verifies that precision, recall, and confusion matrix are mathematically consistent."""
    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 0])
    y_prob = np.array([0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05])
    
    # At threshold 0.5: Predicted positives: idx 0, 1, 2 (probs 0.8, 0.7, 0.6)
    # TP: idx 0 (1), idx 1 (1) => TP = 2
    # FP: idx 2 (0) => FP = 1
    # FN: idx 4 (true 1, prob 0.3) => FN = 1
    # TN: idx 3, 5, 6, 7 => TN = 4
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.50)
    
    assert metrics["confusion_matrix"]["tp"] == 2
    assert metrics["confusion_matrix"]["fp"] == 1
    assert metrics["confusion_matrix"]["fn"] == 1
    assert metrics["confusion_matrix"]["tn"] == 4
    assert metrics["recall"] == round(2 / 3, 4)
    assert metrics["precision"] == round(2 / 3, 4)
