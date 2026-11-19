"""
Evaluation metrics computation module for imbalanced clinical readmission modeling.
Computes Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Specificity, Brier score, and curves.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)

def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes full clinical evaluation metrics at a specific operating threshold.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    total = len(y_true)
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    # Area under curves (computed on continuous probabilities)
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except Exception:
        pr_auc = float(np.mean(y_true))
        
    brier = float(brier_score_loss(y_true, y_prob))
    
    return {
        "threshold": float(threshold),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),          # Sensitivity
        "sensitivity": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        },
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "positive_prevalence": round(float(np.mean(y_true)), 4)
    }

def compute_curves(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Generates coordinate arrays for ROC and Precision-Recall curves."""
    fpr, tpr, roc_thresh = roc_curve(y_true, y_prob)
    precision, recall, pr_thresh = precision_recall_curve(y_true, y_prob)
    
    # Downsample points for efficient JSON serialization and UI rendering
    def downsample(arr, max_pts=100):
        cleaned = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        if len(cleaned) <= max_pts:
            return cleaned.tolist()
        indices = np.linspace(0, len(cleaned) - 1, max_pts, dtype=int)
        return [float(cleaned[i]) for i in indices]

        
    return {
        "roc_curve": {
            "fpr": downsample(fpr),
            "tpr": downsample(tpr),
            "thresholds": downsample(roc_thresh)
        },
        "pr_curve": {
            "recall": downsample(recall),
            "precision": downsample(precision),
            "thresholds": downsample(pr_thresh)
        }
    }
