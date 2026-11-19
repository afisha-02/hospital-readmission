"""
Baseline model suite and class imbalance experiments for Hospital Readmission Risk Intelligence.
Trains Dummy Classifier, Logistic Regression, Random Forest, and default XGBoost.
"""

import sys
import json
import joblib
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.logger import setup_logger
from src.evaluation.metrics import compute_classification_metrics
from src.preprocessing.pipeline import ClinicalDataPreprocessor

logger = setup_logger("baselines")

class BaselineModelSuite:
    """Trains and compares naive and standard machine learning baseline models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.50
    ) -> Dict[str, Any]:
        """Trains Dummy, Logistic Regression, Random Forest, and default XGBoost."""
        logger.info("Training baseline model suite...")
        
        pos_weight = float((len(y_train) - sum(y_train)) / max(sum(y_train), 1))
        logger.info(f"Class imbalance scale_pos_weight calculated: {pos_weight:.2f}")

        # 1. Dummy Classifier (Most frequent / Stratified)
        dummy = DummyClassifier(strategy="prior", random_state=42)
        dummy.fit(X_train, y_train)
        self.models["Dummy Classifier"] = dummy

        # 2. Logistic Regression (L2 regularized, balanced class weights)
        log_reg = LogisticRegression(
            C=1.0, penalty="l2", class_weight="balanced",
            solver="lbfgs", max_iter=1000, random_state=42
        )
        logger.info("Fitting Logistic Regression baseline...")
        log_reg.fit(X_train, y_train)
        self.models["Logistic Regression"] = log_reg

        # 3. Random Forest (Balanced tree ensemble)
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=20,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        )
        logger.info("Fitting Random Forest baseline...")
        rf.fit(X_train, y_train)
        self.models["Random Forest"] = rf

        # 4. Default XGBoost
        xgb_base = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=42, n_jobs=-1
        )
        logger.info("Fitting default XGBoost baseline...")
        xgb_base.fit(X_train, y_train)
        self.models["XGBoost (Default)"] = xgb_base

        # Evaluate all models
        for name, model in self.models.items():
            probs = model.predict_proba(X_test)[:, 1]
            metrics = compute_classification_metrics(y_test, probs, threshold=threshold)
            self.metrics[name] = metrics
            logger.info(
                f"Model [{name}]: PR-AUC={metrics['pr_auc']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, "
                f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}"
            )

        return self.metrics

    def run_imbalance_experiment(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Explicitly compares class imbalance techniques:
        1. Unweighted XGBoost
        2. Cost-Weighted XGBoost (scale_pos_weight)
        3. Synthetic Oversampling (SMOTE) + XGBoost
        """
        logger.info("Running Class Imbalance Strategy Benchmark...")
        imbalance_results = {}
        pos_weight = float((len(y_train) - sum(y_train)) / max(sum(y_train), 1))

        # 1. Unweighted
        m_unweighted = XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
        m_unweighted.fit(X_train, y_train)
        probs_un = m_unweighted.predict_proba(X_val)[:, 1]
        imbalance_results["Unweighted"] = compute_classification_metrics(y_val, probs_un, threshold=0.50)

        # 2. scale_pos_weight
        m_weighted = XGBClassifier(
            n_estimators=80, max_depth=4, learning_rate=0.1,
            scale_pos_weight=pos_weight, random_state=42, n_jobs=-1
        )
        m_weighted.fit(X_train, y_train)
        probs_wt = m_weighted.predict_proba(X_val)[:, 1]
        imbalance_results["Class Weighting (scale_pos_weight)"] = compute_classification_metrics(y_val, probs_wt, threshold=0.50)

        # 3. SMOTE
        logger.info("Synthesizing minority class via SMOTE (sampling_strategy=0.3)...")
        smote = SMOTE(sampling_strategy=0.3, random_state=42)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        m_smote = XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.1, random_state=42, n_jobs=-1)
        m_smote.fit(X_smote, y_smote)
        probs_smote = m_smote.predict_proba(X_val)[:, 1]
        imbalance_results["SMOTE Oversampling"] = compute_classification_metrics(y_val, probs_smote, threshold=0.50)

        logger.info("Class imbalance benchmark complete.")
        return imbalance_results

    def save_artifacts(self, artifact_path: str = None):
        """Saves baseline models to disk."""
        if artifact_path is None:
            artifact_path = resolve_path(self.config["paths"]["baseline_models"])
        else:
            artifact_path = Path(artifact_path)
            
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "models": self.models,
            "metrics": self.metrics
        }, artifact_path)
        logger.info(f"Baseline models and metrics saved to {artifact_path}")

if __name__ == "__main__":
    config = load_config()
    preprocessor = ClinicalDataPreprocessor.load_artifacts()
    
    train_df = pd.read_parquet(resolve_path(config["paths"]["processed_train"]))
    val_df = pd.read_parquet(resolve_path(config["paths"]["processed_val"]))
    test_df = pd.read_parquet(resolve_path(config["paths"]["processed_test"]))
    
    y_train = train_df["target_readmitted_30d"].values
    y_val = val_df["target_readmitted_30d"].values
    y_test = test_df["target_readmitted_30d"].values
    
    X_train_proc = preprocessor.transform(train_df)
    X_val_proc = preprocessor.transform(val_df)
    X_test_proc = preprocessor.transform(test_df)
    
    suite = BaselineModelSuite(config)
    
    # 1. Imbalance experiments
    imb_results = suite.run_imbalance_experiment(X_train_proc, y_train, X_val_proc, y_val)
    print("\n--- CLASS IMBALANCE EXPERIMENT (Validation Set) ---")
    for name, res in imb_results.items():
        print(f"[{name}]: PR-AUC={res['pr_auc']:.4f}, ROC-AUC={res['roc_auc']:.4f}, Recall={res['recall']:.4f}, F1={res['f1']:.4f}")
        
    # 2. Baseline models evaluation
    metrics = suite.train_and_evaluate(X_train_proc, y_train, X_test_proc, y_test)
    suite.save_artifacts()
    
    print("\n--- BASELINE MODEL COMPARISON (Test Set) ---")
    print(f"{'Model':<25} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Recall':<8} | {'F1':<8} | {'Brier':<8}")
    print("-" * 72)
    for model_name, m in metrics.items():
        print(f"{model_name:<25} | {m['pr_auc']:<8.4f} | {m['roc_auc']:<8.4f} | {m['recall']:<8.4f} | {m['f1']:<8.4f} | {m['brier_score']:<8.4f}")
