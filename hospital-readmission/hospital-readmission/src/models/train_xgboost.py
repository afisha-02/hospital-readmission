"""
XGBoost hyperparameter optimization via Optuna with early stopping and MLflow experiment tracking.
Maximizes PR-AUC on held-out validation set.
"""

import sys
import json
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.logger import setup_logger
from src.evaluation.metrics import compute_classification_metrics
from src.preprocessing.pipeline import ClinicalDataPreprocessor

logger = setup_logger("train_xgboost")
optuna.logging.set_verbosity(optuna.logging.WARNING)

class XGBoostTrainer:
    """Orchestrates Optuna Bayesian optimization, early stopping, and MLflow logging for XGBoost."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or load_config()
        self.best_params: Dict[str, Any] = {}
        self.best_model: Any = None
        self.mlflow_uri = f"file:///{str(resolve_path('mlruns')).replace('\\', '/')}"
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment("Hospital_Readmission_Risk_Intelligence")

    def optimize_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 15
    ) -> Dict[str, Any]:
        """Runs Optuna study to optimize XGBoost hyperparameters maximizing validation PR-AUC."""
        logger.info(f"Starting Optuna hyperparameter study ({n_trials} trials)...")
        pos_weight = float((len(y_train) - sum(y_train)) / max(sum(y_train), 1))
        
        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 80, 250),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.20, log=True),
                "subsample": trial.suggest_float("subsample", 0.65, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 0.95),
                "min_child_weight": trial.suggest_int("min_child_weight", 2, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 2.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
                "scale_pos_weight": pos_weight,
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1
            }
            
            model = XGBClassifier(**params, early_stopping_rounds=15)
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
            
            y_val_prob = model.predict_proba(X_val)[:, 1]
            metrics = compute_classification_metrics(y_val, y_val_prob)
            val_pr_auc = metrics["pr_auc"]
            
            # Log trial to MLflow
            with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("val_pr_auc", val_pr_auc)
                mlflow.log_metric("val_roc_auc", metrics["roc_auc"])
                mlflow.log_metric("val_brier", metrics["brier_score"])
                mlflow.log_metric("best_iteration", model.best_iteration if hasattr(model, "best_iteration") else params["n_estimators"])
                
            return val_pr_auc

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        self.best_params["scale_pos_weight"] = pos_weight
        self.best_params["eval_metric"] = "logloss"
        self.best_params["random_state"] = 42
        self.best_params["n_jobs"] = -1
        
        logger.info(f"Optuna optimization complete. Best Val PR-AUC: {study.best_value:.4f}")
        logger.info(f"Best hyperparameters: {self.best_params}")
        return self.best_params

    def train_best_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Any, Dict[str, Any]]:
        """Trains final tuned XGBoost candidate and logs to MLflow."""
        if not self.best_params:
            raise ValueError("Run optimize_hyperparameters before train_best_model.")
            
        logger.info("Training final tuned XGBoost candidate with early stopping...")
        with mlflow.start_run(run_name="xgboost_tuned_candidate"):
            model = XGBClassifier(**self.best_params, early_stopping_rounds=20)
            model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
            self.best_model = model
            
            # Evaluate on Test set
            y_test_prob = model.predict_proba(X_test)[:, 1]
            test_metrics = compute_classification_metrics(y_test, y_test_prob, threshold=0.50)
            
            # MLflow logging
            mlflow.log_params(self.best_params)
            mlflow.log_metrics({
                "test_pr_auc": test_metrics["pr_auc"],
                "test_roc_auc": test_metrics["roc_auc"],
                "test_f1": test_metrics["f1"],
                "test_recall": test_metrics["recall"],
                "test_precision": test_metrics["precision"],
                "test_brier": test_metrics["brier_score"]
            })
            if hasattr(model, "best_iteration"):
                mlflow.log_metric("best_iteration", model.best_iteration)
                
            logger.info(
                f"Tuned XGBoost Evaluation (Test Set): PR-AUC={test_metrics['pr_auc']:.4f}, "
                f"ROC-AUC={test_metrics['roc_auc']:.4f}, Recall={test_metrics['recall']:.4f}, "
                f"F1={test_metrics['f1']:.4f}, Brier={test_metrics['brier_score']:.4f}"
            )
            
            # Persist artifact
            out_path = resolve_path(self.config["paths"]["uncalibrated_model"])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({
                "model": model,
                "best_params": self.best_params,
                "best_iteration": getattr(model, "best_iteration", None),
                "metrics": test_metrics
            }, out_path)
            logger.info(f"Uncalibrated tuned XGBoost saved to {out_path}")
            
            return model, test_metrics

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
    
    trainer = XGBoostTrainer(config)
    trainer.optimize_hyperparameters(X_train_proc, y_train, X_val_proc, y_val, n_trials=12)
    model, metrics = trainer.train_best_model(X_train_proc, y_train, X_val_proc, y_val, X_test_proc, y_test)
    
    print("\n--- TUNED XGBOOST COMPLETE ---")
    print(f"Test PR-AUC:  {metrics['pr_auc']:.4f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Test Brier:   {metrics['brier_score']:.4f}")
