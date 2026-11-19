"""
Reusable Scikit-learn preprocessing pipeline and ColumnTransformer.
Maintains exact feature alignment and prevents train-test leakage.
"""

import sys
import joblib
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, resolve_path
from src.utils.logger import setup_logger

logger = setup_logger("preprocessing")

def build_column_transformer(numerical_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Builds a robust Scikit-learn ColumnTransformer."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    return preprocessor

class ClinicalDataPreprocessor:
    """
    Orchestrates dataset splitting, pipeline fitting, and transformation.
    Strictly prevents data leakage by fitting only on X_train.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.num_cols = self.config["features"]["numerical"]
        self.cat_cols = self.config["features"]["categorical"]
        self.target_col = self.config["data"]["target_column"]
        self.positive_val = self.config["data"]["positive_target_value"]
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names_: List[str] = []

    def prepare_target(self, y_raw: pd.Series) -> np.ndarray:
        """Converts raw readmitted column ('NO', '>30', '<30') to binary (1 for <30, 0 otherwise)."""
        return (y_raw == self.positive_val).astype(int).values

    def split_dataset(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Splits dataset into Train (70%), Validation (15%), and Test (15%) splits
        stratified by the binary target.
        """
        random_state = self.config["data"]["random_state"]
        train_ratio = self.config["data"]["train_ratio"]
        val_ratio = self.config["data"]["val_ratio"]
        test_ratio = self.config["data"]["test_ratio"]
        
        y = self.prepare_target(df[self.target_col])
        X = df.drop(columns=[self.target_col], errors="ignore")
        
        # First split: Train vs Temp (Val + Test)
        temp_ratio = val_ratio + test_ratio
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=temp_ratio,
            random_state=random_state,
            stratify=y
        )
        
        # Second split: Val vs Test
        val_relative_ratio = val_ratio / temp_ratio
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1.0 - val_relative_ratio),
            random_state=random_state,
            stratify=y_temp
        )
        
        logger.info(
            f"Dataset split complete: Train={len(X_train):,} ({y_train.mean()*100:.2f}% positive), "
            f"Val={len(X_val):,} ({y_val.mean()*100:.2f}% positive), "
            f"Test={len(X_test):,} ({y_test.mean()*100:.2f}% positive)"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def fit_transform(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fits preprocessor exclusively on X_train, then transforms train, val, and test."""
        self.preprocessor = build_column_transformer(self.num_cols, self.cat_cols)
        
        logger.info("Fitting ColumnTransformer strictly on X_train...")
        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_val_proc = self.preprocessor.transform(X_val)
        X_test_proc = self.preprocessor.transform(X_test)
        
        # Extract feature names
        try:
            self.feature_names_ = list(self.preprocessor.get_feature_names_out())
        except Exception:
            # Fallback for older scikit-learn versions
            self.feature_names_ = [f"feat_{i}" for i in range(X_train_proc.shape[1])]
            
        logger.info(f"Feature transformation complete. Encoded dimension: {X_train_proc.shape[1]} features.")
        return X_train_proc, X_val_proc, X_test_proc

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms new unlabelled incoming data using the fitted pipeline."""
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted or loaded yet.")
        return self.preprocessor.transform(X)

    def save_splits(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
                    y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray):
        """Saves split DataFrames with target to data/processed for fast reproducible training."""
        train_df = X_train.copy()
        train_df["target_readmitted_30d"] = y_train
        
        val_df = X_val.copy()
        val_df["target_readmitted_30d"] = y_val
        
        test_df = X_test.copy()
        test_df["target_readmitted_30d"] = y_test
        
        train_path = resolve_path(self.config["paths"]["processed_train"])
        val_path = resolve_path(self.config["paths"]["processed_val"])
        test_path = resolve_path(self.config["paths"]["processed_test"])
        drift_path = resolve_path(self.config["paths"]["drift_baseline"])
        
        train_path.parent.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)
        test_df.to_parquet(test_path, index=False)
        # Baseline drift reference is the training distribution
        train_df.to_parquet(drift_path, index=False)
        logger.info(f"Saved processed splits to {train_path.parent}")

    def save_artifacts(self, output_path: Optional[str] = None):
        """Persists fitted preprocessor to disk."""
        if output_path is None:
            output_path = resolve_path(self.config["paths"]["preprocessor"])
        else:
            output_path = Path(output_path)
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "preprocessor": self.preprocessor,
            "feature_names": self.feature_names_,
            "numerical_cols": self.num_cols,
            "categorical_cols": self.cat_cols
        }, output_path)
        logger.info(f"Fitted preprocessor artifact saved to {output_path}")


    @classmethod
    def load_artifacts(cls, artifact_path: Optional[str] = None) -> "ClinicalDataPreprocessor":
        """Loads fitted preprocessor from disk."""
        config = load_config()
        if artifact_path is None:
            artifact_path = resolve_path(config["paths"]["preprocessor"])
        else:
            artifact_path = Path(artifact_path)
            
        if not artifact_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {artifact_path}")
            
        data = joblib.load(artifact_path)
        instance = cls(config)
        instance.preprocessor = data["preprocessor"]
        instance.feature_names_ = data["feature_names"]
        instance.num_cols = data["numerical_cols"]
        instance.cat_cols = data["categorical_cols"]
        return instance

if __name__ == "__main__":
    from src.data.loader import load_mapped_data
    from src.validation.leakage_audit import LeakageAuditor
    from src.features.engineering import FeatureEngineer
    
    # Run full pipeline to verify
    df_raw = load_mapped_data()
    df_clean, _ = LeakageAuditor().audit_and_cleanse(df_raw)
    df_feat = FeatureEngineer().transform(df_clean)
    
    preprocessor = ClinicalDataPreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_dataset(df_feat)
    X_train_proc, X_val_proc, X_test_proc = preprocessor.fit_transform(X_train, X_val, X_test)
    preprocessor.save_artifacts()
    preprocessor.save_splits(X_train, X_val, X_test, y_train, y_val, y_test)
    
    print("\n--- PREPROCESSING COMPLETE ---")

    print(f"X_train shape: {X_train_proc.shape}, y_train mean: {y_train.mean():.4f}")
    print(f"X_val shape:   {X_val_proc.shape}, y_val mean:   {y_val.mean():.4f}")
    print(f"X_test shape:  {X_test_proc.shape}, y_test mean:  {y_test.mean():.4f}")
    print(f"Total encoded features: {len(preprocessor.feature_names_)}")
