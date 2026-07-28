"""
Model Training & Evaluation Module
A modular predictive maintenance framework supporting multiple datasets and models.
"""

import json
import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src.config import DATASET_TYPE, MODEL_TYPE, WINDOW_LENGTH
from src.dataset_adapter import DatasetFactory
from src.model_adapter import ModelFactory
from src.preprocess import Preprocessor, WindowGenerator

# Fallback REPO_ROOT logic
try:
    from src.data_loader import REPO_ROOT, compute_checksums
except ImportError:
    from data_loader import REPO_ROOT, compute_checksums

MODEL_DIR = os.path.join(REPO_ROOT, "models")
MAX_RUL_CAP = 125.0


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute overall MAE/RMSE, segmented RUL performance metrics,
    near-failure MAE (RUL <= 30), and empirical quantiles from residuals.
    """
    overall_mae = float(mean_absolute_error(y_true, y_pred))
    overall_rmse = float(root_mean_squared_error(y_true, y_pred))
    
    # Segmented RUL Metrics
    mask_near = y_true <= 30.0
    mask_mid = (y_true > 30.0) & (y_true <= 75.0)
    mask_early = y_true > 75.0
    
    segmented = {
        "near_failure_le_30": {
            "count": int(np.sum(mask_near)),
            "mae": round(float(mean_absolute_error(y_true[mask_near], y_pred[mask_near])), 4) if np.sum(mask_near) > 0 else None,
            "rmse": round(float(root_mean_squared_error(y_true[mask_near], y_pred[mask_near])), 4) if np.sum(mask_near) > 0 else None
        },
        "mid_life_30_to_75": {
            "count": int(np.sum(mask_mid)),
            "mae": round(float(mean_absolute_error(y_true[mask_mid], y_pred[mask_mid])), 4) if np.sum(mask_mid) > 0 else None,
            "rmse": round(float(root_mean_squared_error(y_true[mask_mid], y_pred[mask_mid])), 4) if np.sum(mask_mid) > 0 else None
        },
        "early_life_gt_75": {
            "count": int(np.sum(mask_early)),
            "mae": round(float(mean_absolute_error(y_true[mask_early], y_pred[mask_early])), 4) if np.sum(mask_early) > 0 else None,
            "rmse": round(float(root_mean_squared_error(y_true[mask_early], y_pred[mask_early])), 4) if np.sum(mask_early) > 0 else None
        }
    }
    
    residuals = y_true - y_pred
    q05 = round(float(np.percentile(residuals, 5)), 2)
    q95 = round(float(np.percentile(residuals, 95)), 2)
    coverage = round(float(np.mean((residuals >= q05) & (residuals <= q95)) * 100), 2)
    
    return {
        "overall_mae": round(overall_mae, 4),
        "overall_rmse": round(overall_rmse, 4),
        "near_failure_mae": segmented["near_failure_le_30"]["mae"],
        "empirical_quantiles": {
            "lower_quantile_5": q05,
            "upper_quantile_95": q95
        },
        "empirical_interval_coverage_pct": coverage,
        "segmented_metrics": segmented
    }


def evaluate_on_test_set(model_adapter, test_df: pd.DataFrame, useful_features: list, target_col: str) -> dict:
    """Evaluate model on the test set."""
    # Since our random forest baseline expects single-cycle inference, 
    # we take the last cycle. For a sequence model (e.g. LSTM), we would extract windows.
    group_col = "engine_id" if "engine_id" in test_df.columns else "unit_nr"
    last_cycles = test_df.groupby(group_col).last()[useful_features]
    true_ruls = test_df.groupby(group_col).last()[target_col]
    
    y_pred = model_adapter.predict(last_cycles.values)
    
    # Bounded comparison
    y_pred_clipped = np.minimum(y_pred, MAX_RUL_CAP)
    y_true_clipped = np.minimum(true_ruls.values, MAX_RUL_CAP)
    
    mae = float(mean_absolute_error(y_true_clipped, y_pred_clipped))
    rmse = float(root_mean_squared_error(y_true_clipped, y_pred_clipped))
    
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4)
    }


def run_pipeline():
    """Execute generic modular pipeline."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Initialize Adapters
    dataset_adapter = DatasetFactory.get_adapter(DATASET_TYPE)
    model_adapter = ModelFactory.get_adapter(MODEL_TYPE)
    
    # 2. Get Data & Meta
    train_df, val_df, test_df = dataset_adapter.get_data_splits()
    feature_names = dataset_adapter.get_feature_names()
    target_col = dataset_adapter.get_rul_column()
    dataset_meta = dataset_adapter.get_metadata()
    model_meta = model_adapter.get_metadata()
    
    # 3. Preprocess (Scale Features)
    preprocessor = Preprocessor()
    train_df, val_df, test_df, scaled_cols = preprocessor.fit_transform(
        train_df, val_df, test_df, feature_names
    )
    
    # 4. For Baseline Random Forest, we just use the raw flattened arrays of the snapshot.
    # In a full sequence model, we would use WindowGenerator.create_windows()
    # For now, we simulate what the old code did: flattening snapshots.
    X_train = train_df[feature_names].values
    y_train = train_df[target_col].values
    X_val = val_df[feature_names].values
    y_val = val_df[target_col].values
    
    # 5. Train Model
    print(f"Training {model_meta['algorithm']} on {DATASET_TYPE} dataset...")
    start_time = time.time()
    model_adapter.train(X_train, y_train)
    training_time_sec = time.time() - start_time
    
    # 6. Evaluate
    y_val_pred = model_adapter.predict(X_val)
    perf_metrics = compute_comprehensive_metrics(y_val, y_val_pred)
    feat_importance = model_adapter.get_feature_importances(feature_names)
    test_metrics = evaluate_on_test_set(model_adapter, test_df, feature_names, target_col)
    
    print(f"Validation -> MAE: {perf_metrics['overall_mae']}, RMSE: {perf_metrics['overall_rmse']}")
    print(f"Test Set   -> MAE: {test_metrics['mae']}, RMSE: {test_metrics['rmse']}")
    
    # 7. Save Artifacts
    model_path = os.path.join(MODEL_DIR, "baseline_model.joblib")
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    feat_order_path = os.path.join(MODEL_DIR, "feature_order.json")
    feat_imp_path = os.path.join(MODEL_DIR, "feature_importance.json")
    metadata_path = os.path.join(MODEL_DIR, "metadata.json")
    
    model_adapter.save(model_path)
    joblib.dump(preprocessor.scaler, scaler_path)
    
    with open(feat_order_path, "w") as f:
        json.dump(feature_names, f, indent=2)
        
    with open(feat_imp_path, "w") as f:
        json.dump(feat_importance, f, indent=2)
        
    env_versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
        "scikit_learn": sklearn.__version__
    }
    
    metadata = {
        "dataset_type": DATASET_TYPE,
        "model_name": model_meta["algorithm"],
        "version": "2.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_time_seconds": round(training_time_sec, 2),
        "primary_target": target_col,
        "environment_versions": env_versions,
        "expected_feature_count": len(feature_names),
        "expected_window_length": WINDOW_LENGTH,
        "feature_order": feature_names,
        "scaled_feature_order": scaled_cols,
        "sequence_conversion_strategy": model_meta.get("sequence_conversion_strategy", "Unknown"),
        "performance_metrics": perf_metrics,
        "test_set_metrics": test_metrics,
        "dataset_metadata": dataset_meta
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"Artifacts saved successfully to {MODEL_DIR}")
    return metadata, val_df, model_adapter, preprocessor, feature_names


if __name__ == "__main__":
    run_pipeline()
