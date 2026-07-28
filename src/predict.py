"""
C-MAPSS FD001 Predict RUL Unified Interface
Author: Member A (Data & ML Lead)
"""

import json
import os
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

try:
    from src.data_loader import REPO_ROOT
except ImportError:
    from data_loader import REPO_ROOT

MODEL_DIR = os.path.join(REPO_ROOT, "models")
REQUIRED_WINDOW_LENGTH = 30
MAX_RUL_CAP = 125.0

MAX_RUL_CAP = 125.0


class RULPredictor:
    """
    Production Predictor for C-MAPSS FD001 Engine RUL Estimation.
    Standardized API Contract: Accepts strictly a 30-cycle sequence shaped (30, 16).
    """

    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.model = joblib.load(os.path.join(model_dir, "baseline_model.joblib"))
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        
        with open(os.path.join(model_dir, "feature_order.json"), "r") as f:
            self.feature_order = json.load(f)
            
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
            
        quantiles = self.metadata.get("performance_metrics", {}).get("empirical_quantiles", {})
        self.q05 = float(quantiles.get("lower_quantile_5", -15.0))
        self.q95 = float(quantiles.get("upper_quantile_95", 25.0))

    def determine_risk_and_recommendation(self, estimated_rul: float) -> tuple:
        """
        Determine risk level classification and actionable maintenance recommendation.
        Uses neutral prototype wording rather than safety-certifying language.
        """
        if estimated_rul <= 15.0:
            risk = "CRITICAL"
            rec = "CRITICAL ALERT: RUL <= 15 cycles. Require immediate qualified engineering inspection and detailed diagnostic checks before further flight operation."
        elif estimated_rul <= 30.0:
            risk = "HIGH"
            rec = "HIGH PRIORITY: RUL <= 30 cycles. Require qualified engineering inspection during next scheduled service window."
        elif estimated_rul <= 60.0:
            risk = "MEDIUM"
            rec = "ELEVATED MONITORING: RUL <= 60 cycles. Routine inspection recommended within validated dataset range."
        else:
            risk = "LOW"
            rec = "NOMINAL: Nominal behavior detected within the model's validated dataset range."
        return risk, rec

    def _convert_input_to_df(self, input_data) -> pd.DataFrame:
        """Convert input data structure into validated DataFrame."""
        if isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        elif isinstance(input_data, np.ndarray):
            if input_data.ndim == 2 and input_data.shape[1] == len(self.feature_order):
                df = pd.DataFrame(input_data, columns=self.feature_order)
            else:
                raise ValueError(f"Array shape {input_data.shape} invalid. Expected (30, {len(self.feature_order)}).")
        else:
            raise TypeError("API contract requires a 30-cycle sequence (list of dicts, numpy array (30, 16), or DataFrame with 30 rows). Single snapshots are not supported.")
            
        return df

    def predict_rul(self, input_data) -> dict:
        """
        Production prediction interface.
        Strict API Contract: Input MUST be a 30-cycle sequence shaped (30, 16).
        """
        df = self._convert_input_to_df(input_data)
        
        # 1. Validate Window Length (strictly 30 cycles)
        if len(df) != REQUIRED_WINDOW_LENGTH:
            raise ValueError(f"Invalid sequence length {len(df)}. Production API contract requires strictly {REQUIRED_WINDOW_LENGTH} cycles.")
            
        # 2. Check for missing required features
        missing_fields = [f for f in self.feature_order if f not in df.columns]
        if missing_fields:
            raise ValueError(f"Missing required feature fields: {missing_fields}")
            
        # 3. Strip extra fields safely (Security & Hygiene) and reorder
        df_clean = df[self.feature_order]
        
        # 4. Check for non-finite values (NaN / Inf)
        if df_clean.isnull().values.any() or not np.isfinite(df_clean.values).all():
            raise ValueError("Input sequence contains non-finite values (NaN or Inf). Inspection rejected.")
            
        # Real data quality score: fraction of values within ±3 standard deviations
        scaled_values = self.scaler.transform(df_clean.values)
        in_bounds_fraction = np.mean((scaled_values >= -3.0) & (scaled_values <= 3.0))
        data_quality_score = float(round(in_bounds_fraction, 4))
        
        # 5. Extract final cycle snapshot (Cycle 30) for Random Forest baseline prediction
        final_cycle_snapshot = df_clean.iloc[[-1]]
        
        # 6. Model Prediction
        raw_pred = float(self.model.predict(final_cycle_snapshot.values)[0])
        
        # 7. Cap Displayed RUL and Bounds to documented target range [0.0, 125.0] using authoritative offsets [-15.01, +24.08]
        estimated_rul = max(0.0, min(MAX_RUL_CAP, raw_pred))
        lower_bound = max(0.0, min(MAX_RUL_CAP, raw_pred + self.q05))
        upper_bound = max(0.0, min(MAX_RUL_CAP, raw_pred + self.q95))
        
        risk_level, recommendation = self.determine_risk_and_recommendation(estimated_rul)
        
        return {
            "model_name": self.metadata.get("model_name", "RandomForestRegressor_FD001"),
            "version": self.metadata.get("version", "1.3.0"),
            "feature_order": self.feature_order,
            "window_length": REQUIRED_WINDOW_LENGTH,
            "sequence_conversion_strategy": "The Random Forest converts the (30, 16) sequence into a single prediction by extracting the final cycle snapshot (cycle 30) from the input window.",
            "model_limitation": "Random Forest baseline evaluates ONLY the final cycle of the 30-cycle window, discarding preceding temporal progression.",
            "estimated_rul": round(estimated_rul, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "risk_level": risk_level,
            "data_quality_score": data_quality_score,
            "recommendation": recommendation
        }





if __name__ == "__main__":
    predictor = RULPredictor()
    sample_seq = [{f: 0.5 for f in predictor.feature_order} for _ in range(30)]
    res = predictor.predict_rul(sample_seq)
    print("Sample 30-Cycle Predict RUL Output:")
    print(json.dumps(res, indent=2))
