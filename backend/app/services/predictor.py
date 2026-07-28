import os
import sys
import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

# Resolve repository root path relative to this file
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.config import MODEL_NAME, MODEL_VERSION, FEATURE_ORDER

class PredictorService:
    def __init__(self):
        self.model_loaded = False
        self.predictor = None
        self.metadata = {}
        
        # Check if the models are available and initialize Member A's RULPredictor
        try:
            model_dir = os.path.join(REPO_ROOT, "models")
            model_file = os.path.join(model_dir, "baseline_model.joblib")
            scaler_file = os.path.join(model_dir, "scaler.joblib")
            
            # Standard check to ensure necessary artifacts exist
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                from src.predict import RULPredictor
                self.predictor = RULPredictor(model_dir=model_dir)
                self.metadata = self.predictor.metadata
                self.model_loaded = True
            else:
                self.model_loaded = False
        except Exception as e:
            # Safe recovery: set model_loaded to False so API remains up in model-unavailable mode
            logger.warning("RULPredictor failed to initialize. Error: %s", type(e).__name__)
            self.model_loaded = False

    def predict_rul(self, engine_id: int, cycle: int, sensor_window: list) -> Dict[str, Any]:
        """
        Uses Member A's predict_rul() to calculate RUL based on the sensor window.
        """
        if not self.model_loaded or self.predictor is None:
            raise RuntimeError("Model is currently unavailable.")
            
        # Map the incoming 30x16 array to a Pandas DataFrame with exact ordered columns
        df = pd.DataFrame(sensor_window, columns=FEATURE_ORDER)
        
        # Call Member A's model prediction routine
        pred_dict = self.predictor.predict_rul(df)
        
        # Return formatted payload aligned with PredictRULResponse Pydantic schema
        return {
            "engine_id": engine_id,
            "cycle": cycle,
            "estimated_rul": pred_dict["estimated_rul"],
            "prediction_interval_lower": pred_dict["lower_bound"],
            "prediction_interval_upper": pred_dict["upper_bound"],
            "prediction_interval_level": 0.90,
            "prediction_interval_coverage": 0.8998,
            "prediction_interval_description": "Bounds use validation-residual 5th and 95th percentiles.",
            "sequence_conversion_strategy": self.metadata.get(
                "sequence_conversion_strategy",
                "The Random Forest converts the (30, 16) sequence into a single prediction by extracting the final cycle snapshot (cycle 30) from the input window."
            ),
            "risk_level": pred_dict["risk_level"].capitalize(),  # Convert e.g., "CRITICAL" -> "Critical"
            "data_quality_score": pred_dict["data_quality_score"],
            "recommendation": pred_dict["recommendation"],
            "model_name": pred_dict["model_name"],
            "model_version": pred_dict["version"]
        }
