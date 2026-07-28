import math
from typing import List, Dict
from pydantic import BaseModel, Field, field_validator
from app.config import WINDOW_SIZE, NUM_FEATURES

class PredictRULRequest(BaseModel):
    engine_id: int = Field(..., gt=0, description="Unique identifier of the engine/machine")
    cycle: int = Field(..., ge=1, description="Current operational cycle of the engine")
    sensor_window: List[List[float]] = Field(
        ...,
        description=f"2D list of shape ({WINDOW_SIZE}, {NUM_FEATURES}) representing historical sensor readings sequence"
    )

    @field_validator("sensor_window")
    @classmethod
    def validate_window_dimensions_and_values(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != WINDOW_SIZE:
            raise ValueError(f"Outer list must have exactly {WINDOW_SIZE} time steps, got {len(v)}")
        
        for i, timestep in enumerate(v):
            if len(timestep) != NUM_FEATURES:
                raise ValueError(
                    f"Timestep {i} must have exactly {NUM_FEATURES} sensor features, got {len(timestep)}"
                )
            for j, val in enumerate(timestep):
                if not math.isfinite(val):
                    raise ValueError(
                        f"Non-finite numeric value found at timestep {i}, sensor {j}: {val}"
                    )
        return v

class PredictRULResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    engine_id: int = Field(..., gt=0)
    cycle: int = Field(..., ge=1)
    estimated_rul: float = Field(..., ge=0.0, description="Estimated Remaining Useful Life in cycles")
    prediction_interval_lower: float = Field(..., ge=0.0, description="Lower bound of RUL prediction interval")
    prediction_interval_upper: float = Field(..., ge=0.0, description="Upper bound of RUL prediction interval")
    prediction_interval_level: float = Field(0.90, description="Prediction level of interval")
    prediction_interval_coverage: float = Field(0.8998, description="Numeric measured validation coverage of interval")
    prediction_interval_description: str = Field(
        "Bounds use validation-residual 5th and 95th percentiles.",
        description="Description of the prediction interval bounds"
    )
    sequence_conversion_strategy: str = Field(..., description="Strategy used to convert sequence data for prediction")
    risk_level: str = Field(..., description="Risk category: Low, Medium, High, Critical")
    data_quality_score: float = Field(..., ge=0.0, le=1.0, description="Data quality confidence index")
    recommendation: str = Field(..., description="Actionable maintenance advice based on risk level")
    model_name: str = Field(..., description="Name of the model that served the prediction")
    model_version: str = Field(..., description="Version of the model that served the prediction")

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    status: str = Field("healthy", description="Status of the API service")
    api_version: str = Field(..., description="Version of the API service")
    model_loaded: bool = Field(..., description="Indicates whether the TensorFlow model is loaded and ready")

class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    model_name: str
    model_version: str
    framework: str
    description: str
    expected_window_size: int
    expected_num_features: int
    metrics: Dict[str, float] = Field(..., description="Key validation metrics of the model (e.g. MAE, RMSE)")
