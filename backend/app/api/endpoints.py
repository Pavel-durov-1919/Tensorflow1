from fastapi import APIRouter, HTTPException, status
import logging
from app.schemas import PredictRULRequest, PredictRULResponse, HealthResponse, ModelInfoResponse
from app.services.predictor import PredictorService
from app.config import API_VERSION, MODEL_NAME, MODEL_VERSION, WINDOW_SIZE, NUM_FEATURES

logger = logging.getLogger(__name__)

router = APIRouter()

# Instantiate the predictor service
predictor_service = PredictorService()

@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Returns system operational status, API version, and model load status.
    Remains active (returns 200 OK) even if model load has failed.
    """
    return HealthResponse(
        status="healthy",
        api_version=API_VERSION,
        model_loaded=predictor_service.model_loaded
    )

@router.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """
    Returns architecture and training metadata for the active predictive model.
    Throws 503 if the model artifacts are unavailable.
    """
    if not predictor_service.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictive maintenance model is currently unavailable."
        )
    
    meta = predictor_service.metadata
    metrics = meta.get("validation_metrics", {})
    
    # Retrieve metrics from validation_metrics inside metadata.json
    return ModelInfoResponse(
        model_name=meta.get("model_name", MODEL_NAME),
        model_version=meta.get("version", MODEL_VERSION),
        framework="Scikit-Learn (RandomForestRegressor)",
        description="Predicts the Remaining Useful Life (RUL) of turbofan engines using multivariate sensor time series sequences.",
        expected_window_size=meta.get("window_length", WINDOW_SIZE),
        expected_num_features=meta.get("feature_count", NUM_FEATURES),
        metrics={
            "MAE": round(metrics.get("mae", 12.35), 2),
            "RMSE": round(metrics.get("rmse", 17.06), 2)
        }
    )

@router.post("/predict/rul", response_model=PredictRULResponse)
def predict_rul(payload: PredictRULRequest):
    """
    Accepts an ordered sensor window and predicts the remaining useful life (RUL).
    Throws 503 if the model artifacts are unavailable.
    """
    if not predictor_service.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Predictive maintenance model is currently unavailable."
        )
        
    try:
        prediction = predictor_service.predict_rul(
            engine_id=payload.engine_id,
            cycle=payload.cycle,
            sensor_window=payload.sensor_window
        )
        return PredictRULResponse(**prediction)
    except Exception:
        logger.exception(
            "RUL prediction failed for engine_id=%s cycle=%s",
            payload.engine_id, payload.cycle
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while executing the RUL model prediction."
        )
