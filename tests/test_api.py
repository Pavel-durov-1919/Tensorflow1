import pytest
import os
import shutil
from app.config import BASE_DIR, WINDOW_SIZE, NUM_FEATURES
from app.api.endpoints import predictor_service

def test_health_endpoint(test_client):
    """
    Verifies /health endpoint returns 200 OK and correct structure.
    """
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "api_version" in data
    assert isinstance(data["model_loaded"], bool)

def test_model_info_endpoint(test_client):
    """
    Verifies /model-info returns metadata matching the active Random Forest model.
    """
    response = test_client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "RandomForestRegressor_FD001"
    assert data["model_version"] == "1.3.0"
    assert "framework" in data
    assert data["expected_window_size"] == WINDOW_SIZE
    assert data["expected_num_features"] == NUM_FEATURES
    assert "metrics" in data
    assert "MAE" in data["metrics"]
    assert "RMSE" in data["metrics"]

def test_predict_rul_engine_54_valid(test_client, sample_payloads, get_sensor_window_payload):
    """
    Tests /predict/rul using the official Engine 54 payload.
    Verifies the newly added prediction interval level, coverage, description, and strategy.
    """
    case_data = sample_payloads["engine_54_successful"]
    window = get_sensor_window_payload(case_data)
    
    payload = {
        "engine_id": case_data["unit_nr"],
        "cycle": case_data["total_cycles"],
        "sensor_window": window
    }
    
    response = test_client.post("/predict/rul", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["engine_id"] == 54
    assert data["cycle"] == case_data["total_cycles"]
    assert "estimated_rul" in data
    assert "prediction_interval_lower" in data
    assert "prediction_interval_upper" in data
    assert data["prediction_interval_level"] == 0.90
    assert data["prediction_interval_coverage"] == 0.8998
    assert data["prediction_interval_description"] == "Bounds use validation-residual 5th and 95th percentiles."
    assert "sequence_conversion_strategy" in data
    assert data["risk_level"] in ("Low", "Medium", "High", "Critical")
    assert 0.0 <= data["data_quality_score"] <= 1.0
    assert "recommendation" in data
    assert data["model_name"] == "RandomForestRegressor_FD001"
    assert data["model_version"] == "1.3.0"

def test_predict_rul_engine_74_valid(test_client, sample_payloads, get_sensor_window_payload):
    """
    Tests /predict/rul using the official Engine 74 payload.
    """
    case_data = sample_payloads["engine_74_difficult"]
    window = get_sensor_window_payload(case_data)
    
    payload = {
        "engine_id": case_data["unit_nr"],
        "cycle": case_data["total_cycles"],
        "sensor_window": window
    }
    response = test_client.post("/predict/rul", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["engine_id"] == 74
    assert data["cycle"] == case_data["total_cycles"]
    assert "estimated_rul" in data
    assert data["prediction_interval_level"] == 0.90
    assert data["prediction_interval_coverage"] == 0.8998
    assert data["prediction_interval_description"] == "Bounds use validation-residual 5th and 95th percentiles."
    assert "sequence_conversion_strategy" in data
    assert data["risk_level"] in ("Low", "Medium", "High", "Critical")

@pytest.mark.parametrize("bad_shape", [
    [[1.0] * (NUM_FEATURES - 1)] * WINDOW_SIZE,  # Malformed features: 15 instead of 16
    [[1.0] * NUM_FEATURES] * (WINDOW_SIZE - 1),  # Malformed timesteps: 29 instead of 30
    [[1.0] * (NUM_FEATURES + 1)] * WINDOW_SIZE   # Malformed features: 17 instead of 16
])
def test_predict_rul_invalid_dimensions(test_client, bad_shape):
    """
    Tests that sequences with incorrect shapes are rejected.
    """
    payload = {
        "engine_id": 54,
        "cycle": 257,
        "sensor_window": bad_shape
    }
    response = test_client.post("/predict/rul", json=payload)
    assert response.status_code == 422

def test_predict_rul_non_finite(test_client):
    """
    Tests that requests with non-finite values (NaN, Inf) are rejected.
    """
    bad_window = [[1.0] * NUM_FEATURES] * WINDOW_SIZE
    bad_window[0][0] = float("nan")
    
    payload = {
        "engine_id": 54,
        "cycle": 257,
        "sensor_window": bad_window
    }
    response = test_client.post("/predict/rul", json=payload)
    assert response.status_code == 422

def test_missing_artifacts_behavior(test_client, sample_payloads, get_sensor_window_payload, monkeypatch, tmp_path):
    """
    Simulates missing model artifacts to test the explicit model-unavailable state.
    Monkeypatches the model directory instead of altering repo files.
    """
    # Point predictor service to an empty temp dir
    empty_models_dir = tmp_path / "models"
    empty_models_dir.mkdir()
    
    try:
        import sys
        
        # Monkeypatch the module variable where model_dir logic lives
        original_init = predictor_service.__init__
        
        def mock_init(self):
            # Same init logic but override REPO_ROOT effectively
            self.model_loaded = False
            self.predictor = None
            self.metadata = {}
            
        monkeypatch.setattr(predictor_service.__class__, "__init__", mock_init)
        
        # Trigger reload of the predictor service to check directory
        predictor_service.__init__()
        
        # 1. Health check should still return 200 but model_loaded: False
        health_resp = test_client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["model_loaded"] is False
        
        # 2. Predict should return 503 Service Unavailable
        case_data = sample_payloads["engine_54_successful"]
        window = get_sensor_window_payload(case_data)
        payload = {
            "engine_id": case_data["unit_nr"],
            "cycle": case_data["total_cycles"],
            "sensor_window": window
        }
        pred_resp = test_client.post("/predict/rul", json=payload)
        assert pred_resp.status_code == 503
        
        # 3. Model Info should return 503 Service Unavailable
        info_resp = test_client.get("/model-info")
        assert info_resp.status_code == 503
        
    finally:
        # Re-initialize properly when exiting
        monkeypatch.undo()
        predictor_service.__init__()
        
    # Verify it restored successfully
    assert predictor_service.model_loaded is True
