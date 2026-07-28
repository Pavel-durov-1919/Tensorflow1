import pytest
import os
import sys
import json
from fastapi.testclient import TestClient

# ME-01: Ensure 'backend' is in sys.path
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.main import app
from app.config import BASE_DIR, FEATURE_ORDER

@pytest.fixture(scope="session")
def test_client():
    """Provides a global TestClient instance."""
    return TestClient(app)

@pytest.fixture(scope="session")
def sample_payloads():
    """Loads official sample payloads."""
    payloads_file = os.path.join(BASE_DIR, "models", "sample_payloads.json")
    if not os.path.exists(payloads_file):
        pytest.skip(f"Missing official integration payloads at {payloads_file}")
    with open(payloads_file, "r") as f:
        return json.load(f)

@pytest.fixture
def get_sensor_window_payload():
    """Helper fixture to format payloads to the 30x16 array."""
    def _formatter(raw_case):
        seq = raw_case["sequence_30_cycle_payload"]
        return [[row[feat] for feat in FEATURE_ORDER] for row in seq]
    return _formatter
