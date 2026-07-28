import os
import json
from typing import List

# Base directory of the repository (useful for repository-relative paths)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# API configuration
API_TITLE = "Predictive Maintenance RUL API"
API_VERSION = "1.0.0"

# Load feature order dynamically from Member A's artifacts to avoid duplicating constants
feature_order_path = os.path.join(BASE_DIR, "models", "feature_order.json")
if os.path.exists(feature_order_path):
    try:
        with open(feature_order_path, "r") as f:
            FEATURE_ORDER = json.load(f)
    except Exception:
        FEATURE_ORDER = [
            "op_setting_1", "op_setting_2", "sensor_2", "sensor_3", "sensor_4",
            "sensor_7", "sensor_8", "sensor_9", "sensor_11", "sensor_12",
            "sensor_13", "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21"
        ]
else:
    FEATURE_ORDER = [
        "op_setting_1", "op_setting_2", "sensor_2", "sensor_3", "sensor_4",
        "sensor_7", "sensor_8", "sensor_9", "sensor_11", "sensor_12",
        "sensor_13", "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21"
    ]

NUM_FEATURES = len(FEATURE_ORDER)
WINDOW_SIZE = 30

# Load model metadata for name and version dynamically if available
MODEL_NAME = "RandomForestRegressor_FD001"
MODEL_VERSION = "1.0.0"
metadata_path = os.path.join(BASE_DIR, "models", "metadata.json")
if os.path.exists(metadata_path):
    try:
        with open(metadata_path, "r") as f:
            meta = json.load(f)
            MODEL_NAME = meta.get("model_name", MODEL_NAME)
            MODEL_VERSION = meta.get("version", MODEL_VERSION)
    except Exception:
        pass

# Security settings
# Restrict CORS origins. Default to local Streamlit app ports.
origins_env = os.getenv("ALLOWED_ORIGINS")
if origins_env:
    try:
        ALLOWED_ORIGINS: List[str] = json.loads(origins_env)
    except (json.JSONDecodeError, TypeError):
        ALLOWED_ORIGINS = ["http://localhost:8501", "http://127.0.0.1:8501"]
else:
    ALLOWED_ORIGINS = ["http://localhost:8501", "http://127.0.0.1:8501"]

# Max request body size in bytes (e.g., 1MB to prevent DOS)
MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 1024 * 1024))
