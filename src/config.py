"""
Centralized Configuration Module
"""

import os

# Options: "cmapss", "cruisebench"
DATASET_TYPE = os.getenv("DATASET_TYPE", "cmapss")

# Options: "random_forest"
MODEL_TYPE = os.getenv("MODEL_TYPE", "random_forest")

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
WINDOW_LENGTH = int(os.getenv("WINDOW_LENGTH", "30"))
