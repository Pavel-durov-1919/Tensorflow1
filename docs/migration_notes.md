# Migration Notes: C-MAPSS FD001 to Modular CruiseBench Framework

## Overview
This document tracks the architectural changes introduced to migrate the predictive maintenance pipeline from a monolithic C-MAPSS (FD001) implementation to a highly modular, dataset-agnostic framework supporting both the legacy C-MAPSS dataset and the new CruiseBench (N-CMAPSS) benchmark dataset.

## Architectural Changes

### 1. Configuration Layer (`src/config.py`)
- Extracted hardcoded constants into a dynamic configuration module (`DATASET_TYPE`, `MODEL_TYPE`, etc.).
- The system now uses environment variables to seamlessly switch between datasets and models without altering code.

### 2. Dataset Factory Pattern (`src/dataset_adapter.py`)
- Created a `BaseDatasetAdapter` abstract class to enforce a consistent data loading contract.
- **CMAPSSAdapter**: Refactored the old `data_loader.py` logic. It automatically downloads, filters features, calculates RUL, and dynamically splits the training dataset.
- **CruiseBenchAdapter**: Supports `CruiseBench.h5` and `CruiseBench.csv`. It loads pre-split data (`train`, `val`, `test`) and utilizes pre-computed RUL targets, skipping redundant splitting and calculation steps.

### 3. Model Factory Pattern (`src/model_adapter.py`)
- Created a `BaseModelAdapter` abstract class.
- Encapsulated the existing Random Forest baseline into `RandomForestAdapter`.
- This layer guarantees that `train.py` can swap Random Forests for LSTMs, Transformers, or XGBoost without modifying the main training loop.

### 4. Decoupled Preprocessing (`src/preprocess.py`)
- Transitioned to class-based `Preprocessor` and `WindowGenerator`.
- The `Preprocessor` fits `StandardScaler` strictly on the train set provided by the adapter and applies it uniformly to validation and test sets.
- `WindowGenerator` abstracts the sequence windowing, automatically determining the grouping key (`unit_nr` vs. `engine_id`).

### 5. Dataset-Agnostic Backend (`backend/app/schemas.py`, `backend/app/api/endpoints.py`)
- Removed hardcoded feature dimensions (`NUM_FEATURES=16`).
- The FastAPI schemas now validate incoming sequences dynamically based on the active `model_info` metadata, which inherits its expectations directly from the dataset adapter used during training.

### 6. Dataset-Agnostic Frontend (`frontend/dashboard.py`)
- Removed the hardcoded NASA `SENSOR_DESCRIPTIONS` dictionary.
- The dashboard now polls the `model-info` (and `metadata.json`) for the ordered feature list and their descriptions.
- The UI dynamically scales whether the dataset provides 16 features or 21 features.

## Backward Compatibility
All existing REST API endpoints, evaluation suites, and Apple-inspired Streamlit telemetry charts will function identically as before when `DATASET_TYPE="cmapss"`. 

When swapped to `DATASET_TYPE="cruisebench"`, the pipeline inherits 21 features natively, bypassing older static configurations while maintaining robust statistical coverage reporting.
