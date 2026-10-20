# AeroShield Baseline Results

**Date:** 2026-10-20  
**Phase:** 0 — Ground Truth Establishment

## Application Startup

| Component | Command | Working Directory | Status |
|-----------|---------|-------------------|--------|
| Backend | `python -m uvicorn app.main:app --port 8000 --reload` | `backend/` | ✅ Starts on `http://127.0.0.1:8000` |
| Dashboard | `python -m streamlit run frontend/dashboard.py` | project root | ✅ Starts on `http://localhost:8501` |

## Endpoint Verification

| Endpoint | Method | Status | Response |
|----------|--------|--------|----------|
| `/` | GET | 200 | `{"service":"Predictive Maintenance RUL API","version":"1.0.0","docs":"/docs"}` |
| `/health` | GET | 200 | `{"status":"healthy","api_version":"1.0.0","model_loaded":true}` |
| `/model-info` | GET | 200 | Model name, version, metrics, window size, feature count |
| `/predict/rul` | POST (valid) | 200 | Full prediction with RUL, bounds, risk, recommendation |
| `/predict/rul` | POST (bad dims) | 422 | Descriptive validation error |
| `/predict/rul` | POST (NaN) | 422 | Non-finite value rejection |
| `/docs` | GET | 200 | OpenAPI documentation |

## Training Pipeline

```
Command: python -m src.train
Working directory: project root

Output:
  Downloading train_FD001.txt ... Verified: 3515356 bytes
  Downloading test_FD001.txt  ... Verified: 2228855 bytes
  Downloading RUL_FD001.txt   ... Verified: 429 bytes
  Training RandomForestRegressor on cmapss dataset...
  Validation -> MAE: 12.3476, RMSE: 17.0591
  Test Set   -> MAE: 11.781,  RMSE: 16.8334
  Artifacts saved successfully to models/
```

## Measured Metrics

| Split | MAE (cycles) | RMSE (cycles) |
|-------|-------------|---------------|
| Validation | 12.3476 | 17.0591 |
| Test | 11.7810 | 16.8334 |
| Near-failure (RUL ≤ 30) | 6.0716 | 9.0755 |
| Mid-life (30–75) | 18.3110 | 24.1355 |
| Early-life (> 75) | 11.7688 | 15.4549 |

## Test Results

```
28 passed, 0 failed, 17 warnings in 0.97s
```

All 28 tests pass after correcting 3 stale version/name assertions.

## Dashboard Integration

- Dashboard connects to backend at `http://localhost:8000`
- Health status indicator works (shows "API Connected · Model Active")
- Engine case studies (Engine #054 and #074) load from `models/engine_examples.json`
- Predictions are fetched live from the backend API
- Sensor telemetry, lifetime trajectory, and feature importance charts render correctly
- Risk level, prediction interval, data quality score all display

## Limitations

1. **CruiseBench:** Adapter exists but only works with mock data (no real H5/CSV file available)
2. **Backend startup:** Must be run from `backend/` directory (or set PYTHONPATH)
3. **Training uses unscaled features:** The Random Forest is trained on raw feature values, not the scaled columns — this is intentional for this baseline
