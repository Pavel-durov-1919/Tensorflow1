# AeroShield Experiment Results

**Date:** 2026-10-20  
**Project:** AeroShield — Aircraft Engine Remaining Useful Life (RUL) Prediction  
**Environment:** Python 3.12.10, scikit-learn 1.8.0, numpy 2.3.5, pandas 2.3.3, joblib 1.5.3  

---

## Experiment 001: Random Forest Baseline on C-MAPSS FD001

| Attribute | Details |
|-----------|---------|
| **Experiment ID** | `EXP-001-RF-CMAPSS-FD001` |
| **Status** | Completed |
| **Dataset** | NASA C-MAPSS FD001 (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`) |
| **Engine Units** | Train/Val: 100 engines (20,631 cycles) · Test: 100 engines (13,096 cycles, 100 target points) |
| **Data Split** | GroupShuffleSplit on `unit_nr` (80 train engines / 20 validation engines, `test_size=0.20`, `random_state=42`); predefined test set (100 engines) evaluated at final observed cycle |
| **Model Type** | `RandomForestRegressor` (scikit-learn 1.8.0) |
| **Hyperparameters** | `n_estimators=120`, `max_depth=14`, `min_samples_split=5`, `min_samples_leaf=2`, `max_features="sqrt"`, `random_state=42`, `n_jobs=-1` |
| **Feature Set (16)** | `op_setting_1`, `op_setting_2`, `sensor_2`, `sensor_3`, `sensor_4`, `sensor_7`, `sensor_8`, `sensor_9`, `sensor_11`, `sensor_12`, `sensor_13`, `sensor_14`, `sensor_15`, `sensor_17`, `sensor_20`, `sensor_21` |
| **Excluded Features** | 7 constant/zero-variance sensors: `sensor_1`, `sensor_5`, `sensor_6`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`, plus `op_setting_3` |
| **Target Construction** | Piecewise linear RUL: `RUL = min(max_cycle - current_cycle, 125)` (capped at 125 cycles) |
| **Window Configuration** | Window length $W=30$, stride=1. Final cycle snapshot feature extraction for Random Forest input vector. |
| **Preprocessing** | `StandardScaler` fitted on training split only; applied to validation/test splits. Preprocessing artifacts persisted to `models/scaler.joblib`. |
| **Training Duration** | 5.3 seconds |
| **Reproduction Command** | `python -m src.train` |
| **Artifacts** | `models/baseline_model.joblib`, `models/scaler.joblib`, `models/metadata.json`, `models/feature_order.json`, `models/feature_importance.json`, `models/engine_examples.json`, `models/sample_payloads.json` |

### Measured Performance Metrics

All metrics are measured on real data from the actual model run:

| Split / Segment | Observation Count | MAE (cycles) | RMSE (cycles) |
|---|---|---|---|
| **Validation Set (Overall)** | 4,070 windows | **12.3476** | **17.0591** |
| Validation: Near-failure ($RUL \le 30$) | 620 windows | **6.0716** | **9.0755** |
| Validation: Mid-life ($30 < RUL \le 75$) | 900 windows | **18.3110** | **24.1355** |
| Validation: Early-life ($RUL > 75$) | 2,550 windows | **11.7688** | **15.4549** |
| **Test Set (Ground Truth)** | 100 engines | **11.7810** | **16.8334** |

### Empirical Prediction Intervals (Validation Residuals)

- **5th Percentile Error (Lower bound):** $-34.28$ cycles  
- **95th Percentile Error (Upper bound):** $+24.08$ cycles  
- **90% Interval Empirical Coverage:** $89.98\%$  

---

## Experiment 002: CruiseBench Adapter Smoke Evaluation

| Attribute | Details |
|-----------|---------|
| **Experiment ID** | `EXP-002-CRUISEBENCH-MOCK` |
| **Status** | Partially Verified (Mock Fallback Only) |
| **Dataset** | CruiseBench (N-CMAPSS benchmark format) |
| **Real Data Availability** | `CruiseBench.h5` and `CruiseBench.csv` are not present in repository. |
| **Fallback Mechanism** | Synthetically generated mock DataFrame matching the 18-feature CruiseBench schema (`w, mach, alt, tla, t24, t30, t48, t50, p15, p2, p21, p24, ps30, p40, p50, nf, nc, wf`). |
| **Verification** | Unit tests in `tests/test_core.py::TestCruiseBenchAdapter` pass cleanly with mock generation. |
| **Known Limitation** | Full real-data evaluation is blocked pending provision of the raw benchmark file `CruiseBench.h5`. |
