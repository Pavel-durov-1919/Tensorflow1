# Model Card: C-MAPSS FD001 RandomForest RUL Estimator (v1.3.0)

## Model Overview
- **Model Name**: `RandomForestRegressor_FD001`
- **Model Type**: Ensembled Decision Trees (`RandomForestRegressor`)
- **Version**: `1.3.0`
- **Hyperparameters**: `n_estimators=120`, `max_depth=14`, `min_samples_split=5`, `min_samples_leaf=2`, `random_state=42`
- **Training Framework**: `scikit-learn` (v1.6+)
- **Artifacts**: `models/baseline_model.joblib`, `models/scaler.joblib`, `models/feature_order.json`, `models/feature_importance.json`, `models/metadata.json`

---

## Intended Use & Production API Contract
- **Primary Task**: Predict Remaining Useful Life (RUL) of C-MAPSS turbofan engines.
- **API Contract**: Accepts strictly a 30-cycle sequence shaped `(30, 16)` containing unscaled raw sensor measurements.
- **Output Interface**: Returns point RUL prediction, empirical lower/upper bounds, risk level, data quality score, and actionable engineering recommendation.

---

## Model Performance & Validation Metrics

### Performance Summary
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Overall MAE** | `12.3476` cycles | Mean Absolute Error across all validation cycles |
| **Overall RMSE** | `17.0591` cycles | Root Mean Squared Error across all validation cycles |
| **Near-Failure MAE** | **`6.0716` cycles** | **MAE on engines near end-of-life ($\text{RUL} \le 30$)** |
| **Authoritative Residual Offsets** | `[-15.01, +24.08]` | Empirical 5% and 95% validation residual quantiles |
| **Empirical 90% Interval Coverage** | **`89.98%`** | Proportion of true RULs falling within `[lower_bound, upper_bound]` |

### Segmented RUL Metrics
- **Near-Failure ($\text{RUL} \le 30$)**: `620` samples | MAE: `6.07` cycles | RMSE: `9.08` cycles
- **Mid-Life ($30 < \text{RUL} \le 75$)**: `900` samples | MAE: `18.31` cycles | RMSE: `24.14` cycles
- **Early-Life ($\text{RUL} > 75$)**: `2,550` samples | MAE: `11.77` cycles | RMSE: `15.45` cycles

---

## Feature Importance Ranking (`models/feature_importance.json`)

Top physical sensor predictors identified by the model:
1. `sensor_11` (HPC Outlet Static Pressure): **18.42%**
2. `sensor_9` (NR Speed / Fan Speed): **16.15%**
3. `sensor_4` (LPT Outlet Temperature): **14.88%**
4. `sensor_12` (Bypass Ratio): **12.30%**
5. `sensor_14` (Corrected Core Speed): **10.95%**
6. `sensor_7` (HPC Outlet Pressure): **9.84%**

---

## Model Limitations & Operational Constraints

> [!CAUTION]
> 1. **Final Cycle Snapshot Usage**: The baseline Random Forest model evaluates **ONLY the final cycle (Cycle 30)** of the 30-cycle input window, ignoring preceding temporal progression within the window.
> 2. **Piecewise RUL Target Capping**: Predictions and bounds are strictly capped to `[0.0, 125.0]` cycles.
> 3. **Single Flight Regime**: Model trained exclusively on FD001 (single sea-level operating condition) and cannot be applied directly to multi-condition datasets (FD002/FD004) without re-training.
