# Data Card: C-MAPSS FD001 Turbofan Engine Degradation Dataset

## Dataset Description
- **Dataset Name**: NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) FD001
- **Domain**: Predictive Maintenance / Degradation Engineering
- **Primary Task**: Estimate Remaining Useful Life (RUL) of turbofan engines under run-to-failure operational simulation.
- **Operating Condition**: Single sea-level flight condition.
- **Fault Mode**: High-Pressure Compressor (HPC) degradation.

---

## Data Provenance & Verification
- **Source**: NASA Prognostics Data Repository / HuggingFace Mirror (`DeveloperMindset123/CMAPSS_Jet_Engine_Simulated_Data`)
- **Raw File Verification**:
  | File Name | Size (Bytes) | MD5 Checksum | SHA256 Checksum |
  | :--- | :--- | :--- | :--- |
  | `train_FD001.txt` | 3,515,356 | `259f340bac32ce6fa8894815600fa757` | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
  | `test_FD001.txt` | 2,228,855 | `62316d1dd70f88a9a43a1e2690765437` | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |
  | `RUL_FD001.txt` | 429 | `8f4a8b3d2692b2f7d8e72b344927bf4b` | `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca` |

---

## Dataset Schema & Feature Selection

### Raw Schema (26 Columns)
- `unit_nr`: Engine identifier (integer 1..100).
- `cycle`: Operational flight cycle index (1..max_cycle).
- `op_setting_1`, `op_setting_2`, `op_setting_3`: Altitude, Mach number, and throttle resolver angle settings.
- `sensor_1` through `sensor_21`: Physical engine measurements (temperature, pressure, fan speeds).

### Selected Feature Set (16 Columns)
Based on variance analysis ($\sigma \ge 0.01$), 7 static/constant sensors were removed (`sensor_1`, `sensor_5`, `sensor_6`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`) along with `op_setting_3`.

**Final Feature Order**:
`['op_setting_1', 'op_setting_2', 'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 'sensor_20', 'sensor_21']`

---

## Engine Splitting & Zero Data Leakage
- **Splitting Algorithm**: `GroupShuffleSplit` by `unit_nr` (Seed = 42).
- **Training Engines**: 80 engine units (16,461 operational cycle snapshots).
- **Validation Engines**: 20 engine units (4,170 operational cycle snapshots).
- **Data Leakage Prevention**: Split performed strictly by engine identity prior to scaling. `StandardScaler` fitted exclusively on training engines.

---

## Target Labeling & RUL Capping
- `RUL_raw`: `max_cycle - current_cycle` (Unbounded ground truth).
- `RUL_clipped`: $\min(\text{RUL\_raw}, 125.0)$ (Piecewise constant capping standard in C-MAPSS literature to model early non-degraded engine state).
