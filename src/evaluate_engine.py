"""
C-MAPSS FD001 Engine Trajectory & Case Study Extraction Module
Author: Member A (Data & ML Lead)
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error

try:
    from src.data_loader import REPO_ROOT
    from src.predict import RULPredictor
except ImportError:
    from data_loader import REPO_ROOT
    from predict import RULPredictor

MODEL_DIR = os.path.join(REPO_ROOT, "models")
WINDOW_LENGTH = 30


def extract_engine_case_studies(val_df: pd.DataFrame, predictor: RULPredictor) -> dict:
    """
    Evaluate validation engines cycle-by-cycle using 30-cycle rolling windows
    and extract 1 successful and 1 difficult engine case study for Member B & C UI visualization.
    """
    engine_results = {}
    
    for unit_nr, group in val_df.groupby("unit_nr"):
        group = group.sort_values("cycle").reset_index(drop=True)
        n_samples = len(group)
        if n_samples < WINDOW_LENGTH:
            continue
            
        actual_ruls = []
        pred_ruls = []
        cycles = []
        
        for i in range(WINDOW_LENGTH, n_samples + 1):
            window_df = group.iloc[i - WINDOW_LENGTH:i]
            res = predictor.predict_rul(window_df)
            
            actual_ruls.append(float(group["RUL_clipped"].iloc[i - 1]))
            pred_ruls.append(float(res["estimated_rul"]))
            cycles.append(int(group["cycle"].iloc[i - 1]))
            
        rmse = float(root_mean_squared_error(actual_ruls, pred_ruls))
        mae = float(np.mean(np.abs(np.array(actual_ruls) - np.array(pred_ruls))))
        
        # Sample unscaled features for final cycle
        sample_unscaled = group[predictor.feature_order].iloc[-1].to_dict()
        sample_scaled_cols = [f"{c}_scaled" for c in predictor.feature_order]
        sample_scaled = group[sample_scaled_cols].iloc[-1].to_dict() if all(c in group.columns for c in sample_scaled_cols) else {}
        
        engine_results[int(unit_nr)] = {
            "unit_nr": int(unit_nr),
            "total_cycles": len(group),
            "eval_windows_count": len(cycles),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "cycles": cycles,
            "actual_rul": [round(v, 2) for v in actual_ruls],
            "predicted_rul": [round(v, 2) for v in pred_ruls],
            "final_actual_rul": round(actual_ruls[-1], 2),
            "final_predicted_rul": round(pred_ruls[-1], 2),
            "latest_unscaled_features": sample_unscaled,
            "latest_scaled_features": sample_scaled
        }
        
    sorted_engines = sorted(engine_results.values(), key=lambda x: x["rmse"])
    successful_engine = sorted_engines[0]   # Lowest RMSE
    difficult_engine = sorted_engines[-1]   # Highest RMSE
    
    case_studies = {
        "successful_engine_example": {
            "type": "SUCCESSFUL",
            "description": f"Engine #{successful_engine['unit_nr']} demonstrated smooth degradation curve with low error.",
            "metrics": {"rmse": successful_engine["rmse"], "mae": successful_engine["mae"]},
            "data": successful_engine
        },
        "difficult_engine_example": {
            "type": "DIFFICULT",
            "description": f"Engine #{difficult_engine['unit_nr']} exhibited non-linear degradation leading to higher prediction variance.",
            "metrics": {"rmse": difficult_engine["rmse"], "mae": difficult_engine["mae"]},
            "data": difficult_engine
        }
    }
    
    output_path = os.path.join(MODEL_DIR, "engine_examples.json")
    with open(output_path, "w") as f:
        json.dump(case_studies, f, indent=2)
        
    print(f"Engine case studies saved to {output_path}")
    print(f"Successful Engine: Unit #{successful_engine['unit_nr']} (RMSE: {successful_engine['rmse']})")
    print(f"Difficult Engine: Unit #{difficult_engine['unit_nr']} (RMSE: {difficult_engine['rmse']})")
    
    return case_studies


if __name__ == "__main__":
    from preprocess import prepare_engine_splits, get_validated_dataset
    train_df, _, _, _, _, useful_feats = get_validated_dataset()
    _, val_df, _, _ = prepare_engine_splits(train_df, useful_feats)
    predictor = RULPredictor()
    extract_engine_case_studies(val_df, predictor)
