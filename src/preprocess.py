"""
C-MAPSS FD001 Preprocessing & Engine Splitting Module
Author: Member A (Data & ML Lead)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

try:
    from src.data_loader import get_validated_dataset, PROCESSED_DATA_DIR
except ImportError:
    from data_loader import get_validated_dataset, PROCESSED_DATA_DIR


def prepare_engine_splits(train_df: pd.DataFrame, useful_features: list, test_size: float = 0.2, random_state: int = 42) -> tuple:
    """
    Split engines into Train and Validation sets using GroupShuffleSplit.
    Ensures zero data leakage between engines.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(gss.split(train_df, groups=train_df["unit_nr"]))
    
    train_split = train_df.iloc[train_idx].copy()
    val_split = train_df.iloc[val_idx].copy()
    
    # Fit scaler strictly on training split
    scaler = StandardScaler()
    train_scaled_feats = scaler.fit_transform(train_split[useful_features])
    val_scaled_feats = scaler.transform(val_split[useful_features])
    
    # Create scaled columns
    scaled_feature_cols = [f"{c}_scaled" for c in useful_features]
    
    for i, sc_col in enumerate(scaled_feature_cols):
        train_split[sc_col] = train_scaled_feats[:, i]
        val_split[sc_col] = val_scaled_feats[:, i]
        
    return train_split, val_split, scaler, scaled_feature_cols


def create_sequence_windows(df: pd.DataFrame, feature_cols: list, sequence_length: int = 30, target_col: str = "RUL_clipped") -> tuple:
    """
    Generate rolling sequence windows per engine unit for sequence models.
    """
    X_seq, y_seq = [], []
    
    for unit_nr, group in df.groupby("unit_nr"):
        feats = group[feature_cols].values
        targets = group[target_col].values
        n_samples = len(group)
        
        if n_samples < sequence_length:
            continue
            
        for i in range(sequence_length, n_samples + 1):
            X_seq.append(feats[i - sequence_length:i])
            y_seq.append(targets[i - 1])
            
    return np.array(X_seq), np.array(y_seq)


def export_processed_samples(train_split: pd.DataFrame, useful_features: list, scaled_cols: list, processed_dir: str = PROCESSED_DATA_DIR):
    """Export processed dataset sample for Member B & C integration."""
    os.makedirs(processed_dir, exist_ok=True)
    
    # Select columns: metadata + unscaled features + scaled features + RUL labels
    export_cols = ["unit_nr", "cycle", "RUL_raw", "RUL_clipped"] + useful_features + scaled_cols
    sample_df = train_split[export_cols].head(100)
    
    sample_path = os.path.join(processed_dir, "fd001_processed_sample.csv")
    sample_df.to_csv(sample_path, index=False)
    print(f"Exported processed sample (100 rows) to {sample_path}")
    
    return sample_path


if __name__ == "__main__":
    train_df, _, _, _, _, useful_features = get_validated_dataset()
    train_sp, val_sp, scaler, sc_cols = prepare_engine_splits(train_df, useful_features)
    sample_p = export_processed_samples(train_sp, useful_features, sc_cols)
    print(f"Engine splits created. Train engines: {train_sp['unit_nr'].nunique()}, Val engines: {val_sp['unit_nr'].nunique()}")
