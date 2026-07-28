import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.scaled_cols = []
        
    def fit_transform(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> tuple:
        """
        Fits the scaler on the training set and transforms train, val, and test sets.
        """
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        
        train_scaled = self.scaler.fit_transform(train_df[feature_cols])
        val_scaled = self.scaler.transform(val_df[feature_cols])
        test_scaled = self.scaler.transform(test_df[feature_cols])
        
        self.scaled_cols = [f"{c}_scaled" for c in feature_cols]
        
        for i, sc_col in enumerate(self.scaled_cols):
            train_df[sc_col] = train_scaled[:, i]
            val_df[sc_col] = val_scaled[:, i]
            test_df[sc_col] = test_scaled[:, i]
            
        return train_df, val_df, test_df, self.scaled_cols

class WindowGenerator:
    @staticmethod
    def create_windows(df: pd.DataFrame, feature_cols: list[str], target_col: str, sequence_length: int = 30) -> tuple:
        """
        Generate rolling sequence windows per engine unit for sequence models.
        Dynamically detects the engine ID column (engine_id or unit_nr).
        """
        X_seq, y_seq = [], []
        group_col = "engine_id" if "engine_id" in df.columns else "unit_nr"
        
        for unit, group in df.groupby(group_col):
            feats = group[feature_cols].values
            targets = group[target_col].values
            n_samples = len(group)
            
            if n_samples < sequence_length:
                continue
                
            for i in range(sequence_length, n_samples + 1):
                X_seq.append(feats[i - sequence_length:i])
                y_seq.append(targets[i - 1])
                
        return np.array(X_seq), np.array(y_seq)
