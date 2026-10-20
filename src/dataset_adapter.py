import os
import pandas as pd
from abc import ABC, abstractmethod

# Local imports
try:
    from src.data_loader import get_validated_dataset
except ImportError:
    from data_loader import get_validated_dataset

class BaseDatasetAdapter(ABC):
    @abstractmethod
    def get_data_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Returns train_df, val_df, test_df"""
        pass
        
    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """Returns the list of feature columns"""
        pass
        
    @abstractmethod
    def get_rul_column(self) -> str:
        """Returns the target RUL column name"""
        pass
        
    @abstractmethod
    def get_metadata(self) -> dict:
        """Returns dataset-specific metadata"""
        pass


class CMAPSSAdapter(BaseDatasetAdapter):
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self._useful_features = None
        self._metadata = None
        
    def get_data_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df, test_df, test_rul_series, checksums, sensor_meta, useful_features = get_validated_dataset()
        self._useful_features = useful_features
        self._metadata = {
            "dataset_checksums": checksums,
            "sensor_metadata": sensor_meta
        }
        
        # We temporarily bypass scaling here, since Preprocessor should handle it
        # but we need to split engines.
        # Wait, prepare_engine_splits does scaling. We'll modify prepare_engine_splits later
        # to just split, or we just call GroupShuffleSplit here.
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=self.test_size, random_state=self.random_state)
        train_idx, val_idx = next(gss.split(train_df, groups=train_df["unit_nr"]))
        
        train_split = train_df.iloc[train_idx].copy()
        val_split = train_df.iloc[val_idx].copy()
        
        # For C-MAPSS test set, true RUL is only available for the last cycle of each engine
        test_last_cycles = test_df.groupby('unit_nr').last().reset_index()
        test_last_cycles['RUL_clipped'] = test_rul_series.values
        
        return train_split, val_split, test_last_cycles
        
    def get_feature_names(self) -> list[str]:
        if self._useful_features is None:
            self.get_data_splits()
        return self._useful_features
        
    def get_rul_column(self) -> str:
        return "RUL_clipped"
        
    def get_metadata(self) -> dict:
        if self._metadata is None:
            self.get_data_splits()
        return self._metadata


class CruiseBenchAdapter(BaseDatasetAdapter):
    def __init__(self, data_path="CruiseBench.h5", fallback_path="CruiseBench.csv"):
        self.data_path = data_path
        self.fallback_path = fallback_path
        self._features = [
            "w", "mach", "alt", "tla", "t24", "t30", "t48", "t50", 
            "p15", "p2", "p21", "p24", "ps30", "p40", "p50", "nf", "nc", "wf"
        ]
        
    def get_data_splits(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if os.path.exists(self.data_path):
            try:
                import h5py
            except ImportError:
                raise ImportError("h5py is required to load CruiseBench.h5. Install it with: pip install h5py")
            with h5py.File(self.data_path, "r") as f:
                train_df = pd.DataFrame(f["train_data"][:], columns=f["train_data"].attrs["columns"])
                val_df = pd.DataFrame(f["val_data"][:], columns=f["val_data"].attrs["columns"])
                test_df = pd.DataFrame(f["test_data"][:], columns=f["test_data"].attrs["columns"])
        elif os.path.exists(self.fallback_path):
            # If a combined CSV is provided, we would load it. 
            # For this prototype, we'll assume the files might be split if CSV.
            raise NotImplementedError("CSV fallback requires specific split definitions not yet implemented.")
        else:
            # Generate mock dataset for CI/CD and testing if real dataset is absent
            print(f"Warning: {self.data_path} not found. Returning mock CruiseBench dataset.")
            return self._generate_mock_data()
            
        return train_df, val_df, test_df
        
    def _generate_mock_data(self):
        import numpy as np
        # Create a mock dataframe structure
        cols = ["engine_id", "cycle", "rul"] + self._features
        def make_mock(n_engines):
            data = []
            for e in range(1, n_engines + 1):
                for c in range(1, 31):
                    row = [e, c, max(0, 31-c)] + list(np.random.randn(len(self._features)))
                    data.append(row)
            return pd.DataFrame(data, columns=cols)
        
        return make_mock(5), make_mock(2), make_mock(2)

    def get_feature_names(self) -> list[str]:
        return self._features
        
    def get_rul_column(self) -> str:
        return "rul"
        
    def get_metadata(self) -> dict:
        return {
            "description": "CruiseBench Dataset derived from N-CMAPSS"
        }

class DatasetFactory:
    @staticmethod
    def get_adapter(dataset_type: str) -> BaseDatasetAdapter:
        dataset_type = dataset_type.lower()
        if dataset_type == "cmapss":
            return CMAPSSAdapter()
        elif dataset_type == "cruisebench":
            return CruiseBenchAdapter()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
