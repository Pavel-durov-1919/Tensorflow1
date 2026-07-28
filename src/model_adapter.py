from abc import ABC, abstractmethod
import joblib
import numpy as np

class BaseModelAdapter(ABC):
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the model"""
        pass
        
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the trained model"""
        pass
        
    @abstractmethod
    def save(self, path: str):
        """Save model artifacts to disk"""
        pass
        
    @abstractmethod
    def load(self, path: str):
        """Load model from disk"""
        pass
        
    @abstractmethod
    def get_feature_importances(self, feature_names: list[str]) -> dict:
        """Return feature importances if applicable"""
        pass
        
    @abstractmethod
    def get_metadata(self) -> dict:
        """Return model metadata (e.g., algorithm name)"""
        pass

class RandomForestAdapter(BaseModelAdapter):
    def __init__(self, random_state=42):
        from sklearn.ensemble import RandomForestRegressor
        self.model = RandomForestRegressor(
            n_estimators=120,
            max_depth=14,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
        
    def save(self, path: str):
        joblib.dump(self.model, path)
        
    def load(self, path: str):
        self.model = joblib.load(path)
        
    def get_feature_importances(self, feature_names: list[str]) -> dict:
        importances = self.model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        
        ranked_features = [
            {
                "rank": i + 1,
                "feature": feature_names[idx],
                "importance": round(float(importances[idx]), 6)
            }
            for i, idx in enumerate(sorted_idx)
        ]
        return {
            "ranked_features": ranked_features,
            "feature_importance_dict": {feature_names[i]: round(float(importances[i]), 6) for i in range(len(feature_names))}
        }
        
    def get_metadata(self) -> dict:
        return {
            "algorithm": "RandomForestRegressor",
            "framework": "Scikit-Learn",
            "sequence_conversion_strategy": "The Random Forest converts the sequence into a single prediction by extracting the final cycle snapshot from the input window."
        }

class ModelFactory:
    @staticmethod
    def get_adapter(model_type: str) -> BaseModelAdapter:
        model_type = model_type.lower()
        if model_type == "random_forest":
            return RandomForestAdapter()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
