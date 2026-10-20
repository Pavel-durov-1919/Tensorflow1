"""
Integration tests for AeroShield end-to-end pipeline:
- Dataset loading -> preprocessing -> training -> persistence -> inference -> API response
"""

import os
import sys
import json
import tempfile
import unittest
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DATASET_TYPE
from src.dataset_adapter import DatasetFactory
from src.preprocess import Preprocessor
from src.model_adapter import ModelFactory
from src.predict import RULPredictor
from src.data_loader import REPO_ROOT


class TestDataPipelineIntegration(unittest.TestCase):
    """Test dataset loading through preprocessing to training."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = DatasetFactory.get_adapter("cmapss")
        cls.train_df, cls.val_df, cls.test_df = cls.adapter.get_data_splits()
        cls.features = cls.adapter.get_feature_names()
        cls.target_col = cls.adapter.get_rul_column()

    def test_pipeline_dataset_to_preprocessing(self):
        """Dataset splits flow into Preprocessor without errors."""
        p = Preprocessor()
        train, val, test, scaled_cols = p.fit_transform(
            self.train_df, self.val_df, self.test_df, self.features
        )
        self.assertEqual(len(scaled_cols), len(self.features))
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)

    def test_pipeline_preprocessing_to_training(self):
        """Preprocessed data trains a model successfully."""
        X_train = self.train_df[self.features].values
        y_train = self.train_df[self.target_col].values

        model = ModelFactory.get_adapter("random_forest")
        model.train(X_train, y_train)
        preds = model.predict(X_train[:10])
        self.assertEqual(preds.shape, (10,))
        # Predictions should be non-negative
        self.assertTrue(np.all(preds >= 0))

    def test_pipeline_training_to_persistence_to_inference(self):
        """Model trains, saves, reloads, and produces identical predictions."""
        X_train = self.train_df[self.features].values[:500]
        y_train = self.train_df[self.target_col].values[:500]

        model = ModelFactory.get_adapter("random_forest")
        model.train(X_train, y_train)
        pred_before = model.predict(X_train[:5])

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        try:
            model.save(path)
            model2 = ModelFactory.get_adapter("random_forest")
            model2.load(path)
            pred_after = model2.predict(X_train[:5])
            np.testing.assert_array_almost_equal(pred_before, pred_after)
        finally:
            os.unlink(path)

    def test_rul_values_non_negative(self):
        """All RUL values in training data are non-negative."""
        self.assertTrue((self.train_df[self.target_col] >= 0).all())

    def test_rul_values_clipped(self):
        """RUL values in training data are clipped at 125."""
        self.assertTrue((self.train_df[self.target_col] <= 125).all())


class TestInferencePipelineIntegration(unittest.TestCase):
    """Test the production inference pipeline using saved artifacts."""

    @classmethod
    def setUpClass(cls):
        cls.predictor = RULPredictor()
        payloads_path = os.path.join(cls.predictor.model_dir, "sample_payloads.json")
        with open(payloads_path) as f:
            cls.payloads = json.load(f)

    def test_end_to_end_prediction(self):
        """Full inference: load model -> validate input -> scale -> predict -> format response."""
        seq = self.payloads["engine_54_successful"]["sequence_30_cycle_payload"]
        result = self.predictor.predict_rul(seq)

        self.assertIn("estimated_rul", result)
        self.assertTrue(0 <= result["estimated_rul"] <= 125)
        self.assertIn("risk_level", result)
        self.assertIn("data_quality_score", result)
        self.assertTrue(0 <= result["data_quality_score"] <= 1)

    def test_scaler_consistency(self):
        """Scaler loaded from disk matches the feature order."""
        scaler = self.predictor.scaler
        self.assertEqual(scaler.n_features_in_, len(self.predictor.feature_order))

    def test_model_and_scaler_from_same_training(self):
        """Model and scaler artifact feature counts match."""
        model = self.predictor.model
        scaler = self.predictor.scaler
        self.assertEqual(model.n_features_in_, scaler.n_features_in_)

    def test_metadata_consistency(self):
        """Metadata feature order matches loaded feature_order.json."""
        meta_features = self.predictor.metadata["feature_order"]
        self.assertEqual(meta_features, self.predictor.feature_order)


class TestAPIIntegration(unittest.TestCase):
    """Test FastAPI app integration with the prediction pipeline."""

    @classmethod
    def setUpClass(cls):
        backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)
        from fastapi.testclient import TestClient
        from app.main import app
        cls.client = TestClient(app)

        payloads_path = os.path.join(REPO_ROOT, "models", "sample_payloads.json")
        with open(payloads_path) as f:
            cls.payloads = json.load(f)

        from app.config import FEATURE_ORDER
        cls.feature_order = FEATURE_ORDER

    def test_health_to_predict_flow(self):
        """Health check passes, then prediction succeeds."""
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["model_loaded"])

        case = self.payloads["engine_54_successful"]
        window = [[row[f] for f in self.feature_order]
                   for row in case["sequence_30_cycle_payload"]]
        payload = {
            "engine_id": case["unit_nr"],
            "cycle": case["total_cycles"],
            "sensor_window": window
        }
        resp = self.client.post("/predict/rul", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(0 <= data["estimated_rul"] <= 125)

    def test_model_info_reflects_artifacts(self):
        """Model-info endpoint returns values consistent with metadata.json."""
        resp = self.client.get("/model-info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["expected_window_size"], 30)
        self.assertEqual(data["expected_num_features"], 16)
        self.assertIn("MAE", data["metrics"])
        self.assertIn("RMSE", data["metrics"])


if __name__ == "__main__":
    unittest.main()
