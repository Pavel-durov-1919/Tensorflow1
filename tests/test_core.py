"""
Unit tests for core AeroShield components:
- Configuration
- Dataset adapters
- Preprocessing (scaler + window generator)
- Model pipeline (train, predict, persist, reload)
- Evaluation metrics
"""

import os
import sys
import json
import tempfile
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DATASET_TYPE, MODEL_TYPE, RANDOM_STATE, WINDOW_LENGTH
from src.dataset_adapter import DatasetFactory, CMAPSSAdapter, CruiseBenchAdapter
from src.preprocess import Preprocessor, WindowGenerator
from src.model_adapter import ModelFactory, RandomForestAdapter
from sklearn.metrics import mean_absolute_error


class TestConfiguration(unittest.TestCase):
    """Test centralized configuration module."""

    def test_default_dataset_type(self):
        self.assertEqual(DATASET_TYPE, "cmapss")

    def test_default_model_type(self):
        self.assertEqual(MODEL_TYPE, "random_forest")

    def test_default_random_state(self):
        self.assertEqual(RANDOM_STATE, 42)

    def test_default_window_length(self):
        self.assertEqual(WINDOW_LENGTH, 30)


class TestDatasetFactory(unittest.TestCase):
    """Test dataset factory and adapter selection."""

    def test_cmapss_adapter_selection(self):
        adapter = DatasetFactory.get_adapter("cmapss")
        self.assertIsInstance(adapter, CMAPSSAdapter)

    def test_cruisebench_adapter_selection(self):
        adapter = DatasetFactory.get_adapter("cruisebench")
        self.assertIsInstance(adapter, CruiseBenchAdapter)

    def test_case_insensitive(self):
        adapter = DatasetFactory.get_adapter("CMAPSS")
        self.assertIsInstance(adapter, CMAPSSAdapter)

    def test_unknown_dataset_raises(self):
        with self.assertRaises(ValueError):
            DatasetFactory.get_adapter("nonexistent_dataset")


class TestCMAPSSAdapter(unittest.TestCase):
    """Test C-MAPSS adapter data splits and features."""

    @classmethod
    def setUpClass(cls):
        cls.adapter = CMAPSSAdapter()
        cls.train_df, cls.val_df, cls.test_df = cls.adapter.get_data_splits()
        cls.features = cls.adapter.get_feature_names()

    def test_splits_are_dataframes(self):
        self.assertIsInstance(self.train_df, pd.DataFrame)
        self.assertIsInstance(self.val_df, pd.DataFrame)
        self.assertIsInstance(self.test_df, pd.DataFrame)

    def test_splits_not_empty(self):
        self.assertGreater(len(self.train_df), 0)
        self.assertGreater(len(self.val_df), 0)
        self.assertGreater(len(self.test_df), 0)

    def test_feature_count(self):
        self.assertEqual(len(self.features), 16)

    def test_features_present_in_train(self):
        for f in self.features:
            self.assertIn(f, self.train_df.columns)

    def test_rul_column_name(self):
        self.assertEqual(self.adapter.get_rul_column(), "RUL_clipped")

    def test_rul_column_in_train(self):
        self.assertIn("RUL_clipped", self.train_df.columns)

    def test_no_engine_overlap(self):
        """Verify no engine appears in both train and validation splits."""
        train_engines = set(self.train_df["unit_nr"].unique())
        val_engines = set(self.val_df["unit_nr"].unique())
        overlap = train_engines & val_engines
        self.assertEqual(len(overlap), 0, f"Engines {overlap} appear in both train and val")

    def test_metadata_has_checksums(self):
        meta = self.adapter.get_metadata()
        self.assertIn("dataset_checksums", meta)

    def test_no_missing_values_in_features(self):
        for col in self.features:
            self.assertFalse(self.train_df[col].isnull().any(), f"NaN in train {col}")
            self.assertFalse(self.val_df[col].isnull().any(), f"NaN in val {col}")


class TestCruiseBenchAdapter(unittest.TestCase):
    """Test CruiseBench adapter (mock data fallback)."""

    def test_mock_data_generation(self):
        adapter = CruiseBenchAdapter(data_path="nonexistent.h5")
        train, val, test = adapter.get_data_splits()
        self.assertIsInstance(train, pd.DataFrame)
        self.assertGreater(len(train), 0)

    def test_feature_list(self):
        adapter = CruiseBenchAdapter()
        features = adapter.get_feature_names()
        self.assertEqual(len(features), 18)
        self.assertIn("w", features)
        self.assertIn("wf", features)

    def test_rul_column(self):
        adapter = CruiseBenchAdapter()
        self.assertEqual(adapter.get_rul_column(), "rul")


class TestPreprocessor(unittest.TestCase):
    """Test preprocessing scaler - fit on train, transform on val/test."""

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)
        cls.features = ["f1", "f2", "f3"]
        cls.train = pd.DataFrame({
            "f1": np.random.randn(100) * 10 + 50,
            "f2": np.random.randn(100) * 5 + 100,
            "f3": np.random.randn(100) * 2 + 200,
            "target": np.random.rand(100) * 125
        })
        cls.val = pd.DataFrame({
            "f1": np.random.randn(30) * 10 + 50,
            "f2": np.random.randn(30) * 5 + 100,
            "f3": np.random.randn(30) * 2 + 200,
            "target": np.random.rand(30) * 125
        })
        cls.test = pd.DataFrame({
            "f1": np.random.randn(20) * 10 + 50,
            "f2": np.random.randn(20) * 5 + 100,
            "f3": np.random.randn(20) * 2 + 200,
            "target": np.random.rand(20) * 125
        })

    def test_scaler_fit_on_train(self):
        p = Preprocessor()
        t, v, te, sc = p.fit_transform(self.train, self.val, self.test, self.features)
        # Scaled training features should have ~0 mean, ~1 std
        for col in sc:
            mean = abs(t[col].mean())
            self.assertLess(mean, 0.15, f"{col} mean too far from 0: {mean}")

    def test_scaled_columns_created(self):
        p = Preprocessor()
        t, v, te, sc = p.fit_transform(self.train, self.val, self.test, self.features)
        self.assertEqual(len(sc), 3)
        for col in sc:
            self.assertIn(col, t.columns)
            self.assertIn(col, v.columns)
            self.assertIn(col, te.columns)

    def test_original_columns_preserved(self):
        p = Preprocessor()
        t, v, te, sc = p.fit_transform(self.train, self.val, self.test, self.features)
        for f in self.features:
            self.assertIn(f, t.columns)


class TestWindowGenerator(unittest.TestCase):
    """Test sliding window generation."""

    def _make_df(self, n_engines=2, cycles_per=50):
        rows = []
        for e in range(1, n_engines + 1):
            for c in range(1, cycles_per + 1):
                rows.append({
                    "unit_nr": e, "cycle": c,
                    "f1": np.random.randn(), "f2": np.random.randn(),
                    "RUL": max(0, cycles_per - c)
                })
        return pd.DataFrame(rows)

    def test_window_shape(self):
        df = self._make_df(n_engines=1, cycles_per=50)
        X, y = WindowGenerator.create_windows(df, ["f1", "f2"], "RUL", sequence_length=10)
        self.assertEqual(X.shape[1], 10)  # window length
        self.assertEqual(X.shape[2], 2)   # features
        self.assertEqual(len(X), len(y))

    def test_no_cross_engine_windows(self):
        df = self._make_df(n_engines=3, cycles_per=20)
        X, y = WindowGenerator.create_windows(df, ["f1", "f2"], "RUL", sequence_length=10)
        # Each engine produces (20 - 10 + 1) = 11 windows
        self.assertEqual(len(X), 3 * 11)

    def test_short_engine_skipped(self):
        df = self._make_df(n_engines=1, cycles_per=5)
        X, y = WindowGenerator.create_windows(df, ["f1", "f2"], "RUL", sequence_length=10)
        self.assertEqual(len(X), 0)

    def test_target_alignment(self):
        """Target must correspond to the last cycle in each window."""
        df = self._make_df(n_engines=1, cycles_per=50)
        X, y = WindowGenerator.create_windows(df, ["f1", "f2"], "RUL", sequence_length=10)
        # First window covers cycles 1-10, target is RUL at cycle 10 = 50-10 = 40
        self.assertEqual(y[0], 40)
        # Last window covers cycles 41-50, target is RUL at cycle 50 = 0
        self.assertEqual(y[-1], 0)


class TestModelAdapter(unittest.TestCase):
    """Test model factory and RandomForest adapter."""

    def test_factory_random_forest(self):
        adapter = ModelFactory.get_adapter("random_forest")
        self.assertIsInstance(adapter, RandomForestAdapter)

    def test_factory_unknown_raises(self):
        with self.assertRaises(ValueError):
            ModelFactory.get_adapter("nonexistent_model")

    def test_train_predict_shape(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.rand(100) * 125
        adapter = RandomForestAdapter(random_state=42)
        adapter.train(X, y)
        preds = adapter.predict(X[:10])
        self.assertEqual(preds.shape, (10,))

    def test_feature_importances(self):
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.rand(100)
        adapter = RandomForestAdapter()
        adapter.train(X, y)
        imp = adapter.get_feature_importances(["a", "b", "c"])
        self.assertIn("ranked_features", imp)
        self.assertEqual(len(imp["ranked_features"]), 3)

    def test_save_load_roundtrip(self):
        np.random.seed(42)
        X = np.random.randn(50, 4)
        y = np.random.rand(50) * 100
        adapter = RandomForestAdapter(random_state=42)
        adapter.train(X, y)
        pred_before = adapter.predict(X[:5])

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            path = f.name
        try:
            adapter.save(path)
            adapter2 = RandomForestAdapter()
            adapter2.load(path)
            pred_after = adapter2.predict(X[:5])
            np.testing.assert_array_almost_equal(pred_before, pred_after)
        finally:
            os.unlink(path)

    def test_metadata(self):
        adapter = RandomForestAdapter()
        meta = adapter.get_metadata()
        self.assertEqual(meta["algorithm"], "RandomForestRegressor")
        self.assertEqual(meta["framework"], "Scikit-Learn")


class TestEvaluationMetrics(unittest.TestCase):
    """Test metric calculations."""

    def test_mae_perfect(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([10, 20, 30])
        self.assertAlmostEqual(mean_absolute_error(y_true, y_pred), 0.0)

    def test_mae_known_error(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([15, 25, 35])  # each off by 5
        self.assertAlmostEqual(mean_absolute_error(y_true, y_pred), 5.0)

    def test_rmse_known(self):
        from sklearn.metrics import root_mean_squared_error
        y_true = np.array([10.0, 20.0, 30.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        self.assertAlmostEqual(root_mean_squared_error(y_true, y_pred), 0.0)


if __name__ == "__main__":
    unittest.main()
