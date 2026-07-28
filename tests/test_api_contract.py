"""
Unit & Security Tests for RUL Predictor API Contract
Author: Member A (Data & ML Lead)
"""

import sys
import os
import json
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import RULPredictor


class TestRULPredictorAPIContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictor = RULPredictor()
        cls.valid_feature_order = cls.predictor.feature_order
        cls.valid_30_seq = [
            {f: 0.5 for f in cls.valid_feature_order}
            for _ in range(30)
        ]
        payloads_path = os.path.join(cls.predictor.model_dir, "sample_payloads.json")
        with open(payloads_path, "r") as f:
            cls.sample_payloads = json.load(f)

    def test_valid_sequence_prediction(self):
        """Test prediction with valid (30, 16) sequence."""
        res = self.predictor.predict_rul(self.valid_30_seq)
        self.assertEqual(res["version"], "1.3.0")
        self.assertEqual(res["window_length"], 30)
        self.assertIn("estimated_rul", res)
        self.assertIn("lower_bound", res)
        self.assertIn("upper_bound", res)
        self.assertIn("risk_level", res)
        self.assertIn("recommendation", res)
        self.assertTrue(0.0 <= res["estimated_rul"] <= 125.0)
        self.assertTrue(0.0 <= res["lower_bound"] <= 125.0)
        self.assertTrue(0.0 <= res["upper_bound"] <= 125.0)
        self.assertIn("nominal behavior detected within the model's validated dataset range", res["recommendation"].lower())

    def test_critical_risk_recommendation_text(self):
        """Test that CRITICAL risk level requires immediate qualified engineering inspection."""
        risk, rec = self.predictor.determine_risk_and_recommendation(10.0)
        self.assertEqual(risk, "CRITICAL")
        self.assertIn("immediate qualified engineering inspection", rec)

    def test_window_length_validation(self):
        """Test that non-30 window lengths raise ValueError."""
        short_seq = [{f: 0.5 for f in self.valid_feature_order} for _ in range(10)]
        with self.assertRaises(ValueError) as ctx:
            self.predictor.predict_rul(short_seq)
        self.assertIn("Invalid sequence length", str(ctx.exception))

        long_seq = [{f: 0.5 for f in self.valid_feature_order} for _ in range(35)]
        with self.assertRaises(ValueError) as ctx:
            self.predictor.predict_rul(long_seq)
        self.assertIn("Invalid sequence length", str(ctx.exception))

    def test_single_snapshot_rejection(self):
        """Test that single snapshot dict raises TypeError."""
        single_dict = {f: 0.5 for f in self.valid_feature_order}
        with self.assertRaises(TypeError) as ctx:
            self.predictor.predict_rul(single_dict)
        self.assertIn("API contract requires a 30-cycle sequence", str(ctx.exception))

    def test_missing_fields_validation(self):
        """Test that missing required feature fields raise ValueError."""
        incomplete_seq = []
        for _ in range(30):
            item = {f: 0.5 for f in self.valid_feature_order}
            del item["sensor_2"]
            incomplete_seq.append(item)

        with self.assertRaises(ValueError) as ctx:
            self.predictor.predict_rul(incomplete_seq)
        self.assertIn("Missing required feature fields", str(ctx.exception))

    def test_extra_fields_handling(self):
        """Test that extra unneeded fields are safely stripped without error."""
        extra_fields_seq = []
        for _ in range(30):
            item = {f: 0.5 for f in self.valid_feature_order}
            item["extra_untrusted_field"] = "malicious_string"
            item["extra_sensor_99"] = 999.9
            extra_fields_seq.append(item)

        res = self.predictor.predict_rul(extra_fields_seq)
        self.assertIsInstance(res["data_quality_score"], float)
        self.assertTrue(0.0 <= res["data_quality_score"] <= 1.0)
        self.assertIn("estimated_rul", res)

    def test_non_finite_values_rejection(self):
        """Test that NaN and Inf values raise ValueError."""
        nan_seq = [{f: 0.5 for f in self.valid_feature_order} for _ in range(30)]
        nan_seq[15]["sensor_3"] = np.nan

        with self.assertRaises(ValueError) as ctx:
            self.predictor.predict_rul(nan_seq)
        self.assertIn("non-finite values", str(ctx.exception))

        inf_seq = [{f: 0.5 for f in self.valid_feature_order} for _ in range(30)]
        inf_seq[20]["sensor_4"] = np.inf

        with self.assertRaises(ValueError) as ctx:
            self.predictor.predict_rul(inf_seq)
        self.assertIn("non-finite values", str(ctx.exception))

    def test_deterministic_prediction(self):
        """Test that identical 30-cycle inputs yield identical predictions."""
        res1 = self.predictor.predict_rul(self.valid_30_seq)
        res2 = self.predictor.predict_rul(self.valid_30_seq)
        self.assertEqual(res1["estimated_rul"], res2["estimated_rul"])
        self.assertEqual(res1["lower_bound"], res2["lower_bound"])
        self.assertEqual(res1["upper_bound"], res2["upper_bound"])
        self.assertEqual(res1["risk_level"], res2["risk_level"])

    def test_exact_engine_54_deterministic_prediction(self):
        """Test exact deterministic prediction output for frozen Engine 54 payload."""
        seq_54 = self.sample_payloads["engine_54_successful"]["sequence_30_cycle_payload"]
        res = self.predictor.predict_rul(seq_54)
        self.assertEqual(res["estimated_rul"], 1.82)
        self.assertEqual(res["lower_bound"], 0.0)
        self.assertEqual(res["upper_bound"], 25.90)
        self.assertEqual(res["risk_level"], "CRITICAL")

    def test_exact_engine_74_deterministic_prediction(self):
        """Test exact deterministic prediction output for frozen Engine 74 payload."""
        seq_74 = self.sample_payloads["engine_74_difficult"]["sequence_30_cycle_payload"]
        res = self.predictor.predict_rul(seq_74)
        self.assertEqual(res["estimated_rul"], 3.39)
        self.assertEqual(res["lower_bound"], 0.0)
        self.assertEqual(res["upper_bound"], 27.47)
        self.assertEqual(res["risk_level"], "CRITICAL")

    def test_payload_schema_keys(self):
        """Test presence of all required response payload keys."""
        res = self.predictor.predict_rul(self.valid_30_seq)
        expected_keys = [
            "model_name", "version", "feature_order", "window_length",
            "sequence_conversion_strategy", "model_limitation", "estimated_rul",
            "lower_bound", "upper_bound", "risk_level", "data_quality_score", "recommendation"
        ]
        for k in expected_keys:
            self.assertIn(k, res)

    def test_risk_level_thresholds(self):
        """Test risk level thresholds: CRITICAL <= 15, HIGH <= 30, MEDIUM <= 60, LOW > 60."""
        self.assertEqual(self.predictor.determine_risk_and_recommendation(10.0)[0], "CRITICAL")
        self.assertEqual(self.predictor.determine_risk_and_recommendation(25.0)[0], "HIGH")
        self.assertEqual(self.predictor.determine_risk_and_recommendation(45.0)[0], "MEDIUM")
        self.assertEqual(self.predictor.determine_risk_and_recommendation(80.0)[0], "LOW")


if __name__ == "__main__":
    unittest.main()
