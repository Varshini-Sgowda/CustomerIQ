"""
Unit tests for ModelHealthMonitor and PSI calculation
"""

import unittest
import numpy as np
from customeriq.data_loader import DataLoader
from customeriq.feature_pipeline import FeaturePipeline
from customeriq.ltv_model import LTVModelEngine
from customeriq.drift_monitor import ModelHealthMonitor


class TestDriftMonitor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = DataLoader()
        cls.df = cls.loader.load_data()
        cls.pipeline = FeaturePipeline()
        cls.features = cls.pipeline.transform(cls.df)
        cls.engine = LTVModelEngine()
        cls.engine.load_model()
        cls.features = cls.engine.predict_features(cls.features)
        cls.monitor = ModelHealthMonitor(cls.features, cls.features, cls.engine)

    def test_psi_identical_distributions(self):
        arr = np.random.normal(100, 15, 1000)
        psi = self.monitor._calculate_psi(arr, arr)
        self.assertAlmostEqual(psi, 0.0, places=2)

    def test_psi_shifted_distributions(self):
        base = np.random.normal(100, 15, 1000)
        shifted = np.random.normal(140, 15, 1000)
        psi = self.monitor._calculate_psi(base, shifted)
        self.assertGreater(psi, 0.25, "Shifted distribution should exceed significant drift threshold (PSI > 0.25)")

    def test_health_audit_structure(self):
        audit = self.monitor.compute_health_audit()
        self.assertIn('overall_status', audit)
        self.assertIn('missingness_audit', audit)
        self.assertIn('feature_drift_psi', audit)
        self.assertIn('prediction_drift', audit)
        self.assertIn('segment_slices', audit)
        self.assertIn('drift_explanation', audit)


if __name__ == '__main__':
    unittest.main()

