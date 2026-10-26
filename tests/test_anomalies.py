"""
Unit tests for AnomalyDetector early-warning and Isolation Forest detection
"""

import unittest
from customeriq.data_loader import DataLoader
from customeriq.feature_pipeline import FeaturePipeline
from customeriq.ltv_model import LTVModelEngine
from customeriq.anomaly_detector import AnomalyDetector


class TestAnomalyDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = DataLoader()
        cls.df = cls.loader.load_data()
        cls.pipeline = FeaturePipeline()
        cls.features = cls.pipeline.transform(cls.df)
        cls.engine = LTVModelEngine()
        cls.engine.load_model()
        cls.features = cls.engine.predict_features(cls.features)
        cls.detector = AnomalyDetector(cls.features, min_history_orders=2)

    def test_scan_produces_alerts(self):
        res = self.detector.scan_all()
        self.assertGreater(res['total_alerts'], 0)
        self.assertGreater(res['critical_count'], 0)
        self.assertGreater(res['warning_count'], 0)

        # Check alert structure
        first_alert = res['alerts'][0]
        self.assertIn('severity', first_alert)
        self.assertIn('customer_id', first_alert)
        self.assertIn('evidence', first_alert)
        self.assertIn('comparison_period', first_alert)
        self.assertIn('detection_method', first_alert)
        self.assertIn('recommended_investigation', first_alert)
        self.assertIn('action_url', first_alert)

    def test_isolation_forest_execution(self):
        res = self.detector.scan_all(contamination=0.03)
        iso_alerts = [a for a in res['alerts'] if 'Isolation Forest' in a['detection_method']]
        self.assertGreater(len(iso_alerts), 0)


if __name__ == '__main__':
    unittest.main()

