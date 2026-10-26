"""
Unit tests for CustomerFutureLab Digital Twin simulation and constraint enforcement
"""

import unittest
from customeriq.data_loader import DataLoader
from customeriq.feature_pipeline import FeaturePipeline
from customeriq.ltv_model import LTVModelEngine
from customeriq.future_lab import CustomerFutureLab


class TestFutureLab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = DataLoader()
        cls.df = cls.loader.load_data()
        cls.pipeline = FeaturePipeline()
        cls.features = cls.pipeline.transform(cls.df)
        cls.engine = LTVModelEngine()
        cls.engine.load_model()
        cls.features = cls.engine.predict_features(cls.features)
        cls.lab = CustomerFutureLab(cls.features, cls.engine)
        cls.test_cust_id = cls.features.iloc[0]['customer_id']

    def test_valid_simulation(self):
        sim = self.lab.simulate(self.test_cust_id, {
            'recency_days': 15,
            'avg_order_value': 800.0,
            'frequency': 12
        })
        self.assertEqual(sim['status'], 'SUCCESS')
        self.assertIn('predicted_ltv', sim['baseline'])
        self.assertIn('predicted_ltv', sim['simulation'])
        self.assertIn('delta_ltv', sim['variance'])
        self.assertIn('causal_disclaimer', sim)

    def test_negative_recency_rejected(self):
        sim = self.lab.simulate(self.test_cust_id, {'recency_days': -5})
        self.assertEqual(sim['status'], 'INVALID_INPUT')
        self.assertTrue(any('negative' in err.lower() for err in sim['errors']))

    def test_recency_exceeding_tenure_rejected(self):
        tenure = self.features.iloc[0]['tenure_days']
        sim = self.lab.simulate(self.test_cust_id, {'recency_days': tenure + 100})
        self.assertEqual(sim['status'], 'INVALID_INPUT')
        self.assertTrue(any('cannot exceed customer tenure' in err.lower() for err in sim['errors']))

    def test_dependent_feature_recalculation(self):
        sim = self.lab.simulate(self.test_cust_id, {
            'frequency': 8,
            'avg_order_value': 250.0
        })
        comp_dict = {f['feature']: f['simulated'] for f in sim['feature_comparison']}
        self.assertEqual(comp_dict['total_monetary'], 2000.0)


if __name__ == '__main__':
    unittest.main()

