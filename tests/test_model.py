"""
Unit tests for LTVModelEngine and SHAP explainability
"""

import unittest
from customeriq.data_loader import DataLoader
from customeriq.feature_pipeline import FeaturePipeline
from customeriq.ltv_model import LTVModelEngine


class TestLTVModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = DataLoader()
        cls.df = cls.loader.load_data()
        cls.pipeline = FeaturePipeline()
        cls.features = cls.pipeline.transform(cls.df)
        cls.engine = LTVModelEngine()
        if not cls.engine.load_model():
            cls.engine.train(cls.loader)

    def test_model_is_trained(self):
        self.assertTrue(self.engine.is_trained)
        self.assertIsNotNone(self.engine.model)
        self.assertIsNotNone(self.engine.explainer)

    def test_batch_prediction_and_intervals(self):
        preds_df = self.engine.predict_features(self.features)
        self.assertIn('predicted_ltv_12m', preds_df.columns)
        self.assertIn('ltv_lower_bound', preds_df.columns)
        self.assertIn('ltv_upper_bound', preds_df.columns)

        # Lower bound should not exceed upper bound
        self.assertTrue((preds_df['ltv_lower_bound'] <= preds_df['ltv_upper_bound']).all())
        # Predictions non-negative
        self.assertTrue((preds_df['predicted_ltv_12m'] >= 0.0).all())

    def test_single_customer_shap_explanation(self):
        sample_cust = self.features.iloc[0]
        explanation = self.engine.explain_customer(sample_cust)

        self.assertIn('predicted_ltv', explanation)
        self.assertIn('base_value', explanation)
        self.assertIn('all_contributions', explanation)
        self.assertIn('top_positive_drivers', explanation)
        self.assertIn('top_negative_drivers', explanation)
        self.assertGreater(len(explanation['all_contributions']), 0)


if __name__ == '__main__':
    unittest.main()

