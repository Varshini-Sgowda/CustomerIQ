"""
Unit tests for CopilotEngine deterministic natural language query routing
"""

import unittest
from customeriq.data_loader import DataLoader
from customeriq.feature_pipeline import FeaturePipeline
from customeriq.ltv_model import LTVModelEngine
from customeriq.copilot import CopilotEngine


class TestCopilotEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = DataLoader()
        cls.df = cls.loader.load_data()
        cls.pipeline = FeaturePipeline()
        cls.features = cls.pipeline.transform(cls.df)
        cls.engine = LTVModelEngine()
        cls.engine.load_model()
        cls.copilot = CopilotEngine(cls.df, cls.features, cls.engine)

    def test_highest_predicted_ltv_query(self):
        res = self.copilot.ask("Which customers have the highest predicted future LTV?")
        self.assertEqual(res['status'], 'SUCCESS')
        self.assertGreater(len(res['data_table']), 0)
        self.assertIsNotNone(res['chart'])
        self.assertIn('filters_applied', res['calculation_details'])

    def test_high_value_inactive_query(self):
        res = self.copilot.ask("Which high-value customers have not purchased recently?")
        self.assertEqual(res['status'], 'SUCCESS')
        self.assertGreater(len(res['data_table']), 0)

    def test_segment_revenue_query(self):
        res = self.copilot.ask("Which customer segments contribute the most historical revenue?")
        self.assertEqual(res['status'], 'SUCCESS')
        self.assertEqual(res['chart']['type'], 'pie')

    def test_ltv_factors_query(self):
        res = self.copilot.ask("What factors influence the LTV predictions?")
        self.assertEqual(res['status'], 'SUCCESS')
        self.assertGreater(len(res['data_table']), 0)

    def test_declining_frequency_query(self):
        res = self.copilot.ask("Which customer groups have declining purchase frequency?")
        self.assertEqual(res['status'], 'SUCCESS')

    def test_aov_by_segment_query(self):
        res = self.copilot.ask("How does average order value vary across customer segments?")
        self.assertEqual(res['status'], 'SUCCESS')

    def test_prediction_horizon_query(self):
        res = self.copilot.ask("What changes when the prediction horizon or business assumptions change?")
        self.assertEqual(res['status'], 'SUCCESS')
        self.assertEqual(res['chart']['type'], 'line')

    def test_unsupported_demographic_query_rejected(self):
        res = self.copilot.ask("What is the average age and gender of high spenders?")
        self.assertEqual(res['status'], 'UNSUPPORTED')
        self.assertIn("cannot be answered", res['explanation'].lower())


if __name__ == '__main__':
    unittest.main()

