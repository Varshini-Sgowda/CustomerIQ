"""
Unit tests for InsightEngine autonomous discovery
"""

import unittest
from customeriq.data_loader import DataLoader
from customeriq.feature_pipeline import FeaturePipeline
from customeriq.ltv_model import LTVModelEngine
from customeriq.insight_engine import InsightEngine


class TestInsightEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = DataLoader()
        cls.df = cls.loader.load_data()
        cls.pipeline = FeaturePipeline()
        cls.features = cls.pipeline.transform(cls.df)
        cls.engine = LTVModelEngine()
        cls.engine.load_model()
        cls.features = cls.engine.predict_features(cls.features)
        cls.insights = InsightEngine(cls.df, cls.features, cls.engine, cls.loader)

    def test_eight_point_insight_structure(self):
        cards = self.insights.generate_all_insights()
        self.assertEqual(len(cards), 8)

        required_keys = [
            'id', 'category', 'title', 'supporting_metric',
            'relevant_population', 'comparison_baseline',
            'detection_method', 'possible_interpretation',
            'recommended_next_step', 'evidence_qualification'
        ]
        for c in cards:
            for k in required_keys:
                self.assertIn(k, c, f"Insight card missing required key: {k}")


if __name__ == '__main__':
    unittest.main()

