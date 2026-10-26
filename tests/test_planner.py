"""
Unit tests for MarketingActionPlanner budget optimization and uplift modeling
"""

import unittest
from customeriq.data_loader import DataLoader
from customeriq.feature_pipeline import FeaturePipeline
from customeriq.ltv_model import LTVModelEngine
from customeriq.action_planner import MarketingActionPlanner


class TestActionPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = DataLoader()
        cls.df = cls.loader.load_data()
        cls.pipeline = FeaturePipeline()
        cls.features = cls.pipeline.transform(cls.df)
        cls.engine = LTVModelEngine()
        cls.engine.load_model()
        cls.features = cls.engine.predict_features(cls.features)
        cls.planner = MarketingActionPlanner(cls.features)

    def test_budget_knapsack_allocation(self):
        budget = 4000.0
        plan = self.planner.plan_campaign(budget=budget, priority_goal='PREVENT_CHURN_HIGH_LTV')

        self.assertEqual(plan['status'], 'SUCCESS')
        self.assertLessEqual(plan['summary']['budget_utilized'], budget)
        self.assertGreater(plan['summary']['targeted_customers_count'], 0)
        self.assertGreater(plan['summary']['estimated_incremental_revenue'], 0)
        self.assertIn('comparison_with_baseline', plan)
        self.assertIn('uplift_disclaimer', plan)

    def test_all_goals_executable(self):
        goals = ['PREVENT_CHURN_HIGH_LTV', 'MAXIMIZE_EXPECTED_ROI', 'WINBACK_DORMANT', 'NURTURE_POTENTIAL_LOYALISTS']
        for g in goals:
            plan = self.planner.plan_campaign(budget=2500.0, priority_goal=g)
            self.assertEqual(plan['status'], 'SUCCESS', f"Goal {g} failed")


if __name__ == '__main__':
    unittest.main()

