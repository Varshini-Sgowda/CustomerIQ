"""
Unit tests for DataLoader and FeaturePipeline
"""

import unittest
import pandas as pd
from customeriq.data_loader import DataLoader
from customeriq.feature_pipeline import FeaturePipeline, FEATURE_COLUMNS


class TestFeaturePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = DataLoader()
        cls.df = cls.loader.load_data()
        cls.pipeline = FeaturePipeline()
        cls.features = cls.pipeline.transform(cls.df)

    def test_data_loaded_correctly(self):
        self.assertFalse(self.df.empty)
        self.assertEqual(len(self.df), 9994)
        self.assertEqual(self.loader.metadata['unique_customers'], 793)

    def test_feature_columns_present(self):
        for col in FEATURE_COLUMNS:
            self.assertIn(col, self.features.columns, f"Missing feature column: {col}")

    def test_rfm_segments_assigned(self):
        self.assertIn('customer_segment', self.features.columns)
        self.assertIn('r_score', self.features.columns)
        self.assertIn('f_score', self.features.columns)
        self.assertIn('m_score', self.features.columns)

        unique_segs = self.features['customer_segment'].unique()
        self.assertTrue(len(unique_segs) >= 5)
        self.assertTrue((self.features['r_score'].between(1, 5)).all())

    def test_no_null_customer_features(self):
        null_counts = self.features[FEATURE_COLUMNS].isnull().sum().sum()
        self.assertEqual(null_counts, 0, "Feature table should have zero nulls.")


if __name__ == '__main__':
    unittest.main()

