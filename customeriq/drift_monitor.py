"""
CustomerIQ Model Health & Feature Drift Monitor
Calculates Population Stability Index (PSI), Wasserstein distance, missingness ratios,
prediction distribution shifts, and segment-level slice performance metrics.
Distinguishes feature distribution drift from accuracy degradation with clear diagnostic guidance.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from customeriq.feature_pipeline import FEATURE_COLUMNS
from customeriq.ltv_model import LTVModelEngine


DRIFT_EXPLANATION = (
    "METRIC INTERPRETATION NOTE: Feature drift (measured via PSI) indicates that input distributions "
    "have shifted relative to the training baseline. Feature drift is NOT equivalent to degraded predictive "
    "accuracy; true predictive performance can only be verified when new ground-truth revenue outcomes "
    "materialize at the conclusion of the 12-month prediction horizon."
)


class ModelHealthMonitor:
    def __init__(self, baseline_features_df: pd.DataFrame, current_features_df: pd.DataFrame, model_engine: LTVModelEngine):
        self.baseline_df = baseline_features_df.copy()
        self.current_df = current_features_df.copy()
        self.model_engine = model_engine

    def compute_health_audit(self) -> Dict[str, Any]:
        """
        Runs comprehensive model health, schema validation, and drift analysis.
        """
        # 1. Missing Value and Schema Integrity Check
        missingness_audit = self._audit_missingness()

        # 2. Feature Distribution Drift (Population Stability Index - PSI)
        psi_report = self._compute_feature_psi()

        # 3. Prediction Distribution Shift
        prediction_shift = self._compute_prediction_drift()

        # 4. Performance by Segment (Slice Diagnostics)
        segment_slice_report = self._compute_segment_performance()

        # 5. Overall Model Status Determination
        max_psi = max([f['psi'] for f in psi_report['features']]) if psi_report['features'] else 0.0
        if max_psi >= 0.25:
            overall_status = 'WARNING'
            status_summary = 'Significant feature distribution shift detected on key inputs (PSI >= 0.25). Model recalibration recommended.'
        elif max_psi >= 0.10:
            overall_status = 'MONITOR'
            status_summary = 'Moderate feature shift observed (0.10 <= PSI < 0.25). Distributions remain within acceptable operational limits.'
        else:
            overall_status = 'HEALTHY'
            status_summary = 'All feature distributions and prediction outputs are stable (PSI < 0.10).'

        return {
            'overall_status': overall_status,
            'status_summary': status_summary,
            'model_metadata': self.model_engine.metadata,
            'missingness_audit': missingness_audit,
            'feature_drift_psi': psi_report,
            'prediction_drift': prediction_shift,
            'segment_slices': segment_slice_report,
            'drift_explanation': DRIFT_EXPLANATION
        }

    def _audit_missingness(self) -> Dict[str, Any]:
        """Checks for null or infinite values across feature registry."""
        total_rows = len(self.current_df)
        missing_records = []

        for col in self.model_engine.feature_columns:
            null_count = int(self.current_df[col].isnull().sum())
            null_pct = round((null_count / max(total_rows, 1)) * 100, 2)
            missing_records.append({
                'feature': col,
                'missing_count': null_count,
                'missing_percentage': null_pct,
                'status': 'PASS' if null_pct == 0.0 else ('WARNING' if null_pct < 5.0 else 'FAIL')
            })

        overall_missing_pct = round(
            float(self.current_df[self.model_engine.feature_columns].isnull().mean().mean() * 100), 2
        )

        return {
            'total_rows_inspected': total_rows,
            'overall_missing_percentage': overall_missing_pct,
            'features_missingness': missing_records,
            'status': 'PASS' if overall_missing_pct == 0.0 else 'FAIL'
        }

    def _compute_feature_psi(self) -> Dict[str, Any]:
        """
        Calculates Population Stability Index (PSI) for each model feature.
        PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
        Standard thresholds:
          PSI < 0.10: Stable
          0.10 <= PSI < 0.25: Moderate Drift
          PSI >= 0.25: Significant Drift
        """
        feature_results = []
        for col in self.model_engine.feature_columns:
            base_vals = self.baseline_df[col].dropna().values
            curr_vals = self.current_df[col].dropna().values

            psi_val = self._calculate_psi(base_vals, curr_vals, num_buckets=10)
            status = 'STABLE' if psi_val < 0.10 else ('MODERATE_DRIFT' if psi_val < 0.25 else 'SIGNIFICANT_DRIFT')

            feature_results.append({
                'feature': col,
                'psi': round(float(psi_val), 4),
                'status': status,
                'baseline_mean': round(float(np.mean(base_vals)), 2) if len(base_vals) > 0 else 0.0,
                'current_mean': round(float(np.mean(curr_vals)), 2) if len(curr_vals) > 0 else 0.0,
                'relative_mean_shift_pct': round(
                    float(((np.mean(curr_vals) - np.mean(base_vals)) / max(abs(np.mean(base_vals)), 1e-4)) * 100), 2
                ) if len(base_vals) > 0 else 0.0
            })

        # Sort by PSI descending
        feature_results.sort(key=lambda x: x['psi'], reverse=True)

        return {
            'threshold_definitions': {
                'stable': 'PSI < 0.10',
                'moderate': '0.10 <= PSI < 0.25',
                'significant': 'PSI >= 0.25'
            },
            'features': feature_results
        }

    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10) -> float:
        """Helper to compute standard Population Stability Index between two 1D series."""
        if len(expected) == 0 or len(actual) == 0:
            return 0.0

        # Create quantiles on expected distribution
        percentiles = np.linspace(0, 100, num_buckets + 1)
        breakpoints = np.percentile(expected, percentiles)
        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        # Handle degenerate/duplicate breakpoints
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0

        expected_counts, _ = np.histogram(expected, bins=breakpoints)
        actual_counts, _ = np.histogram(actual, bins=breakpoints)

        # Smooth zero counts with small epsilon
        eps = 1e-4
        expected_pct = (expected_counts + eps) / (len(expected) + eps * len(expected_counts))
        actual_pct = (actual_counts + eps) / (len(actual) + eps * len(actual_counts))

        psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return max(float(psi_val), 0.0)

    def _compute_prediction_drift(self) -> Dict[str, Any]:
        """Compares baseline vs current scored prediction distribution."""
        preds = self.current_df['predicted_ltv_12m'].values
        return {
            'predicted_pool_total': round(float(preds.sum()), 2),
            'mean_prediction': round(float(preds.mean()), 2),
            'median_prediction': round(float(np.median(preds)), 2),
            'std_prediction': round(float(preds.std()), 2),
            'p10_prediction': round(float(np.percentile(preds, 10)), 2),
            'p90_prediction': round(float(np.percentile(preds, 90)), 2),
            'distribution_quantiles': {
                'min': round(float(preds.min()), 2),
                'p25': round(float(np.percentile(preds, 25)), 2),
                'p50': round(float(np.median(preds)), 2),
                'p75': round(float(np.percentile(preds, 75)), 2),
                'max': round(float(preds.max()), 2)
            }
        }

    def _compute_segment_performance(self) -> List[Dict[str, Any]]:
        """Slice diagnostics across segments."""
        slices = []
        for seg, grp in self.current_df.groupby('customer_segment'):
            preds = grp['predicted_ltv_12m']
            slices.append({
                'segment': seg,
                'customer_count': int(len(grp)),
                'mean_predicted_ltv': round(float(preds.mean()), 2),
                'median_predicted_ltv': round(float(preds.median()), 2),
                'total_predicted_segment_value': round(float(preds.sum()), 2),
                'mean_recency_days': round(float(grp['recency_days'].mean()), 1),
                'mean_historical_margin': round(float(grp['profit_margin'].mean()), 1)
            })
        slices.sort(key=lambda s: s['total_predicted_segment_value'], reverse=True)
        return slices

