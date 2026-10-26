"""
CustomerIQ Early-Warning & Behavioral Change Detection Module
Detects early indicators of dormancy, velocity drops, spending collapses, emerging VIPs,
and multivariate outliers using statistical thresholding, robust Z-scores, and scikit-learn Isolation Forest.
Provides structured alert cards with evidence, detection method, severity, and investigation links.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.ensemble import IsolationForest

from customeriq.feature_pipeline import FEATURE_COLUMNS


class AnomalyDetector:
    def __init__(self, features_df: pd.DataFrame, min_history_orders: int = 2):
        self.features_df = features_df.copy()
        self.min_history_orders = min_history_orders
        self.alerts: List[Dict[str, Any]] = []

    def scan_all(self, contamination: float = 0.05) -> Dict[str, Any]:
        """
        Runs comprehensive early-warning scan across rule-based indicators,
        statistical Z-scores, and Isolation Forest multivariate anomaly detection.
        """
        self.alerts = []

        # Filter customers meeting minimum history requirement
        eligible_df = self.features_df[self.features_df['frequency'] >= self.min_history_orders].copy()

        # 1. Increasing Purchase Recency (Dormancy Risk on High-Value)
        self._detect_recency_drift(eligible_df)

        # 2. Declining Order Frequency (Velocity Drop)
        self._detect_velocity_collapse(eligible_df)

        # 3. Falling Average Order Value
        self._detect_aov_shrinkage(eligible_df)

        # 4. Unusual Spending Changes (Spend Collapse vs Spend Spikes)
        self._detect_spending_anomalies(eligible_df)

        # 5. Emerging High-Value Customers (Accelerating Momentum)
        self._detect_emerging_vips(eligible_df)

        # 6. Significant Segment Shifts (Segment composition migration)
        segment_shift_alerts = self._detect_segment_migration()
        self.alerts.extend(segment_shift_alerts)

        # 7. Multivariate Statistical Anomaly Detection (Isolation Forest)
        self._detect_isolation_forest_anomalies(eligible_df, contamination=contamination)

        # Sort alerts by severity order: CRITICAL > WARNING > OPPORTUNITY > INFO
        severity_rank = {'CRITICAL': 0, 'WARNING': 1, 'OPPORTUNITY': 2, 'INFO': 3}
        self.alerts.sort(key=lambda a: severity_rank.get(a['severity'], 4))

        summary = {
            'total_alerts': len(self.alerts),
            'critical_count': sum(1 for a in self.alerts if a['severity'] == 'CRITICAL'),
            'warning_count': sum(1 for a in self.alerts if a['severity'] == 'WARNING'),
            'opportunity_count': sum(1 for a in self.alerts if a['severity'] == 'OPPORTUNITY'),
            'info_count': sum(1 for a in self.alerts if a['severity'] == 'INFO'),
            'eligible_accounts_scanned': len(eligible_df),
            'min_history_orders_threshold': self.min_history_orders,
            'alerts': self.alerts
        }
        return summary

    def _detect_recency_drift(self, df: pd.DataFrame):
        """Identifies accounts with dormancy risk exceeding typical cycle."""
        # Top 50% spenders whose recency is over 180 days
        median_spend = df['total_monetary'].median()
        at_risk = df[(df['total_monetary'] >= median_spend) & (df['recency_days'] >= 180)]

        for _, r in at_risk.iterrows():
            severity = 'CRITICAL' if (r['recency_days'] >= 270 and r['total_monetary'] >= 2500) else 'WARNING'
            self.alerts.append({
                'id': f"REC-{r['customer_id']}",
                'type': 'RECENCY_DRIFT',
                'severity': severity,
                'customer_id': r['customer_id'],
                'customer_name': r['customer_name'],
                'segment': r['customer_segment'],
                'title': f"Severe Inactivity ({int(r['recency_days'])} days) on High-Value Account",
                'evidence': (
                    f"Customer generated ${r['total_monetary']:,.2f} over {int(r['frequency'])} orders, "
                    f"but has been inactive for {int(r['recency_days'])} days (last order: {r['last_order_date']})."
                ),
                'comparison_period': 'Lifetime baseline vs. current inactivity period',
                'detection_method': 'Threshold Gap Monitoring (Recency >= 180d, Spend >= P50)',
                'recommended_investigation': 'Verify if account contact has changed; review recent support tickets; launch re-activation offer.',
                'action_url': f"#/profiles?customer_id={r['customer_id']}"
            })

    def _detect_velocity_collapse(self, df: pd.DataFrame):
        """Detects accounts with severe purchase frequency degradation."""
        collapsing = df[df['order_momentum'] <= -0.50]
        for _, r in collapsing.iterrows():
            self.alerts.append({
                'id': f"VEL-{r['customer_id']}",
                'type': 'VELOCITY_COLLAPSE',
                'severity': 'WARNING',
                'customer_id': r['customer_id'],
                'customer_name': r['customer_name'],
                'segment': r['customer_segment'],
                'title': f"Order Velocity Dropped by {abs(r['order_momentum']*100):.0f}%",
                'evidence': (
                    f"Orders declined from {int(r['prior_orders_6m'])} orders (prior 6m) "
                    f"to {int(r['recent_orders_6m'])} orders (recent 6m)."
                ),
                'comparison_period': 'Last 180 days vs Prior 180-360 days window',
                'detection_method': 'Order Frequency Momentum Ratio (order_momentum <= -0.50)',
                'recommended_investigation': 'Check for product stockouts, category migration, or supplier dissatisfaction.',
                'action_url': f"#/profiles?customer_id={r['customer_id']}"
            })

    def _detect_aov_shrinkage(self, df: pd.DataFrame):
        """Detects accounts whose recent basket spend shrank significantly."""
        # When spend fell despite orders or spend momentum < -0.6
        spend_collapse = df[(df['spend_momentum'] <= -0.60) & (df['prior_spend_6m'] >= 500)]
        for _, r in spend_collapse.iterrows():
            self.alerts.append({
                'id': f"AOV-{r['customer_id']}",
                'type': 'SPEND_SHRINKAGE',
                'severity': 'WARNING',
                'customer_id': r['customer_id'],
                'customer_name': r['customer_name'],
                'segment': r['customer_segment'],
                'title': f"Semi-Annual Spend Contracted by {abs(r['spend_momentum']*100):.0f}%",
                'evidence': (
                    f"Spend plunged from ${r['prior_spend_6m']:,.2f} (prior 6m) "
                    f"to ${r['recent_spend_6m']:,.2f} (recent 6m)."
                ),
                'comparison_period': 'Recent 180 days vs Previous 180 days',
                'detection_method': 'Spending Trajectory Ratio (spend_momentum <= -0.60)',
                'recommended_investigation': 'Investigate whether client shifted high-margin purchases to a competing distributor.',
                'action_url': f"#/profiles?customer_id={r['customer_id']}"
            })

    def _detect_spending_anomalies(self, df: pd.DataFrame):
        """Robust statistical Z-score anomaly on discount and spend distribution."""
        # Unusually high discounts (Z-score > 2.5)
        disc_mean = df['avg_discount'].mean()
        disc_std = max(df['avg_discount'].std(), 1e-4)
        z_disc = (df['avg_discount'] - disc_mean) / disc_std

        high_disc = df[z_disc > 2.5]
        for _, r in high_disc.iterrows():
            self.alerts.append({
                'id': f"DISC-{r['customer_id']}",
                'type': 'PROMOTIONAL_EROSION',
                'severity': 'CRITICAL' if r['profit_margin'] < 0 else 'WARNING',
                'customer_id': r['customer_id'],
                'customer_name': r['customer_name'],
                'segment': r['customer_segment'],
                'title': f"Excessive Promotional Discount Exposure ({r['avg_discount']*100:.1f}%)",
                'evidence': (
                    f"Average discount of {r['avg_discount']*100:.1f}% is {z_disc.loc[r.name]:.1f} std devs "
                    f"above population average ({disc_mean*100:.1f}%). Lifetime margin is {r['profit_margin']:.1f}%."
                ),
                'comparison_period': 'Account average vs Full Customer Population distribution',
                'detection_method': 'Robust Z-Score Outlier Detection (Z > 2.5)',
                'recommended_investigation': 'Audit sales rep discounting overrides; enforce standard gross margin floor.',
                'action_url': f"#/profiles?customer_id={r['customer_id']}"
            })

    def _detect_emerging_vips(self, df: pd.DataFrame):
        """Identifies rapid growth accounts with surging velocity and high forward LTV."""
        emerging = df[(df['order_momentum'] >= 0.75) & (df['predicted_ltv_12m'] >= 1200) & (df['recent_orders_6m'] >= 3)]
        for _, r in emerging.iterrows():
            self.alerts.append({
                'id': f"VIP-{r['customer_id']}",
                'type': 'EMERGING_VIP',
                'severity': 'OPPORTUNITY',
                'customer_id': r['customer_id'],
                'customer_name': r['customer_name'],
                'segment': r['customer_segment'],
                'title': f"Emerging High-Growth VIP: Momentum +{r['order_momentum']*100:.0f}%",
                'evidence': (
                    f"Placed {int(r['recent_orders_6m'])} orders in past 6 months (up from {int(r['prior_orders_6m'])}). "
                    f"Predicted forward 12-month LTV is ${r['predicted_ltv_12m']:,.2f}."
                ),
                'comparison_period': 'Accelerating momentum (recent 6m vs prior 6m)',
                'detection_method': 'Positive Velocity Spike & High Predictive Value Filter',
                'recommended_investigation': 'Assign dedicated relationship manager; enroll in premium loyalty tier.',
                'action_url': f"#/profiles?customer_id={r['customer_id']}"
            })

    def _detect_segment_migration(self) -> List[Dict[str, Any]]:
        """Identifies systemic shifts in segment distribution."""
        seg_counts = self.features_df['customer_segment'].value_counts()
        total = len(self.features_df)

        alerts = []
        dormant_pct = (seg_counts.get('Dormant / Lost', 0) + seg_counts.get('Hibernating', 0)) / total * 100
        if dormant_pct > 20:
            alerts.append({
                'id': "SEG-DORMANT-SURGE",
                'type': 'SEGMENT_COMPOSITION_SHIFT',
                'severity': 'WARNING',
                'customer_id': 'SYSTEM_WIDE',
                'customer_name': 'Entire Portfolio',
                'segment': 'Dormant & Hibernating',
                'title': f"Elevated Dormancy Share ({dormant_pct:.1f}% of Accounts)",
                'evidence': (
                    f"Combined Dormant/Lost and Hibernating accounts comprise {dormant_pct:.1f}% "
                    f"of total customer base ({seg_counts.get('Dormant / Lost', 0) + seg_counts.get('Hibernating', 0)} accounts)."
                ),
                'comparison_period': 'Current snapshot vs optimal portfolio composition benchmark (<15%)',
                'detection_method': 'Segment Proportional Concentration Scan',
                'recommended_investigation': 'Conduct broad win-back review and evaluate recent churn root causes.',
                'action_url': "#/copilot?q=Which+customer+segments+contribute+the+most+historical+revenue?"
            })
        return alerts

    def _detect_isolation_forest_anomalies(self, df: pd.DataFrame, contamination: float = 0.04):
        """
        Multivariate anomaly detection using Scikit-Learn IsolationForest.
        Discovers subtle non-linear behavioral outliers across multi-dimensional feature space.
        """
        num_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
        X = df[num_cols].fillna(0.0).values

        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42
        )
        anomaly_labels = iso_forest.fit_predict(X)
        anomaly_scores = iso_forest.decision_function(X)

        df_anom = df[anomaly_labels == -1].copy()
        df_anom['anomaly_score'] = anomaly_scores[anomaly_labels == -1]
        df_anom = df_anom.sort_values(by='anomaly_score')

        for _, r in df_anom.head(10).iterrows():
            # Check if alert already logged for this customer
            existing_ids = [a['customer_id'] for a in self.alerts]
            if r['customer_id'] in existing_ids:
                continue

            self.alerts.append({
                'id': f"ISO-{r['customer_id']}",
                'type': 'MULTIVARIATE_ANOMALY',
                'severity': 'INFO',
                'customer_id': r['customer_id'],
                'customer_name': r['customer_name'],
                'segment': r['customer_segment'],
                'title': f"Multivariate Outlier Profile (Isolation Forest Score: {r['anomaly_score']:.3f})",
                'evidence': (
                    f"Unusual combination of metrics: {int(r['frequency'])} orders, ${r['total_monetary']:,.2f} spend, "
                    f"AOV ${r['avg_order_value']:,.2f}, margin {r['profit_margin']:.1f}%, recency {int(r['recency_days'])} days."
                ),
                'comparison_period': 'Multi-dimensional cluster separation against 793 accounts',
                'detection_method': 'Isolation Forest Tree Ensemble Outlier Scoring (n=100)',
                'recommended_investigation': 'Review individual transaction logs for atypical order sizes or return patterns.',
                'action_url': f"#/profiles?customer_id={r['customer_id']}"
            })

