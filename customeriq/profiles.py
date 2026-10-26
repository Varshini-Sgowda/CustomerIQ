"""
CustomerIQ Explainable Customer Intelligence Profiles Module
Combines observed facts, model predictions, SHAP feature attributions,
historical spending trends, behavioral shifts, data quality warnings,
and recommended next actions for single and dual-customer comparison.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

from customeriq.feature_pipeline import FeaturePipeline
from customeriq.ltv_model import LTVModelEngine


class CustomerProfileEngine:
    def __init__(self, raw_df: pd.DataFrame, features_df: pd.DataFrame, model_engine: LTVModelEngine):
        self.raw_df = raw_df
        self.features_df = features_df
        self.model_engine = model_engine

        # Ensure predictions are attached
        if 'predicted_ltv_12m' not in self.features_df.columns:
            self.features_df = self.model_engine.predict_features(self.features_df)

    def get_customer_profile(self, customer_id: str) -> Dict[str, Any]:
        """
        Builds a comprehensive 360-degree explainable customer profile.
        Labels every metric by its epistemological status:
        [OBSERVED_FACT], [MODEL_ESTIMATE], [HEURISTIC_RULE], [DATA_WARNING].
        """
        cust_rows = self.features_df[self.features_df['customer_id'] == customer_id]
        if cust_rows.empty:
            raise KeyError(f"Customer ID '{customer_id}' not found.")

        feat_row = cust_rows.iloc[0]
        cust_tx = self.raw_df[self.raw_df['Customer ID'] == customer_id].sort_values('Order Date')

        # 1. Observed Transaction History & Facts
        orders_summary = []
        for order_id, ogroup in cust_tx.groupby('Order ID'):
            orders_summary.append({
                'order_id': order_id,
                'order_date': ogroup['Order Date'].iloc[0].strftime('%Y-%m-%d'),
                'items_count': int(len(ogroup)),
                'sales': float(round(ogroup['Sales'].sum(), 2)),
                'profit': float(round(ogroup['Profit'].sum(), 2)),
                'avg_discount': float(round(ogroup['Discount'].mean(), 3)),
                'categories': ogroup['Category'].unique().tolist()
            })
        orders_summary.sort(key=lambda x: x['order_date'], reverse=True)

        # Monthly spending trend
        tx_monthly = cust_tx.set_index('Order Date').resample('QE').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
        quarterly_trend = [
            {
                'quarter': f"Q{r['Order Date'].quarter} {r['Order Date'].year}",
                'sales': float(round(r['Sales'], 2)),
                'profit': float(round(r['Profit'], 2))
            }
            for _, r in tx_monthly.iterrows()
        ]

        # Category spend breakdown
        cat_breakdown = cust_tx.groupby('Category').agg({'Sales': 'sum', 'Profit': 'sum', 'Quantity': 'sum'}).reset_index()
        category_distribution = [
            {
                'category': r['Category'],
                'sales': float(round(r['Sales'], 2)),
                'profit': float(round(r['Profit'], 2)),
                'quantity': int(r['Quantity'])
            }
            for _, r in cat_breakdown.iterrows()
        ]

        # 2. Model Prediction & Explainability
        shap_explanation = self.model_engine.explain_customer(feat_row)

        # 3. Data Quality Warnings
        data_quality_warnings = []
        if feat_row['frequency'] <= 1:
            data_quality_warnings.append({
                'type': 'SPARSE_HISTORY',
                'severity': 'HIGH',
                'message': 'Only 1 historical order recorded. Model prediction carries wider uncertainty due to limited behavioral sample.'
            })
        if feat_row['tenure_days'] < 90:
            data_quality_warnings.append({
                'type': 'SHORT_TENURE',
                'severity': 'MEDIUM',
                'message': f"Account is relatively new ({feat_row['tenure_days']} days). Trajectory is still stabilizing."
            })
        if feat_row['profit_margin'] < 0:
            data_quality_warnings.append({
                'type': 'UNPROFITABLE_ACCOUNT',
                'severity': 'CRITICAL',
                'message': f"Account generated negative lifetime profit (${feat_row['total_profit']:,.2f}, margin {feat_row['profit_margin']:.1f}%). High discount exposure."
            })
        if feat_row['avg_discount'] > 0.35:
            data_quality_warnings.append({
                'type': 'DISCOUNT_DEPENDENCY',
                'severity': 'WARNING',
                'message': f"Average discount received is {feat_row['avg_discount']*100:.1f}%. Customer rarely converts at full list price."
            })

        # 4. Behavioral Momentum & Shifts
        order_shift_desc = (
            f"Orders declined by {abs(feat_row['order_momentum']*100):.0f}% compared to prior 6 months"
            if feat_row['order_momentum'] < 0 else
            f"Orders grew by {feat_row['order_momentum']*100:.0f}% compared to prior 6 months"
            if feat_row['order_momentum'] > 0 else "Order velocity remained steady"
        )
        spend_shift_desc = (
            f"Spending dropped from ${feat_row['prior_spend_6m']:,.2f} to ${feat_row['recent_spend_6m']:,.2f}"
            if feat_row['spend_momentum'] < 0 else
            f"Spending increased from ${feat_row['prior_spend_6m']:,.2f} to ${feat_row['recent_spend_6m']:,.2f}"
            if feat_row['spend_momentum'] > 0 else "Spend level remained consistent"
        )

        # 5. Tailored Next Steps (Heuristic Decision Support)
        recommended_actions = self._generate_recommendations(feat_row)

        return {
            'metadata': {
                'customer_id': feat_row['customer_id'],
                'customer_name': feat_row['customer_name'],
                'segment': feat_row['segment'],
                'region': feat_row['region'],
                'status_tag': 'OBSERVED_FACT'
            },
            'rfm_classification': {
                'customer_segment': feat_row['customer_segment'],
                'r_score': int(feat_row['r_score']),
                'f_score': int(feat_row['f_score']),
                'm_score': int(feat_row['m_score']),
                'rfm_code': str(feat_row['rfm_score']),
                'rule_explanation': feat_row['segment_explanation'],
                'status_tag': 'HEURISTIC_RULE'
            },
            'model_predictions': {
                'predicted_ltv_12m': float(feat_row['predicted_ltv_12m']),
                'lower_bound_80': float(feat_row['ltv_lower_bound']),
                'upper_bound_80': float(feat_row['ltv_upper_bound']),
                'prediction_horizon_months': 12,
                'status_tag': 'MODEL_ESTIMATE',
                'shap_base_value': shap_explanation['base_value'],
                'top_positive_drivers': shap_explanation['top_positive_drivers'],
                'top_negative_drivers': shap_explanation['top_negative_drivers'],
                'all_shap_contributions': shap_explanation['all_contributions']
            },
            'observed_metrics': {
                'recency_days': int(feat_row['recency_days']),
                'frequency': int(feat_row['frequency']),
                'tenure_days': int(feat_row['tenure_days']),
                'total_monetary': float(feat_row['total_monetary']),
                'avg_order_value': float(feat_row['avg_order_value']),
                'total_profit': float(feat_row['total_profit']),
                'profit_margin': float(feat_row['profit_margin']),
                'avg_discount': float(feat_row['avg_discount']),
                'items_per_order': float(feat_row['items_per_order']),
                'status_tag': 'OBSERVED_FACT'
            },
            'behavioral_momentum': {
                'recent_orders_6m': int(feat_row['recent_orders_6m']),
                'prior_orders_6m': int(feat_row['prior_orders_6m']),
                'order_momentum_pct': float(round(feat_row['order_momentum'] * 100, 1)),
                'order_shift_description': order_shift_desc,
                'recent_spend_6m': float(feat_row['recent_spend_6m']),
                'prior_spend_6m': float(feat_row['prior_spend_6m']),
                'spend_momentum_pct': float(round(feat_row['spend_momentum'] * 100, 1)),
                'spend_shift_description': spend_shift_desc,
                'status_tag': 'OBSERVED_FACT'
            },
            'category_distribution': category_distribution,
            'quarterly_trend': quarterly_trend,
            'recent_orders': orders_summary[:10],
            'data_quality_warnings': data_quality_warnings,
            'recommended_next_steps': recommended_actions
        }

    def compare_customers(self, cust_id_a: str, cust_id_b: str) -> Dict[str, Any]:
        """
        Performs rigorous side-by-side analytical comparison between two customer profiles.
        """
        prof_a = self.get_customer_profile(cust_id_a)
        prof_b = self.get_customer_profile(cust_id_b)

        metrics_diff = {
            'predicted_ltv_diff': round(prof_a['model_predictions']['predicted_ltv_12m'] - prof_b['model_predictions']['predicted_ltv_12m'], 2),
            'total_spend_diff': round(prof_a['observed_metrics']['total_monetary'] - prof_b['observed_metrics']['total_monetary'], 2),
            'frequency_diff': prof_a['observed_metrics']['frequency'] - prof_b['observed_metrics']['frequency'],
            'recency_diff_days': prof_a['observed_metrics']['recency_days'] - prof_b['observed_metrics']['recency_days'],
            'aov_diff': round(prof_a['observed_metrics']['avg_order_value'] - prof_b['observed_metrics']['avg_order_value'], 2),
            'margin_diff_pct': round(prof_a['observed_metrics']['profit_margin'] - prof_b['observed_metrics']['profit_margin'], 2)
        }

        return {
            'customer_a': prof_a,
            'customer_b': prof_b,
            'variance_summary': metrics_diff
        }

    def _generate_recommendations(self, feat_row: pd.Series) -> List[Dict[str, str]]:
        """Generates transparent, deterministic next-best actions."""
        actions = []
        seg = feat_row['customer_segment']
        r = feat_row['r_score']
        margin = feat_row['profit_margin']
        pred_ltv = feat_row['predicted_ltv_12m']

        if seg == "Champions":
            actions.append({
                'action': 'VIP Executive Concierge & Loyalty Rewards',
                'priority': 'HIGH',
                'rational': f"Top-tier LTV (${pred_ltv:,.0f}). Provide early product previews, dedicated account support, and zero margin-eroding discounts."
            })
        elif seg in ["At Risk", "Need Attention"]:
            actions.append({
                'action': 'Targeted Win-Back Re-engagement Campaign',
                'priority': 'CRITICAL',
                'rational': f"Customer was previously high-value but has slipped to {feat_row['recency_days']} days inactive. Send personalized reactivation offer."
            })
        elif seg == "New Customers":
            actions.append({
                'action': 'Welcome Onboarding Sequence',
                'priority': 'HIGH',
                'rational': "Customer made their 1st purchase recently. Trigger educational product guide to incentivize 2nd order within 45 days."
            })
        elif margin < 0:
            actions.append({
                'action': 'Discount Threshold Restructuring',
                'priority': 'HIGH',
                'rational': f"Negative account margin ({margin:.1f}%). Limit maximum promotional discount to 10% on future purchases to restore unit profitability."
            })
        else:
            actions.append({
                'action': 'Cross-Category Discovery Trigger',
                'priority': 'MEDIUM',
                'rational': f"Promote adjacent categories based on purchase affinity to increase frequency and basket diversification."
            })

        return actions
