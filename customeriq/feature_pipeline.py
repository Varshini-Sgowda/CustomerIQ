"""
CustomerIQ Feature Engineering Pipeline
Transforms transaction logs into customer-level analytical feature tables.
Supports leak-free temporal point-in-time calculation and RFM segmentation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple


FEATURE_COLUMNS = [
    'recency_days',
    'frequency',
    'tenure_days',
    'total_monetary',
    'avg_order_value',
    'total_profit',
    'profit_margin',
    'avg_discount',
    'discount_order_ratio',
    'items_per_order',
    'order_velocity_annual',
    'recent_orders_6m',
    'prior_orders_6m',
    'order_momentum',
    'recent_spend_6m',
    'prior_spend_6m',
    'spend_momentum',
    'furniture_spend_ratio',
    'office_spend_ratio',
    'technology_spend_ratio'
]


class FeaturePipeline:
    def __init__(self, reference_date: Optional[datetime] = None):
        self.reference_date = reference_date

    def transform(self, df: pd.DataFrame, ref_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Transforms transaction slice into a validated customer feature table.
        ref_date: Point-in-time anchor. If None, uses max(Order Date) in df.
        """
        if df.empty:
            raise ValueError("Input transaction dataframe is empty.")

        if ref_date is None:
            ref_date = self.reference_date if self.reference_date is not None else df['Order Date'].max()
        ref_date = pd.to_datetime(ref_date)

        # Cutoff: only consider orders on or before ref_date
        valid_df = df[df['Order Date'] <= ref_date].copy()
        if valid_df.empty:
            raise ValueError(f"No transactions found on or before reference date {ref_date}")

        # Six-month windows for momentum
        d_180 = ref_date - pd.Timedelta(days=180)
        d_360 = ref_date - pd.Timedelta(days=360)

        # Customer grouping
        records = []
        for cust_id, group in valid_df.groupby('Customer ID'):
            cust_name = group['Customer Name'].iloc[-1]
            segment = group['Segment'].iloc[-1]
            # Primary region (mode or latest)
            region = group['Region'].mode().iloc[0] if not group['Region'].empty else group['Region'].iloc[-1]

            orders = group.drop_duplicates(subset=['Order ID'])
            frequency = int(len(orders))
            first_order_dt = group['Order Date'].min()
            last_order_dt = group['Order Date'].max()

            recency_days = max(int((ref_date - last_order_dt).days), 0)
            tenure_days = max(int((ref_date - first_order_dt).days), 1)

            total_monetary = float(round(group['Sales'].sum(), 2))
            total_profit = float(round(group['Profit'].sum(), 2))
            avg_order_value = float(round(total_monetary / max(frequency, 1), 2))
            profit_margin = float(round((total_profit / max(total_monetary, 1e-6)) * 100, 2))

            avg_discount = float(round(group['Discount'].mean(), 4))
            discount_order_ratio = float(round((group['Discount'] > 0.15).mean(), 4))
            total_qty = float(group['Quantity'].sum())
            items_per_order = float(round(total_qty / max(frequency, 1), 2))

            order_velocity_annual = float(round((frequency / max(tenure_days, 1)) * 365.25, 2))

            # Momentum features (last 180d vs prior 180d)
            recent_orders_slice = group[group['Order Date'] > d_180]
            prior_orders_slice = group[(group['Order Date'] > d_360) & (group['Order Date'] <= d_180)]

            recent_orders_6m = int(recent_orders_slice['Order ID'].nunique())
            prior_orders_6m = int(prior_orders_slice['Order ID'].nunique())
            order_momentum = float(round((recent_orders_6m - prior_orders_6m) / max(prior_orders_6m, 1), 4))

            recent_spend_6m = float(round(recent_orders_slice['Sales'].sum(), 2))
            prior_spend_6m = float(round(prior_orders_slice['Sales'].sum(), 2))
            spend_momentum = float(round((recent_spend_6m - prior_spend_6m) / max(prior_spend_6m, 1.0), 4))

            # Category spending distribution
            cat_spend = group.groupby('Category')['Sales'].sum().to_dict()
            furniture_spend_ratio = float(round(cat_spend.get('Furniture', 0.0) / max(total_monetary, 1e-6), 4))
            office_spend_ratio = float(round(cat_spend.get('Office Supplies', 0.0) / max(total_monetary, 1e-6), 4))
            technology_spend_ratio = float(round(cat_spend.get('Technology', 0.0) / max(total_monetary, 1e-6), 4))

            records.append({
                'customer_id': cust_id,
                'customer_name': cust_name,
                'segment': segment,
                'region': region,
                'first_order_date': first_order_dt.strftime('%Y-%m-%d'),
                'last_order_date': last_order_dt.strftime('%Y-%m-%d'),
                'recency_days': recency_days,
                'frequency': frequency,
                'tenure_days': tenure_days,
                'total_monetary': total_monetary,
                'avg_order_value': avg_order_value,
                'total_profit': total_profit,
                'profit_margin': profit_margin,
                'avg_discount': avg_discount,
                'discount_order_ratio': discount_order_ratio,
                'items_per_order': items_per_order,
                'order_velocity_annual': order_velocity_annual,
                'recent_orders_6m': recent_orders_6m,
                'prior_orders_6m': prior_orders_6m,
                'order_momentum': order_momentum,
                'recent_spend_6m': recent_spend_6m,
                'prior_spend_6m': prior_spend_6m,
                'spend_momentum': spend_momentum,
                'furniture_spend_ratio': furniture_spend_ratio,
                'office_spend_ratio': office_spend_ratio,
                'technology_spend_ratio': technology_spend_ratio
            })

        features_df = pd.DataFrame(records)

        # Compute RFM Scores and Segment Classifications
        features_df = self._assign_rfm_segments(features_df)
        return features_df

    def _assign_rfm_segments(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes robust quintile RFM scores and rule-based segment names with explanations.
        """
        # Recency score (inverted: lower recency days = higher score 5)
        df['r_score'] = pd.qcut(df['recency_days'].rank(method='first', ascending=False), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        df['f_score'] = pd.qcut(df['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        df['m_score'] = pd.qcut(df['total_monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5]).astype(int)
        df['rfm_score'] = df['r_score'].astype(str) + df['f_score'].astype(str) + df['m_score'].astype(str)

        segments = []
        explanations = []

        for _, row in df.iterrows():
            r, f, m = row['r_score'], row['f_score'], row['m_score']
            rec_days = row['recency_days']
            freq = row['frequency']
            mon = row['total_monetary']

            if r >= 4 and f >= 4 and m >= 4:
                seg = "Champions"
                exp = f"Top-tier engagement: purchased within {rec_days} days, high frequency ({freq} orders), total spend ${mon:,.0f} (RFM {r}{f}{m})."
            elif r >= 3 and f >= 3 and m >= 3:
                seg = "Loyal Customers"
                exp = f"Consistently active: {freq} lifetime orders, spend ${mon:,.0f}, recent order {rec_days} days ago (RFM {r}{f}{m})."
            elif r >= 4 and f >= 2 and m < 3:
                seg = "Potential Loyalists"
                exp = f"Recent repeat customer ({rec_days} days ago, {freq} orders) with growth capacity for higher order values."
            elif r >= 4 and f == 1:
                seg = "New Customers"
                exp = f"Acquired recently ({rec_days} days ago) with 1 initial order. High priority for onboarding."
            elif r >= 3 and f <= 2:
                seg = "Promising"
                exp = f"Recent engagement ({rec_days} days ago) but low order volume ({freq} orders). Nurture for loyalty."
            elif r == 2 and f >= 3:
                seg = "Need Attention"
                exp = f"Above-average previous engagement ({freq} orders) but dormant for {rec_days} days. High re-activation urgency."
            elif r <= 2 and f >= 3 and m >= 3:
                seg = "At Risk"
                exp = f"High-value account (${mon:,.0f} spend, {freq} orders) slipping into churn ({rec_days} days since last purchase)."
            elif r <= 2 and f <= 2 and m >= 3:
                seg = "About to Sleep"
                exp = f"High historical value (${mon:,.0f}) but low order count ({freq}) and inactive for {rec_days} days."
            elif r == 1 and f <= 2 and m <= 2:
                seg = "Hibernating"
                exp = f"Low lifetime value (${mon:,.0f}, {freq} orders) and absent for {rec_days} days."
            else:
                seg = "Dormant / Lost"
                exp = f"Prolonged inactivity ({rec_days} days since last order), RFM score {r}{f}{m}."

            segments.append(seg)
            explanations.append(exp)

        df['customer_segment'] = segments
        df['segment_explanation'] = explanations
        return df

