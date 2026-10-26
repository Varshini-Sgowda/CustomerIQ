"""
CustomerIQ Scenario-Based Customer Digital Twin ("Customer Future Lab")
Simulates hypothetical behavioral adjustments (frequency, basket size, recency, discounts),
strictly enforces mathematical consistency and feature dependencies, runs real trained model inference,
and computes defensible uncertainty intervals with clear causal vs correlational disclaimers.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

from customeriq.feature_pipeline import FEATURE_COLUMNS
from customeriq.ltv_model import LTVModelEngine


CAUSAL_DISCLAIMER = (
    "STATISTICAL SIMULATION NOTICE: Changes simulated in this Digital Twin reflect observational "
    "associations learned by the machine learning model. Adjusting a model parameter (e.g., lower discount "
    "or higher frequency) demonstrates the model's conditional expectation for an account with those features; "
    "it does NOT guarantee causal impact. Real-world interventions require controlled experimentation (e.g., A/B tests)."
)


class CustomerFutureLab:
    def __init__(self, features_df: pd.DataFrame, model_engine: LTVModelEngine):
        self.features_df = features_df.copy()
        self.model_engine = model_engine

    def simulate(self, customer_id: str, hypothetical_changes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs digital twin simulation for a specific customer with modified behavioral inputs.
        Validates constraints and recalculates dependent features.
        """
        rows = self.features_df[self.features_df['customer_id'] == customer_id]
        if rows.empty:
            raise KeyError(f"Customer '{customer_id}' not found in feature registry.")

        baseline_row = rows.iloc[0].to_dict()

        # 1. Validate constraints
        validated_inputs, validation_errors = self._validate_constraints(baseline_row, hypothetical_changes)
        if validation_errors:
            return {
                'status': 'INVALID_INPUT',
                'customer_id': customer_id,
                'errors': validation_errors,
                'baseline': None,
                'simulation': None
            }

        # 2. Recompute dependent features
        sim_row = self._recompute_dependent_features(baseline_row, validated_inputs)

        # 3. Model Inference on Baseline vs Simulated
        baseline_X = np.array([baseline_row.get(col, 0.0) for col in self.model_engine.feature_columns], dtype=float).reshape(1, -1)
        sim_X = np.array([sim_row.get(col, 0.0) for col in self.model_engine.feature_columns], dtype=float).reshape(1, -1)

        base_pred = float(max(self.model_engine.model.predict(baseline_X)[0], 0.0))
        sim_pred = float(max(self.model_engine.model.predict(sim_X)[0], 0.0))

        q_low, q_high = self.model_engine.residual_quantiles

        base_interval = [float(round(max(base_pred + q_low, 0.0), 2)), float(round(max(base_pred + q_high, base_pred), 2))]
        sim_interval = [float(round(max(sim_pred + q_low, 0.0), 2)), float(round(max(sim_pred + q_high, sim_pred), 2))]

        # Delta metrics
        delta_ltv = round(sim_pred - base_pred, 2)
        delta_ltv_pct = round((delta_ltv / max(base_pred, 1e-4)) * 100, 1)

        comparison_features = [
            {
                'feature': 'recency_days',
                'label': 'Recency (Days)',
                'baseline': int(baseline_row['recency_days']),
                'simulated': int(sim_row['recency_days']),
                'status': 'MODIFIED' if 'recency_days' in validated_inputs else 'UNCHANGED'
            },
            {
                'feature': 'frequency',
                'label': 'Lifetime Frequency (Orders)',
                'baseline': int(baseline_row['frequency']),
                'simulated': int(sim_row['frequency']),
                'status': 'MODIFIED' if 'frequency' in validated_inputs else 'UNCHANGED'
            },
            {
                'feature': 'avg_order_value',
                'label': 'Average Order Value ($)',
                'baseline': round(baseline_row['avg_order_value'], 2),
                'simulated': round(sim_row['avg_order_value'], 2),
                'status': 'MODIFIED' if 'avg_order_value' in validated_inputs else 'UNCHANGED'
            },
            {
                'feature': 'avg_discount',
                'label': 'Average Discount Rate',
                'baseline': f"{baseline_row['avg_discount']*100:.1f}%",
                'simulated': f"{sim_row['avg_discount']*100:.1f}%",
                'status': 'MODIFIED' if 'avg_discount' in validated_inputs else 'UNCHANGED'
            },
            {
                'feature': 'total_monetary',
                'label': 'Total Historical Spend ($)',
                'baseline': round(baseline_row['total_monetary'], 2),
                'simulated': round(sim_row['total_monetary'], 2),
                'status': 'RECOMPUTED_DEPENDENT'
            },
            {
                'feature': 'order_momentum',
                'label': 'Order Momentum',
                'baseline': round(baseline_row['order_momentum'], 2),
                'simulated': round(sim_row['order_momentum'], 2),
                'status': 'RECOMPUTED_DEPENDENT' if 'recent_orders_6m' in validated_inputs else 'UNCHANGED'
            }
        ]

        return {
            'status': 'SUCCESS',
            'customer_id': customer_id,
            'customer_name': baseline_row['customer_name'],
            'segment': baseline_row['customer_segment'],
            'baseline': {
                'predicted_ltv': round(base_pred, 2),
                'prediction_interval_80': base_interval,
                'status_tag': 'MODEL_ESTIMATE_OBSERVED'
            },
            'simulation': {
                'predicted_ltv': round(sim_pred, 2),
                'prediction_interval_80': sim_interval,
                'status_tag': 'SIMULATION_HYPOTHETICAL'
            },
            'variance': {
                'delta_ltv': delta_ltv,
                'delta_ltv_pct': delta_ltv_pct,
                'direction': 'INCREASE' if delta_ltv > 0 else 'DECREASE' if delta_ltv < 0 else 'FLAT'
            },
            'feature_comparison': comparison_features,
            'causal_disclaimer': CAUSAL_DISCLAIMER
        }

    def _validate_constraints(self, baseline: Dict[str, Any], changes: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Strict validation of behavioral inputs to prevent impossible states."""
        errors = []
        valid_changes = {}

        # 1. Recency
        if 'recency_days' in changes:
            rec = float(changes['recency_days'])
            if rec < 0:
                errors.append("Recency cannot be negative (must be >= 0 days).")
            elif rec > 1500:
                errors.append("Recency cannot exceed 1,500 days.")
            else:
                valid_changes['recency_days'] = int(rec)

        # 2. Frequency
        if 'frequency' in changes:
            freq = int(changes['frequency'])
            if freq < 1:
                errors.append("Lifetime order frequency must be at least 1 order.")
            elif freq > 100:
                errors.append("Order frequency input exceeds maximum realistic cap (100 orders).")
            else:
                valid_changes['frequency'] = freq

        # 3. Average Order Value
        if 'avg_order_value' in changes:
            aov = float(changes['avg_order_value'])
            if aov <= 0:
                errors.append("Average order value must be strictly positive.")
            elif aov > 25000:
                errors.append("Average order value exceeds realistic threshold ($25,000).")
            else:
                valid_changes['avg_order_value'] = round(aov, 2)

        # 4. Average Discount
        if 'avg_discount' in changes:
            disc = float(changes['avg_discount'])
            if disc < 0.0 or disc > 0.80:
                errors.append("Average discount rate must be between 0.0 (0%) and 0.80 (80%).")
            else:
                valid_changes['avg_discount'] = round(disc, 4)

        # 5. Recent 6-Month Orders
        if 'recent_orders_6m' in changes:
            r6 = int(changes['recent_orders_6m'])
            if r6 < 0:
                errors.append("Recent 6-month orders cannot be negative.")
            else:
                valid_changes['recent_orders_6m'] = r6

        # Cross-field integrity: tenure vs recency
        eff_recency = valid_changes.get('recency_days', baseline['recency_days'])
        eff_tenure = baseline['tenure_days']
        if eff_recency > eff_tenure:
            errors.append(f"Inconsistent inputs: Recency ({eff_recency} days) cannot exceed customer tenure ({eff_tenure} days).")

        return valid_changes, errors

    def _recompute_dependent_features(self, baseline: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures mathematical consistency across dependent engineered features."""
        sim = baseline.copy()
        sim.update(changes)

        # Recompute total monetary if frequency or AOV modified
        if 'avg_order_value' in changes or 'frequency' in changes:
            sim['total_monetary'] = round(sim['frequency'] * sim['avg_order_value'], 2)

        # Recompute order velocity
        tenure = max(sim['tenure_days'], 1)
        sim['order_velocity_annual'] = round((sim['frequency'] / tenure) * 365.25, 2)

        # Recompute order momentum if recent orders adjusted
        if 'recent_orders_6m' in changes:
            prior = max(sim['prior_orders_6m'], 1)
            sim['order_momentum'] = round((sim['recent_orders_6m'] - prior) / prior, 4)

        return sim

