"""
CustomerIQ Intelligent Marketing Action Planner ("Opportunity Center")
Prioritizes high-conviction commercial interventions, optimizes budget allocation,
benchmarks prioritized strategies against blanket mass campaigns,
and implements uplift modeling simulation with explicit transparent assumptions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


ACTION_CATALOG = {
    'VIP_CONCIERGE': {
        'name': 'Executive VIP Concierge & Personalized Gifting',
        'cost_per_customer': 65.0,
        'target_segments': ['Champions'],
        'base_retention_lift_assumed': 0.12,
        'description': 'High-touch personal outreach with bespoke enterprise gifting and dedicated account management.'
    },
    'WINBACK_PROMO': {
        'name': 'Targeted Reactivation Campaign & Margin-Safe Voucher',
        'cost_per_customer': 25.0,
        'target_segments': ['At Risk', 'Need Attention', 'About to Sleep'],
        'base_retention_lift_assumed': 0.18,
        'description': 'Time-sensitive 10% voucher with personalized category recommendations based on prior purchases.'
    },
    'ONBOARDING_NURTURE': {
        'name': '2nd-Purchase Fast-Track Automation',
        'cost_per_customer': 8.0,
        'target_segments': ['New Customers', 'Potential Loyalists'],
        'base_retention_lift_assumed': 0.22,
        'description': 'Automated product education sequence with complimentary accessory credit upon second order.'
    },
    'CATEGORY_CROSS_SELL': {
        'name': 'Cross-Category Expansion Trigger',
        'cost_per_customer': 12.0,
        'target_segments': ['Loyal Customers', 'Promising'],
        'base_retention_lift_assumed': 0.14,
        'description': 'Data-driven recommendations for unpurchased categories (e.g., Technology accessories for Office buyers).'
    },
    'MARGIN_DISCIPLINE': {
        'name': 'Account Profitability Restructuring',
        'cost_per_customer': 5.0,
        'target_segments': ['Dormant / Lost', 'Need Attention'],
        'base_retention_lift_assumed': 0.05,
        'description': 'Removal of unapproved discounts and transition to net-margin pricing agreements.'
    }
}

UPLIFT_DISCLAIMER = (
    "CAUSAL & UPLIFT MODELING METHODOLOGY NOTICE: Incremental ROI estimates are computed using a "
    "heterogeneous uplift simulation model based on explicit behavioral priors (RFM responsiveness, discount affinity, "
    "and recency decay). These projections are decision-support heuristics and do NOT substitute for true randomized "
    "A/B control trials. Marketing teams must run champion/challenger campaign holdout groups to measure actual causal lift."
)


class MarketingActionPlanner:
    def __init__(self, features_df: pd.DataFrame):
        self.features_df = features_df.copy()

    def plan_campaign(
        self,
        budget: float = 3000.0,
        priority_goal: str = 'PREVENT_CHURN_HIGH_LTV',
        custom_cost_overrides: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Builds an optimized marketing action allocation given budget, priorities, and costs.
        """
        budget = max(float(budget), 100.0)
        catalog = {k: v.copy() for k, v in ACTION_CATALOG.items()}
        if custom_cost_overrides:
            for k, cost in custom_cost_overrides.items():
                if k in catalog and cost > 0:
                    catalog[k]['cost_per_customer'] = float(cost)

        candidates = []
        for _, r in self.features_df.iterrows():
            action_key, reason = self._select_best_action(r, priority_goal)
            action_meta = catalog[action_key]
            cost = action_meta['cost_per_customer']

            # Estimated outcome using explicit uplift heuristic
            # Expected Retained Value = Predicted LTV * Assumed Lift * Responsiveness Factor
            responsiveness = self._compute_uplift_responsiveness(r)
            assumed_lift = action_meta['base_retention_lift_assumed']
            estimated_incremental_value = round(r['predicted_ltv_12m'] * assumed_lift * responsiveness, 2)
            expected_net_roi = round((estimated_incremental_value - cost) / max(cost, 1.0), 2)

            priority_score = self._compute_priority_score(r, priority_goal, expected_net_roi)

            candidates.append({
                'customer_id': r['customer_id'],
                'customer_name': r['customer_name'],
                'segment': r['customer_segment'],
                'predicted_ltv_12m': round(float(r['predicted_ltv_12m']), 2),
                'recency_days': int(r['recency_days']),
                'action_key': action_key,
                'action_name': action_meta['name'],
                'action_reason': reason,
                'cost': cost,
                'assumed_lift_pct': round(assumed_lift * 100, 1),
                'responsiveness_index': round(responsiveness, 2),
                'estimated_incremental_value': estimated_incremental_value,
                'expected_net_roi': expected_net_roi,
                'priority_score': priority_score
            })

        cand_df = pd.DataFrame(candidates)
        # Sort candidates by priority score descending
        cand_df = cand_df.sort_values(by='priority_score', ascending=False).reset_index(drop=True)

        # Greedy Budget Knapsack Allocation
        allocated = []
        cumulative_cost = 0.0
        for _, c in cand_df.iterrows():
            if cumulative_cost + c['cost'] <= budget:
                allocated.append(c.to_dict())
                cumulative_cost += c['cost']

        alloc_df = pd.DataFrame(allocated) if allocated else pd.DataFrame()

        total_targeted = len(alloc_df)
        total_cost_allocated = round(cumulative_cost, 2)
        total_projected_incremental_value = round(alloc_df['estimated_incremental_value'].sum(), 2) if not alloc_df.empty else 0.0
        portfolio_roi = round((total_projected_incremental_value - total_cost_allocated) / max(total_cost_allocated, 1.0), 2) if total_cost_allocated > 0 else 0.0

        # Benchmark Comparison: Blanket Mass Outreach (uniform blast to random customers until budget spent)
        blanket_comparison = self._benchmark_against_blanket(budget, catalog)

        # Action Breakdown Summary
        action_counts = alloc_df['action_name'].value_counts().to_dict() if not alloc_df.empty else {}

        return {
            'status': 'SUCCESS',
            'parameters': {
                'budget': budget,
                'priority_goal': priority_goal,
                'total_eligible_population': len(self.features_df)
            },
            'summary': {
                'targeted_customers_count': total_targeted,
                'budget_utilized': total_cost_allocated,
                'budget_remaining': round(budget - total_cost_allocated, 2),
                'estimated_incremental_revenue': total_projected_incremental_value,
                'expected_portfolio_roi': portfolio_roi,
                'intervention_breakdown': action_counts
            },
            'comparison_with_baseline': blanket_comparison,
            'targeted_customers': allocated[:50],  # Top 50 in return payload
            'action_catalog': catalog,
            'uplift_disclaimer': UPLIFT_DISCLAIMER
        }

    def _select_best_action(self, row: pd.Series, goal: str) -> Tuple[str, str]:
        """Maps customer profile to best action and plain-English rational."""
        seg = row['customer_segment']
        rec = row['recency_days']
        ltv = row['predicted_ltv_12m']
        margin = row['profit_margin']

        if margin < 0:
            return 'MARGIN_DISCIPLINE', f"Negative margin ({margin:.1f}%). Restructure discount terms to protect unit economics."

        if goal == 'PREVENT_CHURN_HIGH_LTV':
            if ltv >= 1500 and rec >= 90:
                return 'WINBACK_PROMO', f"High-LTV account (${ltv:,.0f}) dormant for {rec} days. Critical churn prevention."
            elif seg == 'Champions':
                return 'VIP_CONCIERGE', f"Premier VIP customer. Protect relationship with executive concierge."
            elif seg in ['At Risk', 'Need Attention']:
                return 'WINBACK_PROMO', f"Engagement slipping ({rec} days inactive). Re-engage before complete churn."
            else:
                return 'CATEGORY_CROSS_SELL', "Maintain engagement with cross-category product discovery."

        elif goal == 'MAXIMIZE_EXPECTED_ROI':
            if seg in ['New Customers', 'Potential Loyalists']:
                return 'ONBOARDING_NURTURE', "High conversion lift per dollar: secure 2nd purchase to maximize customer lifetime ROI."
            elif seg in ['Loyal Customers', 'Promising']:
                return 'CATEGORY_CROSS_SELL', "Strong purchase frequency: drive basket size via category expansion."
            elif seg == 'Champions':
                return 'VIP_CONCIERGE', "Solid recurring revenue: reinforce VIP loyalty."
            else:
                return 'WINBACK_PROMO', "Reactivate lapsed account to restore revenue baseline."

        elif goal == 'WINBACK_DORMANT':
            if rec >= 120:
                return 'WINBACK_PROMO', f"Dormant account ({rec} days inactive). Target with reactivation offer."
            else:
                return 'CATEGORY_CROSS_SELL', "Active account: stimulate purchase frequency."

        else: # NURTURE_POTENTIAL_LOYALISTS
            if seg in ['Potential Loyalists', 'New Customers', 'Promising']:
                return 'ONBOARDING_NURTURE', "High potential account: guide into core loyal customer tier."
            elif seg == 'Champions':
                return 'VIP_CONCIERGE', "Maintain VIP relationship."
            else:
                return 'CATEGORY_CROSS_SELL', "Broaden catalog purchase exposure."

    def _compute_uplift_responsiveness(self, row: pd.Series) -> float:
        """
        Estimates customer responsiveness factor (0.5 to 1.5)
        based on historical discount sensitivity and activity recency.
        """
        # More recent customers and moderately discount-sensitive customers respond better
        rec_factor = 1.2 if row['recency_days'] < 90 else (0.9 if row['recency_days'] < 200 else 0.6)
        disc_factor = 1.1 if (0.05 <= row['avg_discount'] <= 0.25) else 0.85
        return float(np.clip(rec_factor * disc_factor, 0.4, 1.6))

    def _compute_priority_score(self, row: pd.Series, goal: str, expected_roi: float) -> float:
        """Calculates normalized sorting score based on active goal."""
        ltv = row['predicted_ltv_12m']
        rec = row['recency_days']

        if goal == 'PREVENT_CHURN_HIGH_LTV':
            # Weight: High LTV * High Recency Risk
            churn_risk_weight = min(rec / 180.0, 2.0)
            return float(ltv * (1.0 + churn_risk_weight))

        elif goal == 'MAXIMIZE_EXPECTED_ROI':
            return float(expected_roi * 100.0 + ltv * 0.05)

        elif goal == 'WINBACK_DORMANT':
            return float(rec * 5.0 + ltv * 0.1)

        else: # NURTURE_POTENTIAL_LOYALISTS
            loyalty_potential = 2.0 if row['customer_segment'] in ['Potential Loyalists', 'New Customers'] else 0.8
            return float(ltv * loyalty_potential)

    def _benchmark_against_blanket(self, budget: float, catalog: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates uniform blanket outreach to quantify prioritized ROI uplift."""
        # Blanket campaign: average cost $20, uniform lower lift (6%)
        blanket_cost = 20.0
        blanket_targeted = int(budget // blanket_cost)
        avg_ltv = self.features_df['predicted_ltv_12m'].mean()
        blanket_incremental = round(blanket_targeted * avg_ltv * 0.06, 2)
        blanket_roi = round((blanket_incremental - (blanket_targeted * blanket_cost)) / max(blanket_targeted * blanket_cost, 1.0), 2)

        return {
            'strategy_name': 'Blanket Untargeted Outreach (Uniform Random Blast)',
            'targeted_count': blanket_targeted,
            'cost': round(blanket_targeted * blanket_cost, 2),
            'estimated_incremental_revenue': blanket_incremental,
            'estimated_roi': blanket_roi,
            'narrative': (
                f"A uniform blanket campaign would contact {blanket_targeted} accounts indiscriminately, "
                f"yielding an estimated ${blanket_incremental:,.2f} incremental return (ROI {blanket_roi}x). "
                f"Prioritized allocation delivers significantly higher capital efficiency by concentrating spend on accounts with the highest elasticity."
            )
        }

