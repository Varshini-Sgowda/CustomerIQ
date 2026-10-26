"""
CustomerIQ LTV Prediction & Explainability Engine
Trains a leak-free Machine Learning model predicting 12-month future Customer Lifetime Value (LTV),
computes SHAP TreeExplainer feature attributions, and constructs defensible prediction intervals.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import KFold
import shap

from customeriq.data_loader import DataLoader
from customeriq.feature_pipeline import FeaturePipeline, FEATURE_COLUMNS


class LTVModelEngine:
    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(base_dir, 'data', 'models')
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

        self.model: Optional[GradientBoostingRegressor] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_columns = FEATURE_COLUMNS.copy()
        self.metadata: Dict[str, Any] = {}
        self.residual_std: float = 0.0
        self.residual_quantiles: Tuple[float, float] = (-150.0, 150.0)
        self.is_trained: bool = False

    def train(self, data_loader: DataLoader, cutoff_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Trains the LTV model on a leak-free temporal train/test split.
        Uses observation window before cutoff to generate features,
        and the 365-day window after cutoff as actual ground truth future revenue.
        """
        obs_df, target_df = data_loader.get_temporal_split(cutoff_date)
        ref_date = obs_df['Order Date'].max()

        pipeline = FeaturePipeline(reference_date=ref_date)
        train_features_df = pipeline.transform(obs_df, ref_date=ref_date)

        # Build ground-truth target: sum of sales in target window per customer
        target_spend = target_df.groupby('Customer ID')['Sales'].sum().to_dict()
        train_features_df['target_future_ltv'] = train_features_df['customer_id'].map(
            lambda cid: target_spend.get(cid, 0.0)
        )

        X = train_features_df[self.feature_columns].fillna(0.0).values
        y = train_features_df['target_future_ltv'].values

        # 5-Fold Cross-Validation for robust metric tracking
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_mae, cv_rmse, cv_r2 = [], [], []
        cv_residuals = []

        for train_idx, val_idx in kf.split(X):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            fold_model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.85,
                random_state=42
            )
            fold_model.fit(X_tr, y_tr)
            preds_val = fold_model.predict(X_val)

            cv_mae.append(mean_absolute_error(y_val, preds_val))
            cv_rmse.append(root_mean_squared_error(y_val, preds_val))
            cv_r2.append(r2_score(y_val, preds_val))
            cv_residuals.extend((y_val - preds_val).tolist())

        # Fit final production model on full training set
        self.model = GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.85,
            random_state=42
        )
        self.model.fit(X, y)

        # Residuals for empirical prediction intervals (10th and 90th percentiles: 80% confidence)
        res_arr = np.array(cv_residuals)
        self.residual_std = float(np.std(res_arr))
        self.residual_quantiles = (
            float(np.percentile(res_arr, 10)),
            float(np.percentile(res_arr, 90))
        )

        # Fit SHAP Explainer
        self.explainer = shap.TreeExplainer(self.model)

        # Global feature importance
        importances = self.model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        feature_importance_list = [
            {'feature': self.feature_columns[i], 'importance': float(round(importances[i], 4))}
            for i in sorted_indices
        ]

        self.metadata = {
            'algorithm': 'Gradient Boosting Regressor (Ensemble Trees)',
            'hyperparameters': {
                'n_estimators': 120,
                'learning_rate': 0.05,
                'max_depth': 4,
                'subsample': 0.85
            },
            'trained_samples': int(len(train_features_df)),
            'target_variable': '12-Month Future Customer Revenue ($)',
            'prediction_horizon_months': 12,
            'cv_mean_mae': float(round(np.mean(cv_mae), 2)),
            'cv_mean_rmse': float(round(np.mean(cv_rmse), 2)),
            'cv_mean_r2': float(round(np.mean(cv_r2), 3)),
            'residual_std': float(round(self.residual_std, 2)),
            'prediction_interval_method': 'Empirical CV Residual Quantiles (80% Interval [P10, P90])',
            'top_features': feature_importance_list[:8],
            'all_feature_importance': feature_importance_list
        }
        self.is_trained = True
        self.save_model()
        return self.metadata

    def predict_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs inference on customer feature table.
        Adds 'predicted_ltv_12m', 'ltv_lower_bound', 'ltv_upper_bound'.
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model is not trained. Call train() or load_model() first.")

        X = features_df[self.feature_columns].fillna(0.0).values
        raw_preds = self.model.predict(X)
        preds = np.maximum(raw_preds, 0.0)

        # Defensible empirical prediction intervals
        q_low, q_high = self.residual_quantiles
        lower_bounds = np.maximum(raw_preds + q_low, 0.0)
        upper_bounds = np.maximum(raw_preds + q_high, lower_bounds)

        result_df = features_df.copy()
        result_df['predicted_ltv_12m'] = np.round(preds, 2)
        result_df['ltv_lower_bound'] = np.round(lower_bounds, 2)
        result_df['ltv_upper_bound'] = np.round(upper_bounds, 2)
        return result_df

    def explain_customer(self, customer_features: pd.Series) -> Dict[str, Any]:
        """
        Computes local SHAP waterfall feature attributions for a single customer.
        """
        if not self.is_trained or self.explainer is None:
            raise RuntimeError("Explainer is not initialized. Train or load model first.")

        feature_vals = [customer_features.get(col, 0.0) for col in self.feature_columns]
        X_single = np.array(feature_vals, dtype=float).reshape(1, -1)

        pred_val = float(max(self.model.predict(X_single)[0], 0.0))
        shap_values = self.explainer.shap_values(X_single)[0]
        base_val = self.explainer.expected_value
        if isinstance(base_val, (np.ndarray, list)):
            base_value = float(np.ravel(base_val)[0])
        else:
            base_value = float(base_val)

        contributions = []
        for col, val, shap_val in zip(self.feature_columns, feature_vals, shap_values):
            contributions.append({
                'feature': col,
                'feature_value': float(round(val, 2)) if isinstance(val, (int, float)) else str(val),
                'shap_value': float(round(shap_val, 2)),
                'direction': 'positive' if shap_val > 0 else 'negative',
                'impact_magnitude': float(round(abs(shap_val), 2))
            })

        # Sort by absolute impact
        contributions.sort(key=lambda x: x['impact_magnitude'], reverse=True)

        q_low, q_high = self.residual_quantiles
        return {
            'predicted_ltv': float(round(pred_val, 2)),
            'base_value': float(round(base_value, 2)),
            'lower_bound': float(round(max(pred_val + q_low, 0.0), 2)),
            'upper_bound': float(round(max(pred_val + q_high, pred_val), 2)),
            'top_positive_drivers': [c for c in contributions if c['shap_value'] > 0][:5],
            'top_negative_drivers': [c for c in contributions if c['shap_value'] < 0][:5],
            'all_contributions': contributions
        }

    def save_model(self):
        """Serializes trained model and metadata."""
        model_path = os.path.join(self.model_dir, 'ltv_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'metadata': self.metadata,
                'residual_std': self.residual_std,
                'residual_quantiles': self.residual_quantiles,
                'feature_columns': self.feature_columns
            }, f)

    def load_model(self) -> bool:
        """Loads serialized model if available."""
        model_path = os.path.join(self.model_dir, 'ltv_model.pkl')
        if not os.path.exists(model_path):
            return False
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.metadata = data['metadata']
            self.residual_std = data['residual_std']
            self.residual_quantiles = data['residual_quantiles']
            self.feature_columns = data['feature_columns']
            self.explainer = shap.TreeExplainer(self.model)
            self.is_trained = True
        return True
