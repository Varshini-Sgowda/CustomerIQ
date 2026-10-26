"""
CustomerIQ Data Loader and Schema Validation Module
Handles robust loading, schema verification, type conversion, and temporal splitting.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple, Optional


REQUIRED_COLUMNS = [
    'Order ID', 'Order Date', 'Customer ID', 'Customer Name',
    'Segment', 'Sales', 'Profit', 'Quantity', 'Discount',
    'Category', 'Sub-Category', 'Region'
]


class DataLoader:
    def __init__(self, filepath: Optional[str] = None):
        if filepath is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(base_dir, 'data', 'Superstore_Cleaned.csv')
        self.filepath = filepath
        self.raw_df: Optional[pd.DataFrame] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.schema_warnings: list = []
        self.metadata: Dict[str, Any] = {}

    def load_data(self) -> pd.DataFrame:
        """Loads and validates the transaction dataset."""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Dataset file not found at: {self.filepath}")

        df = pd.read_csv(self.filepath)
        self.raw_df = df.copy()

        # Validate schema
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Dataset is missing required columns: {missing_cols}")

        # Check for missing values in core identifier columns
        null_cust = df['Customer ID'].isnull().sum()
        if null_cust > 0:
            self.schema_warnings.append(f"Dropped {null_cust} records with missing Customer ID.")
            df = df.dropna(subset=['Customer ID'])

        # Type conversions
        df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
        invalid_dates = df['Order Date'].isnull().sum()
        if invalid_dates > 0:
            self.schema_warnings.append(f"Dropped {invalid_dates} records with unparseable Order Date.")
            df = df.dropna(subset=['Order Date'])

        # Numeric conversions and validations
        for num_col in ['Sales', 'Profit', 'Quantity', 'Discount']:
            df[num_col] = pd.to_numeric(df[num_col], errors='coerce').fillna(0.0)

        # String cleanups
        for str_col in ['Customer ID', 'Customer Name', 'Segment', 'Category', 'Sub-Category', 'Region']:
            df[str_col] = df[str_col].astype(str).str.strip()

        # Sort chronologically
        df = df.sort_values('Order Date').reset_index(drop=True)
        self.cleaned_df = df

        # Compute metadata
        min_date = df['Order Date'].min()
        max_date = df['Order Date'].max()
        days_span = (max_date - min_date).days

        self.metadata = {
            'filepath': self.filepath,
            'total_transactions': int(len(df)),
            'unique_customers': int(df['Customer ID'].nunique()),
            'unique_orders': int(df['Order ID'].nunique()),
            'total_sales': float(round(df['Sales'].sum(), 2)),
            'total_profit': float(round(df['Profit'].sum(), 2)),
            'overall_margin_pct': float(round((df['Profit'].sum() / max(df['Sales'].sum(), 1e-6)) * 100, 2)),
            'date_min': min_date.strftime('%Y-%m-%d'),
            'date_max': max_date.strftime('%Y-%m-%d'),
            'days_span': int(days_span),
            'segments': sorted(df['Segment'].unique().tolist()),
            'categories': sorted(df['Category'].unique().tolist()),
            'regions': sorted(df['Region'].unique().tolist()),
            'warnings': self.schema_warnings
        }
        return df

    def get_temporal_split(self, cutoff_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits transactions into an observation window (before cutoff)
        and an outcome window (after cutoff) for leakage-free model training.
        Defaults to 1 year before the maximum date in the dataset.
        """
        if self.cleaned_df is None:
            self.load_data()

        df = self.cleaned_df
        if cutoff_date is None:
            max_dt = df['Order Date'].max()
            # Default cutoff is 1 year before end: 365 days
            cutoff_dt = max_dt - pd.Timedelta(days=365)
        else:
            cutoff_dt = pd.to_datetime(cutoff_date)

        obs_df = df[df['Order Date'] <= cutoff_dt].copy()
        target_df = df[df['Order Date'] > cutoff_dt].copy()

        return obs_df, target_df

