# time_feature_creator.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

class TimeFeatureCreator:
    """
    Create time-based features from datetime columns.
    Supports temporal, cyclical, and lagged features.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def create_time_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Create time-based features from datetime columns.
        """
        time_df = df.copy()
        details = {'features_created': 0, 'datetime_columns_processed': []}
        
        datetime_columns = config.get('datetime_columns', [])
        feature_types = config.get('feature_types', ['temporal'])
        
        for datetime_column in datetime_columns:
            if datetime_column not in df.columns:
                continue
            
            print(f"    ⏰ Creating time features from: {datetime_column}")
            details['datetime_columns_processed'].append(datetime_column)
            
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(time_df[datetime_column]):
                time_df[datetime_column] = pd.to_datetime(time_df[datetime_column], errors='coerce')
            
            # Extract features based on types
            for feature_type in feature_types:
                try:
                    if feature_type == 'temporal':
                        time_df, count = self._create_temporal_features(time_df, datetime_column)
                        details['features_created'] += count
                    
                    elif feature_type == 'cyclical':
                        time_df, count = self._create_cyclical_features(time_df, datetime_column)
                        details['features_created'] += count
                    
                    elif feature_type == 'lagged':
                        time_df, count = self._create_lagged_features(time_df, datetime_column, config)
                        details['features_created'] += count
                    
                    elif feature_type == 'rolling':
                        time_df, count = self._create_rolling_features(time_df, datetime_column, config)
                        details['features_created'] += count
                    
                    print(f"      ✅ {feature_type}: {count} features")
                    
                except Exception as e:
                    print(f"      ❌ Error creating {feature_type} from {datetime_column}: {e}")
        
        return time_df, details
    
    def _create_temporal_features(self, df: pd.DataFrame, datetime_column: str) -> Tuple[pd.DataFrame, int]:
        """Create basic temporal features from datetime."""
        dt_series = df[datetime_column]
        features_created = 0
        
        # Basic date components
        df[f'{datetime_column}_year'] = dt_series.dt.year
        df[f'{datetime_column}_month'] = dt_series.dt.month
        df[f'{datetime_column}_day'] = dt_series.dt.day
        df[f'{datetime_column}_dayofweek'] = dt_series.dt.dayofweek  # Monday=0, Sunday=6
        df[f'{datetime_column}_dayofyear'] = dt_series.dt.dayofyear
        df[f'{datetime_column}_week'] = dt_series.dt.isocalendar().week
        df[f'{datetime_column}_quarter'] = dt_series.dt.quarter
        features_created += 7
        
        # Time components (if available)
        if any(dt_series.dt.time != pd.Timestamp('00:00:00').time()):
            df[f'{datetime_column}_hour'] = dt_series.dt.hour
            df[f'{datetime_column}_minute'] = dt_series.dt.minute
            df[f'{datetime_column}_second'] = dt_series.dt.second
            features_created += 3
        
        # Boolean features for special days
        df[f'{datetime_column}_is_weekend'] = (dt_series.dt.dayofweek >= 5).astype(int)
        df[f'{datetime_column}_is_month_start'] = dt_series.dt.is_month_start.astype(int)
        df[f'{datetime_column}_is_month_end'] = dt_series.dt.is_month_end.astype(int)
        df[f'{datetime_column}_is_quarter_start'] = dt_series.dt.is_quarter_start.astype(int)
        df[f'{datetime_column}_is_quarter_end'] = dt_series.dt.is_quarter_end.astype(int)
        df[f'{datetime_column}_is_year_start'] = dt_series.dt.is_year_start.astype(int)
        df[f'{datetime_column}_is_year_end'] = dt_series.dt.is_year_end.astype(int)
        features_created += 7
        
        return df, features_created
    
    def _create_cyclical_features(self, df: pd.DataFrame, datetime_column: str) -> Tuple[pd.DataFrame, int]:
        """Create cyclical features for time components."""
        dt_series = df[datetime_column]
        features_created = 0
        
        # Cyclical encoding for periodic features
        def create_cyclical_features(series, period):
            sin_component = np.sin(2 * np.pi * series / period)
            cos_component = np.cos(2 * np.pi * series / period)
            return sin_component, cos_component
        
        # Month cyclical (12-month cycle)
        month_sin, month_cos = create_cyclical_features(dt_series.dt.month, 12)
        df[f'{datetime_column}_month_sin'] = month_sin
        df[f'{datetime_column}_month_cos'] = month_cos
        features_created += 2
        
        # Day of month cyclical (approx 30-day cycle)
        day_sin, day_cos = create_cyclical_features(dt_series.dt.day, 30)
        df[f'{datetime_column}_day_sin'] = day_sin
        df[f'{datetime_column}_day_cos'] = day_cos
        features_created += 2
        
        # Day of week cyclical (7-day cycle)
        dow_sin, dow_cos = create_cyclical_features(dt_series.dt.dayofweek, 7)
        df[f'{datetime_column}_dayofweek_sin'] = dow_sin
        df[f'{datetime_column}_dayofweek_cos'] = dow_cos
        features_created += 2
        
        # Day of year cyclical (365-day cycle)
        doy_sin, doy_cos = create_cyclical_features(dt_series.dt.dayofyear, 365)
        df[f'{datetime_column}_dayofyear_sin'] = doy_sin
        df[f'{datetime_column}_dayofyear_cos'] = doy_cos
        features_created += 2
        
        # Hour cyclical (if time component exists)
        if any(dt_series.dt.time != pd.Timestamp('00:00:00').time()):
            hour_sin, hour_cos = create_cyclical_features(dt_series.dt.hour, 24)
            df[f'{datetime_column}_hour_sin'] = hour_sin
            df[f'{datetime_column}_hour_cos'] = hour_cos
            features_created += 2
        
        return df, features_created
    
    def _create_lagged_features(self, df: pd.DataFrame, datetime_column: str, config: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
        """Create lagged features for time series data."""
        # Sort by datetime to ensure proper lagging
        if not df.index.is_monotonic_increasing:
            df = df.sort_values(by=datetime_column)
        
        numeric_columns = config.get('numeric_columns_for_lag', [])
        lag_periods = config.get('lag_periods', [1, 2, 3, 5, 7, 14, 21, 30])
        features_created = 0
        
        for numeric_col in numeric_columns:
            if numeric_col not in df.columns:
                continue
            
            for lag in lag_periods:
                df[f'{numeric_col}_lag_{lag}'] = df[numeric_col].shift(lag)
                features_created += 1
            
            # Lag differences (change from previous period)
            for lag in [1, 2, 3]:
                df[f'{numeric_col}_diff_{lag}'] = df[numeric_col] - df[numeric_col].shift(lag)
                features_created += 1
            
            # Percentage changes
            for lag in [1, 2, 3, 5]:
                df[f'{numeric_col}_pct_change_{lag}'] = df[numeric_col].pct_change(periods=lag)
                features_created += 1
        
        return df, features_created
    
    def _create_rolling_features(self, df: pd.DataFrame, datetime_column: str, config: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
        """Create rolling window features for time series data."""
        # Sort by datetime
        if not df.index.is_monotonic_increasing:
            df = df.sort_values(by=datetime_column)
        
        numeric_columns = config.get('numeric_columns_for_rolling', [])
        windows = config.get('rolling_windows', [3, 5, 7, 10, 14, 20, 30])
        features_created = 0
        
        for numeric_col in numeric_columns:
            if numeric_col not in df.columns:
                continue
            
            for window in windows:
                if len(df) >= window:
                    # Rolling statistics
                    df[f'{numeric_col}_rolling_mean_{window}'] = df[numeric_col].rolling(window=window).mean()
                    df[f'{numeric_col}_rolling_std_{window}'] = df[numeric_col].rolling(window=window).std()
                    df[f'{numeric_col}_rolling_min_{window}'] = df[numeric_col].rolling(window=window).min()
                    df[f'{numeric_col}_rolling_max_{window}'] = df[numeric_col].rolling(window=window).max()
                    
                    # Rolling quantiles
                    df[f'{numeric_col}_rolling_median_{window}'] = df[numeric_col].rolling(window=window).median()
                    df[f'{numeric_col}_rolling_q25_{window}'] = df[numeric_col].rolling(window=window).quantile(0.25)
                    df[f'{numeric_col}_rolling_q75_{window}'] = df[numeric_col].rolling(window=window).quantile(0.75)
                    
                    features_created += 7
                    
                    # Rolling comparisons
                    df[f'{numeric_col}_vs_rolling_mean_{window}'] = (
                        df[numeric_col] / df[f'{numeric_col}_rolling_mean_{window}'] - 1
                    )
                    df[f'{numeric_col}_zscore_{window}'] = (
                        (df[numeric_col] - df[f'{numeric_col}_rolling_mean_{window}']) / 
                        df[f'{numeric_col}_rolling_std_{window}']
                    )
                    features_created += 2
        
        return df, features_created
    
    def create_time_based_splits(self, df: pd.DataFrame, datetime_column: str, 
                               test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create time-based train/test splits."""
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_column]):
            df[datetime_column] = pd.to_datetime(df[datetime_column])
        
        df_sorted = df.sort_values(by=datetime_column)
        split_index = int(len(df_sorted) * (1 - test_size))
        
        train_df = df_sorted.iloc[:split_index]
        test_df = df_sorted.iloc[split_index:]
        
        return train_df, test_df
    
    def create_seasonal_features(self, df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
        """Create seasonal and holiday-related features."""
        dt_series = df[datetime_column]
        
        # Seasonal features
        df[f'{datetime_column}_season'] = (dt_series.dt.month % 12 + 3) // 3  # 1=Winter, 2=Spring, etc.
        
        # US Holiday approximations (simplified)
        def is_holiday(dt):
            month, day = dt.month, dt.day
            # New Year's
            if month == 1 and day == 1:
                return 1
            # Independence Day
            elif month == 7 and day == 4:
                return 1
            # Christmas
            elif month == 12 and day == 25:
                return 1
            return 0
        
        df[f'{datetime_column}_is_holiday'] = dt_series.apply(is_holiday)
        
        # Business days (Monday to Friday)
        df[f'{datetime_column}_is_business_day'] = (dt_series.dt.dayofweek < 5).astype(int)
        
        return df
