# feature_engineer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from technical_indicators import TechnicalIndicatorCalculator

class FeatureEngineer:
    """
    Feature engineering for financial and text data.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.technical_calculator = TechnicalIndicatorCalculator(config)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive feature engineering to dataset.
        """
        result_df = df.copy()
        
        print(f"   🔧 Starting with {result_df.shape[1]} features")
        
        # 1. Technical indicators for stock data
        result_df = self._add_technical_indicators(result_df)
        
        # 2. Time-based features
        result_df = self._add_time_features(result_df)
        
        # 3. Statistical features
        result_df = self._add_statistical_features(result_df)
        
        # 4. Interaction features
        result_df = self._add_interaction_features(result_df)
        
        # 5. Text-based features (if text columns exist)
        result_df = self._add_text_features(result_df)
        
        # 6. Lag features
        result_df = self._add_lag_features(result_df)
        
        # 7. Rolling statistics
        result_df = self._add_rolling_features(result_df)
        
        print(f"   ✅ Finished with {result_df.shape[1]} features")
        print(f"   📈 Added {result_df.shape[1] - df.shape[1]} new features")
        
        return result_df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for stock data."""
        result_df = df.copy()
        
        # Check if this looks like stock data
        price_cols = [col for col in df.columns if any(x in col.lower() for x in ['close', 'price', 'adj_close'])]
        volume_cols = [col for col in df.columns if 'volume' in col.lower()]
        
        if price_cols and len(df) > 50:  # Only add if we have sufficient data
            price_col = price_cols[0]
            volume_col = volume_cols[0] if volume_cols else None
            
            try:
                result_df = self.technical_calculator.calculate_all_indicators(
                    result_df, price_col=price_col, volume_col=volume_col
                )
                print(f"   📈 Added technical indicators using {price_col}")
            except Exception as e:
                print(f"   ⚠️  Could not add technical indicators: {e}")
        
        return result_df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        result_df = df.copy()
        
        # Date features
        date_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time', 'created', 'published'])]
        
        for date_col in date_cols[:1]:  # Use first date column
            try:
                if df[date_col].dtype == 'object':
                    # Try to parse as datetime
                    result_df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                
                if pd.api.types.is_datetime64_any_dtype(result_df[date_col]):
                    result_df[f'{date_col}_year'] = result_df[date_col].dt.year
                    result_df[f'{date_col}_month'] = result_df[date_col].dt.month
                    result_df[f'{date_col}_day'] = result_df[date_col].dt.day
                    result_df[f'{date_col}_dayofweek'] = result_df[date_col].dt.dayofweek
                    result_df[f'{date_col}_quarter'] = result_df[date_col].dt.quarter
                    result_df[f'{date_col}_is_weekend'] = (result_df[date_col].dt.dayofweek >= 5).astype(int)
                    print(f"   📅 Added time features from {date_col}")
            except Exception as e:
                print(f"   ⚠️  Could not add time features from {date_col}: {e}")
        
        return result_df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        result_df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        stats_added = 0
        for col in numeric_cols[:10]:  # Limit to first 10 columns to avoid too many features
            try:
                # Z-score normalization
                mean_val = df[col].mean()
                std_val = df[col].std()
                if std_val > 0:
                    result_df[f'{col}_zscore'] = (df[col] - mean_val) / std_val
                    stats_added += 1
                
                # Log transformation (for positive values)
                if (df[col] > 0).all():
                    result_df[f'{col}_log'] = np.log(df[col])
                    stats_added += 1
                
                # Square root transformation (for non-negative values)
                if (df[col] >= 0).all():
                    result_df[f'{col}_sqrt'] = np.sqrt(df[col])
                    stats_added += 1
                    
            except:
                pass
        
        if stats_added > 0:
            print(f"   📊 Added {stats_added} statistical transformations")
        
        return result_df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between important columns."""
        result_df = df.copy()
        
        # Look for pairs of columns that might have meaningful interactions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        interactions_added = 0
        # Create some basic interactions between top correlated columns
        if len(numeric_cols) >= 2:
            try:
                # Use first few numeric columns for interactions
                for i, col1 in enumerate(numeric_cols[:3]):
                    for col2 in numeric_cols[i+1:4]:  # Limit to avoid too many features
                        result_df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                        result_df[f'{col1}_div_{col2}'] = df[col1] / df[col2].replace(0, np.nan)
                        interactions_added += 2
            except Exception as e:
                print(f"   ⚠️  Could not add interaction features: {e}")
        
        if interactions_added > 0:
            print(f"   🔄 Added {interactions_added} interaction features")
        
        return result_df
    
    def _add_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add text-based features if text columns exist."""
        result_df = df.copy()
        
        text_cols = [col for col in df.columns if df[col].dtype == 'object' and 
                   any(x in col.lower() for x in ['title', 'content', 'text', 'summary', 'body'])]
        
        text_features_added = 0
        for text_col in text_cols[:2]:  # Limit to first 2 text columns
            try:
                # Basic text statistics
                result_df[f'{text_col}_length'] = df[text_col].str.len().fillna(0)
                result_df[f'{text_col}_word_count'] = df[text_col].str.split().str.len().fillna(0)
                result_df[f'{text_col}_has_question'] = df[text_col].str.contains('\?').fillna(False).astype(int)
                result_df[f'{text_col}_has_exclamation'] = df[text_col].str.contains('!').fillna(False).astype(int)
                text_features_added += 4
                print(f"   📝 Added text features from {text_col}")
            except Exception as e:
                print(f"   ⚠️  Could not add text features from {text_col}: {e}")
        
        return result_df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag features for time series data."""
        result_df = df.copy()
        
        # Only add lag features if data is sorted and has sufficient length
        if len(df) > 100:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            lag_features_added = 0
            for col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                try:
                    result_df[f'{col}_lag_1'] = df[col].shift(1)
                    result_df[f'{col}_lag_3'] = df[col].shift(3)
                    result_df[f'{col}_lag_7'] = df[col].shift(7)
                    
                    # Momentum features
                    result_df[f'{col}_momentum_1'] = df[col] - df[col].shift(1)
                    result_df[f'{col}_momentum_3'] = df[col] - df[col].shift(3)
                    lag_features_added += 5
                except:
                    pass
            
            if lag_features_added > 0:
                print(f"   ⏰ Added {lag_features_added} lag and momentum features")
        
        return result_df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics."""
        result_df = df.copy()
        
        if len(df) > 50:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            rolling_features_added = 0
            for col in numeric_cols[:2]:  # Limit to first 2 numeric columns
                try:
                    windows = [5, 10, 20]
                    for window in windows:
                        result_df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                        result_df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                        result_df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
                        result_df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
                        rolling_features_added += 4
                except:
                    pass
            
            if rolling_features_added > 0:
                print(f"   📊 Added {rolling_features_added} rolling statistics")
        
        return result_df
