# technical_indicators.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicatorCalculator:
    """
    Calculator for financial technical indicators.
    Supports common indicators used in quantitative finance.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def add_technical_indicators(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Add technical indicators to dataframe.
        """
        indicator_df = df.copy()
        details = {'features_created': 0, 'indicators_applied': []}
        
        price_column = config.get('price_column', 'Close')
        volume_column = config.get('volume_column')
        indicators = config.get('indicators', [])
        
        if price_column not in df.columns:
            print(f"    ⚠️  Price column '{price_column}' not found")
            return indicator_df, details
        
        print(f"    📈 Calculating technical indicators for {price_column}...")
        
        # Ensure data is sorted by index (assuming time series)
        if not indicator_df.index.is_monotonic_increasing:
            indicator_df = indicator_df.sort_index()
        
        price_series = indicator_df[price_column]
        
        for indicator in indicators:
            try:
                if indicator == 'sma':
                    indicator_df, sma_count = self._add_sma_indicators(indicator_df, price_series)
                    details['features_created'] += sma_count
                    details['indicators_applied'].extend([f'SMA_{period}' for period in self.config.get('sma_periods', [20, 50, 200])])
                
                elif indicator == 'ema':
                    indicator_df, ema_count = self._add_ema_indicators(indicator_df, price_series)
                    details['features_created'] += ema_count
                    details['indicators_applied'].extend([f'EMA_{period}' for period in self.config.get('ema_periods', [12, 26])])
                
                elif indicator == 'rsi':
                    indicator_df = self._add_rsi(indicator_df, price_series)
                    details['features_created'] += 1
                    details['indicators_applied'].append('RSI')
                
                elif indicator == 'macd':
                    indicator_df, macd_count = self._add_macd(indicator_df, price_series)
                    details['features_created'] += macd_count
                    details['indicators_applied'].extend(['MACD', 'MACD_Signal', 'MACD_Histogram'])
                
                elif indicator == 'bollinger_bands':
                    indicator_df, bb_count = self._add_bollinger_bands(indicator_df, price_series)
                    details['features_created'] += bb_count
                    details['indicators_applied'].extend(['BB_Upper', 'BB_Lower', 'BB_Middle', 'BB_Width', 'BB_Position'])
                
                elif indicator == 'stochastic':
                    indicator_df, stoch_count = self._add_stochastic(indicator_df, config)
                    details['features_created'] += stoch_count
                    details['indicators_applied'].extend(['Stoch_K', 'Stoch_D'])
                
                elif indicator == 'volume_indicators' and volume_column:
                    indicator_df, volume_count = self._add_volume_indicators(indicator_df, volume_column)
                    details['features_created'] += volume_count
                    details['indicators_applied'].extend(['Volume_SMA', 'Volume_Ratio'])
                
                print(f"      ✅ {indicator}: {self._get_indicator_count(indicator)} features")
                
            except Exception as e:
                print(f"      ❌ Error calculating {indicator}: {e}")
        
        return indicator_df, details
    
    def _get_indicator_count(self, indicator: str) -> int:
        """Get number of features created by each indicator."""
        indicator_counts = {
            'sma': len(self.config.get('sma_periods', [20, 50, 200])),
            'ema': len(self.config.get('ema_periods', [12, 26])),
            'rsi': 1,
            'macd': 3,
            'bollinger_bands': 5,
            'stochastic': 2,
            'volume_indicators': 2
        }
        return indicator_counts.get(indicator, 0)
    
    def _add_sma_indicators(self, df: pd.DataFrame, price_series: pd.Series) -> Tuple[pd.DataFrame, int]:
        """Add Simple Moving Average indicators."""
        periods = self.config.get('sma_periods', [20, 50, 200])
        features_created = 0
        
        for period in periods:
            if len(price_series) >= period:
                df[f'SMA_{period}'] = price_series.rolling(window=period, min_periods=1).mean()
                
                # Add price relative to SMA
                df[f'Price_vs_SMA_{period}'] = (price_series / df[f'SMA_{period}'] - 1) * 100
                features_created += 2
        
        return df, features_created
    
    def _add_ema_indicators(self, df: pd.DataFrame, price_series: pd.Series) -> Tuple[pd.DataFrame, int]:
        """Add Exponential Moving Average indicators."""
        periods = self.config.get('ema_periods', [12, 26])
        features_created = 0
        
        for period in periods:
            if len(price_series) >= period:
                df[f'EMA_{period}'] = price_series.ewm(span=period, adjust=False).mean()
                features_created += 1
        
        return df, features_created
    
    def _add_rsi(self, df: pd.DataFrame, price_series: pd.Series, period: int = 14) -> pd.DataFrame:
        """Add Relative Strength Index."""
        if len(price_series) < period + 1:
            return df
        
        delta = price_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _add_macd(self, df: pd.DataFrame, price_series: pd.Series) -> Tuple[pd.DataFrame, int]:
        """Add Moving Average Convergence Divergence."""
        ema_12 = price_series.ewm(span=12, adjust=False).mean()
        ema_26 = price_series.ewm(span=26, adjust=False).mean()
        
        df['MACD'] = ema_12 - ema_26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        return df, 3
    
    def _add_bollinger_bands(self, df: pd.DataFrame, price_series: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.DataFrame, int]:
        """Add Bollinger Bands and related features."""
        if len(price_series) < period:
            return df, 0
        
        # Calculate Bollinger Bands
        df['BB_Middle'] = price_series.rolling(window=period).mean()
        bb_std = price_series.rolling(window=period).std()
        
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * std_dev)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * std_dev)
        
        # Additional Bollinger Band features
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (price_series - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        return df, 5
    
    def _add_stochastic(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, int]:
        """Add Stochastic Oscillator."""
        high_col = config.get('high_column', 'High')
        low_col = config.get('low_column', 'Low')
        close_col = config.get('price_column', 'Close')
        
        if not all(col in df.columns for col in [high_col, low_col, close_col]):
            return df, 0
        
        period = config.get('stochastic_period', 14)
        
        # Calculate highest high and lowest low
        highest_high = df[high_col].rolling(window=period).max()
        lowest_low = df[low_col].rolling(window=period).min()
        
        # Stochastic %K
        df['Stoch_K'] = ((df[close_col] - lowest_low) / (highest_high - lowest_low)) * 100
        
        # Stochastic %D (3-period SMA of %K)
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        return df, 2
    
    def _add_volume_indicators(self, df: pd.DataFrame, volume_column: str) -> Tuple[pd.DataFrame, int]:
        """Add volume-based indicators."""
        if volume_column not in df.columns:
            return df, 0
        
        volume_series = df[volume_column]
        
        # Volume SMA
        df['Volume_SMA_20'] = volume_series.rolling(window=20).mean()
        
        # Volume ratio (current volume vs average)
        df['Volume_Ratio'] = volume_series / df['Volume_SMA_20']
        
        return df, 2
    
    def calculate_momentum_indicators(self, df: pd.DataFrame, price_column: str) -> pd.DataFrame:
        """Calculate momentum-based indicators."""
        if price_column not in df.columns:
            return df
        
        price_series = df[price_column]
        
        # Rate of Change
        df['ROC_1'] = price_series.pct_change(periods=1) * 100
        df['ROC_5'] = price_series.pct_change(periods=5) * 100
        df['ROC_20'] = price_series.pct_change(periods=20) * 100
        
        # Price momentum
        df['Momentum_5'] = price_series / price_series.shift(5) - 1
        df['Momentum_20'] = price_series / price_series.shift(20) - 1
        
        return df
    
    def calculate_volatility_indicators(self, df: pd.DataFrame, price_column: str) -> pd.DataFrame:
        """Calculate volatility indicators."""
        if price_column not in df.columns:
            return df
        
        price_series = df[price_column]
        
        # Historical volatility (annualized)
        returns = price_series.pct_change()
        df['Volatility_20'] = returns.rolling(window=20).std() * np.sqrt(252) * 100
        df['Volatility_50'] = returns.rolling(window=50).std() * np.sqrt(252) * 100
        
        # True Range (requires High and Low)
        if all(col in df.columns for col in ['High', 'Low']):
            high_low = df['High'] - df['Low']
            high_close = (df['High'] - df[price_column].shift()).abs()
            low_close = (df['Low'] - df[price_column].shift()).abs()
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['ATR_14'] = true_range.rolling(window=14).mean()
        
        return df
