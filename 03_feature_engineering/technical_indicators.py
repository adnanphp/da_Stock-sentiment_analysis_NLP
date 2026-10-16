# technical_indicators.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

class TechnicalIndicatorCalculator:
    """
    Calculate technical indicators for financial data.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def calculate_all_indicators(self, df: pd.DataFrame, price_col: str = 'Close', 
                               volume_col: str = 'Volume') -> pd.DataFrame:
        """
        Calculate all technical indicators for a dataframe.
        """
        result_df = df.copy()
        
        # Price-based indicators
        result_df = self.calculate_moving_averages(result_df, price_col)
        result_df = self.calculate_rsi(result_df, price_col)
        result_df = self.calculate_macd(result_df, price_col)
        result_df = self.calculate_bollinger_bands(result_df, price_col)
        result_df = self.calculate_price_volatility(result_df, price_col)
        
        # Volume-based indicators
        if volume_col in df.columns:
            result_df = self.calculate_volume_indicators(result_df, price_col, volume_col)
        
        # Support/Resistance
        result_df = self.calculate_support_resistance(result_df, price_col)
        
        return result_df
    
    def calculate_moving_averages(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Calculate various moving averages."""
        result_df = df.copy()
        
        # Simple Moving Averages
        periods = [5, 10, 20, 50, 200]
        for period in periods:
            result_df[f'SMA_{period}'] = result_df[price_col].rolling(window=period).mean()
            result_df[f'EMA_{period}'] = result_df[price_col].ewm(span=period, adjust=False).mean()
        
        # Moving average crossovers
        result_df['SMA_20_vs_50'] = result_df['SMA_20'] - result_df['SMA_50']
        result_df['SMA_50_vs_200'] = result_df['SMA_50'] - result_df['SMA_200']
        
        # Price vs moving averages
        result_df['Price_vs_SMA_20'] = result_df[price_col] / result_df['SMA_20']
        result_df['Price_vs_SMA_50'] = result_df[price_col] / result_df['SMA_50']
        result_df['Price_vs_SMA_200'] = result_df[price_col] / result_df['SMA_200']
        
        return result_df
    
    def calculate_rsi(self, df: pd.DataFrame, price_col: str, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        result_df = df.copy()
        
        delta = result_df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        result_df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI-based signals
        result_df['RSI_oversold'] = (result_df[f'RSI_{period}'] < 30).astype(int)
        result_df['RSI_overbought'] = (result_df[f'RSI_{period}'] > 70).astype(int)
        
        return result_df
    
    def calculate_macd(self, df: pd.DataFrame, price_col: str, 
                      fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD indicator."""
        result_df = df.copy()
        
        ema_fast = result_df[price_col].ewm(span=fast, adjust=False).mean()
        ema_slow = result_df[price_col].ewm(span=slow, adjust=False).mean()
        
        result_df['MACD'] = ema_fast - ema_slow
        result_df['MACD_signal'] = result_df['MACD'].ewm(span=signal, adjust=False).mean()
        result_df['MACD_histogram'] = result_df['MACD'] - result_df['MACD_signal']
        
        # MACD signals
        result_df['MACD_cross_above'] = (result_df['MACD'] > result_df['MACD_signal']).astype(int)
        result_df['MACD_cross_below'] = (result_df['MACD'] < result_df['MACD_signal']).astype(int)
        
        return result_df
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, price_col: str, 
                                period: int = 20, std: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        result_df = df.copy()
        
        sma = result_df[price_col].rolling(window=period).mean()
        rolling_std = result_df[price_col].rolling(window=period).std()
        
        result_df['BB_upper'] = sma + (rolling_std * std)
        result_df['BB_lower'] = sma - (rolling_std * std)
        result_df['BB_middle'] = sma
        
        # Bollinger Band position
        result_df['BB_position'] = (result_df[price_col] - result_df['BB_lower']) / (
            result_df['BB_upper'] - result_df['BB_lower'])
        
        # Band width
        result_df['BB_width'] = (result_df['BB_upper'] - result_df['BB_lower']) / result_df['BB_middle']
        
        return result_df
    
    def calculate_price_volatility(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """Calculate price volatility measures."""
        result_df = df.copy()
        
        # Daily returns
        result_df['daily_return'] = result_df[price_col].pct_change()
        result_df['log_return'] = np.log(result_df[price_col] / result_df[price_col].shift(1))
        
        # Volatility (standard deviation of returns)
        result_df['volatility_5d'] = result_df['daily_return'].rolling(window=5).std()
        result_df['volatility_20d'] = result_df['daily_return'].rolling(window=20).std()
        
        return result_df
    
    def calculate_volume_indicators(self, df: pd.DataFrame, price_col: str, volume_col: str) -> pd.DataFrame:
        """Calculate volume-based indicators."""
        result_df = df.copy()
        
        # Volume moving averages
        result_df['volume_SMA_5'] = result_df[volume_col].rolling(window=5).mean()
        result_df['volume_SMA_20'] = result_df[volume_col].rolling(window=20).mean()
        
        # Volume vs average
        result_df['volume_ratio_5'] = result_df[volume_col] / result_df['volume_SMA_5']
        result_df['volume_ratio_20'] = result_df[volume_col] / result_df['volume_SMA_20']
        
        return result_df
    
    def calculate_support_resistance(self, df: pd.DataFrame, price_col: str, 
                                   window: int = 20) -> pd.DataFrame:
        """Calculate support and resistance levels."""
        result_df = df.copy()
        
        # Rolling highs and lows
        result_df['rolling_high'] = result_df[price_col].rolling(window=window).max()
        result_df['rolling_low'] = result_df[price_col].rolling(window=window).min()
        
        # Distance from support/resistance
        result_df['dist_from_resistance'] = (result_df['rolling_high'] - result_df[price_col]) / result_df[price_col]
        result_df['dist_from_support'] = (result_df[price_col] - result_df['rolling_low']) / result_df[price_col]
        
        return result_df
