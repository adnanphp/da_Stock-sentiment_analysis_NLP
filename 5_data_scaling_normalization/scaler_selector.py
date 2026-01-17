# scaler_selector.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.exceptions import NotFittedError
import warnings
warnings.filterwarnings('ignore')

class ScalerSelector:
    """
    Intelligent scaler selection and configuration based on data characteristics.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.available_scalers = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler,
            'robust': RobustScaler,
            'maxabs': MaxAbsScaler,
            'power': PowerTransformer,
            'quantile': QuantileTransformer
        }
    
    def select_scaler(self, scaler_type: str, params: Dict[str, Any] = None) -> Any:
        """
        Select and configure appropriate scaler.
        """
        scaler_type = scaler_type.lower()
        params = params or {}
        
        if scaler_type not in self.available_scalers:
            print(f"    ⚠️  Unknown scaler type: {scaler_type}, using StandardScaler")
            scaler_type = 'standard'
        
        try:
            scaler_class = self.available_scalers[scaler_type]
            scaler = scaler_class(**params)
            
            print(f"    🔧 Selected {scaler_type} scaler with params: {params}")
            return scaler
            
        except Exception as e:
            print(f"    ❌ Error creating {scaler_type} scaler: {e}")
            # Fallback to StandardScaler
            return StandardScaler()
    
    def recommend_scaler(self, series: pd.Series, column_name: str = None) -> Dict[str, Any]:
        """
        Recommend the best scaler for a given data series.
        """
        # Basic statistics
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return {'scaler': 'minmax', 'reason': 'Empty series', 'params': {'feature_range': (0, 1)}}
        
        stats = {
            'min': clean_series.min(),
            'max': clean_series.max(),
            'mean': clean_series.mean(),
            'std': clean_series.std(),
            'skewness': clean_series.skew(),
            'kurtosis': clean_series.kurtosis(),
            'q1': clean_series.quantile(0.25),
            'q3': clean_series.quantile(0.75),
            'iqr': clean_series.quantile(0.75) - clean_series.quantile(0.25)
        }
        
        # Check for specific patterns
        column_lower = (column_name or '').lower()
        
        # Binary data
        if clean_series.nunique() == 2:
            return {
                'scaler': 'minmax',
                'reason': 'Binary data',
                'params': {'feature_range': (0, 1)}
            }
        
        # Percentage data (0-100 range)
        if 0 <= stats['min'] <= 100 and 0 <= stats['max'] <= 100:
            return {
                'scaler': 'minmax', 
                'reason': 'Percentage data',
                'params': {'feature_range': (0, 1)}
            }
        
        # Check for outliers using IQR method
        lower_bound = stats['q1'] - 1.5 * stats['iqr']
        upper_bound = stats['q3'] + 1.5 * stats['iqr']
        outliers = ((clean_series < lower_bound) | (clean_series > upper_bound)).sum()
        outlier_ratio = outliers / len(clean_series)
        
        # Highly skewed data
        if abs(stats['skewness']) > 2:
            return {
                'scaler': 'power',
                'reason': f'Highly skewed (skewness: {stats["skewness"]:.2f})',
                'params': {'method': 'yeo-johnson', 'standardize': True}
            }
        
        # Data with significant outliers
        if outlier_ratio > 0.05:  # More than 5% outliers
            return {
                'scaler': 'robust',
                'reason': f'Significant outliers ({outlier_ratio:.1%} of data)',
                'params': {'with_centering': True, 'with_scaling': True, 'quantile_range': (25.0, 75.0)}
            }
        
        # Data with negative values
        if stats['min'] < 0:
            return {
                'scaler': 'standard',
                'reason': 'Data contains negative values',
                'params': {'with_mean': True, 'with_std': True}
            }
        
        # Sparse data (many zeros)
        zero_ratio = (clean_series == 0).sum() / len(clean_series)
        if zero_ratio > 0.8:
            return {
                'scaler': 'maxabs',
                'reason': f'Sparse data ({zero_ratio:.1%} zeros)',
                'params': {}
            }
        
        # Normally distributed data
        if abs(stats['skewness']) < 1 and abs(stats['kurtosis']) < 3:
            return {
                'scaler': 'standard',
                'reason': 'Normally distributed data',
                'params': {'with_mean': True, 'with_std': True}
            }
        
        # Default to robust scaler for general use
        return {
            'scaler': 'robust',
            'reason': 'General purpose scaling',
            'params': {'with_centering': True, 'with_scaling': True, 'quantile_range': (25.0, 75.0)}
        }
    
    def analyze_dataset_scalers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze entire dataset and recommend scalers for each numeric column.
        """
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        recommendations = {}
        
        print("    📊 Analyzing scaler recommendations for dataset...")
        
        for column in numeric_columns:
            recommendation = self.recommend_scaler(df[column], column)
            recommendations[column] = recommendation
            
            print(f"      📈 {column}: {recommendation['scaler']} - {recommendation['reason']}")
        
        # Group columns by recommended scaler
        scaler_groups = {}
        for column, rec in recommendations.items():
            scaler_type = rec['scaler']
            if scaler_type not in scaler_groups:
                scaler_groups[scaler_type] = []
            scaler_groups[scaler_type].append(column)
        
        return {
            'individual_recommendations': recommendations,
            'scaler_groups': scaler_groups,
            'summary': f"Recommended {len(scaler_groups)} different scaler types"
        }
    
    def validate_scaler_fit(self, scaler: Any, data: np.ndarray) -> bool:
        """
        Validate that a scaler has been properly fitted.
        """
        try:
            # Try to access fitted attributes
            if hasattr(scaler, 'n_features_in_'):
                if scaler.n_features_in_ != data.shape[1]:
                    return False
            
            # Try to transform a small sample
            test_data = data[:1] if len(data) > 0 else data
            if len(test_data) > 0:
                scaler.transform(test_data)
            
            return True
            
        except (NotFittedError, AttributeError, ValueError) as e:
            print(f"    ⚠️  Scaler validation failed: {e}")
            return False
    
    def get_scaler_info(self, scaler: Any) -> Dict[str, Any]:
        """
        Get information about a fitted scaler.
        """
        info = {
            'scaler_type': type(scaler).__name__,
            'fitted': False,
            'parameters': {}
        }
        
        try:
            # Check if scaler is fitted
            if hasattr(scaler, 'n_features_in_'):
                info['fitted'] = True
                info['n_features'] = scaler.n_features_in_
            
            # Get scaler-specific parameters
            if hasattr(scaler, 'mean_') and scaler.mean_ is not None:
                info['mean'] = scaler.mean_.tolist()
            if hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                info['scale'] = scaler.scale_.tolist()
            if hasattr(scaler, 'min_') and scaler.min_ is not None:
                info['min'] = scaler.min_.tolist()
            if hasattr(scaler, 'data_min_') and scaler.data_min_ is not None:
                info['data_min'] = scaler.data_min_.tolist()
            if hasattr(scaler, 'data_max_') and scaler.data_max_ is not None:
                info['data_max'] = scaler.data_max_.tolist()
            if hasattr(scaler, 'lambdas_') and scaler.lambdas_ is not None:
                info['lambdas'] = scaler.lambdas_.tolist()
            
            # Get configuration parameters
            if hasattr(scaler, 'get_params'):
                info['parameters'] = scaler.get_params()
            
        except Exception as e:
            print(f"    ⚠️  Error getting scaler info: {e}")
        
        return info
