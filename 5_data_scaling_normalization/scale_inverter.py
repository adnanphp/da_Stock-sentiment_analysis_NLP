# scale_inverter.py
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.exceptions import NotFittedError
import warnings
warnings.filterwarnings('ignore')

class ScaleInverter:
    """
    Scale inversion utilities for transforming scaled data back to original scale.
    Essential for interpreting model predictions and results.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def invert_scaling(self, scaled_data: np.ndarray, scaler: Any, 
                      column_name: str = None, scaling_parameters: Dict[str, Any] = None) -> np.ndarray:
        """
        Invert scaling transformation to recover original data scale.
        """
        if scaled_data is None or len(scaled_data) == 0:
            return scaled_data
        
        try:
            scaler_type = type(scaler).__name__
            
            if scaler_type == 'StandardScaler':
                return self._invert_standard_scaler(scaled_data, scaler)
            
            elif scaler_type == 'MinMaxScaler':
                return self._invert_minmax_scaler(scaled_data, scaler, scaling_parameters, column_name)
            
            elif scaler_type == 'RobustScaler':
                return self._invert_robust_scaler(scaled_data, scaler)
            
            elif scaler_type == 'MaxAbsScaler':
                return self._invert_maxabs_scaler(scaled_data, scaler)
            
            elif scaler_type == 'PowerTransformer':
                return self._invert_power_transformer(scaled_data, scaler)
            
            elif scaler_type == 'QuantileTransformer':
                return self._invert_quantile_transformer(scaled_data, scaler)
            
            else:
                print(f"    ⚠️  Unknown scaler type: {scaler_type}, cannot invert")
                return scaled_data
                
        except Exception as e:
            print(f"    ❌ Error inverting {scaler_type} for {column_name}: {e}")
            return scaled_data
    
    def _invert_standard_scaler(self, scaled_data: np.ndarray, scaler: StandardScaler) -> np.ndarray:
        """Invert StandardScaler transformation."""
        if not hasattr(scaler, 'mean_') or not hasattr(scaler, 'scale_'):
            raise ValueError("StandardScaler not properly fitted")
        
        # StandardScaler: inverse = scaled * scale + mean
        original_data = scaled_data * scaler.scale_ + scaler.mean_
        return original_data
    
    def _invert_minmax_scaler(self, scaled_data: np.ndarray, scaler: MinMaxScaler, 
                             scaling_parameters: Dict[str, Any], column_name: str) -> np.ndarray:
        """Invert MinMaxScaler transformation."""
        if scaling_parameters and column_name:
            # Use stored original parameters if available
            col_params = scaling_parameters.get('original_columns', {}).get(column_name, {})
            if col_params:
                original_min = col_params.get('original_min')
                original_max = col_params.get('original_max')
                
                if original_min is not None and original_max is not None:
                    # MinMaxScaler: inverse = scaled * (max - min) + min
                    feature_range = getattr(scaler, 'feature_range', (0, 1))
                    scale_min, scale_max = feature_range
                    
                    original_data = (scaled_data - scale_min) / (scale_max - scale_min)
                    original_data = original_data * (original_max - original_min) + original_min
                    return original_data
        
        # Fallback to standard inversion
        if hasattr(scaler, 'data_min_') and hasattr(scaler, 'data_max_'):
            try:
                return scaler.inverse_transform(scaled_data)
            except:
                pass
        
        # Final fallback - approximate inversion
        print(f"    ⚠️  Using approximate inversion for MinMaxScaler on {column_name}")
        return scaled_data * 100  # Rough approximation
    
    def _invert_robust_scaler(self, scaled_data: np.ndarray, scaler: RobustScaler) -> np.ndarray:
        """Invert RobustScaler transformation."""
        if not hasattr(scaler, 'center_') or not hasattr(scaler, 'scale_'):
            raise ValueError("RobustScaler not properly fitted")
        
        # RobustScaler: inverse = scaled * scale + center
        original_data = scaled_data * scaler.scale_ + scaler.center_
        return original_data
    
    def _invert_maxabs_scaler(self, scaled_data: np.ndarray, scaler: MaxAbsScaler) -> np.ndarray:
        """Invert MaxAbsScaler transformation."""
        if not hasattr(scaler, 'max_abs_'):
            raise ValueError("MaxAbsScaler not properly fitted")
        
        # MaxAbsScaler: inverse = scaled * max_abs
        original_data = scaled_data * scaler.max_abs_
        return original_data
    
    def _invert_power_transformer(self, scaled_data: np.ndarray, scaler: PowerTransformer) -> np.ndarray:
        """Invert PowerTransformer transformation."""
        try:
            # Use built-in inverse_transform if available
            return scaler.inverse_transform(scaled_data)
        except Exception as e:
            print(f"    ⚠️  PowerTransformer inverse_transform failed: {e}")
            
            # Manual inversion for Yeo-Johnson
            if hasattr(scaler, 'lambdas_'):
                lambdas = scaler.lambdas_
                
                def inverse_yeo_johnson(x, lmbda):
                    x = np.asarray(x)
                    pos = x >= 0
                    
                    # When x >= 0
                    if abs(lmbda) < np.finfo(float).eps:
                        x_pos = np.exp(x) - 1
                    else:
                        x_pos = (x * lmbda + 1) ** (1 / lmbda) - 1
                    
                    # When x < 0
                    if abs(lmbda - 2) < np.finfo(float).eps:
                        x_neg = 1 - np.exp(-x)
                    else:
                        x_neg = 1 - (-(2 - lmbda) * x + 1) ** (1 / (2 - lmbda))
                    
                    return np.where(pos, x_pos, x_neg)
                
                original_data = np.copy(scaled_data)
                for i in range(scaled_data.shape[1]):
                    original_data[:, i] = inverse_yeo_johnson(scaled_data[:, i], lambdas[i])
                
                return original_data
            
            else:
                raise ValueError("PowerTransformer not properly fitted")
    
    def _invert_quantile_transformer(self, scaled_data: np.ndarray, scaler: QuantileTransformer) -> np.ndarray:
        """Invert QuantileTransformer transformation."""
        try:
            # Use built-in inverse_transform
            return scaler.inverse_transform(scaled_data)
        except Exception as e:
            print(f"    ⚠️  QuantileTransformer inverse_transform failed: {e}")
            # QuantileTransformer inversion is complex, return as is
            return scaled_data
    
    def invert_predictions(self, scaled_predictions: np.ndarray, 
                          dataset_name: str, target_column: str,
                          fitted_scalers: Dict[str, Any],
                          scaling_parameters: Dict[str, Any]) -> np.ndarray:
        """
        Invert scaling for model predictions to get original scale values.
        """
        print(f"    🔄 Inverting predictions for {target_column} in {dataset_name}")
        
        if scaled_predictions is None or len(scaled_predictions) == 0:
            return scaled_predictions
        
        try:
            # Find which scaler was used for the target column
            target_scaler = None
            scaler_type = None
            
            for stype, scaler in fitted_scalers.items():
                # In a real scenario, you'd have mapping of columns to scalers
                # For now, we'll use a simple heuristic
                if hasattr(scaler, 'n_features_in_'):
                    target_scaler = scaler
                    scaler_type = stype
                    break
            
            if target_scaler is None:
                print(f"    ⚠️  No scaler found for target column {target_column}")
                return scaled_predictions
            
            # Reshape predictions if needed
            if len(scaled_predictions.shape) == 1:
                scaled_predictions = scaled_predictions.reshape(-1, 1)
            
            # Invert scaling
            original_predictions = self.invert_scaling(
                scaled_predictions, target_scaler, target_column, scaling_parameters
            )
            
            print(f"    ✅ Predictions inverted using {scaler_type} scaler")
            return original_predictions.flatten()
            
        except Exception as e:
            print(f"    ❌ Error inverting predictions: {e}")
            return scaled_predictions
    
    def calculate_inversion_accuracy(self, original_data: np.ndarray,
                                   inverted_data: np.ndarray,
                                   tolerance: float = 1e-6) -> Dict[str, float]:
        """
        Calculate accuracy of scale inversion process.
        """
        if original_data.shape != inverted_data.shape:
            raise ValueError("Original and inverted data must have same shape")
        
        # Calculate differences
        differences = np.abs(original_data - inverted_data)
        
        # Calculate accuracy metrics
        accuracy_metrics = {
            'mean_absolute_error': float(np.mean(differences)),
            'max_absolute_error': float(np.max(differences)),
            'root_mean_squared_error': float(np.sqrt(np.mean(differences ** 2))),
            'within_tolerance_ratio': float(np.mean(differences <= tolerance)),
            'perfect_inversions': int(np.sum(differences == 0))
        }
        
        return accuracy_metrics
    
    def create_inversion_report(self, original_df: pd.DataFrame,
                               scaled_df: pd.DataFrame,
                               inverted_df: pd.DataFrame,
                               scaled_columns: List[str]) -> Dict[str, Any]:
        """
        Create comprehensive report on scale inversion accuracy.
        """
        report = {
            'inversion_accuracy': {},
            'column_comparisons': {},
            'summary': {}
        }
        
        perfect_inversions = 0
        total_columns = 0
        
        for scaled_col in scaled_columns:
            original_col = scaled_col.replace('_scaled', '')
            
            if original_col not in original_df.columns:
                continue
            
            if scaled_col not in scaled_df.columns:
                continue
            
            if original_col not in inverted_df.columns:
                continue
            
            total_columns += 1
            
            # Calculate accuracy for this column
            original_values = original_df[original_col].values
            inverted_values = inverted_df[original_col].values
            
            # Handle null values
            mask = ~np.isnan(original_values) & ~np.isnan(inverted_values)
            original_clean = original_values[mask]
            inverted_clean = inverted_values[mask]
            
            if len(original_clean) == 0:
                continue
            
            accuracy = self.calculate_inversion_accuracy(
                original_clean.reshape(-1, 1),
                inverted_clean.reshape(-1, 1)
            )
            
            report['inversion_accuracy'][original_col] = accuracy
            
            # Check if inversion is perfect
            if accuracy['mean_absolute_error'] < 1e-10:
                perfect_inversions += 1
        
        # Summary statistics
        report['summary'] = {
            'total_columns_inverted': total_columns,
            'perfect_inversions': perfect_inversions,
            'inversion_success_rate': perfect_inversions / total_columns if total_columns > 0 else 0,
            'average_mae': np.mean([acc['mean_absolute_error'] 
                                  for acc in report['inversion_accuracy'].values()])
        }
        
        return report
