# fixed_outlier_treatment.py
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from sklearn.impute import SimpleImputer
import scipy.stats as sp_stats

warnings.filterwarnings('ignore')

class OutlierTreatment:
    """
    Fixed outlier treatment with reasonable bounds and validation.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.reasonable_bounds = {}  # Track reasonable bounds per column
    
    def apply_treatment_strategy(self, df: pd.DataFrame, columns: List[str],
                               method_name: str, method_params: Dict[str, Any],
                               detection_results: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply outlier treatment with reasonable bounds validation.
        """
        treated_df = df.copy()
        treatment_details = {
            'method': method_name,
            'columns_treated': columns,
            'column_results': {},
            'outliers_treated': 0,
            'parameters_used': method_params
        }
        
        total_outliers_treated = 0
        
        try:
            if method_name == 'cap':
                results, details = self._cap_outliers_reasonable(treated_df, columns, method_params, detection_results)
            elif method_name == 'winsorize':
                results, details = self._winsorize_outliers(treated_df, columns, method_params, detection_results)
            elif method_name == 'transform':
                results, details = self._transform_outliers_safe(treated_df, columns, method_params, detection_results)
            elif method_name == 'impute':
                results, details = self._impute_outliers(treated_df, columns, method_params, detection_results)
            elif method_name == 'remove':
                results, details = self._remove_outliers_safe(treated_df, columns, method_params, detection_results)
            else:
                raise ValueError(f"Unknown treatment method: {method_name}")
            
            # Update dataframe with treated values
            for col in columns:
                if col in results.columns:
                    treated_df[col] = results[col]
            
            treatment_details.update(details)
            treatment_details['outliers_treated'] = int(details.get('outliers_treated', 0))
            
        except Exception as e:
            treatment_details['error'] = str(e)
            print(f"      ❌ Error in {method_name} treatment: {e}")
        
        return treated_df, treatment_details
    
    def _cap_outliers_reasonable(self, df: pd.DataFrame, columns: List[str],
                               params: Dict[str, Any], detection_results: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Cap outliers with reasonable bounds validation."""
        results_df = pd.DataFrame(index=df.index)
        details = {'method': 'cap', 'column_results': {}, 'outliers_treated': 0}
        
        cap_method = params.get('method', 'iqr')
        multiplier = params.get('multiplier', 2.0)  # More conservative default
        use_quantiles = params.get('use_quantiles', True)  # Use quantiles by default
        
        total_treated = 0
        
        for column in columns:
            if column not in df.columns:
                continue
            
            series = df[column].dropna()
            if len(series) < 10:  # Need sufficient data
                continue
            
            # Calculate reasonable bounds
            if use_quantiles:
                lower_quantile = params.get('lower_quantile', 0.05)  # 5th percentile
                upper_quantile = params.get('upper_quantile', 0.95)  # 95th percentile
                lower_bound = float(series.quantile(lower_quantile))
                upper_bound = float(series.quantile(upper_quantile))
            else:
                if cap_method == 'iqr':
                    Q1 = float(series.quantile(0.25))
                    Q3 = float(series.quantile(0.75))
                    IQR = float(Q3 - Q1)
                    lower_bound = float(Q1 - multiplier * IQR)
                    upper_bound = float(Q3 + multiplier * IQR)
                elif cap_method == 'zscore':
                    mean_val = float(series.mean())
                    std_val = float(series.std())
                    lower_bound = float(mean_val - multiplier * std_val)
                    upper_bound = float(mean_val + multiplier * std_val)
                elif cap_method == 'mad':
                    median_val = float(series.median())
                    mad = float(np.median(np.abs(series - median_val)))
                    if mad == 0:
                        mad = float(1.4826 * np.mean(np.abs(series - median_val)))
                    lower_bound = float(median_val - multiplier * mad)
                    upper_bound = float(median_val + multiplier * mad)
                else:
                    # Fallback to quantile-based bounds
                    lower_bound = float(series.quantile(0.05))
                    upper_bound = float(series.quantile(0.95))
            
            # Validate bounds are reasonable
            lower_bound, upper_bound = self._validate_bounds(series, lower_bound, upper_bound, column)
            
            # Store reasonable bounds for this column
            self.reasonable_bounds[column] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'method': cap_method
            }
            
            # Cap the values
            treated_series = df[column].copy()
            outliers_below = treated_series < lower_bound
            outliers_above = treated_series > upper_bound
            
            treated_series[outliers_below] = lower_bound
            treated_series[outliers_above] = upper_bound
            
            outliers_treated = int(outliers_below.sum() + outliers_above.sum())
            
            results_df[column] = treated_series
            
            details['column_results'][column] = {
                'outliers_treated': outliers_treated,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'method': cap_method,
                'multiplier': multiplier,
                'outliers_below': int(outliers_below.sum()),
                'outliers_above': int(outliers_above.sum())
            }
            
            total_treated += outliers_treated
            
            print(f"      📊 {column}: capped {outliers_treated} outliers to [{lower_bound:.4f}, {upper_bound:.4f}]")
        
        details['outliers_treated'] = int(total_treated)
        
        return results_df, details
    
    def _validate_bounds(self, series: pd.Series, lower_bound: float, upper_bound: float, column: str) -> Tuple[float, float]:
        """Ensure bounds are reasonable and don't create extreme values."""
        min_val = float(series.min())
        max_val = float(series.max())
        median_val = float(series.median())
        
        # Check if bounds are too extreme
        if lower_bound < min_val * 0.1 and min_val > 0:  # Don't make bounds too small for positive values
            lower_bound = max(lower_bound, min_val * 0.5)
        
        if upper_bound > max_val * 10 and max_val > 0:  # Don't make bounds too large
            upper_bound = min(upper_bound, max_val * 2.0)
        
        # Ensure bounds are not too close or too far from median
        range_size = upper_bound - lower_bound
        if range_size < (max_val - min_val) * 0.01:  # If range is too small
            lower_bound = median_val - (max_val - min_val) * 0.1
            upper_bound = median_val + (max_val - min_val) * 0.1
        
        # Ensure lower bound is not negative for positive data
        if min_val >= 0 and lower_bound < 0:
            lower_bound = 0.0
        
        return lower_bound, upper_bound
    
    def _winsorize_outliers(self, df: pd.DataFrame, columns: List[str],
                           params: Dict[str, Any], detection_results: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Winsorize outliers with safe limits."""
        results_df = pd.DataFrame(index=df.index)
        details = {'method': 'winsorize', 'column_results': {}, 'outliers_treated': 0}
        
        # Use more conservative limits
        limits = params.get('limits', [0.05, 0.05])  # 5% on each tail
        
        total_treated = 0
        
        for column in columns:
            if column not in df.columns:
                continue
            
            series = df[column].dropna()
            if len(series) < 20:  # Need sufficient data for winsorizing
                continue
            
            try:
                # Winsorize the series
                winsorized_values = sp_stats.mstats.winsorize(
                    series, limits=limits
                )
                
                # Count outliers treated (values that changed)
                original_values = series.values
                winsorized_values = winsorized_values.data
                
                outliers_treated = int(np.sum(original_values != winsorized_values))
                
                # Create treated series
                treated_series = df[column].copy()
                mask = ~df[column].isna()
                treated_series[mask] = winsorized_values
                
                results_df[column] = treated_series
                
                details['column_results'][column] = {
                    'outliers_treated': outliers_treated,
                    'limits': [float(x) for x in limits],
                    'lower_percentile': float(limits[0] * 100),
                    'upper_percentile': float((1 - limits[1]) * 100)
                }
                
                total_treated += outliers_treated
                
                print(f"      📊 {column}: winsorized {outliers_treated} outliers")
                
            except Exception as e:
                details['column_results'][column] = {
                    'error': str(e),
                    'outliers_treated': 0
                }
        
        details['outliers_treated'] = int(total_treated)
        
        return results_df, details
    
    def _transform_outliers_safe(self, df: pd.DataFrame, columns: List[str],
                                params: Dict[str, Any], detection_results: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Safe transformation with bounds checking."""
        results_df = pd.DataFrame(index=df.index)
        details = {'method': 'transform', 'column_results': {}, 'outliers_treated': 0}
        
        transform_method = params.get('method', 'log')
        offset = params.get('offset', 1.0)  # Larger default offset
        
        total_treated = 0
        
        for column in columns:
            if column not in df.columns:
                continue
            
            series = df[column].dropna()
            if len(series) < 5:
                continue
            
            try:
                treated_series = df[column].copy()
                mask = ~df[column].isna()
                
                if transform_method == 'log':
                    # Safe log transformation
                    min_val = float(series.min())
                    if min_val <= 0:
                        shift = abs(min_val) + offset
                        transformed = np.log(series + shift)
                    else:
                        transformed = np.log(series)
                    
                    # Check for extreme values after transformation
                    if abs(transformed.min()) > 100 or abs(transformed.max()) > 100:
                        # If transformation creates extremes, use winsorize instead
                        Q1 = float(series.quantile(0.25))
                        Q3 = float(series.quantile(0.75))
                        IQR = float(Q3 - Q1)
                        lower_bound = float(Q1 - 1.5 * IQR)
                        upper_bound = float(Q3 + 1.5 * IQR)
                        
                        treated_series = series.clip(lower_bound, upper_bound)
                        transform_method = 'clip_fallback'
                    else:
                        treated_series[mask] = transformed
                
                elif transform_method == 'sqrt':
                    # Safe square root transformation
                    min_val = float(series.min())
                    if min_val < 0:
                        shift = abs(min_val) + offset
                        treated_series[mask] = np.sqrt(series + shift)
                    else:
                        treated_series[mask] = np.sqrt(series)
                
                else:
                    # For other transformations, use conservative approach
                    Q1 = float(series.quantile(0.25))
                    Q3 = float(series.quantile(0.75))
                    IQR = float(Q3 - Q1)
                    lower_bound = float(Q1 - 1.5 * IQR)
                    upper_bound = float(Q3 + 1.5 * IQR)
                    
                    treated_series = series.clip(lower_bound, upper_bound)
                    transform_method = 'clip_fallback'
                
                # All values are considered treated in transformation
                outliers_treated = int(len(series))
                
                results_df[column] = treated_series
                
                details['column_results'][column] = {
                    'outliers_treated': outliers_treated,
                    'transform_method': transform_method,
                    'offset_used': float(offset) if transform_method in ['log', 'sqrt'] else None
                }
                
                total_treated += outliers_treated
                
                print(f"      📊 {column}: transformed using {transform_method}")
                
            except Exception as e:
                details['column_results'][column] = {
                    'error': str(e),
                    'outliers_treated': 0
                }
        
        details['outliers_treated'] = int(total_treated)
        
        return results_df, details
    
    def _impute_outliers(self, df: pd.DataFrame, columns: List[str],
                        params: Dict[str, Any], detection_results: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Impute outliers with median (safer than mean)."""
        results_df = pd.DataFrame(index=df.index)
        details = {'method': 'impute', 'column_results': {}, 'outliers_treated': 0}
        
        impute_method = params.get('method', 'median')  # Use median by default
        
        total_treated = 0
        
        for column in columns:
            if column not in df.columns:
                continue
            
            series = df[column].dropna()
            if len(series) < 5:
                continue
            
            try:
                # Use IQR detection for imputation
                Q1 = float(series.quantile(0.25))
                Q3 = float(series.quantile(0.75))
                IQR = float(Q3 - Q1)
                outliers = (df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)
                
                outliers_treated = int(outliers.sum())
                
                if outliers_treated > 0:
                    # Calculate imputation value
                    if impute_method == 'mean':
                        impute_value = float(series.mean())
                    elif impute_method == 'median':
                        impute_value = float(series.median())
                    elif impute_method == 'mode':
                        impute_value = float(series.mode()[0] if not series.mode().empty else series.median())
                    else:
                        impute_value = float(series.median())
                    
                    # Impute outliers
                    treated_series = df[column].copy()
                    treated_series[outliers] = impute_value
                    
                    results_df[column] = treated_series
                else:
                    results_df[column] = df[column]
                
                details['column_results'][column] = {
                    'outliers_treated': outliers_treated,
                    'impute_method': impute_method,
                    'impute_value': impute_value
                }
                
                total_treated += outliers_treated
                
                print(f"      📊 {column}: imputed {outliers_treated} outliers with {impute_method}")
                
            except Exception as e:
                details['column_results'][column] = {
                    'error': str(e),
                    'outliers_treated': 0
                }
        
        details['outliers_treated'] = int(total_treated)
        
        return results_df, details
    
    def _remove_outliers_safe(self, df: pd.DataFrame, columns: List[str],
                            params: Dict[str, Any], detection_results: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Safe outlier removal (limit maximum removal)."""
        results_df = pd.DataFrame(index=df.index)
        details = {'method': 'remove', 'column_results': {}, 'outliers_treated': 0}
        
        max_removal_percentage = params.get('max_removal_percentage', 0.05)  # Max 5% removal
        
        total_treated = 0
        
        for column in columns:
            if column not in df.columns:
                continue
            
            series = df[column].dropna()
            if len(series) < 10:
                continue
            
            try:
                # Identify outliers using IQR
                Q1 = float(series.quantile(0.25))
                Q3 = float(series.quantile(0.75))
                IQR = float(Q3 - Q1)
                outliers = (df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)
                
                outliers_treated = int(outliers.sum())
                
                # Check if we're removing too many points
                removal_percentage = outliers_treated / len(series)
                if removal_percentage > max_removal_percentage:
                    # Use winsorize instead if too many outliers
                    limits = [max_removal_percentage, max_removal_percentage]
                    winsorized_values = sp_stats.mstats.winsorize(series, limits=limits)
                    treated_series = df[column].copy()
                    treated_series[~df[column].isna()] = winsorized_values.data
                    method_used = 'winsorize_fallback'
                else:
                    # Remove outliers (set to NaN)
                    treated_series = df[column].copy()
                    treated_series[outliers] = np.nan
                    method_used = 'remove'
                
                results_df[column] = treated_series
                
                details['column_results'][column] = {
                    'outliers_treated': outliers_treated,
                    'method_used': method_used,
                    'removal_percentage': float(removal_percentage * 100)
                }
                
                total_treated += outliers_treated
                
                print(f"      📊 {column}: {method_used} for {outliers_treated} outliers")
                
            except Exception as e:
                details['column_results'][column] = {
                    'error': str(e),
                    'outliers_treated': 0
                }
        
        details['outliers_treated'] = int(total_treated)
        
        return results_df, details
    
    def get_reasonable_bounds(self) -> Dict[str, Any]:
        """Get the reasonable bounds calculated during treatment."""
        return self.reasonable_bounds
