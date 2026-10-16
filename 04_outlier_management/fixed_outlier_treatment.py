# fixed_outlier_treatment.py
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from sklearn.impute import SimpleImputer
import scipy.stats as sp_stats

warnings.filterwarnings('ignore')

class FixedOutlierTreatment:
    """
    Fixed outlier treatment with reasonable bounds and accurate counting.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.reasonable_bounds = {}
    
    def apply_treatment_strategy(self, df: pd.DataFrame, columns: List[str],
                               method_name: str, method_params: Dict[str, Any],
                               detection_results: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply outlier treatment with proper outlier counting.
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
                results, details = self._transform_outliers_accurate(treated_df, columns, method_params, detection_results)
            elif method_name == 'impute':
                results, details = self._impute_outliers(treated_df, columns, method_params, detection_results)
            elif method_name == 'remove':
                results, details = self._remove_outliers_safe(treated_df, columns, method_params, detection_results)
            else:
                raise ValueError(f"Unknown treatment method: {method_name}")
            
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
        """Cap outliers with reasonable bounds."""
        results_df = pd.DataFrame(index=df.index)
        details = {'method': 'cap', 'column_results': {}, 'outliers_treated': 0}
        
        cap_method = params.get('method', 'iqr')
        multiplier = params.get('multiplier', 2.0)
        use_quantiles = params.get('use_quantiles', True)
        
        total_treated = 0
        
        for column in columns:
            if column not in df.columns:
                continue
            
            series = df[column].dropna()
            if len(series) < 10:
                continue
            
            # Calculate bounds
            if use_quantiles:
                lower_quantile = params.get('lower_quantile', 0.05)
                upper_quantile = params.get('upper_quantile', 0.95)
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
                else:
                    lower_bound = float(series.quantile(0.05))
                    upper_bound = float(series.quantile(0.95))
            
            # Validate bounds
            lower_bound, upper_bound = self._validate_bounds(series, lower_bound, upper_bound, column)
            
            # Cap values and count ACTUAL outliers
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
                'outliers_below': int(outliers_below.sum()),
                'outliers_above': int(outliers_above.sum())
            }
            
            total_treated += outliers_treated
            
            if outliers_treated > 0:
                print(f"      📊 {column}: capped {outliers_treated} outliers to [{lower_bound:.4f}, {upper_bound:.4f}]")
        
        details['outliers_treated'] = int(total_treated)
        return results_df, details
    
    def _validate_bounds(self, series: pd.Series, lower_bound: float, upper_bound: float, column: str) -> Tuple[float, float]:
        """Ensure bounds are reasonable."""
        min_val = float(series.min())
        max_val = float(series.max())
        median_val = float(series.median())
        
        # Don't make bounds too extreme
        if lower_bound < min_val * 0.1 and min_val > 0:
            lower_bound = max(lower_bound, min_val * 0.5)
        
        if upper_bound > max_val * 10 and max_val > 0:
            upper_bound = min(upper_bound, max_val * 2.0)
        
        # Ensure reasonable range
        range_size = upper_bound - lower_bound
        if range_size < (max_val - min_val) * 0.01:
            lower_bound = median_val - (max_val - min_val) * 0.1
            upper_bound = median_val + (max_val - min_val) * 0.1
        
        if min_val >= 0 and lower_bound < 0:
            lower_bound = 0.0
        
        return lower_bound, upper_bound
    
    def _winsorize_outliers(self, df: pd.DataFrame, columns: List[str],
                           params: Dict[str, Any], detection_results: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Winsorize outliers with proper counting."""
        results_df = pd.DataFrame(index=df.index)
        details = {'method': 'winsorize', 'column_results': {}, 'outliers_treated': 0}
        
        limits = params.get('limits', [0.05, 0.05])
        total_treated = 0
        
        for column in columns:
            if column not in df.columns:
                continue
            
            series = df[column].dropna()
            if len(series) < 20:
                continue
            
            try:
                # Get original values for comparison
                original_values = series.values.copy()
                
                # Winsorize
                winsorized_values = sp_stats.mstats.winsorize(series, limits=limits)
                winsorized_values = winsorized_values.data
                
                # Count ACTUAL changes (true outliers)
                values_changed = original_values != winsorized_values
                outliers_treated = int(values_changed.sum())
                
                # Apply treatment
                treated_series = df[column].copy()
                mask = ~df[column].isna()
                treated_series[mask] = winsorized_values
                
                results_df[column] = treated_series
                
                details['column_results'][column] = {
                    'outliers_treated': outliers_treated,
                    'limits': [float(x) for x in limits]
                }
                
                total_treated += outliers_treated
                
                if outliers_treated > 0:
                    print(f"      📊 {column}: winsorized {outliers_treated} outliers")
                
            except Exception as e:
                details['column_results'][column] = {'error': str(e), 'outliers_treated': 0}
        
        details['outliers_treated'] = int(total_treated)
        return results_df, details
    
    def _transform_outliers_accurate(self, df: pd.DataFrame, columns: List[str],
                                   params: Dict[str, Any], detection_results: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Transform outliers with accurate counting."""
        results_df = pd.DataFrame(index=df.index)
        details = {'method': 'transform', 'column_results': {}, 'outliers_treated': 0}
        
        transform_method = params.get('method', 'log')
        offset = params.get('offset', 1.0)
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
                
                # Count outliers BEFORE transformation
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_before = ((series < lower_bound) | (series > upper_bound))
                outliers_treated = int(outliers_before.sum())
                
                # Apply transformation
                if transform_method == 'log':
                    min_val = float(series.min())
                    if min_val <= 0:
                        shift = abs(min_val) + offset
                        transformed = np.log(series + shift)
                    else:
                        transformed = np.log(series)
                    treated_series[mask] = transformed
                    
                elif transform_method == 'sqrt':
                    min_val = float(series.min())
                    if min_val < 0:
                        shift = abs(min_val) + offset
                        treated_series[mask] = np.sqrt(series + shift)
                    else:
                        treated_series[mask] = np.sqrt(series)
                        
                else:
                    # Fallback to conservative capping
                    treated_series = series.clip(lower_bound, upper_bound)
                    transform_method = 'clip_fallback'
                
                results_df[column] = treated_series
                
                details['column_results'][column] = {
                    'outliers_treated': outliers_treated,
                    'transform_method': transform_method
                }
                
                total_treated += outliers_treated
                
                if outliers_treated > 0:
                    print(f"      📊 {column}: transformed {outliers_treated} outliers using {transform_method}")
                
            except Exception as e:
                details['column_results'][column] = {'error': str(e), 'outliers_treated': 0}
        
        details['outliers_treated'] = int(total_treated)
        return results_df, details
    
    def _impute_outliers(self, df: pd.DataFrame, columns: List[str],
                        params: Dict[str, Any], detection_results: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Impute outliers accurately."""
        results_df = pd.DataFrame(index=df.index)
        details = {'method': 'impute', 'column_results': {}, 'outliers_treated': 0}
        
        impute_method = params.get('method', 'median')
        total_treated = 0
        
        for column in columns:
            if column not in df.columns:
                continue
            
            series = df[column].dropna()
            if len(series) < 5:
                continue
            
            try:
                # Detect outliers using IQR
                Q1 = float(series.quantile(0.25))
                Q3 = float(series.quantile(0.75))
                IQR = float(Q3 - Q1)
                outliers = (df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)
                outliers_treated = int(outliers.sum())
                
                if outliers_treated > 0:
                    if impute_method == 'mean':
                        impute_value = float(series.mean())
                    elif impute_method == 'median':
                        impute_value = float(series.median())
                    else:
                        impute_value = float(series.median())
                    
                    treated_series = df[column].copy()
                    treated_series[outliers] = impute_value
                    results_df[column] = treated_series
                else:
                    results_df[column] = df[column]
                
                details['column_results'][column] = {
                    'outliers_treated': outliers_treated,
                    'impute_method': impute_method
                }
                
                total_treated += outliers_treated
                
                if outliers_treated > 0:
                    print(f"      📊 {column}: imputed {outliers_treated} outliers with {impute_method}")
                
            except Exception as e:
                details['column_results'][column] = {'error': str(e), 'outliers_treated': 0}
        
        details['outliers_treated'] = int(total_treated)
        return results_df, details
    
    def _remove_outliers_safe(self, df: pd.DataFrame, columns: List[str],
                            params: Dict[str, Any], detection_results: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Safe outlier removal."""
        results_df = pd.DataFrame(index=df.index)
        details = {'method': 'remove', 'column_results': {}, 'outliers_treated': 0}
        
        max_removal = params.get('max_removal_percentage', 0.05)
        total_treated = 0
        
        for column in columns:
            if column not in df.columns:
                continue
            
            series = df[column].dropna()
            if len(series) < 10:
                continue
            
            try:
                Q1 = float(series.quantile(0.25))
                Q3 = float(series.quantile(0.75))
                IQR = float(Q3 - Q1)
                outliers = (df[column] < Q1 - 1.5 * IQR) | (df[column] > Q3 + 1.5 * IQR)
                outliers_treated = int(outliers.sum())
                
                removal_percentage = outliers_treated / len(series)
                
                if removal_percentage > max_removal:
                    # Use winsorize instead
                    limits = [max_removal, max_removal]
                    winsorized = sp_stats.mstats.winsorize(series, limits=limits)
                    treated_series = df[column].copy()
                    treated_series[~df[column].isna()] = winsorized.data
                    method_used = 'winsorize_fallback'
                    # Recalculate outliers treated for winsorize
                    outliers_treated = int((series.values != winsorized.data).sum())
                else:
                    treated_series = df[column].copy()
                    treated_series[outliers] = np.nan
                    method_used = 'remove'
                
                results_df[column] = treated_series
                
                details['column_results'][column] = {
                    'outliers_treated': outliers_treated,
                    'method_used': method_used
                }
                
                total_treated += outliers_treated
                
                if outliers_treated > 0:
                    print(f"      📊 {column}: {method_used} for {outliers_treated} outliers")
                
            except Exception as e:
                details['column_results'][column] = {'error': str(e), 'outliers_treated': 0}
        
        details['outliers_treated'] = int(total_treated)
        return results_df, details
