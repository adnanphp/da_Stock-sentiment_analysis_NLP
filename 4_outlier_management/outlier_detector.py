# outlier_detector.py
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import time

warnings.filterwarnings('ignore')

class OutlierDetector:
    """
    Multi-method outlier detector with various statistical and ML-based approaches.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
    
    def apply_detection_method(self, df: pd.DataFrame, columns: List[str],
                             method_name: str, method_config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply specific outlier detection method to columns.
        """
        start_time = time.time()
        
        detection_df = pd.DataFrame(index=df.index)
        method_details = {
            'method': method_name,
            'columns_analyzed': columns,
            'column_results': {},
            'total_outliers': 0,
            'execution_time': 0
        }
        
        total_outliers = 0
        
        try:
            if method_name == 'iqr':
                results, details = self._detect_iqr(df, columns, method_config)
            elif method_name == 'zscore':
                results, details = self._detect_zscore(df, columns, method_config)
            elif method_name == 'mad':
                results, details = self._detect_mad(df, columns, method_config)
            elif method_name == 'isolation_forest':
                results, details = self._detect_isolation_forest(df, columns, method_config)
            elif method_name == 'lof':
                results, details = self._detect_lof(df, columns, method_config)
            elif method_name == 'elliptic_envelope':
                results, details = self._detect_elliptic_envelope(df, columns, method_config)
            elif method_name == 'dbscan':
                results, details = self._detect_dbscan(df, columns, method_config)
            else:
                raise ValueError(f"Unknown detection method: {method_name}")
            
            # Add results to detection dataframe
            for col in results.columns:
                detection_df[col] = results[col]
            
            method_details.update(details)
            method_details['total_outliers'] = int(details.get('total_outliers', 0))
            
        except Exception as e:
            method_details['error'] = str(e)
            print(f"      ❌ Error in {method_name}: {e}")
        
        method_details['execution_time'] = float(time.time() - start_time)
        
        return detection_df, method_details
    
    def _detect_iqr(self, df: pd.DataFrame, columns: List[str], 
                   config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect outliers using Interquartile Range method."""
        results_df = pd.DataFrame(index=df.index)
        details = {'method': 'iqr', 'column_results': {}, 'total_outliers': 0}
        
        multiplier = config.get('multiplier', 1.5)
        
        for column in columns:
            if column not in df.columns:
                continue
            
            series = df[column].dropna()
            if len(series) < 4:  # Need at least 4 points for IQR
                continue
            
            Q1 = float(series.quantile(0.25))
            Q3 = float(series.quantile(0.75))
            IQR = float(Q3 - Q1)
            
            lower_bound = float(Q1 - multiplier * IQR)
            upper_bound = float(Q3 + multiplier * IQR)
            
            # Detect outliers
            outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
            outlier_count = int(outliers.sum())
            
            results_df[f"{column}_outlier_iqr"] = outliers.astype(int)
            
            details['column_results'][column] = {
                'outlier_count': outlier_count,
                'outlier_percentage': float((outlier_count / len(df)) * 100),
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'q1': Q1,
                'q3': Q3,
                'iqr': IQR
            }
            
            details['total_outliers'] += outlier_count
        
        details['total_outliers'] = int(details['total_outliers'])
        return results_df, details
    
    def _detect_zscore(self, df: pd.DataFrame, columns: List[str],
                      config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect outliers using Z-score method."""
        results_df = pd.DataFrame(index=df.index)
        details = {'method': 'zscore', 'column_results': {}, 'total_outliers': 0}
        
        threshold = config.get('threshold', 3)
        
        for column in columns:
            if column not in df.columns:
                continue
            
            series = df[column].dropna()
            if len(series) < 2:
                continue
            
            # Calculate Z-scores
            mean_val = float(series.mean())
            std_val = float(series.std())
            
            if std_val == 0:  # Constant series
                continue
            
            z_scores = np.abs((df[column] - mean_val) / std_val)
            outliers = z_scores > threshold
            outlier_count = int(outliers.sum())
            
            results_df[f"{column}_outlier_zscore"] = outliers.astype(int)
            
            details['column_results'][column] = {
                'outlier_count': outlier_count,
                'outlier_percentage': float((outlier_count / len(df)) * 100),
                'threshold': float(threshold),
                'mean': mean_val,
                'std': std_val
            }
            
            details['total_outliers'] += outlier_count
        
        details['total_outliers'] = int(details['total_outliers'])
        return results_df, details
    
    def _detect_mad(self, df: pd.DataFrame, columns: List[str],
                   config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect outliers using Median Absolute Deviation method."""
        results_df = pd.DataFrame(index=df.index)
        details = {'method': 'mad', 'column_results': {}, 'total_outliers': 0}
        
        threshold = config.get('threshold', 3)
        
        for column in columns:
            if column not in df.columns:
                continue
            
            series = df[column].dropna()
            if len(series) < 2:
                continue
            
            # Calculate MAD
            median_val = float(series.median())
            mad = float(np.median(np.abs(series - median_val)))
            
            if mad == 0:  # Use modified Z-score approach
                mad = float(1.4826 * np.mean(np.abs(series - median_val)))
                if mad == 0:
                    continue
            
            # Calculate modified Z-scores
            modified_z_scores = 0.6745 * np.abs(df[column] - median_val) / mad
            outliers = modified_z_scores > threshold
            outlier_count = int(outliers.sum())
            
            results_df[f"{column}_outlier_mad"] = outliers.astype(int)
            
            details['column_results'][column] = {
                'outlier_count': outlier_count,
                'outlier_percentage': float((outlier_count / len(df)) * 100),
                'threshold': float(threshold),
                'median': median_val,
                'mad': mad
            }
            
            details['total_outliers'] += outlier_count
        
        details['total_outliers'] = int(details['total_outliers'])
        return results_df, details
    
    def _detect_isolation_forest(self, df: pd.DataFrame, columns: List[str],
                               config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect outliers using Isolation Forest algorithm."""
        results_df = pd.DataFrame(index=df.index)
        details = {'method': 'isolation_forest', 'column_results': {}, 'total_outliers': 0}
        
        # Prepare data
        data = df[columns].dropna()
        if len(data) < 10:  # Need sufficient data
            return results_df, details
        
        contamination = config.get('contamination', 'auto')
        n_estimators = config.get('n_estimators', 100)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )
        
        predictions = iso_forest.fit_predict(data)
        outliers = predictions == -1
        outlier_count = int(outliers.sum())
        
        # Create results for each column (multivariate method)
        for i, column in enumerate(columns):
            if column not in df.columns:
                continue
            
            # For multivariate methods, we assign the same outlier flag to all columns
            column_outliers = pd.Series(False, index=df.index)
            column_outliers[data.index] = outliers
            
            results_df[f"{column}_outlier_isolation_forest"] = column_outliers.astype(int)
            
            column_outlier_count = int(column_outliers.sum())
            
            details['column_results'][column] = {
                'outlier_count': column_outlier_count,
                'outlier_percentage': float((column_outlier_count / len(df)) * 100),
                'contamination': contamination,
                'n_estimators': n_estimators
            }
        
        details['total_outliers'] = outlier_count
        details['multivariate'] = True
        
        return results_df, details
    
    def _detect_lof(self, df: pd.DataFrame, columns: List[str],
                   config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect outliers using Local Outlier Factor algorithm."""
        results_df = pd.DataFrame(index=df.index)
        details = {'method': 'lof', 'column_results': {}, 'total_outliers': 0}
        
        # Prepare data
        data = df[columns].dropna()
        if len(data) < 20:  # Need more data for LOF
            return results_df, details
        
        n_neighbors = config.get('n_neighbors', 20)
        contamination = config.get('contamination', 'auto')
        
        # Fit Local Outlier Factor
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=False
        )
        
        predictions = lof.fit_predict(data)
        outliers = predictions == -1
        outlier_count = int(outliers.sum())
        
        # Create results for each column
        for i, column in enumerate(columns):
            if column not in df.columns:
                continue
            
            column_outliers = pd.Series(False, index=df.index)
            column_outliers[data.index] = outliers
            
            results_df[f"{column}_outlier_lof"] = column_outliers.astype(int)
            
            column_outlier_count = int(column_outliers.sum())
            
            details['column_results'][column] = {
                'outlier_count': column_outlier_count,
                'outlier_percentage': float((column_outlier_count / len(df)) * 100),
                'n_neighbors': n_neighbors,
                'contamination': contamination
            }
        
        details['total_outliers'] = outlier_count
        details['multivariate'] = True
        
        return results_df, details
    
    def _detect_elliptic_envelope(self, df: pd.DataFrame, columns: List[str],
                                config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect outliers using Elliptic Envelope (assuming Gaussian distribution)."""
        results_df = pd.DataFrame(index=df.index)
        details = {'method': 'elliptic_envelope', 'column_results': {}, 'total_outliers': 0}
        
        # Prepare data
        data = df[columns].dropna()
        if len(data) < 10:
            return results_df, details
        
        contamination = config.get('contamination', 0.1)
        
        # Fit Elliptic Envelope
        envelope = EllipticEnvelope(
            contamination=contamination,
            random_state=42
        )
        
        try:
            predictions = envelope.fit_predict(data)
            outliers = predictions == -1
            outlier_count = int(outliers.sum())
            
            # Create results for each column
            for i, column in enumerate(columns):
                if column not in df.columns:
                    continue
                
                column_outliers = pd.Series(False, index=df.index)
                column_outliers[data.index] = outliers
                
                results_df[f"{column}_outlier_elliptic_envelope"] = column_outliers.astype(int)
                
                column_outlier_count = int(column_outliers.sum())
                
                details['column_results'][column] = {
                    'outlier_count': column_outlier_count,
                    'outlier_percentage': float((column_outlier_count / len(df)) * 100),
                    'contamination': contamination
                }
            
            details['total_outliers'] = outlier_count
            details['multivariate'] = True
            
        except Exception as e:
            details['error'] = f"Elliptic Envelope failed: {str(e)}"
        
        return results_df, details
    
    def _detect_dbscan(self, df: pd.DataFrame, columns: List[str],
                      config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Detect outliers using DBSCAN clustering."""
        from sklearn.cluster import DBSCAN
        
        results_df = pd.DataFrame(index=df.index)
        details = {'method': 'dbscan', 'column_results': {}, 'total_outliers': 0}
        
        # Prepare data
        data = df[columns].dropna()
        if len(data) < 10:
            return results_df, details
        
        eps = config.get('eps', 0.5)
        min_samples = config.get('min_samples', 5)
        
        # Fit DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        
        clusters = dbscan.fit_predict(data)
        outliers = clusters == -1  # DBSCAN labels outliers as -1
        outlier_count = int(outliers.sum())
        
        # Create results for each column
        for i, column in enumerate(columns):
            if column not in df.columns:
                continue
            
            column_outliers = pd.Series(False, index=df.index)
            column_outliers[data.index] = outliers
            
            results_df[f"{column}_outlier_dbscan"] = column_outliers.astype(int)
            
            column_outlier_count = int(column_outliers.sum())
            
            details['column_results'][column] = {
                'outlier_count': column_outlier_count,
                'outlier_percentage': float((column_outlier_count / len(df)) * 100),
                'eps': eps,
                'min_samples': min_samples
            }
        
        details['total_outliers'] = outlier_count
        details['multivariate'] = True
        
        return results_df, details
    
    def get_detection_methods(self) -> List[str]:
        """Get list of available detection methods."""
        return [
            'iqr', 'zscore', 'mad', 'isolation_forest', 
            'lof', 'elliptic_envelope', 'dbscan'
        ]
