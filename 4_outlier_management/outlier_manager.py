# outlier_manager.py
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from outlier_detector import OutlierDetector
from outlier_treatment import OutlierTreatment

warnings.filterwarnings('ignore')

class OutlierManager:
    """
    Main outlier detection and treatment management class.
    Coordinates detection methods and treatment strategies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.outlier_report = {}
        self.detection_results = {}
        self.treatment_strategies = {}
        
        # Initialize specialized components
        self.detector = OutlierDetector(config)
        self.treatment = OutlierTreatment(config)
    
    def analyze_outlier_characteristics(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of outlier characteristics in dataset.
        """
        print(f"🔍 Analyzing outlier characteristics for: {dataset_name}")
        
        analysis = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'overview': {
                'total_columns': len(df.columns),
                'numeric_columns': 0,
                'columns_with_outliers': 0,
                'total_outliers_detected': 0,
                'outlier_percentage': 0.0
            },
            'column_analysis': {},
            'outlier_recommendations': []
        }
        
        # Analyze each numeric column
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        analysis['overview']['numeric_columns'] = len(numeric_columns)
        
        total_outliers = 0
        total_values = 0
        
        for column in numeric_columns:
            col_analysis = self._analyze_column_outliers(df[column], column)
            analysis['column_analysis'][column] = col_analysis
            
            if col_analysis['has_outliers']:
                analysis['overview']['columns_with_outliers'] += 1
                total_outliers += col_analysis['outlier_count']
            
            total_values += len(df[column].dropna())
        
        # Calculate overall statistics
        if total_values > 0:
            analysis['overview']['total_outliers_detected'] = int(total_outliers)
            analysis['overview']['outlier_percentage'] = float((total_outliers / total_values) * 100)
        
        # Generate recommendations
        analysis['outlier_recommendations'] = self._generate_outlier_recommendations(analysis, df)
        
        return analysis
    
    def _analyze_column_outliers(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Analyze outlier characteristics for a single column."""
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {
                'has_outliers': False,
                'reason': 'All values are null',
                'outlier_count': 0,
                'outlier_percentage': 0.0,
                'statistics': {}
            }
        
        # Calculate basic statistics
        stats = {
            'min': float(clean_series.min()),
            'max': float(clean_series.max()),
            'mean': float(clean_series.mean()),
            'median': float(clean_series.median()),
            'std': float(clean_series.std()),
            'skewness': float(clean_series.skew()),
            'kurtosis': float(clean_series.kurtosis()),
            'q1': float(clean_series.quantile(0.25)),
            'q3': float(clean_series.quantile(0.75)),
            'iqr': float(clean_series.quantile(0.75) - clean_series.quantile(0.25))
        }
        
        # Quick IQR-based outlier detection for analysis
        lower_bound = stats['q1'] - 1.5 * stats['iqr']
        upper_bound = stats['q3'] + 1.5 * stats['iqr']
        
        outliers = ((clean_series < lower_bound) | (clean_series > upper_bound))
        outlier_count = int(outliers.sum())
        outlier_percentage = float((outlier_count / len(clean_series)) * 100)
        
        has_outliers = outlier_count > 0
        
        # Determine outlier severity
        severity = 'none'
        if has_outliers:
            if outlier_percentage > 10:
                severity = 'high'
            elif outlier_percentage > 5:
                severity = 'medium'
            else:
                severity = 'low'
        
        # Identify outlier type
        outlier_type = self._classify_outlier_type(stats, clean_series)
        
        analysis = {
            'has_outliers': has_outliers,
            'outlier_count': outlier_count,
            'outlier_percentage': outlier_percentage,
            'outlier_severity': severity,
            'outlier_type': outlier_type,
            'statistics': stats,
            'bounds': {
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound)
            },
            'sample_outliers': [float(x) for x in clean_series[outliers].head(3).tolist()] if has_outliers else []
        }
        
        return analysis
    
    def _classify_outlier_type(self, stats: Dict[str, float], series: pd.Series) -> str:
        """Classify the type of outliers present."""
        # Check for extreme skewness
        if abs(stats['skewness']) > 3:
            return 'skewness_induced'
        
        # Check for heavy tails (high kurtosis)
        if stats['kurtosis'] > 5:
            return 'heavy_tailed'
        
        # Check for multimodal distribution
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(series)
            x_range = np.linspace(series.min(), series.max(), 100)
            density = kde(x_range)
            peaks = np.where((density[1:-1] > density[:-2]) & (density[1:-1] > density[2:]))[0] + 1
            if len(peaks) > 2:
                return 'multimodal'
        except:
            pass
        
        # Check for point outliers
        z_scores = np.abs((series - stats['mean']) / stats['std'])
        if (z_scores > 3).any():
            return 'point_outliers'
        
        return 'general'
    
    def _generate_outlier_recommendations(self, analysis: Dict[str, Any], df: pd.DataFrame) -> List[str]:
        """Generate outlier handling recommendations."""
        recommendations = []
        column_analysis = analysis['column_analysis']
        
        # Overall dataset recommendations
        outlier_percentage = analysis['overview']['outlier_percentage']
        columns_with_outliers = analysis['overview']['columns_with_outliers']
        
        if outlier_percentage > 10:
            recommendations.append(f"Aggressive outlier treatment recommended ({outlier_percentage:.1f}% outliers)")
        elif outlier_percentage > 5:
            recommendations.append(f"Moderate outlier treatment recommended ({outlier_percentage:.1f}% outliers)")
        elif outlier_percentage > 0:
            recommendations.append(f"Conservative outlier treatment recommended ({outlier_percentage:.1f}% outliers)")
        
        if columns_with_outliers > 0:
            recommendations.append(f"Treat outliers in {columns_with_outliers} columns")
        
        # Column-specific recommendations
        high_severity_cols = []
        skewed_cols = []
        
        for column, info in column_analysis.items():
            if info['has_outliers']:
                if info['outlier_severity'] == 'high':
                    high_severity_cols.append(column)
                if info['outlier_type'] == 'skewness_induced':
                    skewed_cols.append(column)
        
        if high_severity_cols:
            recommendations.append(f"Aggressive treatment for high-severity columns: {', '.join(high_severity_cols[:3])}")
        
        if skewed_cols:
            recommendations.append(f"Transformation recommended for skewed columns: {', '.join(skewed_cols[:3])}")
        
        return recommendations
    
    def detect_outliers(self, df: pd.DataFrame, dataset_name: str,
                       detection_plan: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Detect outliers using multiple methods.
        """
        print(f"🎯 Detecting outliers for: {dataset_name}")
        
        # Auto-generate detection plan if not provided
        if not detection_plan:
            analysis = self.analyze_outlier_characteristics(df, dataset_name)
            detection_plan = self._create_detection_plan(analysis, df)
            print(f"  📋 Auto-generated detection plan with {len(detection_plan['methods'])} methods")
        
        # Create detection dataframe
        detection_df = pd.DataFrame(index=df.index)
        detection_details = {}
        
        numeric_columns = detection_plan.get('columns_to_analyze', 
                                           df.select_dtypes(include=[np.number]).columns.tolist())
        
        total_outliers_detected = 0
        
        # Apply detection methods
        for method_name, method_config in detection_plan.get('methods', {}).items():
            try:
                print(f"  🔍 Applying {method_name} outlier detection...")
                
                # Detect outliers using specified method
                method_results, method_details = self.detector.apply_detection_method(
                    df, numeric_columns, method_name, method_config
                )
                
                # Add detection results to dataframe
                for col in numeric_columns:
                    outlier_col = f"{col}_outlier_{method_name}"
                    if outlier_col in method_results.columns:
                        detection_df[outlier_col] = method_results[outlier_col]
                
                detection_details[method_name] = method_details
                total_outliers_detected += method_details.get('total_outliers', 0)
                
                print(f"  ✅ {method_name}: {method_details.get('total_outliers', 0)} outliers detected")
                
            except Exception as e:
                print(f"  ❌ Error in {method_name} detection: {e}")
                detection_details[method_name] = {'error': str(e)}
        
        # Store detection results
        self.detection_results[dataset_name] = {
            'detection_plan': detection_plan,
            'detection_details': detection_details,
            'total_outliers_detected': int(total_outliers_detected),
            'detection_columns': detection_df.columns.tolist()
        }
        
        # Combine original data with detection results
        result_df = pd.concat([df, detection_df], axis=1)
        
        print(f"  🎉 Total outliers detected: {total_outliers_detected}")
        
        return result_df, detection_details
    
    def _create_detection_plan(self, analysis: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Create automatic outlier detection plan."""
        detection_plan = {
            'dataset_name': analysis['dataset_name'],
            'methods': {},
            'columns_to_analyze': [],
            'recommendations': analysis['outlier_recommendations']
        }
        
        column_analysis = analysis['column_analysis']
        
        # Select columns with outliers
        columns_with_outliers = [col for col, info in column_analysis.items() 
                               if info['has_outliers']]
        detection_plan['columns_to_analyze'] = columns_with_outliers
        
        # Determine appropriate detection methods based on data characteristics
        methods_to_use = []
        
        # Always use IQR for general detection
        methods_to_use.append(('iqr', {'multiplier': 1.5}))
        
        # Use Z-score for normally distributed data
        normal_cols = [col for col, info in column_analysis.items() 
                      if info['has_outliers'] and abs(info['statistics']['skewness']) < 2]
        if normal_cols:
            methods_to_use.append(('zscore', {'threshold': 3}))
        
        # Use isolation forest for complex patterns
        if len(columns_with_outliers) >= 3:
            methods_to_use.append(('isolation_forest', {'contamination': 'auto'}))
        
        # Use local outlier factor for local density-based detection
        if len(columns_with_outliers) >= 2:
            methods_to_use.append(('lof', {'n_neighbors': 20, 'contamination': 'auto'}))
        
        # Use MAD for robust detection
        methods_to_use.append(('mad', {'threshold': 3}))
        
        # Create method configurations
        for method_name, params in methods_to_use:
            detection_plan['methods'][method_name] = {
                'columns': columns_with_outliers,
                'params': params
            }
        
        return detection_plan
    
    def treat_outliers(self, df: pd.DataFrame, dataset_name: str,
                      treatment_plan: Dict[str, Any] = None,
                      detection_results: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Treat outliers based on detection results and treatment plan.
        """
        print(f"🔄 Treating outliers for: {dataset_name}")
        
        # Use provided detection results or detect if not provided
        if detection_results is None:
            _, detection_results = self.detect_outliers(df, dataset_name)
        
        # Auto-generate treatment plan if not provided
        if not treatment_plan:
            analysis = self.analyze_outlier_characteristics(df, dataset_name)
            treatment_plan = self._create_treatment_plan(analysis, detection_results, df)
            print(f"  📋 Auto-generated treatment plan with {len(treatment_plan['strategies'])} strategies")
        
        treated_df = df.copy()
        treatment_details = {}
        
        total_outliers_treated = 0
        total_columns_treated = 0
        
        # Apply treatment strategies
        for strategy_name, strategy_config in treatment_plan.get('strategies', {}).items():
            try:
                columns_to_treat = strategy_config.get('columns', [])
                method_name = strategy_config.get('method', 'winsorize')
                method_params = strategy_config.get('params', {})
                
                if not columns_to_treat:
                    continue
                
                print(f"  🎯 Applying {strategy_name} treatment to {len(columns_to_treat)} columns...")
                
                # Apply treatment
                strategy_results, strategy_details = self.treatment.apply_treatment_strategy(
                    treated_df, columns_to_treat, method_name, method_params, detection_results
                )
                
                # Update dataframe with treated values
                for col in columns_to_treat:
                    if col in strategy_results.columns:
                        treated_df[col] = strategy_results[col]
                
                treatment_details[strategy_name] = strategy_details
                total_outliers_treated += strategy_details.get('outliers_treated', 0)
                total_columns_treated += len(columns_to_treat)
                
                print(f"  ✅ {strategy_name}: {strategy_details.get('outliers_treated', 0)} outliers treated")
                
            except Exception as e:
                print(f"  ❌ Error in {strategy_name} treatment: {e}")
                treatment_details[strategy_name] = {'error': str(e)}
        
        # Store treatment results
        self.treatment_strategies[dataset_name] = {
            'treatment_plan': treatment_plan,
            'treatment_details': treatment_details,
            'total_outliers_treated': int(total_outliers_treated),
            'total_columns_treated': int(total_columns_treated)
        }
        
        # Calculate treatment effectiveness properly
        total_outliers_detected = self.detection_results.get(dataset_name, {}).get('total_outliers_detected', 0)
        if total_outliers_detected > 0:
            treatment_effectiveness = (total_outliers_treated / total_outliers_detected) * 100
        else:
            treatment_effectiveness = 0.0
        
        # Store overall report
        self.outlier_report[dataset_name] = {
            'timestamp': datetime.now().isoformat(),
            'detection_results': self._convert_to_serializable(self.detection_results.get(dataset_name, {})),
            'treatment_strategies': self._convert_to_serializable(self.treatment_strategies.get(dataset_name, {})),
            'original_shape': list(df.shape),
            'treated_shape': list(treated_df.shape),
            'summary': {
                'total_outliers_detected': int(total_outliers_detected),
                'total_outliers_treated': int(total_outliers_treated),
                'treatment_effectiveness': float(treatment_effectiveness)
            }
        }
        
        print(f"  🎉 Total outliers treated: {total_outliers_treated}")
        
        return treated_df, treatment_details
    
    def _create_treatment_plan(self, analysis: Dict[str, Any], 
                             detection_results: Dict[str, Any], 
                             df: pd.DataFrame) -> Dict[str, Any]:
        """Create automatic outlier treatment plan."""
        treatment_plan = {
            'dataset_name': analysis['dataset_name'],
            'strategies': {},
            'recommendations': analysis['outlier_recommendations']
        }
        
        column_analysis = analysis['column_analysis']
        
        # Group columns by treatment strategy
        winsorize_cols = []
        cap_cols = []
        transform_cols = []
        impute_cols = []
        
        for column, info in column_analysis.items():
            if not info['has_outliers']:
                continue
            
            severity = info['outlier_severity']
            outlier_type = info['outlier_type']
            
            if severity == 'high' or outlier_type == 'skewness_induced':
                # Aggressive treatment for high severity or skewed data
                transform_cols.append(column)
            elif severity == 'medium':
                # Moderate treatment
                winsorize_cols.append(column)
            else:
                # Conservative treatment for low severity
                cap_cols.append(column)
            
            # Always consider imputation for extreme cases
            if info['outlier_percentage'] > 20:
                impute_cols.append(column)
        
        # Create strategy configurations
        if winsorize_cols:
            treatment_plan['strategies']['winsorize_moderate'] = {
                'columns': winsorize_cols,
                'method': 'winsorize',
                'params': {'limits': [0.05, 0.05]}  # 5% on each tail
            }
        
        if cap_cols:
            treatment_plan['strategies']['cap_conservative'] = {
                'columns': cap_cols,
                'method': 'cap',
                'params': {'method': 'iqr', 'multiplier': 1.5}
            }
        
        if transform_cols:
            treatment_plan['strategies']['transform_aggressive'] = {
                'columns': transform_cols,
                'method': 'transform',
                'params': {'method': 'log'}  # Log transformation
            }
        
        if impute_cols:
            treatment_plan['strategies']['impute_extreme'] = {
                'columns': impute_cols,
                'method': 'impute',
                'params': {'method': 'median'}
            }
        
        return treatment_plan
    
    def compare_methods(self, dataset_name: str) -> pd.DataFrame:
        """Compare different outlier detection methods."""
        if dataset_name not in self.detection_results:
            print(f"⚠️  No detection results found for {dataset_name}")
            return pd.DataFrame()
        
        detection_details = self.detection_results[dataset_name]['detection_details']
        
        comparison_data = []
        
        for method_name, details in detection_details.items():
            if 'error' in details:
                continue
            
            comparison_data.append({
                'method': method_name,
                'outliers_detected': details.get('total_outliers', 0),
                'columns_analyzed': len(details.get('column_results', {})),
                'execution_time': details.get('execution_time', 0),
                'success_rate': details.get('success_rate', 0)
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_outlier_summary(self) -> Dict[str, Any]:
        """Get summary of all outlier management operations."""
        summary = {
            'total_datasets_processed': len(self.outlier_report),
            'total_outliers_detected': 0,
            'total_outliers_treated': 0,
            'average_treatment_effectiveness': 0.0,
            'detection_methods_used': {},
            'treatment_strategies_used': {}
        }
        
        total_effectiveness = 0.0
        dataset_count = 0
        
        for dataset, report in self.outlier_report.items():
            dataset_summary = report.get('summary', {})
            
            summary['total_outliers_detected'] += dataset_summary.get('total_outliers_detected', 0)
            summary['total_outliers_treated'] += dataset_summary.get('total_outliers_treated', 0)
            
            effectiveness = dataset_summary.get('treatment_effectiveness', 0.0)
            if effectiveness > 0:
                total_effectiveness += effectiveness
                dataset_count += 1
            
            # Count method usage
            detection_results = report.get('detection_results', {})
            detection_details = detection_results.get('detection_details', {})
            for method_name in detection_details.keys():
                summary['detection_methods_used'][method_name] = summary['detection_methods_used'].get(method_name, 0) + 1
            
            # Count strategy usage
            treatment_strategies = report.get('treatment_strategies', {})
            treatment_details = treatment_strategies.get('treatment_details', {})
            for strategy_name in treatment_details.keys():
                summary['treatment_strategies_used'][strategy_name] = summary['treatment_strategies_used'].get(strategy_name, 0) + 1
        
        if dataset_count > 0:
            summary['average_treatment_effectiveness'] = float(total_effectiveness / dataset_count)
        
        return summary
    
    def _convert_to_serializable(self, obj):
        """Convert numpy/pandas types to JSON serializable types."""
        if isinstance(obj, (np.ndarray, pd.Series)):
            return [self._convert_to_serializable(x) for x in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        
        try:
            if pd.isna(obj):
                return None
        except (ValueError, TypeError):
            pass
        
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(x) for x in obj]
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        else:
            return str(obj)
    
    def save_outlier_report(self, output_path: str = "outlier_management_report.json"):
        """Save outlier management report to JSON."""
        serializable_report = self._convert_to_serializable(self.outlier_report)
        with open(output_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        print(f"✓ Outlier report saved to: {output_path}")
