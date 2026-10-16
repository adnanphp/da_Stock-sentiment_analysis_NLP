# data_scaler.py
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.preprocessing import PowerTransformer, QuantileTransformer
from sklearn.exceptions import NotFittedError
import joblib
import os

warnings.filterwarnings('ignore')

class DataScaler:
    """
    Main data normalization and scaling class with intelligent scaler selection
    and scale inversion capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.scaling_report = {}
        self.fitted_scalers = {}  # Store fitted scalers for inversion
        self.scaling_parameters = {}  # Store scaling parameters
        
        # Initialize specialized components
        from scaler_selector import ScalerSelector
        from scale_inverter import ScaleInverter
        
        self.scaler_selector = ScalerSelector(config)
        self.scale_inverter = ScaleInverter(config)
    
    def analyze_scaling_needs(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Analyze dataset to determine scaling requirements and recommendations.
        """
        print(f"🔍 Analyzing scaling needs for: {dataset_name}")
        
        analysis = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'overview': {
                'total_columns': len(df.columns),
                'numeric_columns': 0,
                'columns_needing_scaling': 0,
                'columns_to_exclude': 0
            },
            'column_analysis': {},
            'scaling_recommendations': []
        }
        
        # Analyze each numeric column
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            col_analysis = self._analyze_column_for_scaling(df[column], column)
            analysis['column_analysis'][column] = col_analysis
            
            analysis['overview']['numeric_columns'] += 1
            
            if col_analysis['needs_scaling']:
                analysis['overview']['columns_needing_scaling'] += 1
            else:
                analysis['overview']['columns_to_exclude'] += 1
        
        # Generate scaling recommendations
        analysis['scaling_recommendations'] = self._generate_scaling_recommendations(analysis, df)
        
        return analysis
    
    def _analyze_column_for_scaling(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Analyze a single column for scaling requirements."""
        # Remove nulls for analysis
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return {
                'needs_scaling': False,
                'reason': 'All values are null',
                'data_type': 'numeric',
                'statistics': {}
            }
        
        # Calculate statistics
        stats = {
            'min': float(clean_series.min()),
            'max': float(clean_series.max()),
            'mean': float(clean_series.mean()),
            'std': float(clean_series.std()),
            'skewness': float(clean_series.skew()),
            'kurtosis': float(clean_series.kurtosis()),
            'q1': float(clean_series.quantile(0.25)),
            'q3': float(clean_series.quantile(0.75)),
            'iqr': float(clean_series.quantile(0.75) - clean_series.quantile(0.25)),
            'unique_values': int(clean_series.nunique()),
            'zero_count': int((clean_series == 0).sum()),
            'negative_count': int((clean_series < 0).sum())
        }
        
        # Determine if scaling is needed
        needs_scaling, reason = self._determine_scaling_need(stats, clean_series, column_name)
        
        analysis = {
            'needs_scaling': needs_scaling,
            'reason': reason,
            'data_type': 'numeric',
            'statistics': stats,
            'sample_values': clean_series.head(3).tolist()
        }
        
        return analysis
    
    def _determine_scaling_need(self, stats: Dict[str, float], series: pd.Series, column_name: str) -> Tuple[bool, str]:
        """Determine if a column needs scaling and why."""
        column_lower = column_name.lower()
        
        # Check for specific column patterns that might not need scaling
        exclude_patterns = [
            'id', 'index', 'count', 'flag', 'binary', 'dummy', 'encoded',
            'year', 'month', 'day', 'week', 'quarter', 'season'
        ]
        
        if any(pattern in column_lower for pattern in exclude_patterns):
            return False, "Column matches exclusion pattern"
        
        # Check if data is already normalized (0-1 range)
        if 0 <= stats['min'] <= 1 and 0 <= stats['max'] <= 1:
            return False, "Data already in 0-1 range"
        
        # Check if data is already standardized (mean ~0, std ~1)
        if abs(stats['mean']) < 0.1 and 0.9 < stats['std'] < 1.1:
            return False, "Data already standardized"
        
        # Check for constant values
        if stats['unique_values'] <= 1:
            return False, "Constant or near-constant values"
        
        # Check for binary data
        if stats['unique_values'] == 2 and set(series.unique()) in [{0, 1}, {-1, 1}, {0.0, 1.0}]:
            return False, "Binary data"
        
        # Check for extreme ranges that need scaling
        value_range = stats['max'] - stats['min']
        if value_range > 1000 or stats['std'] > 100:
            return True, "Large value range or high variance"
        
        # Check for skewed data
        if abs(stats['skewness']) > 2:
            return True, "Highly skewed data"
        
        # Default to scaling for numeric columns
        return True, "General numeric scaling recommended"
    
    def _generate_scaling_recommendations(self, analysis: Dict[str, Any], df: pd.DataFrame) -> List[str]:
        """Generate scaling recommendations based on analysis."""
        recommendations = []
        column_analysis = analysis['column_analysis']
        
        # Count different scaling needs
        scaling_columns = [col for col, info in column_analysis.items() if info['needs_scaling']]
        
        if scaling_columns:
            recommendations.append(f"Scale {len(scaling_columns)} numeric columns")
            
            # Analyze distribution types for scaler recommendations
            normal_count = 0
            skewed_count = 0
            outlier_count = 0
            
            for col, info in column_analysis.items():
                if info['needs_scaling']:
                    stats = info['statistics']
                    if abs(stats['skewness']) < 1:
                        normal_count += 1
                    else:
                        skewed_count += 1
                    
                    # Check for outliers using IQR method
                    iqr = stats['iqr']
                    lower_bound = stats['q1'] - 1.5 * iqr
                    upper_bound = stats['q3'] + 1.5 * iqr
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    
                    if outliers > len(df) * 0.05:  # More than 5% outliers
                        outlier_count += 1
            
            if normal_count > 0:
                recommendations.append(f"Use StandardScaler for {normal_count} normally distributed columns")
            if skewed_count > 0:
                recommendations.append(f"Use PowerTransformer for {skewed_count} skewed columns")
            if outlier_count > 0:
                recommendations.append(f"Use RobustScaler for {outlier_count} columns with outliers")
        
        return recommendations
    
    def scale_dataset(self, df: pd.DataFrame, dataset_name: str,
                     scaling_plan: Dict[str, Any] = None,
                     fit_scalers: bool = True) -> pd.DataFrame:
        """
        Scale dataset based on scaling plan.
        """
        scaled_df = df.copy()
        scaling_details = {}
        
        print(f"📊 Scaling dataset: {dataset_name}")
        
        # Auto-generate scaling plan if not provided
        if not scaling_plan:
            analysis = self.analyze_scaling_needs(df, dataset_name)
            scaling_plan = self._create_scaling_plan(analysis, df)
            print(f"  📋 Auto-generated scaling plan for {len(scaling_plan['columns_to_scale'])} columns")
        
        # Store original data for inversion
        if fit_scalers:
            self.scaling_parameters[dataset_name] = {
                'original_columns': {},
                'scaler_metadata': {}
            }
        
        total_columns_scaled = 0
        
        # Apply scaling to each column group
        for scaler_type, column_config in scaling_plan.get('scaler_groups', {}).items():
            try:
                columns_to_scale = column_config.get('columns', [])
                scaler_params = column_config.get('scaler_params', {})
                
                if not columns_to_scale:
                    continue
                
                print(f"  🎯 Applying {scaler_type} to {len(columns_to_scale)} columns")
                
                # Select and fit scaler
                scaler = self.scaler_selector.select_scaler(scaler_type, scaler_params)
                
                # Scale the columns
                scaled_columns, scaler_details = self._apply_scaler(
                    scaled_df, columns_to_scale, scaler, fit_scalers
                )
                
                # Update dataframe with scaled columns
                for col in columns_to_scale:
                    if f"{col}_scaled" in scaled_columns.columns:
                        scaled_df[f"{col}_scaled"] = scaled_columns[f"{col}_scaled"]
                
                scaling_details[scaler_type] = scaler_details
                total_columns_scaled += len(columns_to_scale)
                
                # Store fitted scaler for inversion
                if fit_scalers:
                    self.fitted_scalers[dataset_name] = self.fitted_scalers.get(dataset_name, {})
                    self.fitted_scalers[dataset_name][scaler_type] = scaler
                    
                    # Store scaling parameters
                    for col in columns_to_scale:
                        self.scaling_parameters[dataset_name]['original_columns'][col] = {
                            'original_min': float(df[col].min()),
                            'original_max': float(df[col].max()),
                            'original_mean': float(df[col].mean()),
                            'original_std': float(df[col].std())
                        }
                
                print(f"  ✅ {scaler_type}: {len(columns_to_scale)} columns scaled")
                
            except Exception as e:
                print(f"  ❌ Error applying {scaler_type}: {e}")
                scaling_details[scaler_type] = {'error': str(e)}
        
        # Store report
        self.scaling_report[dataset_name] = {
            'timestamp': datetime.now().isoformat(),
            'scaling_plan': scaling_plan,
            'total_columns_scaled': total_columns_scaled,
            'scaling_details': scaling_details,
            'original_shape': list(df.shape),
            'scaled_shape': list(scaled_df.shape),
            'scaled_columns': [col for col in scaled_df.columns if col.endswith('_scaled')]
        }
        
        print(f"  🎉 Total columns scaled: {total_columns_scaled}")
        
        return scaled_df
    
    def _create_scaling_plan(self, analysis: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Create automatic scaling plan based on analysis."""
        scaling_plan = {
            'dataset_name': analysis['dataset_name'],
            'scaler_groups': {},
            'columns_to_scale': [],
            'recommendations': analysis['scaling_recommendations']
        }
        
        column_analysis = analysis['column_analysis']
        
        # Group columns by scaling needs
        standard_columns = []
        robust_columns = []
        power_columns = []
        minmax_columns = []
        
        for column, info in column_analysis.items():
            if not info['needs_scaling']:
                continue
            
            stats = info['statistics']
            scaling_plan['columns_to_scale'].append(column)
            
            # Determine appropriate scaler based on data characteristics
            if abs(stats['skewness']) > 2:
                power_columns.append(column)
            elif stats['iqr'] > 0 and (stats['max'] - stats['min']) / stats['iqr'] > 10:
                robust_columns.append(column)
            elif 0 <= stats['min'] and stats['max'] <= 1:
                # Already in good range, maybe minmax to be safe
                minmax_columns.append(column)
            else:
                standard_columns.append(column)
        
        # Create scaler groups
        if standard_columns:
            scaling_plan['scaler_groups']['standard'] = {
                'columns': standard_columns,
                'scaler_params': {'with_mean': True, 'with_std': True}
            }
        
        if robust_columns:
            scaling_plan['scaler_groups']['robust'] = {
                'columns': robust_columns,
                'scaler_params': {'with_centering': True, 'with_scaling': True, 'quantile_range': (25.0, 75.0)}
            }
        
        if power_columns:
            scaling_plan['scaler_groups']['power'] = {
                'columns': power_columns,
                'scaler_params': {'method': 'yeo-johnson', 'standardize': True}
            }
        
        if minmax_columns:
            scaling_plan['scaler_groups']['minmax'] = {
                'columns': minmax_columns,
                'scaler_params': {'feature_range': (0, 1)}
            }
        
        return scaling_plan
    
    def _apply_scaler(self, df: pd.DataFrame, columns: List[str], 
                     scaler: Any, fit_scalers: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply scaler to specified columns."""
        scaled_df = pd.DataFrame(index=df.index)
        details = {'columns_scaled': [], 'scaler_type': type(scaler).__name__}
        
        # Extract data for scaling
        data_to_scale = df[columns].values
        
        try:
            # Fit and transform or just transform
            if fit_scalers:
                scaled_data = scaler.fit_transform(data_to_scale)
            else:
                scaled_data = scaler.transform(data_to_scale)
            
            # Create scaled columns
            for i, col in enumerate(columns):
                scaled_col_name = f"{col}_scaled"
                scaled_df[scaled_col_name] = scaled_data[:, i]
                details['columns_scaled'].append(scaled_col_name)
            
            # Store scaler statistics
            if hasattr(scaler, 'mean_') and scaler.mean_ is not None:
                details['scaler_mean'] = scaler.mean_.tolist()
            if hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                details['scaler_scale'] = scaler.scale_.tolist()
            if hasattr(scaler, 'n_features_in_'):
                details['n_features'] = scaler.n_features_in_
            
        except Exception as e:
            print(f"      ⚠️  Error in scaling: {e}")
            details['error'] = str(e)
        
        return scaled_df, details
    
    def invert_scaling(self, scaled_data: pd.DataFrame, dataset_name: str, 
                      columns: List[str] = None) -> pd.DataFrame:
        """
        Invert scaling transformation to get back original scale.
        """
        print(f"🔄 Inverting scaling for: {dataset_name}")
        
        if dataset_name not in self.fitted_scalers:
            print(f"  ⚠️  No fitted scalers found for {dataset_name}")
            return scaled_data
        
        inverted_df = scaled_data.copy()
        
        # Get all scaled columns if specific columns not provided
        if columns is None:
            columns = [col.replace('_scaled', '') for col in scaled_data.columns 
                      if col.endswith('_scaled')]
        
        inverted_count = 0
        
        for column in columns:
            scaled_col = f"{column}_scaled"
            original_col = column
            
            if scaled_col not in scaled_data.columns:
                print(f"  ⚠️  Scaled column {scaled_col} not found")
                continue
            
            try:
                # Find which scaler was used for this column
                scaler_used = None
                scaler_type = None
                
                for scaler_type, scaler in self.fitted_scalers[dataset_name].items():
                    # Check if this scaler was used for any of our columns
                    scaling_plan = self.scaling_report.get(dataset_name, {}).get('scaling_plan', {})
                    scaler_columns = scaling_plan.get('scaler_groups', {}).get(scaler_type, {}).get('columns', [])
                    
                    if column in scaler_columns:
                        scaler_used = scaler
                        break
                
                if scaler_used is None:
                    print(f"  ⚠️  No scaler found for {column}")
                    continue
                
                # Invert scaling
                inverted_values = self.scale_inverter.invert_scaling(
                    scaled_data[scaled_col].values.reshape(-1, 1),
                    scaler_used,
                    column,
                    self.scaling_parameters.get(dataset_name, {})
                )
                
                inverted_df[original_col] = inverted_values.flatten()
                inverted_count += 1
                print(f"  ✅ Inverted scaling for {column}")
                
            except Exception as e:
                print(f"  ❌ Error inverting {column}: {e}")
        
        print(f"  🎉 Total columns inverted: {inverted_count}")
        return inverted_df
    
    def save_scalers(self, output_dir: str = "fitted_scalers"):
        """Save fitted scalers to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        for dataset_name, scalers in self.fitted_scalers.items():
            dataset_dir = os.path.join(output_dir, dataset_name.replace('.', '_'))
            os.makedirs(dataset_dir, exist_ok=True)
            
            for scaler_type, scaler in scalers.items():
                scaler_path = os.path.join(dataset_dir, f"{scaler_type}_scaler.pkl")
                joblib.dump(scaler, scaler_path)
            
            # Save scaling parameters
            params_path = os.path.join(dataset_dir, "scaling_parameters.json")
            with open(params_path, 'w') as f:
                json.dump(self.scaling_parameters.get(dataset_name, {}), f, indent=2)
        
        print(f"✓ Fitted scalers saved to: {output_dir}")
    
    def load_scalers(self, input_dir: str = "fitted_scalers"):
        """Load fitted scalers from disk."""
        if not os.path.exists(input_dir):
            print(f"⚠️  Scaler directory not found: {input_dir}")
            return
        
        for dataset_dir in os.listdir(input_dir):
            dataset_path = os.path.join(input_dir, dataset_dir)
            if not os.path.isdir(dataset_path):
                continue
            
            dataset_name = dataset_dir.replace('_', '.')
            self.fitted_scalers[dataset_name] = {}
            
            # Load scalers
            for file in os.listdir(dataset_path):
                if file.endswith('_scaler.pkl'):
                    scaler_type = file.replace('_scaler.pkl', '')
                    scaler_path = os.path.join(dataset_path, file)
                    self.fitted_scalers[dataset_name][scaler_type] = joblib.load(scaler_path)
            
            # Load scaling parameters
            params_path = os.path.join(dataset_path, "scaling_parameters.json")
            if os.path.exists(params_path):
                with open(params_path, 'r') as f:
                    self.scaling_parameters[dataset_name] = json.load(f)
        
        print(f"✓ Fitted scalers loaded from: {input_dir}")
    
    def get_scaling_summary(self) -> Dict[str, Any]:
        """Get summary of all scaling operations."""
        summary = {
            'total_datasets_processed': len(self.scaling_report),
            'total_columns_scaled': 0,
            'scaler_distribution': {},
            'average_columns_per_dataset': 0
        }
        
        total_datasets = len(self.scaling_report)
        
        for dataset, report in self.scaling_report.items():
            columns_scaled = report.get('total_columns_scaled', 0)
            summary['total_columns_scaled'] += columns_scaled
            
            # Count scaler usage
            details = report.get('scaling_details', {})
            for scaler_type in details.keys():
                summary['scaler_distribution'][scaler_type] = summary['scaler_distribution'].get(scaler_type, 0) + 1
        
        if total_datasets > 0:
            summary['average_columns_per_dataset'] = summary['total_columns_scaled'] / total_datasets
        
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
    
    def save_scaling_report(self, output_path: str = "data_scaling_report.json"):
        """Save scaling report to JSON."""
        serializable_report = self._convert_to_serializable(self.scaling_report)
        with open(output_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        print(f"✓ Scaling report saved to: {output_path}")
