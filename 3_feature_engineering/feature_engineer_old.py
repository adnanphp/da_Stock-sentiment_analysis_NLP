# feature_engineer.py
import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime
from technical_indicators import TechnicalIndicatorCalculator
from text_feature_extractor import TextFeatureExtractor
from time_feature_creator import TimeFeatureCreator

warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Main feature engineering class with specialized feature creators for different data types.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.feature_engineering_report = {}
        
        # Initialize specialized feature creators
        self.technical_calculator = TechnicalIndicatorCalculator(config)
        self.text_extractor = TextFeatureExtractor(config)
        self.time_creator = TimeFeatureCreator(config)
        
        # Feature registry to track created features
        self.feature_registry = {}
    
    def analyze_dataset_features(self, df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
        """
        Analyze dataset to determine feature engineering opportunities.
        """
        print(f"🔍 Analyzing feature engineering opportunities for: {dataset_name}")
        
        analysis = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'overview': {
                'total_columns': len(df.columns),
                'total_rows': len(df),
                'numeric_columns': 0,
                'text_columns': 0,
                'datetime_columns': 0,
                'categorical_columns': 0
            },
            'column_analysis': {},
            'feature_recommendations': []
        }
        
        # Analyze each column
        for column in df.columns:
            col_analysis = self._analyze_column(df[column], column)
            analysis['column_analysis'][column] = col_analysis
            
            # Update overview counts
            if col_analysis['data_type'] == 'numeric':
                analysis['overview']['numeric_columns'] += 1
            elif col_analysis['data_type'] == 'text':
                analysis['overview']['text_columns'] += 1
            elif col_analysis['data_type'] == 'datetime':
                analysis['overview']['datetime_columns'] += 1
            elif col_analysis['data_type'] == 'categorical':
                analysis['overview']['categorical_columns'] += 1
        
        # Generate feature recommendations
        analysis['feature_recommendations'] = self._generate_feature_recommendations(analysis, df)
        
        return analysis
    
    def _analyze_column(self, series: pd.Series, column_name: str) -> Dict[str, Any]:
        """Analyze a single column for feature engineering potential."""
        analysis = {
            'column_name': column_name,
            'data_type': self._classify_data_type(series),
            'null_count': int(series.isnull().sum()),
            'null_percentage': float((series.isnull().sum() / len(series)) * 100),
            'unique_values': int(series.nunique()),
            'sample_values': series.head(3).tolist()
        }
        
        # Type-specific analysis
        if analysis['data_type'] == 'numeric':
            analysis.update(self._analyze_numeric_column(series))
        elif analysis['data_type'] == 'text':
            analysis.update(self._analyze_text_column(series))
        elif analysis['data_type'] == 'datetime':
            analysis.update(self._analyze_datetime_column(series))
        elif analysis['data_type'] == 'categorical':
            analysis.update(self._analyze_categorical_column(series))
        
        return analysis
    
    def _classify_data_type(self, series: pd.Series) -> str:
        """Classify the data type of a series."""
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(series):
            return 'datetime'
        elif pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            # Check if it's categorical (low cardinality) or text (high cardinality)
            if series.nunique() / len(series) < 0.1 and series.nunique() < 100:
                return 'categorical'
            else:
                return 'text'
        else:
            return 'unknown'
    
    def _analyze_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numeric column for feature engineering."""
        stats = {
            'min': float(series.min()) if not series.empty else 0,
            'max': float(series.max()) if not series.empty else 0,
            'mean': float(series.mean()) if not series.empty else 0,
            'std': float(series.std()) if not series.empty else 0,
            'skewness': float(series.skew()) if not series.empty else 0,
            'is_constant': series.nunique() == 1,
            'has_negative': (series < 0).any() if not series.empty else False,
            'has_zero': (series == 0).any() if not series.empty else False
        }
        return stats
    
    def _analyze_text_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze text column for feature engineering."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return {
                'avg_length': 0,
                'word_count_avg': 0,
                'has_special_chars': False,
                'is_mostly_numeric': False
            }
        
        sample_texts = non_null.head(100)
        lengths = sample_texts.str.len()
        word_counts = sample_texts.str.split().str.len()
        
        return {
            'avg_length': float(lengths.mean()),
            'word_count_avg': float(word_counts.mean()),
            'has_special_chars': sample_texts.str.contains(r'[^\w\s]').any(),
            'is_mostly_numeric': sample_texts.str.match(r'^\d+$').any()
        }
    
    def _analyze_datetime_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze datetime column for feature engineering."""
        non_null = series.dropna()
        if len(non_null) == 0:
            return {
                'date_range': None,
                'has_time_component': False,
                'is_regular_frequency': False
            }
        
        try:
            dt_series = pd.to_datetime(non_null)
            has_time = any(dt.time() != pd.Timestamp('00:00:00').time() for dt in dt_series.head(10))
            
            return {
                'date_range': [dt_series.min().isoformat(), dt_series.max().isoformat()],
                'has_time_component': has_time,
                'is_regular_frequency': self._check_regular_frequency(dt_series)
            }
        except:
            return {
                'date_range': None,
                'has_time_component': False,
                'is_regular_frequency': False
            }
    
    def _analyze_categorical_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze categorical column for feature engineering."""
        value_counts = series.value_counts()
        return {
            'top_categories': value_counts.head(5).to_dict(),
            'category_balance': float(value_counts.std() / value_counts.mean()) if value_counts.mean() > 0 else 0
        }
    
    def _check_regular_frequency(self, dt_series: pd.Series) -> bool:
        """Check if datetime series has regular frequency."""
        if len(dt_series) < 3:
            return False
        
        try:
            diffs = dt_series.sort_values().diff().dropna()
            if len(diffs) == 0:
                return False
            
            # Check if differences are consistent
            most_common_diff = diffs.mode().iloc[0] if not diffs.mode().empty else diffs.iloc[0]
            consistent_count = (diffs == most_common_diff).sum()
            return consistent_count / len(diffs) > 0.8
        except:
            return False
    
    def _generate_feature_recommendations(self, analysis: Dict[str, Any], df: pd.DataFrame) -> List[str]:
        """Generate feature engineering recommendations."""
        recommendations = []
        column_analysis = analysis['column_analysis']
        
        # Check for technical indicators
        price_columns = [col for col, info in column_analysis.items() 
                        if info['data_type'] == 'numeric' and any(pattern in col.lower() 
                        for pattern in ['price', 'close', 'open', 'high', 'low', 'volume'])]
        
        if len(price_columns) >= 4:  # Need OHLC at minimum
            recommendations.append("Add technical indicators (SMA, RSI, MACD, Bollinger Bands)")
        
        # Check for text features
        text_columns = [col for col, info in column_analysis.items() 
                       if info['data_type'] == 'text' and info['avg_length'] > 10]
        
        if text_columns:
            recommendations.append(f"Extract text features from: {', '.join(text_columns)}")
        
        # Check for datetime features
        datetime_columns = [col for col, info in column_analysis.items() 
                          if info['data_type'] == 'datetime']
        
        if datetime_columns:
            recommendations.append(f"Create time-based features from: {', '.join(datetime_columns)}")
        
        # Check for categorical features
        categorical_columns = [col for col, info in column_analysis.items() 
                             if info['data_type'] == 'categorical']
        
        if categorical_columns:
            recommendations.append(f"Encode categorical features: {', '.join(categorical_columns)}")
        
        return recommendations
    
    def engineer_features(self, df: pd.DataFrame, dataset_name: str,
                         feature_plan: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Engineer features based on analysis and feature plan.
        """
        engineered_df = df.copy()
        engineering_details = {}
        
        print(f"🔧 Engineering features for: {dataset_name}")
        
        # Auto-generate feature plan if not provided
        if not feature_plan:
            analysis = self.analyze_dataset_features(df, dataset_name)
            feature_plan = self._create_feature_plan(analysis, df)
            print(f"  📋 Auto-generated feature plan with {len(feature_plan['feature_groups'])} groups")
        
        total_features_created = 0
        
        # Execute feature engineering groups
        for group_name, group_config in feature_plan.get('feature_groups', {}).items():
            try:
                print(f"  🎯 Executing feature group: {group_name}")
                
                if group_name == 'technical_indicators':
                    result_df, details = self.technical_calculator.add_technical_indicators(
                        engineered_df, group_config
                    )
                    engineered_df = result_df
                    engineering_details[group_name] = details
                    total_features_created += details.get('features_created', 0)
                
                elif group_name == 'text_features':
                    result_df, details = self.text_extractor.extract_text_features(
                        engineered_df, group_config
                    )
                    engineered_df = result_df
                    engineering_details[group_name] = details
                    total_features_created += details.get('features_created', 0)
                
                elif group_name == 'time_features':
                    result_df, details = self.time_creator.create_time_features(
                        engineered_df, group_config
                    )
                    engineered_df = result_df
                    engineering_details[group_name] = details
                    total_features_created += details.get('features_created', 0)
                
                elif group_name == 'basic_transformations':
                    result_df, details = self._apply_basic_transformations(
                        engineered_df, group_config
                    )
                    engineered_df = result_df
                    engineering_details[group_name] = details
                    total_features_created += details.get('features_created', 0)
                
                print(f"  ✅ {group_name}: {engineering_details[group_name].get('features_created', 0)} features created")
                
            except Exception as e:
                print(f"  ❌ Error in feature group {group_name}: {e}")
                engineering_details[group_name] = {'error': str(e)}
        
        # Store report
        self.feature_engineering_report[dataset_name] = {
            'timestamp': datetime.now().isoformat(),
            'feature_plan': feature_plan,
            'total_features_created': total_features_created,
            'engineering_details': engineering_details,
            'original_shape': list(df.shape),
            'engineered_shape': list(engineered_df.shape),
            'feature_columns': [col for col in engineered_df.columns if col not in df.columns]
        }
        
        print(f"  🎉 Total features created: {total_features_created}")
        
        return engineered_df
    
    def _create_feature_plan(self, analysis: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
        """Create automatic feature engineering plan based on analysis."""
        feature_plan = {
            'dataset_name': analysis['dataset_name'],
            'feature_groups': {},
            'recommendations': analysis['feature_recommendations']
        }
        
        column_analysis = analysis['column_analysis']
        
        # Technical indicators for price data
        price_columns = [col for col, info in column_analysis.items() 
                        if info['data_type'] == 'numeric' and any(pattern in col.lower() 
                        for pattern in ['close', 'price'])]
        
        if price_columns:
            feature_plan['feature_groups']['technical_indicators'] = {
                'price_column': price_columns[0],
                'volume_column': next((col for col in df.columns if 'volume' in col.lower()), None),
                'indicators': ['sma', 'rsi', 'macd', 'bollinger_bands']
            }
        
        # Text features
        text_columns = [col for col, info in column_analysis.items() 
                       if info['data_type'] == 'text' and info['avg_length'] > 10]
        
        if text_columns:
            feature_plan['feature_groups']['text_features'] = {
                'text_columns': text_columns,
                'feature_types': ['basic_stats', 'linguistic', 'readability']
            }
        
        # Time features
        datetime_columns = [col for col, info in column_analysis.items() 
                          if info['data_type'] == 'datetime']
        
        if datetime_columns:
            feature_plan['feature_groups']['time_features'] = {
                'datetime_columns': datetime_columns,
                'feature_types': ['temporal', 'cyclical', 'lagged']
            }
        
        # Basic transformations
        numeric_columns = [col for col, info in column_analysis.items() 
                          if info['data_type'] == 'numeric' and not info['is_constant']]
        
        if numeric_columns:
            feature_plan['feature_groups']['basic_transformations'] = {
                'numeric_columns': numeric_columns,
                'transformations': ['log', 'standardize', 'interactions']
            }
        
        return feature_plan
    
    def _apply_basic_transformations(self, df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply basic numeric transformations."""
        transformed_df = df.copy()
        details = {'features_created': 0, 'transformations_applied': []}
        
        numeric_columns = config.get('numeric_columns', [])
        transformations = config.get('transformations', [])
        
        for col in numeric_columns:
            if col not in df.columns:
                continue
            
            series = df[col].dropna()
            if len(series) == 0:
                continue
            
            for transformation in transformations:
                try:
                    if transformation == 'log' and (series > 0).all():
                        # Log transformation (add small constant to avoid log(0))
                        transformed_df[f'{col}_log'] = np.log(series + 1e-8)
                        details['features_created'] += 1
                        details['transformations_applied'].append(f'{col}_log')
                    
                    elif transformation == 'standardize':
                        # Z-score standardization
                        mean_val = series.mean()
                        std_val = series.std()
                        if std_val > 0:
                            transformed_df[f'{col}_zscore'] = (series - mean_val) / std_val
                            details['features_created'] += 1
                            details['transformations_applied'].append(f'{col}_zscore')
                    
                    elif transformation == 'interactions':
                        # Create interaction features with other numeric columns
                        for other_col in numeric_columns:
                            if other_col != col and other_col in df.columns:
                                transformed_df[f'{col}_x_{other_col}'] = series * df[other_col]
                                details['features_created'] += 1
                                details['transformations_applied'].append(f'{col}_x_{other_col}')
                
                except Exception as e:
                    print(f"      ⚠️  Error applying {transformation} to {col}: {e}")
        
        return transformed_df, details
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of all feature engineering operations."""
        summary = {
            'total_datasets_processed': len(self.feature_engineering_report),
            'total_features_created': 0,
            'feature_groups_used': {},
            'average_features_per_dataset': 0
        }
        
        total_datasets = len(self.feature_engineering_report)
        
        for dataset, report in self.feature_engineering_report.items():
            features_created = report.get('total_features_created', 0)
            summary['total_features_created'] += features_created
            
            # Count feature groups used
            details = report.get('engineering_details', {})
            for group_name in details.keys():
                summary['feature_groups_used'][group_name] = summary['feature_groups_used'].get(group_name, 0) + 1
        
        if total_datasets > 0:
            summary['average_features_per_dataset'] = summary['total_features_created'] / total_datasets
        
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
    
    def save_engineering_report(self, output_path: str = "feature_engineering_report.json"):
        """Save feature engineering report to JSON."""
        serializable_report = self._convert_to_serializable(self.feature_engineering_report)
        with open(output_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        print(f"✓ Feature engineering report saved to: {output_path}")
