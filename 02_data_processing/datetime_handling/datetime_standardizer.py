# datetime_standardizer.py
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import pytz
from typing import Dict, List, Any, Optional, Union
import re
import warnings
from multi_format_parser import MultiFormatDateTimeParser
from timezone_normalizer import TimezoneNormalizer

class DateTimeStandardizer:
    """
    Main class for standardizing datetime columns with multi-format parsing
    and timezone normalization capabilities.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.multi_parser = MultiFormatDateTimeParser(self.config)
        self.timezone_normalizer = TimezoneNormalizer(self.config)
        self.standardization_report = {}
    
    def _convert_to_serializable(self, obj):
        """Convert numpy/pandas types to JSON serializable types"""
        # Handle numpy arrays and pandas Series first
        if isinstance(obj, (np.ndarray, pd.Series)):
            return [self._convert_to_serializable(x) for x in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        
        # Handle scalar values
        try:
            if pd.isna(obj):
                return None
        except (ValueError, TypeError):
            # If pd.isna fails (e.g., for arrays), treat as not null
            pass
        
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(x) for x in obj]
        else:
            # For any other type, try to convert to string
            try:
                return str(obj)
            except:
                return None
    
    def detect_datetime_columns(self, df: pd.DataFrame, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Detect columns that likely contain datetime data.
        """
        datetime_columns = {}
        
        for column in df.columns:
            # Skip non-string/object columns for efficiency
            if not pd.api.types.is_string_dtype(df[column]) and not pd.api.types.is_object_dtype(df[column]):
                continue
            
            # Check column name patterns
            column_lower = column.lower()
            datetime_patterns = [
                r'.*date.*', r'.*time.*', r'.*timestamp.*', r'created.*', 
                r'updated.*', r'published.*', r'modified.*', r'^dt$', r'^ts$'
            ]
            
            is_datetime_like = any(re.match(pattern, column_lower) for pattern in datetime_patterns)
            
            if is_datetime_like:
                # Sample data for analysis
                sample_data = df[column].dropna().head(sample_size)
                
                if len(sample_data) > 0:
                    # Try to parse as datetime
                    parse_success_rate = self.multi_parser.assess_parsability(sample_data)
                    
                    datetime_columns[column] = {
                        'detection_confidence': parse_success_rate,
                        'sample_values': self._convert_to_serializable(sample_data.head(3).tolist()),
                        'null_count': int(df[column].isnull().sum()),
                        'null_percentage': float((df[column].isnull().sum() / len(df)) * 100),
                        'data_type': str(df[column].dtype)
                    }
        
        return datetime_columns
    
    def standardize_dataset(self, df: pd.DataFrame, dataset_name: str, 
                          datetime_columns: List[str] = None,
                          target_timezone: str = 'UTC',
                          output_format: str = 'iso') -> pd.DataFrame:
        """
        Standardize datetime columns in a dataset.
        """
        standardized_df = df.copy()
        standardization_details = {}
        
        print(f"🕒 Standardizing datetime columns for: {dataset_name}")
        
        # Auto-detect datetime columns if not provided
        if not datetime_columns:
            detected_columns = self.detect_datetime_columns(df)
            datetime_columns = [col for col, info in detected_columns.items() 
                              if info['detection_confidence'] > 0.5]
            print(f"  🔍 Auto-detected datetime columns: {datetime_columns}")
        
        for column in datetime_columns:
            if column not in df.columns:
                print(f"  ⚠️  Column '{column}' not found in dataset")
                continue
            
            try:
                print(f"  📅 Processing: {column}")
                original_dtype = str(df[column].dtype)
                original_sample = self._convert_to_serializable(df[column].head(3).tolist())
                
                # Step 1: Parse using multi-format parser
                parsed_series = self.multi_parser.parse_series(df[column])
                parse_success_rate = (parsed_series.notna().sum() / len(parsed_series)) * 100
                
                # Step 2: Normalize timezone
                normalized_series = self.timezone_normalizer.normalize_timezone(
                    parsed_series, target_timezone
                )
                
                # Step 3: Format output
                if output_format == 'iso':
                    formatted_series = normalized_series.dt.strftime('%Y-%m-%dT%H:%M:%S%z')
                elif output_format == 'standard':
                    formatted_series = normalized_series.dt.strftime('%Y-%m-%d %H:%M:%S %Z')
                else:
                    formatted_series = normalized_series
                
                # Update dataframe
                standardized_df[column] = formatted_series
                
                # Record details
                standardization_details[column] = {
                    'original_dtype': original_dtype,
                    'parsed_dtype': 'datetime64[ns, UTC]',
                    'parse_success_rate': float(parse_success_rate),
                    'target_timezone': target_timezone,
                    'output_format': output_format,
                    'original_sample': original_sample,
                    'standardized_sample': self._convert_to_serializable(standardized_df[column].head(3).tolist()),
                    'null_count_before': int(df[column].isnull().sum()),
                    'null_count_after': int(standardized_df[column].isnull().sum())
                }
                
                print(f"  ✅ {column}: {original_dtype} → datetime (success: {parse_success_rate:.1f}%)")
                
            except Exception as e:
                print(f"  ❌ Error standardizing {column}: {e}")
                standardization_details[column] = {
                    'error': str(e),
                    'original_dtype': str(df[column].dtype)
                }
        
        # Store report
        self.standardization_report[dataset_name] = {
            'timestamp': datetime.now().isoformat(),
            'target_timezone': target_timezone,
            'output_format': output_format,
            'columns_processed': len(datetime_columns),
            'standardization_details': standardization_details,
            'original_shape': list(df.shape),
            'standardized_shape': list(standardized_df.shape)
        }
        
        return standardized_df
    
    def batch_standardize(self, data_dict: Dict[str, pd.DataFrame],
                         datetime_column_mapping: Dict[str, List[str]] = None,
                         target_timezone: str = 'UTC',
                         output_format: str = 'iso') -> Dict[str, pd.DataFrame]:
        """
        Standardize multiple datasets in batch.
        """
        standardized_data = {}
        
        for dataset_name, df in data_dict.items():
            datetime_columns = None
            if datetime_column_mapping and dataset_name in datetime_column_mapping:
                datetime_columns = datetime_column_mapping[dataset_name]
            
            standardized_df = self.standardize_dataset(
                df, dataset_name, datetime_columns, target_timezone, output_format
            )
            standardized_data[dataset_name] = standardized_df
        
        return standardized_data
    
    def get_standardization_summary(self) -> Dict[str, Any]:
        """Get summary of all standardization operations."""
        summary = {
            'total_datasets_processed': len(self.standardization_report),
            'total_columns_standardized': 0,
            'average_success_rate': 0.0,
            'timezone_distribution': {},
            'format_distribution': {}
        }
        
        total_success_rate = 0.0
        total_columns = 0
        
        for dataset, report in self.standardization_report.items():
            timezone = report.get('target_timezone', 'unknown')
            output_format = report.get('output_format', 'unknown')
            
            summary['timezone_distribution'][timezone] = summary['timezone_distribution'].get(timezone, 0) + 1
            summary['format_distribution'][output_format] = summary['format_distribution'].get(output_format, 0) + 1
            
            columns_processed = report.get('columns_processed', 0)
            summary['total_columns_standardized'] += columns_processed
            
            # Calculate average success rate for this dataset
            details = report.get('standardization_details', {})
            if details:
                dataset_success_rates = [
                    col_info.get('parse_success_rate', 0.0) 
                    for col_info in details.values() 
                    if 'parse_success_rate' in col_info
                ]
                if dataset_success_rates:
                    total_success_rate += sum(dataset_success_rates) / len(dataset_success_rates)
                    total_columns += len(dataset_success_rates)
        
        if total_columns > 0:
            summary['average_success_rate'] = round(total_success_rate / total_columns, 2)
        
        return summary
    
    def save_standardization_report(self, output_path: str = "datetime_standardization_report.json"):
        """Save standardization report to JSON."""
        import json
        
        serializable_report = self._convert_to_serializable(self.standardization_report)
        with open(output_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        print(f"✓ Standardization report saved to: {output_path}")
