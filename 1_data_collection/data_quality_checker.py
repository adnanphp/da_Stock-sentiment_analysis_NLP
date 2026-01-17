# data_quality_checker.py
# data_quality_checker.py
print("=== RUNNING SIMPLIFIED VERSION 3.0 ===")
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

class DataQualityChecker:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.quality_report = {}
        
    def _convert_to_serializable(self, obj):
        """Convert numpy/pandas types to JSON serializable types"""
        if pd.isna(obj):
            return None
        elif isinstance(obj, (np.integer, pd.Int64Dtype)):
            return int(obj)
        elif isinstance(obj, (np.floating, pd.Float64Dtype)):
            return float(obj)
        elif isinstance(obj, (np.bool_, pd.BooleanDtype)):
            return bool(obj)
        elif isinstance(obj, (np.ndarray, pd.Series)):
            return [self._convert_to_serializable(x) for x in obj]
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(x) for x in obj]
        else:
            return obj
    
    def analyze_dataset_quality(self, df, dataset_name):
        """Simplified data quality analysis to avoid ambiguous truth errors"""
        quality_metrics = {
            'dataset_name': dataset_name,
            'analysis_timestamp': datetime.now().isoformat(),
            'basic_stats': {},
            'quality_issues': {},
            'recommendations': []
        }
        
        try:
            # Basic statistics - safe operations only
            quality_metrics['basic_stats'] = {
                'total_records': len(df),
                'total_columns': len(df.columns),
                'memory_usage_mb': float(df.memory_usage(deep=True).sum() / (1024 * 1024)),
                'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
            
            # Null value analysis - using safe methods
            null_counts = {}
            null_percentages = {}
            total_nulls = 0
            
            for col in df.columns:
                null_count = df[col].isnull().sum()
                null_counts[col] = int(null_count)
                if len(df) > 0:
                    null_percent = (null_count / len(df)) * 100
                    null_percentages[col] = round(float(null_percent), 2)
                total_nulls += null_count
            
            high_null_cols = [col for col, pct in null_percentages.items() if pct > 50]
            
            quality_metrics['quality_issues']['null_values'] = {
                'total_null_count': int(total_nulls),
                'null_by_column': null_counts,
                'null_percentage_by_column': null_percentages,
                'columns_with_high_nulls': high_null_cols
            }
            
            # Duplicate analysis
            duplicate_count = df.duplicated().sum()
            duplicate_pct = (duplicate_count / len(df)) * 100 if len(df) > 0 else 0
            
            quality_metrics['quality_issues']['duplicates'] = {
                'exact_duplicates': int(duplicate_count),
                'duplicate_percentage': float(duplicate_pct)
            }
            
            # Data type consistency - simplified
            type_consistency = {}
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        sample_values = df[col].dropna().head(10)
                        types_found = set()
                        for val in sample_values:
                            types_found.add(type(val).__name__)
                        type_consistency[col] = {
                            'is_mixed_types': len(types_found) > 1,
                            'types_found': list(types_found)
                        }
                    except:
                        type_consistency[col] = {
                            'is_mixed_types': False,
                            'types_found': ['unknown']
                        }
            
            quality_metrics['quality_issues']['type_consistency'] = type_consistency
            
            # Generate recommendations
            recommendations = []
            
            # Null value recommendations
            for col, pct in null_percentages.items():
                if pct > 50:
                    recommendations.append(f"Column '{col}' has {pct}% null values - consider removal")
                elif pct > 20:
                    recommendations.append(f"Column '{col}' has {pct}% null values - consider imputation")
            
            # Duplicate recommendations
            if duplicate_count > 0:
                recommendations.append(f"Dataset has {duplicate_count} duplicate rows ({duplicate_pct:.1f}%)")
            
            # Type consistency recommendations
            for col, consistency in type_consistency.items():
                if consistency.get('is_mixed_types', False):
                    recommendations.append(f"Column '{col}' has mixed data types: {consistency.get('types_found', [])}")
            
            quality_metrics['recommendations'] = recommendations
            
            # Calculate quality score
            quality_score = self._calculate_simple_quality_score(quality_metrics)
            quality_metrics['overall_quality_score'] = int(quality_score)
            
        except Exception as e:
            quality_metrics['error'] = str(e)
            quality_metrics['overall_quality_score'] = 0
        
        return self._convert_to_serializable(quality_metrics)
    
    def _calculate_simple_quality_score(self, quality_metrics):
        """Calculate quality score using simple metrics"""
        try:
            score = 100
            issues = quality_metrics['quality_issues']
            
            # Penalize for null values
            total_records = quality_metrics['basic_stats']['total_records']
            total_columns = quality_metrics['basic_stats']['total_columns']
            
            if total_records > 0 and total_columns > 0:
                total_null_pct = (issues['null_values']['total_null_count'] / 
                                 (total_columns * total_records)) * 100
                score -= min(total_null_pct * 2, 40)
            
            # Penalize for duplicates
            dup_pct = issues['duplicates']['duplicate_percentage']
            score -= min(dup_pct, 20)
            
            # Penalize for type inconsistencies
            mixed_type_cols = sum(1 for consistency in issues['type_consistency'].values() 
                                 if consistency.get('is_mixed_types', False))
            score -= min(mixed_type_cols * 5, 20)
            
            return max(0, round(score))
            
        except:
            return 50
    
    def check_all_datasets(self):
        """Run quality check on all datasets in the directory"""
        print("Starting comprehensive data quality assessment...")
        
        for root, dirs, files in os.walk(self.data_directory):
            for file in files:
                if file.endswith(('.csv', '.parquet')):
                    file_path = os.path.join(root, file)
                    dataset_name = f"{os.path.basename(root)}_{file}"
                    
                    try:
                        print(f"Analyzing: {file_path}")
                        
                        # Read data based on file type
                        if file.endswith('.csv'):
                            df = pd.read_csv(file_path, low_memory=False)
                        else:  # parquet
                            df = pd.read_parquet(file_path)
                        
                        # Skip empty dataframes
                        if len(df) == 0 or len(df.columns) == 0:
                            print(f"⚠️  Skipping empty dataset: {dataset_name}")
                            self.quality_report[dataset_name] = {
                                'error': 'Empty dataset',
                                'overall_quality_score': 0
                            }
                            continue
                        
                        # Analyze quality
                        quality_report = self.analyze_dataset_quality(df, dataset_name)
                        self.quality_report[dataset_name] = quality_report
                        
                        print(f"✓ Completed: {dataset_name} (Score: {quality_report['overall_quality_score']}/100)")
                        
                    except Exception as e:
                        print(f"✗ Error analyzing {file_path}: {e}")
                        self.quality_report[dataset_name] = {
                            'error': str(e),
                            'overall_quality_score': 0
                        }
        
        return self.quality_report
    
    def save_quality_report(self, output_path="data_quality_report.json"):
        """Save comprehensive quality report to JSON"""
        serializable_report = self._convert_to_serializable(self.quality_report)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        print(f"Quality report saved to: {output_path}")
    
    def get_summary_statistics(self):
        """Get overall summary of data quality across all datasets"""
        if not self.quality_report:
            return {"error": "No quality report available"}
        
        total_datasets = len(self.quality_report)
        successful_analyses = sum(1 for report in self.quality_report.values() 
                                 if report.get('overall_quality_score', 0) > 0)
        
        if successful_analyses == 0:
            return {
                'total_datasets_analyzed': total_datasets,
                'successful_analyses': 0,
                'average_quality_score': 0.0,
                'note': 'No successful analyses'
            }
        
        quality_scores = [report['overall_quality_score'] 
                         for report in self.quality_report.values() 
                         if report.get('overall_quality_score', 0) > 0]
        
        avg_score = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        return {
            'total_datasets_analyzed': total_datasets,
            'successful_analyses': successful_analyses,
            'average_quality_score': round(avg_score, 2),
            'datasets_below_70': sum(1 for score in quality_scores if score < 70),
            'failed_analyses': total_datasets - successful_analyses
        }
