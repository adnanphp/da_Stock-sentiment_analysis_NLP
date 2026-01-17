# simple_missing_analyzer.py
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

class SimpleMissingValueAnalyzer:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.analysis_results = {}
    
    def _convert_to_serializable(self, obj):
        """Convert numpy/pandas types to JSON serializable types"""
        if pd.isna(obj):
            return None
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
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
    
    def ultra_safe_missing_check(self, df):
        """Ultra-safe check for missing values without any pandas operations that could fail"""
        try:
            # Method 1: Convert to numpy and check (most reliable)
            total_missing = 0
            for col in df.columns:
                try:
                    col_data = df[col].values  # Convert to numpy array
                    col_missing = np.sum(pd.isna(col_data))
                    total_missing += col_missing
                except:
                    continue
            return total_missing > 0
        except:
            return False
    
    def ultra_safe_analyze(self, df, dataset_name):
        """Ultra-safe analysis that avoids all problematic pandas operations"""
        try:
            print(f"  Ultra-safe analyzing: {dataset_name} (shape: {df.shape})")
            
            # Basic info
            total_records = len(df)
            total_columns = len(df.columns)
            total_cells = total_records * total_columns
            
            # Calculate missing values using numpy (bypass pandas issues)
            total_missing = 0
            missing_by_column = {}
            
            for col in df.columns:
                try:
                    # Convert column to numpy array to avoid pandas issues
                    col_array = df[col].values
                    col_missing = np.sum(pd.isna(col_array))
                    total_missing += col_missing
                    
                    if col_missing > 0:
                        missing_percentage = (col_missing / total_records) * 100
                        missing_by_column[col] = {
                            'missing_count': int(col_missing),
                            'missing_percentage': float(round(missing_percentage, 2)),
                            'data_type': str(df[col].dtype)
                        }
                except Exception as col_error:
                    print(f"    Warning: Could not analyze column {col}: {col_error}")
                    continue
            
            missing_percentage = (total_missing / total_cells) * 100 if total_cells > 0 else 0
            
            # Calculate records with missing (ultra-safe method)
            records_with_missing = 0
            try:
                for i in range(total_records):
                    row_has_missing = False
                    for col in df.columns:
                        try:
                            if pd.isna(df.iloc[i][col]):
                                row_has_missing = True
                                break
                        except:
                            continue
                    if row_has_missing:
                        records_with_missing += 1
            except:
                # If that fails, estimate
                records_with_missing = int(total_missing / total_columns) if total_columns > 0 else 0
            
            analysis = {
                'dataset_name': dataset_name,
                'analysis_timestamp': datetime.now().isoformat(),
                'overview': {
                    'total_records': total_records,
                    'total_columns': total_columns,
                    'total_cells': total_cells,
                    'total_missing_cells': int(total_missing),
                    'overall_missing_percentage': float(round(missing_percentage, 2)),
                    'records_with_missing': records_with_missing,
                    'records_with_missing_percentage': float(round((records_with_missing / total_records * 100), 2)) if total_records > 0 else 0.0
                },
                'column_analysis': missing_by_column,
                'pattern_analysis': {
                    'columns_with_missing': list(missing_by_column.keys()),
                    'records_with_missing_count': records_with_missing,
                    'complete_records_count': total_records - records_with_missing
                },
                'recommendations': self._generate_recommendations(total_missing, total_cells, missing_by_column)
            }
            
            return self._convert_to_serializable(analysis)
            
        except Exception as e:
            print(f"  Critical error in ultra_safe_analyze for {dataset_name}: {e}")
            return {
                'dataset_name': dataset_name,
                'error': str(e),
                'overview': {
                    'total_records': len(df) if 'df' in locals() else 0,
                    'total_columns': len(df.columns) if 'df' in locals() else 0,
                    'total_cells': 0,
                    'total_missing_cells': 0,
                    'overall_missing_percentage': 0.0,
                    'records_with_missing': 0,
                    'records_with_missing_percentage': 0.0
                },
                'column_analysis': {},
                'pattern_analysis': {},
                'recommendations': ['Analysis failed']
            }
    
    def _generate_recommendations(self, total_missing, total_cells, missing_by_column):
        """Generate simple recommendations"""
        recommendations = []
        
        if total_cells == 0:
            return ["Dataset is empty"]
        
        missing_percentage = (total_missing / total_cells) * 100
        
        if missing_percentage > 50:
            recommendations.append("High missing rate (>50%) - consider dataset removal")
        elif missing_percentage > 20:
            recommendations.append("Moderate missing rate (>20%) - needs careful imputation")
        elif missing_percentage > 5:
            recommendations.append("Low missing rate (>5%) - standard imputation should work")
        else:
            recommendations.append("Minimal missing values - simple imputation sufficient")
        
        # Column-specific recommendations
        for col, info in missing_by_column.items():
            if info['missing_percentage'] > 50:
                recommendations.append(f"Column '{col}' has {info['missing_percentage']}% missing - consider removal")
        
        return recommendations
    
    def analyze_all_datasets(self):
        """Analyze missing values across all datasets"""
        print("Starting ultra-safe missing value analysis...")
        
        for root, dirs, files in os.walk(self.data_directory):
            for file in files:
                if file.endswith(('.csv', '.parquet')):
                    file_path = os.path.join(root, file)
                    dataset_name = f"{os.path.basename(root)}_{file}"
                    
                    try:
                        print(f"Analyzing: {file_path}")
                        
                        # Read data
                        if file.endswith('.csv'):
                            df = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
                        else:
                            df = pd.read_parquet(file_path)
                        
                        # Skip empty datasets
                        if len(df) == 0 or len(df.columns) == 0:
                            print(f"⏭️  {dataset_name} - Empty dataset, skipping")
                            continue
                            
                        # Analyze missing values
                        analysis = self.ultra_safe_analyze(df, dataset_name)
                        self.analysis_results[dataset_name] = analysis
                        
                        if 'error' not in analysis:
                            missing_pct = analysis['overview']['overall_missing_percentage']
                            missing_cols = len(analysis['column_analysis'])
                            print(f"✓ {dataset_name} - Missing: {missing_pct}% ({missing_cols} columns)")
                        else:
                            print(f"⚠️ {dataset_name} - Analysis failed")
                        
                    except Exception as e:
                        print(f"✗ Error reading {file_path}: {e}")
                        self.analysis_results[dataset_name] = {'error': f"File read error: {str(e)}"}
        
        return self.analysis_results
    
    def save_analysis_report(self, output_path="missing_value_analysis.json"):
        """Save missing value analysis report"""
        serializable_results = self._convert_to_serializable(self.analysis_results)
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Missing value analysis saved to: {output_path}")
    
    def get_summary_statistics(self):
        """Get summary statistics across all datasets"""
        if not self.analysis_results:
            return {"error": "No analysis results available"}
        
        datasets_with_missing = []
        total_datasets = len(self.analysis_results)
        
        for dataset_name, analysis in self.analysis_results.items():
            if 'overview' in analysis and 'error' not in analysis:
                overview = analysis['overview']
                if overview['total_missing_cells'] > 0:
                    datasets_with_missing.append({
                        'dataset': dataset_name,
                        'missing_percentage': float(overview['overall_missing_percentage']),
                        'missing_columns': len(analysis['column_analysis'])
                    })
        
        avg_missing = float(np.mean([d['missing_percentage'] for d in datasets_with_missing])) if datasets_with_missing else 0
        
        return self._convert_to_serializable({
            'total_datasets_analyzed': total_datasets,
            'datasets_with_missing_values': len(datasets_with_missing),
            'average_missing_percentage': avg_missing,
            'most_problematic_datasets': sorted(datasets_with_missing, key=lambda x: x['missing_percentage'], reverse=True)[:5]
        })
