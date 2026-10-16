# bulletproof_missing_value_analyzer.py
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

class BulletproofMissingValueAnalyzer:
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
    
    def _safe_has_any_missing(self, row):
        """Completely safe check if a row has any missing values"""
        try:
            # Convert to list and check each value
            values = list(row.values) if hasattr(row, 'values') else list(row)
            for value in values:
                if pd.isna(value):
                    return True
            return False
        except:
            return False
    
    def _safe_column_has_missing(self, column_data):
        """Completely safe check if a column has missing values"""
        try:
            # Use sum which returns a scalar
            return column_data.isnull().sum() > 0
        except:
            # Fallback: check sample of values
            try:
                for i in range(min(100, len(column_data))):
                    if pd.isna(column_data.iloc[i]):
                        return True
                return False
            except:
                return False
    
    def analyze_missing_patterns(self, df, dataset_name):
        """Comprehensive missing value analysis using only safe operations"""
        try:
            print(f"  Analyzing: {dataset_name} (shape: {df.shape})")
            
            analysis = {
                'dataset_name': dataset_name,
                'analysis_timestamp': datetime.now().isoformat(),
                'overview': {},
                'column_analysis': {},
                'pattern_analysis': {},
                'recommendations': []
            }
            
            # 1. OVERVIEW STATISTICS - Using only safe operations
            total_records = len(df)
            total_columns = len(df.columns)
            total_cells = total_records * total_columns
            
            # Calculate total missing using sum (returns scalar)
            total_missing = 0
            for col in df.columns:
                total_missing += df[col].isnull().sum()
            
            missing_percentage = (total_missing / total_cells) * 100 if total_cells > 0 else 0
            
            # Calculate records with missing using safe method
            records_with_missing_count = 0
            for i in range(total_records):
                row_has_missing = False
                for col in df.columns:
                    if pd.isna(df[col].iloc[i]):
                        row_has_missing = True
                        break
                if row_has_missing:
                    records_with_missing_count += 1
            
            records_with_missing_percentage = (records_with_missing_count / total_records) * 100 if total_records > 0 else 0
            
            analysis['overview'] = {
                'total_records': total_records,
                'total_columns': total_columns,
                'total_cells': total_cells,
                'total_missing_cells': int(total_missing),
                'overall_missing_percentage': float(round(missing_percentage, 2)),
                'records_with_missing': records_with_missing_count,
                'records_with_missing_percentage': float(round(records_with_missing_percentage, 2))
            }
            
            # 2. COLUMN-LEVEL ANALYSIS
            for column in df.columns:
                col_data = df[column]
                null_count = col_data.isnull().sum()  # This returns a scalar
                null_percentage = (null_count / total_records) * 100 if total_records > 0 else 0
                
                column_info = {
                    'data_type': str(col_data.dtype),
                    'missing_count': int(null_count),
                    'missing_percentage': float(round(null_percentage, 2)),
                    'unique_values': int(col_data.nunique()) if not col_data.dtype == 'object' else 'N/A for object',
                }
                
                # Get sample values safely
                try:
                    non_null_values = col_data.dropna().head(3).tolist()
                    column_info['sample_values'] = [str(x) for x in non_null_values]
                except:
                    column_info['sample_values'] = []
                
                # Numeric stats only for numeric columns
                if pd.api.types.is_numeric_dtype(col_data):
                    try:
                        numeric_stats = col_data.describe()
                        column_info['numeric_stats'] = {
                            'mean': float(numeric_stats.get('mean', 0)),
                            'std': float(numeric_stats.get('std', 0)),
                            'min': float(numeric_stats.get('min', 0)),
                            'max': float(numeric_stats.get('max', 0))
                        }
                    except:
                        column_info['numeric_stats'] = {'error': 'Could not compute stats'}
                
                analysis['column_analysis'][column] = column_info
            
            # 3. PATTERN ANALYSIS - Simplified but safe
            missing_columns = []
            for col in df.columns:
                if df[col].isnull().sum() > 0:
                    missing_columns.append(col)
            
            analysis['pattern_analysis'] = {
                'columns_with_missing': missing_columns,
                'records_with_missing_count': records_with_missing_count,
                'complete_records_count': total_records - records_with_missing_count,
                'missing_correlation_matrix': {},
                'missing_patterns': {}
            }
            
            # 4. RECOMMENDATIONS
            analysis['recommendations'] = self._generate_missing_value_recommendations(analysis)
            
            return self._convert_to_serializable(analysis)
            
        except Exception as e:
            print(f"  Critical error in analyze_missing_patterns for {dataset_name}: {e}")
            # Return minimal analysis
            return {
                'dataset_name': dataset_name,
                'error': str(e),
                'overview': {
                    'total_records': len(df) if 'df' in locals() else 0,
                    'total_columns': len(df.columns) if 'df' in locals() else 0,
                    'total_cells': (len(df) * len(df.columns)) if 'df' in locals() else 0,
                    'total_missing_cells': 0,
                    'overall_missing_percentage': 0.0,
                    'records_with_missing': 0,
                    'records_with_missing_percentage': 0.0
                },
                'column_analysis': {},
                'pattern_analysis': {},
                'recommendations': ['Error during analysis - data may be corrupted']
            }
    
    def _generate_missing_value_recommendations(self, analysis):
        """Generate recommendations for handling missing values"""
        recommendations = []
        
        if 'error' in analysis:
            return ["❌ Error during analysis - check data format"]
            
        overview = analysis['overview']
        column_analysis = analysis['column_analysis']
        
        # Overall recommendations
        if overview['overall_missing_percentage'] > 50:
            recommendations.append("⚠️  High overall missing rate (>50%) - consider dataset removal or aggressive imputation")
        elif overview['overall_missing_percentage'] > 20:
            recommendations.append("⚠️  Moderate overall missing rate (>20%) - needs careful imputation strategy")
        elif overview['overall_missing_percentage'] > 5:
            recommendations.append("ℹ️  Low overall missing rate (>5%) - standard imputation methods should work")
        else:
            recommendations.append("✅ Minimal missing values - simple imputation or deletion sufficient")
        
        # Column-specific recommendations
        for col, info in column_analysis.items():
            if info['missing_percentage'] > 50:
                recommendations.append(f"🚨 Column '{col}' has {info['missing_percentage']}% missing values - consider removal")
            elif info['missing_percentage'] > 30:
                recommendations.append(f"⚠️  Column '{col}' has {info['missing_percentage']}% missing values - needs advanced imputation")
            elif info['missing_percentage'] > 10:
                recommendations.append(f"ℹ️  Column '{col}' has {info['missing_percentage']}% missing values - standard imputation recommended")
        
        return recommendations
    
    def analyze_all_datasets(self):
        """Analyze missing values across all datasets"""
        print("Starting comprehensive missing value analysis...")
        
        for root, dirs, files in os.walk(self.data_directory):
            for file in files:
                if file.endswith(('.csv', '.parquet')):
                    file_path = os.path.join(root, file)
                    dataset_name = f"{os.path.basename(root)}_{file}"
                    
                    try:
                        print(f"Analyzing missing values: {file_path}")
                        
                        # Read data
                        if file.endswith('.csv'):
                            df = pd.read_csv(file_path, low_memory=False, encoding='utf-8')
                        else:
                            df = pd.read_parquet(file_path)
                        
                        # Skip empty datasets
                        if len(df) == 0 or len(df.columns) == 0:
                            print(f"✓ {dataset_name} - Empty dataset, skipping")
                            continue
                            
                        # Analyze missing values
                        analysis = self.analyze_missing_patterns(df, dataset_name)
                        self.analysis_results[dataset_name] = analysis
                        
                        if 'error' not in analysis:
                            missing_pct = analysis['overview']['overall_missing_percentage']
                            missing_cols = len(analysis['pattern_analysis']['columns_with_missing'])
                            print(f"✓ {dataset_name} - Missing: {missing_pct}% ({missing_cols} columns affected)")
                        else:
                            print(f"⚠️ {dataset_name} - Analysis completed with errors")
                        
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
                        'missing_columns': len(analysis['pattern_analysis']['columns_with_missing'])
                    })
        
        avg_missing = float(np.mean([d['missing_percentage'] for d in datasets_with_missing])) if datasets_with_missing else 0
        
        return self._convert_to_serializable({
            'total_datasets_analyzed': total_datasets,
            'datasets_with_missing_values': len(datasets_with_missing),
            'average_missing_percentage': avg_missing,
            'most_problematic_datasets': sorted(datasets_with_missing, key=lambda x: x['missing_percentage'], reverse=True)[:5]
        })
