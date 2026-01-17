# data_profiler.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class DataProfiler:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.profile_reports = {}
        
    def generate_comprehensive_profile(self, df, dataset_name):
        """Generate comprehensive data profile for a dataset"""
        profile = {
            'dataset_name': dataset_name,
            'profile_timestamp': datetime.now().isoformat(),
            'overview': {},
            'variables': {},
            'correlations': {},
            'missing_values': {},
            'interactions': {}
        }
        
        # Overview statistics
        profile['overview'] = self._get_overview_stats(df, dataset_name)
        
        # Variable-level analysis
        profile['variables'] = self._analyze_variables(df)
        
        # Correlation analysis
        profile['correlations'] = self._analyze_correlations(df)
        
        # Missing values pattern
        profile['missing_values'] = self._analyze_missing_patterns(df)
        
        # Variable interactions
        profile['interactions'] = self._analyze_interactions(df)
        
        return profile
    
    def _get_overview_stats(self, df, dataset_name):
        """Get high-level overview statistics"""
        return {
            'number_of_variables': len(df.columns),
            'number_of_observations': len(df),
            'missing_cells': df.isnull().sum().sum(),
            'missing_cells_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'duplicate_rows_percentage': (df.duplicated().sum() / len(df)) * 100,
            'total_size_memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
            'average_record_size_kb': (df.memory_usage(deep=True).sum() / len(df)) / 1024
        }
    
    def _analyze_variables(self, df):
        """Analyze each variable in detail"""
        variables = {}
        
        for column in df.columns:
            col_data = df[column]
            dtype = str(col_data.dtype)
            
            variable_profile = {
                'data_type': dtype,
                'missing_values': col_data.isnull().sum(),
                'missing_percentage': (col_data.isnull().sum() / len(col_data)) * 100,
                'unique_values': col_data.nunique(),
                'unique_percentage': (col_data.nunique() / len(col_data)) * 100
            }
            
            # Numeric variables
            if pd.api.types.is_numeric_dtype(col_data):
                numeric_stats = self._get_numeric_stats(col_data)
                variable_profile.update(numeric_stats)
                
            # Categorical/text variables
            elif pd.api.types.is_string_dtype(col_data) or pd.api.types.is_object_dtype(col_data):
                categorical_stats = self._get_categorical_stats(col_data)
                variable_profile.update(categorical_stats)
                
            # DateTime variables
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                datetime_stats = self._get_datetime_stats(col_data)
                variable_profile.update(datetime_stats)
            
            variables[column] = variable_profile
        
        return variables
    
    def _get_numeric_stats(self, series):
        """Get detailed statistics for numeric variables"""
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return {'error': 'No valid numeric data'}
        
        stats_dict = {
            'mean': float(series_clean.mean()),
            'std': float(series_clean.std()),
            'variance': float(series_clean.var()),
            'min': float(series_clean.min()),
            'max': float(series_clean.max()),
            'range': float(series_clean.max() - series_clean.min()),
            'median': float(series_clean.median()),
            'skewness': float(series_clean.skew()),
            'kurtosis': float(series_clean.kurtosis()),
            'zeros_count': int((series_clean == 0).sum()),
            'negative_count': int((series_clean < 0).sum()),
            'infinite_count': int(np.isinf(series_clean).sum()),
            'q1': float(series_clean.quantile(0.25)),
            'q3': float(series_clean.quantile(0.75)),
            'iqr': float(series_clean.quantile(0.75) - series_clean.quantile(0.25))
        }
        
        # Outlier detection using IQR method
        Q1 = stats_dict['q1']
        Q3 = stats_dict['q3']
        IQR = stats_dict['iqr']
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series_clean[(series_clean < lower_bound) | (series_clean > upper_bound)]
        stats_dict['outliers_count'] = int(len(outliers))
        stats_dict['outliers_percentage'] = (len(outliers) / len(series_clean)) * 100
        
        return stats_dict
    
    def _get_categorical_stats(self, series):
        """Get detailed statistics for categorical variables"""
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return {'error': 'No valid categorical data'}
        
        # Convert to string and handle whitespace
        series_str = series_clean.astype(str).str.strip()
        
        stats_dict = {
            'distinct_count': series_str.nunique(),
            'top_frequencies': series_str.value_counts().head(10).to_dict(),
            'top_categories_percentage': (series_str.value_counts().head(5).sum() / len(series_str)) * 100,
            'avg_string_length': float(series_str.str.len().mean()),
            'min_string_length': int(series_str.str.len().min()),
            'max_string_length': int(series_str.str.len().max()),
            'empty_strings_count': int((series_str == '').sum()),
            'whitespace_only_count': int(series_str.str.isspace().sum())
        }
        
        return stats_dict
    
    def _get_datetime_stats(self, series):
        """Get detailed statistics for datetime variables"""
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            return {'error': 'No valid datetime data'}
        
        stats_dict = {
            'min_date': series_clean.min().isoformat(),
            'max_date': series_clean.max().isoformat(),
            'date_range_days': (series_clean.max() - series_clean.min()).days,
            'weekend_count': int(series_clean.dt.dayofweek.isin([5, 6]).sum()),
            'weekday_count': int(series_clean.dt.dayofweek.isin([0, 1, 2, 3, 4]).sum()),
            'invalid_dates_count': len(series) - len(series_clean)
        }
        
        return stats_dict
    
    def _analyze_correlations(self, df):
        """Analyze correlations between numeric variables"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {'message': 'Insufficient numeric variables for correlation analysis'}
        
        correlation_matrix = numeric_df.corr()
        
        # Find highly correlated pairs
        highly_correlated = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:  # High correlation threshold
                    highly_correlated.append({
                        'variable1': correlation_matrix.columns[i],
                        'variable2': correlation_matrix.columns[j],
                        'correlation': round(corr_value, 3)
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'highly_correlated_pairs': highly_correlated,
            'most_correlated_pair': max(highly_correlated, key=lambda x: abs(x['correlation'])) if highly_correlated else None
        }
    
    def _analyze_missing_patterns(self, df):
        """Analyze patterns in missing values"""
        missing_matrix = df.isnull()
        
        return {
            'total_missing': missing_matrix.sum().sum(),
            'missing_by_column': missing_matrix.sum().to_dict(),
            'columns_with_missing': list(missing_matrix.columns[missing_matrix.sum() > 0]),
            'rows_with_missing': missing_matrix.any(axis=1).sum(),
            'complete_rows': (~missing_matrix.any(axis=1)).sum()
        }
    
    def _analyze_interactions(self, df):
        """Analyze interactions between variables"""
        interactions = {}
        
        # For financial data, analyze key relationships
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Sample some key interactions
            sample_interactions = []
            for i in range(min(3, len(numeric_cols))):
                for j in range(i+1, min(6, len(numeric_cols))):
                    if i != j:
                        col1, col2 = numeric_cols[i], numeric_cols[j]
                        correlation = df[col1].corr(df[col2])
                        sample_interactions.append({
                            'variables': f"{col1} vs {col2}",
                            'correlation': round(correlation, 3),
                            'relationship_strength': 'strong' if abs(correlation) > 0.7 else 
                                                   'moderate' if abs(correlation) > 0.3 else 'weak'
                        })
            
            interactions['numeric_interactions'] = sample_interactions
        
        return interactions
    
    def profile_all_datasets(self):
        """Generate profiles for all datasets in the directory"""
        print("Starting comprehensive data profiling...")
        
        for root, dirs, files in os.walk(self.data_directory):
            for file in files:
                if file.endswith(('.csv', '.parquet')):
                    file_path = os.path.join(root, file)
                    dataset_name = f"{os.path.basename(root)}_{file}"
                    
                    try:
                        print(f"Profiling: {file_path}")
                        
                        # Read data based on file type
                        if file.endswith('.csv'):
                            df = pd.read_csv(file_path, low_memory=False)
                        else:  # parquet
                            df = pd.read_parquet(file_path)
                        
                        # Generate profile
                        profile = self.generate_comprehensive_profile(df, dataset_name)
                        self.profile_reports[dataset_name] = profile
                        
                        print(f"✓ Completed profiling: {dataset_name}")
                        
                    except Exception as e:
                        print(f"✗ Error profiling {file_path}: {e}")
                        self.profile_reports[dataset_name] = {'error': str(e)}
        
        return self.profile_reports
    
    def generate_summary_report(self):
        """Generate a summary report across all datasets"""
        if not self.profile_reports:
            return "No profile reports available. Run profile_all_datasets() first."
        
        summary = {
            'total_datasets_profiled': len(self.profile_reports),
            'profiling_timestamp': datetime.now().isoformat(),
            'dataset_overview': {},
            'data_type_summary': {},
            'quality_insights': {}
        }
        
        # Aggregate statistics
        total_records = 0
        total_columns = 0
        data_type_counts = {}
        
        for dataset_name, profile in self.profile_reports.items():
            if 'overview' in profile:
                overview = profile['overview']
                total_records += overview['number_of_observations']
                total_columns += overview['number_of_variables']
                
                # Count data types
                if 'variables' in profile:
                    for var_name, var_profile in profile['variables'].items():
                        dtype = var_profile.get('data_type', 'unknown')
                        data_type_counts[dtype] = data_type_counts.get(dtype, 0) + 1
        
        summary['dataset_overview'] = {
            'total_records_across_datasets': total_records,
            'total_columns_across_datasets': total_columns,
            'average_records_per_dataset': total_records / len(self.profile_reports),
            'average_columns_per_dataset': total_columns / len(self.profile_reports)
        }
        
        summary['data_type_summary'] = data_type_counts
        
        return summary
    
    def save_profiles(self, output_dir="profiling_reports"):
        """Save all profile reports to individual JSON files"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for dataset_name, profile in self.profile_reports.items():
            safe_name = "".join(c for c in dataset_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"profile_{safe_name}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(profile, f, indent=2)
        
        print(f"All profiles saved to: {output_dir}")
