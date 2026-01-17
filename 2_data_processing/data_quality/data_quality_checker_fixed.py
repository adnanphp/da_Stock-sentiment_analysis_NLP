# data_quality_checker_fixed.py
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json

print("=== RUNNING COMPLETELY NEW FIXED VERSION ===")

class DataQualityChecker:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.quality_report = {}
    
    def safe_analyze(self, df, dataset_name):
        """Ultra-safe analysis that avoids all pandas ambiguous truth issues"""
        try:
            # Basic info only - no complex pandas operations
            result = {
                'dataset_name': dataset_name,
                'total_records': len(df),
                'total_columns': len(df.columns),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # Safe null count calculation
            null_counts = {}
            total_nulls = 0
            for col in df.columns:
                null_count = 0
                for val in df[col]:
                    if pd.isna(val):
                        null_count += 1
                null_counts[col] = null_count
                total_nulls += null_count
            
            result['null_counts'] = null_counts
            result['total_nulls'] = total_nulls
            
            # Safe duplicate check
            seen_rows = set()
            duplicates = 0
            for _, row in df.iterrows():
                row_tuple = tuple(row)
                if row_tuple in seen_rows:
                    duplicates += 1
                else:
                    seen_rows.add(row_tuple)
            
            result['duplicates'] = duplicates
            
            # Calculate simple quality score
            quality_score = 100
            
            # Penalize for nulls
            if len(df) > 0 and len(df.columns) > 0:
                null_pct = (total_nulls / (len(df) * len(df.columns))) * 100
                quality_score -= min(null_pct, 40)
            
            # Penalize for duplicates
            if len(df) > 0:
                dup_pct = (duplicates / len(df)) * 100
                quality_score -= min(dup_pct, 20)
            
            result['quality_score'] = max(0, int(quality_score))
            
            return result
            
        except Exception as e:
            return {
                'dataset_name': dataset_name,
                'error': str(e),
                'quality_score': 0
            }
    
    def check_all_datasets(self):
        """Run quality check on all datasets"""
        print("Starting ULTRA-SAFE data quality assessment...")
        
        for root, dirs, files in os.walk(self.data_directory):
            for file in files:
                if file.endswith(('.csv', '.parquet')):
                    file_path = os.path.join(root, file)
                    dataset_name = f"{os.path.basename(root)}_{file}"
                    
                    try:
                        print(f"Analyzing: {file_path}")
                        
                        # Read data
                        if file.endswith('.csv'):
                            df = pd.read_csv(file_path, low_memory=False)
                        else:
                            df = pd.read_parquet(file_path)
                        
                        # Analyze
                        report = self.safe_analyze(df, dataset_name)
                        self.quality_report[dataset_name] = report
                        
                        print(f"✓ {dataset_name} (Score: {report['quality_score']}/100)")
                        
                    except Exception as e:
                        print(f"✗ Error: {file_path} - {e}")
                        self.quality_report[dataset_name] = {
                            'error': str(e),
                            'quality_score': 0
                        }
        
        return self.quality_report
    
    def save_report(self, output_path="data_quality_report_fixed.json"):
        """Save quality report"""
        with open(output_path, 'w') as f:
            json.dump(self.quality_report, f, indent=2)
        print(f"Report saved to: {output_path}")
    
    def get_summary(self):
        """Get summary statistics"""
        total = len(self.quality_report)
        successful = sum(1 for r in self.quality_report.values() if r.get('quality_score', 0) > 0)
        
        if successful == 0:
            return {"error": "No successful analyses"}
        
        scores = [r['quality_score'] for r in self.quality_report.values() if r.get('quality_score', 0) > 0]
        avg_score = sum(scores) / len(scores)
        
        return {
            'total_datasets': total,
            'successful_analyses': successful,
            'average_quality_score': round(avg_score, 2),
            'failed_analyses': total - successful
        }

def main():
    checker = DataQualityChecker("financial_data_large")
    checker.check_all_datasets()
    checker.save_report()
    
    summary = checker.get_summary()
    print("\n=== SUMMARY ===")
    print(f"Total datasets: {summary['total_datasets']}")
    print(f"Successful: {summary['successful_analyses']}")
    print(f"Failed: {summary['failed_analyses']}")
    print(f"Average score: {summary['average_quality_score']}/100")

if __name__ == "__main__":
    main()
