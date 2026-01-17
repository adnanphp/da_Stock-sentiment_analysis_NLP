# verify_fixed_data.py
import pandas as pd
import numpy as np
import os

def verify_fixed_data():
    """Verify that the fixed data is clean and ready for feature engineering."""
    fixed_dir = "FIXED_outlier_data"
    
    print("🔍 VERIFYING FIXED DATA QUALITY")
    print("=" * 50)
    
    for root, dirs, files in os.walk(fixed_dir):
        for file in files:
            if file.startswith('FIXED_') and file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                
                print(f"\n📊 {file}:")
                print(f"   Shape: {df.shape}")
                
                # Check for remaining issues
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                issues_found = 0
                
                for col in numeric_cols:
                    series = df[col].dropna()
                    if len(series) == 0:
                        continue
                    
                    # Check for extreme values
                    extreme_neg = (series < -10).sum()
                    extreme_pos = (series > 1e6).sum()  # Very large values
                    infinite = np.isinf(series).sum()
                    nan_count = series.isna().sum()
                    
                    if any([extreme_neg > 0, extreme_pos > 0, infinite > 0]):
                        issues_found += 1
                        print(f"   ⚠️  {col}: {extreme_neg} extreme neg, {extreme_pos} extreme pos, {infinite} infinite")
                
                if issues_found == 0:
                    print("   ✅ Data quality: EXCELLENT")
                else:
                    print(f"   ⚠️  Data quality: {issues_found} columns need attention")
    
    print("\n🎉 VERIFICATION COMPLETED!")

if __name__ == "__main__":
    verify_fixed_data()
