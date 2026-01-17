# fix_specific_columns.py
import pandas as pd
import numpy as np
import os

def fix_specific_issues():
    """Fix the specific column issues identified in verification."""
    input_dir = "FIXED_outlier_data"
    output_dir = "FINAL_cleaned_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔧 FIXING SPECIFIC COLUMN ISSUES")
    print("=" * 50)
    
    fixed_count = 0
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.startswith('FIXED_') and file.endswith('.csv'):
                input_path = os.path.join(root, file)
                output_filename = f"FINAL_{file.replace('FIXED_', '')}"
                
                # Create same directory structure
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_path = os.path.join(output_subdir, output_filename)
                
                try:
                    print(f"\n📊 Processing: {file}")
                    
                    # Load with better error handling
                    df = pd.read_csv(input_path, low_memory=False)
                    original_shape = df.shape
                    
                    # Fix 1: Handle Flesch score columns (negative values are valid)
                    flesch_cols = [col for col in df.columns if 'flesch_score' in col.lower()]
                    for col in flesch_cols:
                        if col in df.columns:
                            # Flesch scores can legitimately be negative for complex text
                            # Just ensure they're not extremely negative (below -100 is unreasonable)
                            extreme_neg_mask = df[col] < -100
                            if extreme_neg_mask.any():
                                # Cap at reasonable minimum for readability scores
                                df.loc[extreme_neg_mask, col] = -100
                                print(f"   🔧 {col}: Capped {extreme_neg_mask.sum()} extreme negative values to -100")
                    
                    # Fix 2: Handle market cap columns (large values are normal)
                    market_cap_cols = [col for col in df.columns if 'market_cap' in col.lower()]
                    for col in market_cap_cols:
                        if col in df.columns:
                            # Market cap can be very large, but check for impossibly large values
                            # Assuming values > 1e15 (quadrillions) are errors
                            extreme_pos_mask = df[col] > 1e15
                            if extreme_pos_mask.any():
                                # Replace with 99th percentile
                                cap_99 = df[col].quantile(0.99)
                                df.loc[extreme_pos_mask, col] = cap_99
                                print(f"   🔧 {col}: Fixed {extreme_pos_mask.sum()} extreme large values")
                    
                    # Fix 3: Handle mixed data types
                    problematic_cols = []
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            try:
                                # Try to convert to numeric, keep as string if fails
                                converted = pd.to_numeric(df[col], errors='coerce')
                                if not converted.isna().all():  # If some values converted successfully
                                    df[col] = converted
                                    problematic_cols.append(col)
                            except:
                                pass
                    
                    if problematic_cols:
                        print(f"   🔧 Converted {len(problematic_cols)} columns from object to numeric: {problematic_cols[:3]}...")
                    
                    # Fix 4: Remove any remaining infinite values
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        inf_mask = np.isinf(df[col])
                        if inf_mask.any():
                            median_val = df[col].median()
                            df.loc[inf_mask, col] = median_val
                            print(f"   🔧 {col}: Fixed {inf_mask.sum()} infinite values")
                    
                    # Save the final cleaned data
                    df.to_csv(output_path, index=False)
                    fixed_count += 1
                    
                    print(f"   ✅ Saved: {output_filename}")
                    print(f"   📊 Final shape: {df.shape}")
                    
                except Exception as e:
                    print(f"   ❌ Error processing {file}: {e}")
    
    print(f"\n�️ FINAL CLEANING COMPLETED!")
    print(f"📊 Datasets processed: {fixed_count}")
    print(f"📁 Output directory: {output_dir}")

def create_data_quality_report():
    """Create a final data quality report."""
    input_dir = "FINAL_cleaned_data"
    
    print("\n📋 FINAL DATA QUALITY REPORT")
    print("=" * 50)
    
    quality_summary = {}
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.startswith('FINAL_') and file.endswith('.csv'):
                file_path = os.path.join(root, file)
                
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    
                    # Analyze data quality
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    issues = {
                        'extreme_negative': 0,
                        'extreme_positive': 0,
                        'infinite': 0,
                        'null_percentage': 0
                    }
                    
                    for col in numeric_cols:
                        series = df[col].dropna()
                        if len(series) == 0:
                            continue
                        
                        # Reasonable bounds for different column types
                        if 'flesch_score' in col.lower():
                            # Flesch scores: -100 to 100 is reasonable
                            issues['extreme_negative'] += (series < -100).sum()
                            issues['extreme_positive'] += (series > 100).sum()
                        elif 'market_cap' in col.lower():
                            # Market cap: > 1e15 is unreasonable
                            issues['extreme_positive'] += (series > 1e15).sum()
                        else:
                            # General numeric columns
                            issues['extreme_negative'] += (series < -1000).sum()
                            Q3 = series.quantile(0.75)
                            IQR = series.quantile(0.75) - series.quantile(0.25)
                            upper_bound = Q3 + 5 * IQR
                            issues['extreme_positive'] += (series > upper_bound).sum()
                        
                        issues['infinite'] += np.isinf(series).sum()
                    
                    total_values = len(numeric_cols) * len(df)
                    issues['null_percentage'] = (df[numeric_cols].isnull().sum().sum() / total_values) * 100
                    
                    quality_summary[file] = {
                        'shape': df.shape,
                        'issues': issues,
                        'quality_score': max(0, 100 - (sum(issues.values()) / total_values * 100)) if total_values > 0 else 100
                    }
                    
                    print(f"\n📊 {file}:")
                    print(f"   Shape: {df.shape}")
                    print(f"   Extreme negatives: {issues['extreme_negative']}")
                    print(f"   Extreme positives: {issues['extreme_positive']}")
                    print(f"   Infinite values: {issues['infinite']}")
                    print(f"   Null percentage: {issues['null_percentage']:.2f}%")
                    print(f"   Quality score: {quality_summary[file]['quality_score']:.1f}%")
                    
                except Exception as e:
                    print(f"   ❌ Error analyzing {file}: {e}")
    
    # Overall summary
    avg_quality = np.mean([info['quality_score'] for info in quality_summary.values()])
    print(f"\n📈 OVERALL DATA QUALITY: {avg_quality:.1f}%")
    
    return quality_summary

if __name__ == "__main__":
    # Step 1: Fix specific column issues
    fix_specific_issues()
    
    # Step 2: Generate final quality report
    create_data_quality_report()
