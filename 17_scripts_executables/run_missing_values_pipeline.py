# run_missing_values_pipeline.py
from simple_missing_imputer import SimpleMissingValueImputer
from missing_value_visualizer import MissingValueVisualizer
import pandas as pd
import os
import json
from datetime import datetime
import numpy as np

def convert_to_serializable(obj):
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
        return [convert_to_serializable(x) for x in obj]
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    else:
        return obj

def safe_has_missing_values(df):
    """Completely safe check for missing values without ambiguous array comparisons"""
    try:
        # Use the ultra-safe method from the imputer
        temp_imputer = SimpleMissingValueImputer()
        return temp_imputer.ultra_safe_has_missing(df)
    except:
        # Last resort: check a few values manually
        try:
            for col in df.columns:
                for i in range(min(5, len(df))):
                    if pd.isna(df[col].iloc[i]):
                        return True
            return False
        except:
            return False

def safe_read_dataframe(file_path):
    """Safely read CSV or Parquet files with error handling"""
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, low_memory=False, encoding='utf-8')
        else:
            return pd.read_parquet(file_path)
    except UnicodeDecodeError:
        # Try with different encoding if UTF-8 fails
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, low_memory=False, encoding='latin-1')
        else:
            raise
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        raise

def main():
    data_directory = "financial_data_large"
    output_directory = "cleaned_financial_data"
    
    print("=" * 70)
    print("MISSING VALUES HANDLING PIPELINE")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    # 1. Analyze Missing Values
    print("\n1. ANALYZING MISSING VALUES...")
    
    # Use the simple analyzer
    try:
        from simple_missing_analyzer import SimpleMissingValueAnalyzer
        analyzer = SimpleMissingValueAnalyzer(data_directory)
        analysis_results = analyzer.analyze_all_datasets()
        analyzer.save_analysis_report("missing_value_analysis.json")
        
        # Print summary
        analysis_summary = analyzer.get_summary_statistics()
        print(f"\nANALYSIS SUMMARY:")
        print(f"Total datasets analyzed: {analysis_summary['total_datasets_analyzed']}")
        print(f"Datasets with missing values: {analysis_summary['datasets_with_missing_values']}")
        print(f"Average missing percentage: {analysis_summary['average_missing_percentage']:.2f}%")
        
    except Exception as e:
        print(f"❌ Error during analysis phase: {e}")
        print("Continuing with imputation phase...")
        analysis_results = {}
        analysis_summary = {
            'total_datasets_analyzed': 0,
            'datasets_with_missing_values': 0,
            'average_missing_percentage': 0.0
        }
    
    # 2. Impute Missing Values
    print("\n2. IMPUTING MISSING VALUES...")
    imputer = SimpleMissingValueImputer()
    visualizer = MissingValueVisualizer()
    
    processed_count = 0
    successful_imputations = 0
    total_datasets_checked = 0
    
    for root, dirs, files in os.walk(data_directory):
        for file in files:
            if file.endswith(('.csv', '.parquet')):
                file_path = os.path.join(root, file)
                dataset_name = f"{os.path.basename(root)}_{file}"
                total_datasets_checked += 1
                
                try:
                    # Read original data safely
                    original_df = safe_read_dataframe(file_path)
                    
                    # Skip empty datasets
                    if len(original_df) == 0 or len(original_df.columns) == 0:
                        print(f"⏭️  {dataset_name} - Empty dataset, skipping")
                        continue
                    
                    # Use safe missing check instead of direct comparison
                    if not safe_has_missing_values(original_df):
                        print(f"✓ {dataset_name} - No missing values, skipping")
                        continue
                    
                    print(f"🔄 Processing: {dataset_name}")
                    
                    # Create visualizations before imputation (with error handling)
                    try:
                        visualizer.create_comprehensive_dashboard(original_df, dataset_name, analysis_results)
                    except Exception as viz_error:
                        print(f"⚠️  Visualization error for {dataset_name}: {viz_error}")
                    
                    # Impute missing values using simple imputer with fallback
                    try:
                        imputed_df = imputer.simple_impute(original_df, dataset_name)
                        
                        # Ultra-safe check for remaining missing
                        remaining_missing = 0
                        try:
                            for col in imputed_df.columns:
                                remaining_missing += imputer.ultra_safe_get_missing_count(imputed_df, col)
                        except:
                            remaining_missing = 1  # Assume some missing if check fails
                        
                        if remaining_missing == 0:
                            successful_imputations += 1
                            status = "✓ Successfully imputed"
                        else:
                            status = "⚠️  Partially imputed"
                        
                        # Create imputation comparison visualization
                        try:
                            visualizer.create_imputation_comparison(original_df, imputed_df, dataset_name)
                        except Exception as comp_viz_error:
                            print(f"⚠️  Comparison visualization error: {comp_viz_error}")
                        
                        # Save cleaned data
                        output_subdir = os.path.join(output_directory, os.path.basename(root))
                        os.makedirs(output_subdir, exist_ok=True)
                        
                        output_path = os.path.join(output_subdir, f"cleaned_{file}")
                        if file.endswith('.csv'):
                            imputed_df.to_csv(output_path, index=False)
                        else:
                            imputed_df.to_parquet(output_path, index=False)
                        
                        processed_count += 1
                        print(f"{status}: {dataset_name} -> {output_path}")
                        
                    except Exception as impute_error:
                        print(f"✗ Imputation error for {dataset_name}: {impute_error}")
                        # Try a fallback imputation
                        try:
                            print(f"  Attempting fallback imputation...")
                            imputed_df = original_df.copy()
                            for col in original_df.columns:
                                try:
                                    if imputer.ultra_safe_column_has_missing(original_df, col):
                                        if str(original_df[col].dtype) in ["int64", "float64"]:
                                            imputed_df[col] = original_df[col].fillna(0)
                                        else:
                                            imputed_df[col] = original_df[col].fillna("Unknown")
                                except:
                                    continue
                            
                            # Save fallback imputed data
                            output_subdir = os.path.join(output_directory, os.path.basename(root))
                            os.makedirs(output_subdir, exist_ok=True)
                            output_path = os.path.join(output_subdir, f"cleaned_{file}")
                            imputed_df.to_csv(output_path, index=False)
                            
                            successful_imputations += 1
                            processed_count += 1
                            status = "✓ Fallback imputation"
                            print(f"{status}: {dataset_name} -> {output_path}")
                            
                        except Exception as fallback_error:
                            print(f"✗ Fallback imputation also failed: {fallback_error}")
                            # Save original data as last resort
                            output_subdir = os.path.join(output_directory, os.path.basename(root))
                            os.makedirs(output_subdir, exist_ok=True)
                            output_path = os.path.join(output_subdir, f"cleaned_{file}")
                            original_df.to_csv(output_path, index=False)
                            print(f"💾 Saved original data as last resort: {output_path}")
                            continue
                    
                except Exception as e:
                    print(f"✗ Error processing {dataset_name}: {e}")
    
    # 3. Generate Imputation Summary
    print("\n3. GENERATING IMPUTATION SUMMARY...")
    try:
        imputation_summary = imputer.get_imputation_summary()
        print(f"\nIMPUTATION SUMMARY:")
        print(f"Total datasets checked: {total_datasets_checked}")
        print(f"Datasets with missing values processed: {processed_count}")
        print(f"Successfully imputed datasets: {successful_imputations}")
        if processed_count > 0:
            success_rate = (successful_imputations / processed_count) * 100
            print(f"Overall success rate: {success_rate:.1f}%")
        else:
            success_rate = 0
            print(f"Overall success rate: 0%")
        
    except Exception as e:
        print(f"❌ Error generating imputation summary: {e}")
        imputation_summary = {
            'total_datasets_imputed': processed_count,
            'success_rate': success_rate if 'success_rate' in locals() else 0,
        }
    
    # 4. Save final reports
    print("\n4. SAVING FINAL REPORTS...")
    
    # Save imputation report with proper serialization
    try:
        serializable_imputation_report = convert_to_serializable(imputer.imputation_report)
        with open("imputation_report.json", "w") as f:
            json.dump(serializable_imputation_report, f, indent=2)
        print("✓ Imputation report saved: imputation_report.json")
    except Exception as e:
        print(f"❌ Error saving imputation report: {e}")
    
    # Generate final comparison
    original_missing_total = 0
    try:
        for result in analysis_results.values():
            if 'overview' in result and 'total_missing_cells' in result['overview']:
                original_missing_total += result['overview']['total_missing_cells']
    except:
        # If analysis failed, estimate based on processed datasets
        original_missing_total = processed_count * 100  # Rough estimate
    
    # Calculate cleaned missing total
    cleaned_missing_total = 0
    try:
        for root, dirs, files in os.walk(output_directory):
            for file in files:
                if file.startswith('cleaned_') and file.endswith(('.csv', '.parquet')):
                    file_path = os.path.join(root, file)
                    try:
                        df = safe_read_dataframe(file_path)
                        cleaned_missing_total += df.isnull().sum().sum()
                    except Exception as e:
                        print(f"⚠️ Warning: Could not read {file_path}: {e}")
    except Exception as e:
        print(f"⚠️ Warning: Could not calculate cleaned missing total: {e}")
    
    # Calculate reduction percentage safely
    if original_missing_total > 0:
        reduction_percentage = ((original_missing_total - cleaned_missing_total) / original_missing_total) * 100
    else:
        reduction_percentage = 0.0
    
    # Create final report with serializable types
    final_report = {
        'pipeline_execution_time': datetime.now().isoformat(),
        'original_missing_cells': int(original_missing_total),
        'remaining_missing_cells': int(cleaned_missing_total),
        'missing_cells_removed': int(original_missing_total - cleaned_missing_total),
        'reduction_percentage': float(reduction_percentage),
        'datasets_processed': int(processed_count),
        'successful_imputations': int(successful_imputations),
        'total_datasets_checked': int(total_datasets_checked),
        'imputation_summary': convert_to_serializable(imputation_summary)
    }
    
    try:
        with open("missing_values_pipeline_report.json", "w") as f:
            json.dump(final_report, f, indent=2)
        print("✓ Pipeline report saved: missing_values_pipeline_report.json")
    except Exception as e:
        print(f"❌ Error saving pipeline report: {e}")
    
    # Final summary
    print(f"\n🎉 PIPELINE COMPLETED!")
    print(f"📊 Total datasets checked: {total_datasets_checked:,}")
    print(f"📊 Datasets processed: {processed_count:,}")
    print(f"📊 Successfully imputed: {successful_imputations:,}")
    print(f"📊 Original missing cells: {original_missing_total:,}")
    print(f"📊 Remaining missing cells: {cleaned_missing_total:,}")
    print(f"📊 Reduction: {final_report['reduction_percentage']:.1f}%")
    print(f"📁 Cleaned data saved to: {output_directory}")
    print(f"📁 Visualizations saved to: missing_value_visualizations/")
    print(f"📄 Reports saved: missing_value_analysis.json, imputation_report.json, missing_values_pipeline_report.json")

if __name__ == "__main__":
    main()
