# data_type_pipeline.py
from data_type_converter import DataTypeConverter
import pandas as pd
import os
import json
from datetime import datetime
import numpy as np

def safe_read_dataframe(file_path):
    """Safely read CSV or Parquet files with error handling"""
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, low_memory=False, encoding='utf-8')
        else:
            return pd.read_parquet(file_path)
    except UnicodeDecodeError:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, low_memory=False, encoding='latin-1')
        else:
            raise
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        raise

def main():
    data_directory = "cleaned_financial_data"  # Using cleaned data from previous step
    output_directory = "typed_financial_data"
    
    print("=" * 70)
    print("DATA TYPE CONVERSION PIPELINE")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    # Initialize converter
    converter = DataTypeConverter()
    
    # Manual type mappings for specific datasets (customize as needed)
    MANUAL_TYPE_MAPPINGS = {
        # Example: Map specific columns to specific types
        "stock_data_all_stocks_combined.csv": {
            # "date": "datetime",
            # "volume": "integer", 
            # "price": "float"
        },
        "reddit_data_reddit_dividends_expanded.csv": {
            # "created_utc": "datetime",
            # "score": "integer"
        }
    }
    
    # 1. Analysis Phase
    print("\n1. ANALYZING DATA TYPES...")
    analysis_results = {}
    
    analysis_count = 0
    for root, dirs, files in os.walk(data_directory):
        for file in files:
            if file.startswith('cleaned_') and file.endswith(('.csv', '.parquet')):
                file_path = os.path.join(root, file)
                dataset_name = f"{os.path.basename(root)}_{file.replace('cleaned_', '')}"
                
                try:
                    print(f"📊 Analyzing: {dataset_name}")
                    df = safe_read_dataframe(file_path)
                    
                    # Perform type analysis
                    analysis = converter.analyze_dataset_types(df, dataset_name)
                    analysis_results[dataset_name] = analysis
                    analysis_count += 1
                    
                    print(f"✓ Analyzed: {dataset_name} ({len(df.columns)} columns)")
                    
                except Exception as e:
                    print(f"❌ Error analyzing {dataset_name}: {e}")
                    analysis_results[dataset_name] = {"error": str(e)}
    
    # Save analysis report
    with open("data_type_analysis_report.json", "w") as f:
        json.dump(converter._convert_to_serializable(analysis_results), f, indent=2)
    print(f"✓ Type analysis report saved: data_type_analysis_report.json")
    
    # 2. Conversion Phase
    print("\n2. CONVERTING DATA TYPES...")
    
    processed_count = 0
    automatic_conversions = 0
    manual_conversions = 0
    
    for root, dirs, files in os.walk(data_directory):
        for file in files:
            if file.startswith('cleaned_') and file.endswith(('.csv', '.parquet')):
                file_path = os.path.join(root, file)
                original_dataset_name = file.replace('cleaned_', '')
                dataset_name = f"{os.path.basename(root)}_{original_dataset_name}"
                
                try:
                    print(f"\n🔄 Processing: {dataset_name}")
                    df = safe_read_dataframe(file_path)
                    
                    # Determine conversion strategy
                    if dataset_name in MANUAL_TYPE_MAPPINGS and MANUAL_TYPE_MAPPINGS[dataset_name]:
                        # Use manual conversion if mapping specified
                        print(f"  Using MANUAL conversion strategy")
                        type_mapping = MANUAL_TYPE_MAPPINGS[dataset_name]
                        converted_df = converter.manual_type_conversion(df, dataset_name, type_mapping)
                        manual_conversions += 1
                    else:
                        # Use automatic conversion
                        print(f"  Using AUTOMATIC conversion strategy")
                        converted_df = converter.automatic_type_conversion(df, dataset_name, confidence_threshold=0.7)
                        automatic_conversions += 1
                    
                    # Save converted data
                    output_subdir = os.path.join(output_directory, os.path.basename(root))
                    os.makedirs(output_subdir, exist_ok=True)
                    
                    output_filename = f"typed_{original_dataset_name}"
                    output_path = os.path.join(output_subdir, output_filename)
                    
                    if file.endswith('.csv'):
                        converted_df.to_csv(output_path, index=False)
                    else:
                        converted_df.to_parquet(output_path, index=False)
                    
                    processed_count += 1
                    print(f"✓ Converted: {dataset_name} -> {output_path}")
                    
                except Exception as e:
                    print(f"❌ Error converting {dataset_name}: {e}")
    
    # 3. Generate Summary
    print("\n3. GENERATING CONVERSION SUMMARY...")
    
    conversion_summary = converter.get_conversion_summary()
    print(f"\nCONVERSION SUMMARY:")
    print(f"Total datasets processed: {conversion_summary['total_datasets_processed']}")
    print(f"Total columns converted: {conversion_summary['total_columns_converted']}")
    print(f"Overall success rate: {conversion_summary['success_rate']}%")
    print(f"Automatic conversions: {automatic_conversions}")
    print(f"Manual conversions: {manual_conversions}")
    
    # 4. Save Reports
    print("\n4. SAVING FINAL REPORTS...")
    
    converter.save_conversion_report("data_type_conversion_report.json")
    
    # Create final pipeline report
    final_report = {
        'pipeline_execution_time': datetime.now().isoformat(),
        'datasets_analyzed': analysis_count,
        'datasets_processed': processed_count,
        'automatic_conversions': automatic_conversions,
        'manual_conversions': manual_conversions,
        'total_columns_converted': conversion_summary['total_columns_converted'],
        'success_rate': conversion_summary['success_rate'],
        'conversion_summary': conversion_summary
    }
    
    with open("data_type_pipeline_report.json", "w") as f:
        json.dump(final_report, f, indent=2)
    print("✓ Pipeline report saved: data_type_pipeline_report.json")
    
    # Final summary
    print(f"\n🎉 DATA TYPE CONVERSION PIPELINE COMPLETED!")
    print(f"📊 Datasets analyzed: {analysis_count}")
    print(f"📊 Datasets processed: {processed_count}")
    print(f"📊 Automatic conversions: {automatic_conversions}")
    print(f"📊 Manual conversions: {manual_conversions}")
    print(f"📊 Total columns converted: {conversion_summary['total_columns_converted']}")
    print(f"📊 Success rate: {conversion_summary['success_rate']}%")
    print(f"📁 Typed data saved to: {output_directory}")
    print(f"📄 Reports saved: data_type_analysis_report.json, data_type_conversion_report.json, data_type_pipeline_report.json")

if __name__ == "__main__":
    main()
