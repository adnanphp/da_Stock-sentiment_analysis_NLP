# datetime_pipeline.py
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from datetime_standardizer import DateTimeStandardizer

def safe_read_dataframe(file_path):
    """Safely read CSV or Parquet files with error handling."""
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
    # Configuration
    input_directory = "typed_financial_data"  # From previous step
    output_directory = "standardized_financial_data"
    config_file = "datetime_config.py"
    
    print("=" * 70)
    print("DATETIME STANDARDIZATION PIPELINE")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    # Load configuration
    config = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_content = f.read()
            # Simple config parsing (for demo purposes)
            exec(config_content, config)
            print("✓ Configuration loaded")
        except Exception as e:
            print(f"⚠️  Error loading config: {e}")
    
    # Initialize standardizer
    standardizer = DateTimeStandardizer(config)
    
    # Manual datetime column mappings (customize as needed)
    DATETIME_COLUMN_MAPPINGS = {
        "stock_data_all_stocks_combined.csv": ["Date"],
        "stock_data_COST_full_year.csv": ["Date"],
        "stock_data_V_full_year.csv": ["Date"],
        "reddit_data_reddit_dividends_expanded.csv": ["created_utc"],
        "news_data_alpha_vantage_news_expanded.csv": ["time_published"]
    }
    
    # 1. Detection Phase
    print("\n1. DETECTING DATETIME COLUMNS...")
    datetime_detection_results = {}
    
    detection_count = 0
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.startswith('typed_') and file.endswith(('.csv', '.parquet')):
                file_path = os.path.join(root, file)
                dataset_name = file.replace('typed_', '')
                
                try:
                    print(f"🔍 Detecting datetime columns in: {dataset_name}")
                    df = safe_read_dataframe(file_path)
                    
                    # Detect datetime columns
                    detected_columns = standardizer.detect_datetime_columns(df)
                    datetime_detection_results[dataset_name] = detected_columns
                    detection_count += 1
                    
                    print(f"✓ Detected {len(detected_columns)} potential datetime columns")
                    for col, info in detected_columns.items():
                        print(f"  📅 {col}: {info['detection_confidence']:.1%} confidence")
                    
                except Exception as e:
                    print(f"❌ Error detecting datetime columns in {dataset_name}: {e}")
                    datetime_detection_results[dataset_name] = {"error": str(e)}
    
    # Save detection report
    with open("datetime_detection_report.json", "w") as f:
        json.dump(datetime_detection_results, f, indent=2, default=str)
    print(f"✓ Datetime detection report saved: datetime_detection_report.json")
    
    # 2. Standardization Phase
    print("\n2. STANDARDIZING DATETIME COLUMNS...")
    
    processed_count = 0
    total_columns_standardized = 0
    
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.startswith('typed_') and file.endswith(('.csv', '.parquet')):
                file_path = os.path.join(root, file)
                original_dataset_name = file.replace('typed_', '')
                dataset_name = original_dataset_name
                
                try:
                    print(f"\n🔄 Processing: {dataset_name}")
                    df = safe_read_dataframe(file_path)
                    
                    # Get datetime columns for this dataset
                    datetime_columns = None
                    if dataset_name in DATETIME_COLUMN_MAPPINGS:
                        datetime_columns = DATETIME_COLUMN_MAPPINGS[dataset_name]
                        print(f"  Using manual column mapping: {datetime_columns}")
                    else:
                        # Auto-detect from detection results
                        if dataset_name in datetime_detection_results:
                            detected_info = datetime_detection_results[dataset_name]
                            if not isinstance(detected_info, dict) or 'error' in detected_info:
                                datetime_columns = []
                            else:
                                datetime_columns = [
                                    col for col, info in detected_info.items() 
                                    if info.get('detection_confidence', 0) > 0.5
                                ]
                                print(f"  Using auto-detected columns: {datetime_columns}")
                    
                    if not datetime_columns:
                        print(f"  ⏭️  No datetime columns to standardize")
                        continue
                    
                    # Standardize datetime columns
                    standardized_df = standardizer.standardize_dataset(
                        df=df,
                        dataset_name=dataset_name,
                        datetime_columns=datetime_columns,
                        target_timezone='UTC',  # Standardize to UTC
                        output_format='iso'     # ISO 8601 format
                    )
                    
                    # Save standardized data
                    output_subdir = os.path.join(output_directory, os.path.basename(root))
                    os.makedirs(output_subdir, exist_ok=True)
                    
                    output_filename = f"standardized_{original_dataset_name}"
                    output_path = os.path.join(output_subdir, output_filename)
                    
                    if file.endswith('.csv'):
                        standardized_df.to_csv(output_path, index=False)
                    else:
                        standardized_df.to_parquet(output_path, index=False)
                    
                    processed_count += 1
                    total_columns_standardized += len(datetime_columns)
                    print(f"✓ Standardized: {dataset_name} -> {output_path}")
                    print(f"  📊 Columns standardized: {len(datetime_columns)}")
                    
                except Exception as e:
                    print(f"❌ Error standardizing {dataset_name}: {e}")
    
    # 3. Generate Summary
    print("\n3. GENERATING STANDARDIZATION SUMMARY...")
    
    standardization_summary = standardizer.get_standardization_summary()
    print(f"\nSTANDARDIZATION SUMMARY:")
    print(f"Total datasets processed: {standardization_summary['total_datasets_processed']}")
    print(f"Total columns standardized: {standardization_summary['total_columns_standardized']}")
    print(f"Average success rate: {standardization_summary['average_success_rate']}%")
    print(f"Target timezone: {standardization_summary['timezone_distribution']}")
    print(f"Output format: {standardization_summary['format_distribution']}")
    
    # 4. Save Reports
    print("\n4. SAVING FINAL REPORTS...")
    
    standardizer.save_standardization_report("datetime_standardization_report.json")
    
    # Create final pipeline report
    final_report = {
        'pipeline_execution_time': datetime.now().isoformat(),
        'datasets_analyzed': detection_count,
        'datasets_processed': processed_count,
        'total_columns_standardized': total_columns_standardized,
        'average_success_rate': standardization_summary['average_success_rate'],
        'standardization_summary': standardization_summary,
        'datetime_detection_results': datetime_detection_results
    }
    
    with open("datetime_pipeline_report.json", "w") as f:
        json.dump(final_report, f, indent=2, default=str)
    print("✓ Pipeline report saved: datetime_pipeline_report.json")
    
    # Final summary
    print(f"\n🎉 DATETIME STANDARDIZATION PIPELINE COMPLETED!")
    print(f"📊 Datasets analyzed: {detection_count}")
    print(f"📊 Datasets processed: {processed_count}")
    print(f"📊 Total columns standardized: {total_columns_standardized}")
    print(f"📊 Average success rate: {standardization_summary['average_success_rate']}%")
    print(f"📁 Standardized data saved to: {output_directory}")
    print(f"📄 Reports saved: datetime_detection_report.json, datetime_standardization_report.json, datetime_pipeline_report.json")

if __name__ == "__main__":
    main()
