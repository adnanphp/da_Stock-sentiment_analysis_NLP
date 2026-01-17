# text_pipeline.py
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from text_preprocessor import TextPreprocessor

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
    input_directory = "standardized_financial_data"  # From previous step
    output_directory = "preprocessed_financial_data"
    config_file = "text_config.py"
    
    print("=" * 70)
    print("TEXT PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    # Load configuration
    config = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_content = f.read()
            # Simple config parsing
            exec(config_content, config)
            print("✓ Configuration loaded")
        except Exception as e:
            print(f"⚠️  Error loading config: {e}")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(config)
    
    # Manual text column mappings (customize as needed)
    TEXT_COLUMN_MAPPINGS = {
        "reddit_dividends_expanded.csv": ["title", "selftext", "body"],
        "alpha_vantage_news_expanded.csv": ["title", "summary", "content"],
        "all_stocks_combined.csv": ["description", "company_name"],
        "COST_full_year.csv": ["description", "company_name"],
        "V_full_year.csv": ["description", "company_name"]
    }
    
    # Manual processor mappings (override auto-detection if needed)
    PROCESSOR_MAPPINGS = {
        "reddit_dividends_expanded.csv": "reddit_cleaner",
        "alpha_vantage_news_expanded.csv": "news_preprocessor",
        "all_stocks_combined.csv": "news_preprocessor",
        "COST_full_year.csv": "news_preprocessor", 
        "V_full_year.csv": "news_preprocessor"
    }
    
    # 1. Detection Phase
    print("\n1. DETECTING TEXT COLUMNS...")
    text_detection_results = {}
    
    detection_count = 0
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.startswith('standardized_') and file.endswith(('.csv', '.parquet')):
                file_path = os.path.join(root, file)
                dataset_name = file.replace('standardized_', '')
                
                try:
                    print(f"🔍 Detecting text columns in: {dataset_name}")
                    df = safe_read_dataframe(file_path)
                    
                    # Detect text columns
                    detected_columns = preprocessor.detect_text_columns(df)
                    text_detection_results[dataset_name] = detected_columns
                    detection_count += 1
                    
                    print(f"✓ Detected {len(detected_columns)} text columns")
                    for col, info in detected_columns.items():
                        print(f"  📄 {col}: {info['text_type']} ({info['recommended_processor']})")
                    
                except Exception as e:
                    print(f"❌ Error detecting text columns in {dataset_name}: {e}")
                    text_detection_results[dataset_name] = {"error": str(e)}
    
    # Save detection report
    with open("text_detection_report.json", "w") as f:
        json.dump(preprocessor._convert_to_serializable(text_detection_results), f, indent=2)
    print(f"✓ Text detection report saved: text_detection_report.json")
    
    # 2. Preprocessing Phase
    print("\n2. PREPROCESSING TEXT COLUMNS...")
    
    processed_count = 0
    total_columns_preprocessed = 0
    
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.startswith('standardized_') and file.endswith(('.csv', '.parquet')):
                file_path = os.path.join(root, file)
                original_dataset_name = file.replace('standardized_', '')
                dataset_name = original_dataset_name
                
                try:
                    print(f"\n🔄 Processing: {dataset_name}")
                    df = safe_read_dataframe(file_path)
                    
                    # Get text columns for this dataset
                    text_columns = None
                    processor_type = 'auto'
                    
                    if dataset_name in TEXT_COLUMN_MAPPINGS:
                        text_columns = TEXT_COLUMN_MAPPINGS[dataset_name]
                        print(f"  Using manual column mapping: {text_columns}")
                    
                    if dataset_name in PROCESSOR_MAPPINGS:
                        processor_type = PROCESSOR_MAPPINGS[dataset_name]
                        print(f"  Using manual processor: {processor_type}")
                    
                    if not text_columns:
                        # Auto-detect from detection results
                        if dataset_name in text_detection_results:
                            detected_info = text_detection_results[dataset_name]
                            if not isinstance(detected_info, dict) or 'error' in detected_info:
                                text_columns = []
                            else:
                                text_columns = list(detected_info.keys())
                                print(f"  Using auto-detected columns: {text_columns}")
                    
                    if not text_columns:
                        print(f"  ⏭️  No text columns to preprocess")
                        continue
                    
                    # Preprocess text columns
                    processed_df = preprocessor.preprocess_dataset(
                        df=df,
                        dataset_name=dataset_name,
                        text_columns=text_columns,
                        processor_type=processor_type
                    )
                    
                    # Save preprocessed data
                    output_subdir = os.path.join(output_directory, os.path.basename(root))
                    os.makedirs(output_subdir, exist_ok=True)
                    
                    output_filename = f"preprocessed_{original_dataset_name}"
                    output_path = os.path.join(output_subdir, output_filename)
                    
                    if file.endswith('.csv'):
                        processed_df.to_csv(output_path, index=False)
                    else:
                        processed_df.to_parquet(output_path, index=False)
                    
                    processed_count += 1
                    total_columns_preprocessed += len(text_columns)
                    print(f"✓ Preprocessed: {dataset_name} -> {output_path}")
                    print(f"  📊 Columns preprocessed: {len(text_columns)}")
                    
                except Exception as e:
                    print(f"❌ Error preprocessing {dataset_name}: {e}")
    
    # 3. Generate Summary
    print("\n3. GENERATING PREPROCESSING SUMMARY...")
    
    preprocessing_summary = preprocessor.get_preprocessing_summary()
    print(f"\nPREPROCESSING SUMMARY:")
    print(f"Total datasets processed: {preprocessing_summary['total_datasets_processed']}")
    print(f"Total columns preprocessed: {preprocessing_summary['total_columns_preprocessed']}")
    print(f"Average success rate: {preprocessing_summary['average_success_rate']}%")
    print(f"Processor distribution: {preprocessing_summary['processor_distribution']}")
    
    # 4. Save Reports
    print("\n4. SAVING FINAL REPORTS...")
    
    preprocessor.save_preprocessing_report("text_preprocessing_report.json")
    
    # Create final pipeline report
    final_report = {
        'pipeline_execution_time': datetime.now().isoformat(),
        'datasets_analyzed': detection_count,
        'datasets_processed': processed_count,
        'total_columns_preprocessed': total_columns_preprocessed,
        'average_success_rate': preprocessing_summary['average_success_rate'],
        'preprocessing_summary': preprocessing_summary,
        'text_detection_results': preprocessor._convert_to_serializable(text_detection_results)
    }
    
    with open("text_pipeline_report.json", "w") as f:
        json.dump(final_report, f, indent=2)
    print("✓ Pipeline report saved: text_pipeline_report.json")
    
    # Final summary
    print(f"\n🎉 TEXT PREPROCESSING PIPELINE COMPLETED!")
    print(f"📊 Datasets analyzed: {detection_count}")
    print(f"📊 Datasets processed: {processed_count}")
    print(f"📊 Total columns preprocessed: {total_columns_preprocessed}")
    print(f"📊 Average success rate: {preprocessing_summary['average_success_rate']}%")
    print(f"📁 Preprocessed data saved to: {output_directory}")
    print(f"📄 Reports saved: text_detection_report.json, text_preprocessing_report.json, text_pipeline_report.json")

if __name__ == "__main__":
    main()
