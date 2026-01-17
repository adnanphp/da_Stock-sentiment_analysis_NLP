# outlier_pipeline.py
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from outlier_manager import OutlierManager

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
    input_directory = "normalized_financial_data"  # From previous step
    output_directory = "outlier_treated_data"
    config_file = "outlier_config.py"
    
    print("=" * 70)
    print("OUTLIER DETECTION & TREATMENT PIPELINE")
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
    
    # Initialize outlier manager
    outlier_manager = OutlierManager(config)
    
    # Manual detection and treatment plans (customize as needed)
    DETECTION_PLANS = {
        "normalized_all_stocks_combined.csv": {
            "methods": {
                "iqr": {
                    "columns": ["Close_scaled", "Open_scaled", "High_scaled", "Low_scaled", "Volume_scaled"],
                    "params": {"multiplier": 1.5}
                },
                "zscore": {
                    "columns": ["RSI_scaled", "MACD_scaled", "BB_Position_scaled"],
                    "params": {"threshold": 3}
                },
                "isolation_forest": {
                    "columns": ["Close_scaled", "Volume_scaled", "RSI_scaled", "MACD_scaled"],
                    "params": {"contamination": "auto", "n_estimators": 100}
                }
            }
        }
    }
    
    TREATMENT_PLANS = {
        "normalized_all_stocks_combined.csv": {
            "strategies": {
                "cap_prices": {
                    "columns": ["Close_scaled", "Open_scaled", "High_scaled", "Low_scaled"],
                    "method": "cap",
                    "params": {"method": "iqr", "multiplier": 2.0}
                },
                "winsorize_indicators": {
                    "columns": ["RSI_scaled", "MACD_scaled", "BB_Position_scaled"],
                    "method": "winsorize",
                    "params": {"limits": [0.05, 0.05]}
                },
                "transform_volume": {
                    "columns": ["Volume_scaled"],
                    "method": "transform",
                    "params": {"method": "log"}
                }
            }
        }
    }
    
    # 1. Analysis Phase
    print("\n1. ANALYZING OUTLIER CHARACTERISTICS...")
    outlier_analysis_results = {}
    
    analysis_count = 0
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.startswith('normalized_') and file.endswith(('.csv', '.parquet')):
                file_path = os.path.join(root, file)
                dataset_name = file.replace('normalized_', '')
                
                try:
                    print(f"🔍 Analyzing outliers in: {dataset_name}")
                    df = safe_read_dataframe(file_path)
                    
                    # Analyze outlier characteristics
                    analysis = outlier_manager.analyze_outlier_characteristics(df, dataset_name)
                    outlier_analysis_results[dataset_name] = analysis
                    analysis_count += 1
                    
                    print(f"✓ Analyzed: {dataset_name}")
                    print(f"  📊 Columns with outliers: {analysis['overview']['columns_with_outliers']}")
                    print(f"  📊 Total outliers: {analysis['overview']['total_outliers_detected']}")
                    print(f"  📊 Outlier percentage: {analysis['overview']['outlier_percentage']:.2f}%")
                    
                except Exception as e:
                    print(f"❌ Error analyzing {dataset_name}: {e}")
                    outlier_analysis_results[dataset_name] = {"error": str(e)}
    
    # Save analysis report
    with open("outlier_analysis_report.json", "w") as f:
        json.dump(outlier_manager._convert_to_serializable(outlier_analysis_results), f, indent=2)
    print(f"✓ Outlier analysis report saved: outlier_analysis_report.json")
    
    # 2. Detection & Treatment Phase
    print("\n2. DETECTING AND TREATING OUTLIERS...")
    
    processed_count = 0
    total_outliers_treated = 0
    
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.startswith('normalized_') and file.endswith(('.csv', '.parquet')):
                file_path = os.path.join(root, file)
                original_dataset_name = file.replace('normalized_', '')
                dataset_name = original_dataset_name
                
                try:
                    print(f"\n🔄 Processing: {dataset_name}")
                    df = safe_read_dataframe(file_path)
                    
                    # Get detection and treatment plans
                    detection_plan = DETECTION_PLANS.get(dataset_name)
                    treatment_plan = TREATMENT_PLANS.get(dataset_name)
                    
                    if detection_plan:
                        print(f"  Using manual detection plan")
                    else:
                        print(f"  Using auto-generated detection plan")
                    
                    if treatment_plan:
                        print(f"  Using manual treatment plan")
                    else:
                        print(f"  Using auto-generated treatment plan")
                    
                    # Detect outliers
                    detection_df, detection_results = outlier_manager.detect_outliers(
                        df, dataset_name, detection_plan
                    )
                    
                    # Treat outliers
                    treated_df, treatment_results = outlier_manager.treat_outliers(
                        detection_df, dataset_name, treatment_plan, detection_results
                    )
                    
                    # Save treated data
                    output_subdir = os.path.join(output_directory, os.path.basename(root))
                    os.makedirs(output_subdir, exist_ok=True)
                    
                    output_filename = f"outlier_treated_{original_dataset_name}"
                    output_path = os.path.join(output_subdir, output_filename)
                    
                    if file.endswith('.csv'):
                        treated_df.to_csv(output_path, index=False)
                    else:
                        treated_df.to_parquet(output_path, index=False)
                    
                    # Get treatment statistics
                    treatment_stats = outlier_manager.outlier_report.get(
                        dataset_name, {}
                    ).get('summary', {})
                    
                    outliers_treated = treatment_stats.get('total_outliers_treated', 0)
                    
                    processed_count += 1
                    total_outliers_treated += outliers_treated
                    print(f"✓ Treated: {dataset_name} -> {output_path}")
                    print(f"  📊 Outliers treated: {outliers_treated}")
                    print(f"  📊 Treatment effectiveness: {treatment_stats.get('treatment_effectiveness', 0):.1f}%")
                    
                except Exception as e:
                    print(f"❌ Error processing {dataset_name}: {e}")
    
    # 3. Generate Summary
    print("\n3. GENERATING OUTLIER MANAGEMENT SUMMARY...")
    
    outlier_summary = outlier_manager.get_outlier_summary()
    print(f"\nOUTLIER MANAGEMENT SUMMARY:")
    print(f"Total datasets processed: {outlier_summary['total_datasets_processed']}")
    print(f"Total outliers detected: {outlier_summary['total_outliers_detected']}")
    print(f"Total outliers treated: {outlier_summary['total_outliers_treated']}")
    print(f"Average treatment effectiveness: {outlier_summary['average_treatment_effectiveness']:.1f}%")
    print(f"Detection methods used: {outlier_summary['detection_methods_used']}")
    print(f"Treatment strategies used: {outlier_summary['treatment_strategies_used']}")
    
    # 4. Save Reports
    print("\n4. SAVING FINAL REPORTS...")
    
    outlier_manager.save_outlier_report("outlier_management_report.json")
    
    # Create final pipeline report
    final_report = {
        'pipeline_execution_time': datetime.now().isoformat(),
        'datasets_analyzed': int(analysis_count),
        'datasets_processed': int(processed_count),
        'total_outliers_treated': int(total_outliers_treated),
        'average_treatment_effectiveness': float(outlier_summary['average_treatment_effectiveness']),
        'outlier_summary': outlier_manager._convert_to_serializable(outlier_summary),
        'outlier_analysis_results': outlier_manager._convert_to_serializable(outlier_analysis_results)
    }
    
    with open("outlier_pipeline_report.json", "w") as f:
        json.dump(final_report, f, indent=2)
    print("✓ Pipeline report saved: outlier_pipeline_report.json")
    
    # Final summary
    print(f"\n🎉 OUTLIER MANAGEMENT PIPELINE COMPLETED!")
    print(f"📊 Datasets analyzed: {analysis_count}")
    print(f"📊 Datasets processed: {processed_count}")
    print(f"📊 Total outliers treated: {total_outliers_treated}")
    print(f"📊 Average treatment effectiveness: {outlier_summary['average_treatment_effectiveness']:.1f}%")
    print(f"📁 Treated data saved to: {output_directory}")
    print(f"📄 Reports saved: outlier_analysis_report.json, outlier_management_report.json, outlier_pipeline_report.json")

if __name__ == "__main__":
    main()
