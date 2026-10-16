# normalization_pipeline.py
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from data_scaler import DataScaler

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
    input_directory = "feature_engineered_data"  # From previous step
    output_directory = "normalized_financial_data"
    config_file = "scaling_config.py"
    
    print("=" * 70)
    print("DATA NORMALIZATION PIPELINE")
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
    
    # Initialize data scaler
    data_scaler = DataScaler(config)
    
    # Manual scaling plans (customize as needed)
    SCALING_PLANS = {
        "engineered_all_stocks_combined.csv": {
            "scaler_groups": {
                "standard": {
                    "columns": ["Close", "Open", "High", "Low", "Volume", "market_cap"],
                    "scaler_params": {"with_mean": True, "with_std": True}
                },
                "minmax": {
                    "columns": ["RSI", "MACD", "BB_Position", "Stoch_K", "Stoch_D"],
                    "scaler_params": {"feature_range": (0, 1)}
                },
                "robust": {
                    "columns": ["Volume_Ratio", "Volatility_20", "Volatility_50"],
                    "scaler_params": {"with_centering": True, "with_scaling": True}
                }
            }
        },
        "engineered_COST_full_year.csv": {
            "scaler_groups": {
                "standard": {
                    "columns": ["Close", "Open", "High", "Low", "Volume"],
                    "scaler_params": {"with_mean": True, "with_std": True}
                },
                "minmax": {
                    "columns": ["RSI", "BB_Position"],
                    "scaler_params": {"feature_range": (0, 1)}
                }
            }
        },
        "engineered_V_full_year.csv": {
            "scaler_groups": {
                "standard": {
                    "columns": ["Close", "Open", "High", "Low", "Volume"],
                    "scaler_params": {"with_mean": True, "with_std": True}
                },
                "minmax": {
                    "columns": ["RSI", "BB_Position"],
                    "scaler_params": {"feature_range": (0, 1)}
                }
            }
        },
        "engineered_reddit_dividends_expanded.csv": {
            "scaler_groups": {
                "minmax": {
                    "columns": ["title_char_count", "title_word_count", "title_sentiment_balance"],
                    "scaler_params": {"feature_range": (0, 1)}
                },
                "standard": {
                    "columns": ["created_utc_year", "created_utc_month", "created_utc_day"],
                    "scaler_params": {"with_mean": True, "with_std": True}
                }
            }
        }
    }
    
    # 1. Analysis Phase
    print("\n1. ANALYZING SCALING NEEDS...")
    scaling_analysis_results = {}
    
    analysis_count = 0
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.startswith('engineered_') and file.endswith(('.csv', '.parquet')):
                file_path = os.path.join(root, file)
                dataset_name = file.replace('engineered_', '')
                
                try:
                    print(f"🔍 Analyzing scaling needs for: {dataset_name}")
                    df = safe_read_dataframe(file_path)
                    
                    # Analyze scaling needs
                    analysis = data_scaler.analyze_scaling_needs(df, dataset_name)
                    scaling_analysis_results[dataset_name] = analysis
                    analysis_count += 1
                    
                    print(f"✓ Analyzed: {dataset_name}")
                    print(f"  📊 Columns needing scaling: {analysis['overview']['columns_needing_scaling']}")
                    for rec in analysis['scaling_recommendations']:
                        print(f"  💡 {rec}")
                    
                except Exception as e:
                    print(f"❌ Error analyzing {dataset_name}: {e}")
                    scaling_analysis_results[dataset_name] = {"error": str(e)}
    
    # Save analysis report
    with open("scaling_analysis_report.json", "w") as f:
        json.dump(data_scaler._convert_to_serializable(scaling_analysis_results), f, indent=2)
    print(f"✓ Scaling analysis report saved: scaling_analysis_report.json")
    
    # 2. Scaling Phase
    print("\n2. SCALING DATASETS...")
    
    processed_count = 0
    total_columns_scaled = 0
    
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.startswith('engineered_') and file.endswith(('.csv', '.parquet')):
                file_path = os.path.join(root, file)
                original_dataset_name = file.replace('engineered_', '')
                dataset_name = original_dataset_name
                
                try:
                    print(f"\n🔄 Processing: {dataset_name}")
                    df = safe_read_dataframe(file_path)
                    
                    # Get scaling plan for this dataset
                    scaling_plan = None
                    if dataset_name in SCALING_PLANS:
                        scaling_plan = SCALING_PLANS[dataset_name]
                        print(f"  Using manual scaling plan")
                    else:
                        print(f"  Using auto-generated scaling plan")
                    
                    # Scale dataset
                    scaled_df = data_scaler.scale_dataset(
                        df=df,
                        dataset_name=dataset_name,
                        scaling_plan=scaling_plan,
                        fit_scalers=True
                    )
                    
                    # Save scaled data
                    output_subdir = os.path.join(output_directory, os.path.basename(root))
                    os.makedirs(output_subdir, exist_ok=True)
                    
                    output_filename = f"normalized_{original_dataset_name}"
                    output_path = os.path.join(output_subdir, output_filename)
                    
                    if file.endswith('.csv'):
                        scaled_df.to_csv(output_path, index=False)
                    else:
                        scaled_df.to_parquet(output_path, index=False)
                    
                    # Get columns scaled for this dataset
                    columns_scaled = data_scaler.scaling_report.get(
                        dataset_name, {}
                    ).get('total_columns_scaled', 0)
                    
                    processed_count += 1
                    total_columns_scaled += columns_scaled
                    print(f"✓ Scaled: {dataset_name} -> {output_path}")
                    print(f"  📊 Columns scaled: {columns_scaled}")
                    
                except Exception as e:
                    print(f"❌ Error scaling {dataset_name}: {e}")
    
    # 3. Save fitted scalers
    print("\n3. SAVING FITTED SCALERS...")
    data_scaler.save_scalers("fitted_scalers")
    
    # 4. Generate Summary
    print("\n4. GENERATING SCALING SUMMARY...")
    
    scaling_summary = data_scaler.get_scaling_summary()
    print(f"\nSCALING SUMMARY:")
    print(f"Total datasets processed: {scaling_summary['total_datasets_processed']}")
    print(f"Total columns scaled: {scaling_summary['total_columns_scaled']}")
    print(f"Average columns per dataset: {scaling_summary['average_columns_per_dataset']:.1f}")
    print(f"Scaler distribution: {scaling_summary['scaler_distribution']}")
    
    # 5. Save Reports
    print("\n5. SAVING FINAL REPORTS...")
    
    data_scaler.save_scaling_report("data_scaling_report.json")
    
    # Create final pipeline report
    final_report = {
        'pipeline_execution_time': datetime.now().isoformat(),
        'datasets_analyzed': analysis_count,
        'datasets_processed': processed_count,
        'total_columns_scaled': total_columns_scaled,
        'average_columns_per_dataset': scaling_summary['average_columns_per_dataset'],
        'scaling_summary': scaling_summary,
        'scaling_analysis_results': data_scaler._convert_to_serializable(scaling_analysis_results)
    }
    
    with open("normalization_pipeline_report.json", "w") as f:
        json.dump(final_report, f, indent=2)
    print("✓ Pipeline report saved: normalization_pipeline_report.json")
    
    # Final summary
    print(f"\n🎉 DATA NORMALIZATION PIPELINE COMPLETED!")
    print(f"📊 Datasets analyzed: {analysis_count}")
    print(f"📊 Datasets processed: {processed_count}")
    print(f"📊 Total columns scaled: {total_columns_scaled}")
    print(f"📊 Average columns per dataset: {scaling_summary['average_columns_per_dataset']:.1f}")
    print(f"📁 Normalized data saved to: {output_directory}")
    print(f"📁 Fitted scalers saved to: fitted_scalers/")
    print(f"📄 Reports saved: scaling_analysis_report.json, data_scaling_report.json, normalization_pipeline_report.json")

if __name__ == "__main__":
    main()
