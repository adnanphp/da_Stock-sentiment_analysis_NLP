# feature_pipeline.py
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from feature_engineer import FeatureEngineer

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
    input_directory = "preprocessed_financial_data"  # From previous step
    output_directory = "feature_engineered_data"
    config_file = "feature_config.py"
    
    print("=" * 70)
    print("FEATURE ENGINEERING PIPELINE")
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
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer(config)
    
    # Manual feature plans (customize as needed)
    FEATURE_PLANS = {
        "all_stocks_combined.csv": {
            "feature_groups": {
                "technical_indicators": {
                    "price_column": "Close",
                    "volume_column": "Volume",
                    "indicators": ["sma", "rsi", "macd", "bollinger_bands"]
                },
                "time_features": {
                    "datetime_columns": ["Date"],
                    "feature_types": ["temporal", "cyclical", "lagged", "rolling"],
                    "numeric_columns_for_lag": ["Close", "Volume", "Open", "High", "Low"],
                    "numeric_columns_for_rolling": ["Close", "Volume"]
                }
            }
        },
        "COST_full_year.csv": {
            "feature_groups": {
                "technical_indicators": {
                    "price_column": "Close",
                    "volume_column": "Volume",
                    "indicators": ["sma", "rsi", "bollinger_bands"]
                },
                "time_features": {
                    "datetime_columns": ["Date"],
                    "feature_types": ["temporal", "cyclical"]
                }
            }
        },
        "V_full_year.csv": {
            "feature_groups": {
                "technical_indicators": {
                    "price_column": "Close",
                    "volume_column": "Volume",
                    "indicators": ["sma", "rsi", "bollinger_bands"]
                },
                "time_features": {
                    "datetime_columns": ["Date"],
                    "feature_types": ["temporal", "cyclical"]
                }
            }
        },
        "reddit_dividends_expanded.csv": {
            "feature_groups": {
                "text_features": {
                    "text_columns": ["title", "selftext"],
                    "feature_types": ["basic_stats", "linguistic", "sentiment"]
                },
                "time_features": {
                    "datetime_columns": ["created_utc"],
                    "feature_types": ["temporal", "cyclical"]
                }
            }
        },
        "alpha_vantage_news_expanded.csv": {
            "feature_groups": {
                "text_features": {
                    "text_columns": ["title", "summary"],
                    "feature_types": ["basic_stats", "linguistic", "readability", "sentiment"]
                },
                "time_features": {
                    "datetime_columns": ["time_published"],
                    "feature_types": ["temporal", "cyclical"]
                }
            }
        }
    }
    
    # 1. Analysis Phase
    print("\n1. ANALYZING FEATURE ENGINEERING OPPORTUNITIES...")
    feature_analysis_results = {}
    
    analysis_count = 0
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.startswith('preprocessed_') and file.endswith(('.csv', '.parquet')):
                file_path = os.path.join(root, file)
                dataset_name = file.replace('preprocessed_', '')
                
                try:
                    print(f"🔍 Analyzing feature opportunities in: {dataset_name}")
                    df = safe_read_dataframe(file_path)
                    
                    # Analyze feature engineering opportunities
                    analysis = feature_engineer.analyze_dataset_features(df, dataset_name)
                    feature_analysis_results[dataset_name] = analysis
                    analysis_count += 1
                    
                    print(f"✓ Analyzed: {dataset_name}")
                    print(f"  📊 Recommendations: {len(analysis['feature_recommendations'])}")
                    for rec in analysis['feature_recommendations']:
                        print(f"  💡 {rec}")
                    
                except Exception as e:
                    print(f"❌ Error analyzing {dataset_name}: {e}")
                    feature_analysis_results[dataset_name] = {"error": str(e)}
    
    # Save analysis report
    with open("feature_analysis_report.json", "w") as f:
        json.dump(feature_engineer._convert_to_serializable(feature_analysis_results), f, indent=2)
    print(f"✓ Feature analysis report saved: feature_analysis_report.json")
    
    # 2. Feature Engineering Phase
    print("\n2. ENGINEERING FEATURES...")
    
    processed_count = 0
    total_features_created = 0
    
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            if file.startswith('preprocessed_') and file.endswith(('.csv', '.parquet')):
                file_path = os.path.join(root, file)
                original_dataset_name = file.replace('preprocessed_', '')
                dataset_name = original_dataset_name
                
                try:
                    print(f"\n🔄 Processing: {dataset_name}")
                    df = safe_read_dataframe(file_path)
                    
                    # Get feature plan for this dataset
                    feature_plan = None
                    if dataset_name in FEATURE_PLANS:
                        feature_plan = FEATURE_PLANS[dataset_name]
                        print(f"  Using manual feature plan")
                    else:
                        print(f"  Using auto-generated feature plan")
                    
                    # Engineer features
                    engineered_df = feature_engineer.engineer_features(
                        df=df,
                        dataset_name=dataset_name,
                        feature_plan=feature_plan
                    )
                    
                    # Save engineered data
                    output_subdir = os.path.join(output_directory, os.path.basename(root))
                    os.makedirs(output_subdir, exist_ok=True)
                    
                    output_filename = f"engineered_{original_dataset_name}"
                    output_path = os.path.join(output_subdir, output_filename)
                    
                    if file.endswith('.csv'):
                        engineered_df.to_csv(output_path, index=False)
                    else:
                        engineered_df.to_parquet(output_path, index=False)
                    
                    # Get features created for this dataset
                    features_created = feature_engineer.feature_engineering_report.get(
                        dataset_name, {}
                    ).get('total_features_created', 0)
                    
                    processed_count += 1
                    total_features_created += features_created
                    print(f"✓ Engineered: {dataset_name} -> {output_path}")
                    print(f"  📊 Features created: {features_created}")
                    
                except Exception as e:
                    print(f"❌ Error engineering features for {dataset_name}: {e}")
    
    # 3. Generate Summary
    print("\n3. GENERATING FEATURE ENGINEERING SUMMARY...")
    
    feature_summary = feature_engineer.get_feature_summary()
    print(f"\nFEATURE ENGINEERING SUMMARY:")
    print(f"Total datasets processed: {feature_summary['total_datasets_processed']}")
    print(f"Total features created: {feature_summary['total_features_created']}")
    print(f"Average features per dataset: {feature_summary['average_features_per_dataset']:.1f}")
    print(f"Feature groups used: {feature_summary['feature_groups_used']}")
    
    # 4. Save Reports
    print("\n4. SAVING FINAL REPORTS...")
    
    feature_engineer.save_engineering_report("feature_engineering_report.json")
    
    # Create final pipeline report
    final_report = {
        'pipeline_execution_time': datetime.now().isoformat(),
        'datasets_analyzed': analysis_count,
        'datasets_processed': processed_count,
        'total_features_created': total_features_created,
        'average_features_per_dataset': feature_summary['average_features_per_dataset'],
        'feature_summary': feature_summary,
        'feature_analysis_results': feature_engineer._convert_to_serializable(feature_analysis_results)
    }
    
    with open("feature_pipeline_report.json", "w") as f:
        json.dump(final_report, f, indent=2)
    print("✓ Pipeline report saved: feature_pipeline_report.json")
    
    # Final summary
    print(f"\n🎉 FEATURE ENGINEERING PIPELINE COMPLETED!")
    print(f"📊 Datasets analyzed: {analysis_count}")
    print(f"📊 Datasets processed: {processed_count}")
    print(f"📊 Total features created: {total_features_created}")
    print(f"📊 Average features per dataset: {feature_summary['average_features_per_dataset']:.1f}")
    print(f"📁 Engineered data saved to: {output_directory}")
    print(f"📄 Reports saved: feature_analysis_report.json, feature_engineering_report.json, feature_pipeline_report.json")

if __name__ == "__main__":
    main()
