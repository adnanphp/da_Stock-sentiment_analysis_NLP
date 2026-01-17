# run_feature_engineering.py
import pandas as pd
import os
from feature_engineer import FeatureEngineer

def run_feature_engineering():
    """Run feature engineering on the final cleaned data."""
    
    input_dir = "FINAL_cleaned_data"
    output_dir = "feature_engineered_data"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 STARTING FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    print("Input: FINAL_cleaned_data/")
    print("Output: feature_engineered_data/")
    print("=" * 60)
    
    # Initialize feature engineering
    feature_engineer = FeatureEngineer()
    
    processed_count = 0
    total_features_added = 0
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.startswith('FINAL_') and file.endswith('.csv'):
                input_path = os.path.join(root, file)
                output_filename = f"feature_engineered_{file.replace('FINAL_', '')}"
                
                # Create same directory structure in output
                relative_path = os.path.relpath(root, input_dir)
                output_subdir = os.path.join(output_dir, relative_path)
                os.makedirs(output_subdir, exist_ok=True)
                output_path = os.path.join(output_subdir, output_filename)
                
                try:
                    print(f"\n🔧 Processing: {file}")
                    
                    # Load the final cleaned data
                    df = pd.read_csv(input_path, low_memory=False)
                    original_shape = df.shape
                    print(f"   📊 Input shape: {original_shape}")
                    
                    # Apply feature engineering
                    engineered_df = feature_engineer.engineer_features(df)
                    
                    final_shape = engineered_df.shape
                    features_added = final_shape[1] - original_shape[1]
                    total_features_added += features_added
                    
                    # Save results
                    engineered_df.to_csv(output_path, index=False)
                    processed_count += 1
                    
                    print(f"   ✅ Saved: {output_filename}")
                    print(f"   📈 Features added: {features_added}")
                    print(f"   📊 Final shape: {final_shape}")
                    
                    # Show sample of new features
                    new_cols = [col for col in engineered_df.columns if col not in df.columns]
                    if new_cols:
                        sample_new = new_cols[:5]
                        print(f"   🆕 Sample new features: {', '.join(sample_new)}" + ("..." if len(new_cols) > 5 else ""))
                    
                except Exception as e:
                    print(f"   ❌ Error processing {file}: {e}")
    
    print(f"\n🎉 FEATURE ENGINEERING COMPLETED!")
    print(f"📊 Datasets processed: {processed_count}")
    print(f"📈 Total features added: {total_features_added}")
    print(f"📁 Output directory: {output_dir}")
    
    return processed_count, total_features_added

def verify_feature_engineering():
    """Verify the feature engineering results."""
    input_dir = "feature_engineered_data"
    
    print("\n🔍 VERIFYING FEATURE ENGINEERING RESULTS")
    print("=" * 50)
    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.startswith('feature_engineered_') and file.endswith('.csv'):
                file_path = os.path.join(root, file)
                
                try:
                    df = pd.read_csv(file_path, low_memory=False)
                    
                    print(f"\n📊 {file}:")
                    print(f"   Shape: {df.shape}")
                    
                    # Check for common feature types
                    feature_types = {
                        'technical_indicators': len([col for col in df.columns if any(x in col.lower() for x in ['sma', 'ema', 'rsi', 'macd', 'bollinger'])]),
                        'interaction_terms': len([col for col in df.columns if any(x in col.lower() for x in ['_x_', '_vs_', '_ratio', '_interaction'])]),
                        'time_based': len([col for col in df.columns if any(x in col.lower() for x in ['lag', 'rolling', 'momentum', 'trend'])]),
                        'statistical': len([col for col in df.columns if any(x in col.lower() for x in ['zscore', 'normalized', 'standardized', 'scaled'])])
                    }
                    
                    print(f"   Feature types:")
                    for feature_type, count in feature_types.items():
                        if count > 0:
                            print(f"     • {feature_type}: {count}")
                    
                    # Data quality check
                    null_percentage = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                    print(f"   📊 Null percentage: {null_percentage:.2f}%")
                    
                    if null_percentage < 5:
                        print("   ✅ Data quality: EXCELLENT")
                    else:
                        print("   ⚠️  Data quality: Some null values present")
                        
                except Exception as e:
                    print(f"   ❌ Error analyzing {file}: {e}")
    
    print("\n🎉 VERIFICATION COMPLETED!")

if __name__ == "__main__":
    # Run feature engineering
    processed_count, total_features = run_feature_engineering()
    
    # Verify results
    verify_feature_engineering()
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"1. Your data is now feature-rich and ready for modeling!")
    print(f"2. Proceed to model training with your modeling pipeline")
    print(f"3. Consider feature selection if you have too many features (>1000)")
