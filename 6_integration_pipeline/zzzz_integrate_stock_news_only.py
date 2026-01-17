"""
integrate_stock_news_only.py - Integrate only stock and news data
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def analyze_data_coverage():
    """Analyze how much data we have for stock and news"""
    print("="*80)
    print("DATA COVERAGE ANALYSIS")
    print("="*80)
    
    # Load your original dataset
    data_file = "./final_dataset/final_integrated_dataset_20251202_221758.csv"
    df = pd.read_csv(data_file)
    
    print(f"📂 Original dataset shape: {df.shape}")
    print(f"   Total records: {len(df)}")
    
    tickers = ['AAPL', 'GOOGL', 'TSLA']
    
    # Stock data analysis
    print(f"\n📊 STOCK DATA ANALYSIS:")
    print(f"{'-'*60}")
    
    stock_cols = [col for col in df.columns 
                 if not any(x in col for x in ['news_', 'reddit_', 'target', 'price_change'])]
    print(f"   Total stock-related columns: {len(stock_cols)}")
    
    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker]
        print(f"\n   {ticker}:")
        print(f"     • Records: {len(ticker_data)}")
        print(f"     • Date range: {ticker_data['date'].min()} to {ticker_data['date'].max()}")
        print(f"     • Stock features: {len([c for c in stock_cols if c in ticker_data.columns])}")
    
    # News data analysis
    print(f"\n📰 NEWS DATA ANALYSIS:")
    print(f"{'-'*60}")
    
    news_cols = [col for col in df.columns if 'news_' in col]
    print(f"   Total news-related columns: {len(news_cols)}")
    
    news_summary = []
    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker]
        
        # Count for this specific ticker
        count_col = f'news_{ticker}_count'
        sentiment_col = f'news_{ticker}_sentiment'
        
        if count_col in df.columns:
            news_count = ticker_data[count_col].notna().sum()
            total_records = len(ticker_data)
            coverage_pct = (news_count / total_records * 100) if total_records > 0 else 0
            
            avg_sentiment = np.nan
            if sentiment_col in df.columns and news_count > 0:
                avg_sentiment = ticker_data[sentiment_col].mean()
            
            print(f"\n   {ticker}:")
            print(f"     • Records with news: {news_count}/{total_records} ({coverage_pct:.1f}%)")
            print(f"     • Average sentiment: {avg_sentiment:.3f}" if not np.isnan(avg_sentiment) else "     • Average sentiment: N/A")
            
            # List available news columns for this ticker
            ticker_news_cols = [col for col in news_cols if ticker in col]
            print(f"     • News columns: {len(ticker_news_cols)}")
            for col in ticker_news_cols[:3]:  # Show first 3
                print(f"       - {col}")
            if len(ticker_news_cols) > 3:
                print(f"       ... and {len(ticker_news_cols)-3} more")
            
            news_summary.append({
                'ticker': ticker,
                'total_records': total_records,
                'news_records': news_count,
                'coverage_pct': coverage_pct,
                'avg_sentiment': avg_sentiment if not np.isnan(avg_sentiment) else None,
                'news_columns': ticker_news_cols
            })
    
    # Reddit data analysis (for comparison)
    print(f"\n📱 REDDIT DATA ANALYSIS (for reference):")
    print(f"{'-'*60}")
    
    reddit_cols = [col for col in df.columns if 'reddit_' in col]
    print(f"   Total reddit-related columns: {len(reddit_cols)}")
    
    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker]
        count_col = f'reddit_{ticker}_post_count'
        
        if count_col in df.columns:
            reddit_count = ticker_data[count_col].notna().sum()
            total_records = len(ticker_data)
            coverage_pct = (reddit_count / total_records * 100) if total_records > 0 else 0
            
            print(f"   {ticker}: {reddit_count}/{total_records} records ({coverage_pct:.1f}%)")
    
    return df, pd.DataFrame(news_summary)

def integrate_stock_news_only(df):
    """Create integrated dataset with only stock and news data"""
    print("\n" + "="*80)
    print("INTEGRATING STOCK + NEWS DATA ONLY")
    print("="*80)
    
    # Keep only stock and news columns
    # Define columns to keep
    
    # 1. Essential metadata columns
    essential_cols = ['date', 'ticker']
    
    # 2. All stock-related columns (excluding news and reddit)
    stock_cols = []
    for col in df.columns:
        if col in essential_cols:
            continue
        if 'news_' not in col and 'reddit_' not in col:
            stock_cols.append(col)
    
    # 3. All news columns
    news_cols = [col for col in df.columns if 'news_' in col]
    
    # Combine all columns to keep
    cols_to_keep = essential_cols + stock_cols + news_cols
    
    # Create the integrated dataset
    df_integrated = df[cols_to_keep].copy()
    
    print(f"📊 DATASET COMPOSITION:")
    print(f"   • Essential columns: {len(essential_cols)}")
    print(f"   • Stock columns: {len(stock_cols)}")
    print(f"   • News columns: {len(news_cols)}")
    print(f"   • Total columns: {len(cols_to_keep)}")
    print(f"   • Total records: {len(df_integrated)}")
    
    # Verify we removed reddit columns
    reddit_cols_in_result = [col for col in df_integrated.columns if 'reddit_' in col]
    if len(reddit_cols_in_result) == 0:
        print(f"   ✅ Successfully removed all Reddit columns")
    else:
        print(f"   ⚠️ Warning: {len(reddit_cols_in_result)} Reddit columns still present")
    
    # Create a price prediction target
    print(f"\n🎯 CREATING PREDICTION TARGET...")
    
    # Sort by ticker and date
    df_integrated = df_integrated.sort_values(['ticker', 'date'])
    
    # Calculate next day's price change
    df_integrated['price_change'] = df_integrated.groupby('ticker')['Close'].transform(
        lambda x: x.pct_change().shift(-1)
    )
    
    # Create binary target (1 = price goes up, 0 = price goes down)
    df_integrated['target'] = (df_integrated['price_change'] > 0).astype(int)
    
    # Remove rows without target
    df_integrated = df_integrated.dropna(subset=['target'])
    
    print(f"   Target created:")
    print(f"     • Total records with target: {len(df_integrated)}")
    print(f"     • UP predictions (target=1): {(df_integrated['target'] == 1).sum()}")
    print(f"     • DOWN predictions (target=0): {(df_integrated['target'] == 0).sum()}")
    print(f"     • Balance: {df_integrated['target'].mean():.2%} UP")
    
    # Show sample of the integrated data
    print(f"\n👀 SAMPLE OF INTEGRATED DATA:")
    
    # Get sample columns to display
    sample_essential = ['date', 'ticker', 'Close']
    sample_stock = ['Open', 'High', 'Low', 'Volume']
    sample_news = [col for col in news_cols if 'count' in col or 'sentiment' in col][:2]  # First 2 news columns
    sample_target = ['price_change', 'target']
    
    sample_cols = sample_essential + sample_stock + sample_news + sample_target
    sample_cols = [col for col in sample_cols if col in df_integrated.columns]
    
    # Show for each ticker
    for ticker in ['AAPL', 'GOOGL', 'TSLA'][:2]:  # Show first 2 tickers
        ticker_data = df_integrated[df_integrated['ticker'] == ticker]
        if len(ticker_data) > 0:
            print(f"\n   {ticker} (first 5 records):")
            print(ticker_data[sample_cols].head(5).to_string())
            print(f"   ...")
    
    return df_integrated

def save_integrated_dataset(df_integrated):
    """Save the integrated dataset"""
    print("\n" + "="*80)
    print("SAVING INTEGRATED DATASET")
    print("="*80)
    
    # Create output directory
    output_dir = "./integrated_stock_news"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save full dataset
    full_file = os.path.join(output_dir, f"stock_news_integrated_{timestamp}.csv")
    df_integrated.to_csv(full_file, index=False)
    
    print(f"💾 Full dataset saved to: {full_file}")
    print(f"   • Records: {len(df_integrated)}")
    print(f"   • Columns: {len(df_integrated.columns)}")
    
    # Create a simplified version (most important columns only)
    print(f"\n📦 CREATING SIMPLIFIED VERSION...")
    
    # Select most important columns
    essential_cols = ['date', 'ticker']
    
    # Key stock columns
    key_stock_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                      'SMA_20', 'SMA_50', 'RSI', 'MACD', 'BB_upper', 'BB_lower']
    key_stock_cols = [col for col in key_stock_cols if col in df_integrated.columns]
    
    # All news columns
    news_cols = [col for col in df_integrated.columns if 'news_' in col]
    
    # Target columns
    target_cols = ['price_change', 'target']
    
    simplified_cols = essential_cols + key_stock_cols + news_cols + target_cols
    
    # Filter to only columns that exist
    simplified_cols = [col for col in simplified_cols if col in df_integrated.columns]
    
    df_simplified = df_integrated[simplified_cols].copy()
    
    simple_file = os.path.join(output_dir, f"stock_news_simplified_{timestamp}.csv")
    df_simplified.to_csv(simple_file, index=False)
    
    print(f"💾 Simplified dataset saved to: {simple_file}")
    print(f"   • Records: {len(df_simplified)}")
    print(f"   • Columns: {len(df_simplified.columns)}")
    
    # Create a summary report
    print(f"\n📊 SUMMARY REPORT:")
    
    report_file = os.path.join(output_dir, f"integration_report_{timestamp}.txt")
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STOCK + NEWS INTEGRATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("DATASET OVERVIEW:\n")
        f.write("-"*40 + "\n")
        f.write(f"Total records: {len(df_integrated)}\n")
        f.write(f"Total columns: {len(df_integrated.columns)}\n")
        f.write(f"Date range: {df_integrated['date'].min()} to {df_integrated['date'].max()}\n")
        f.write(f"Tickers: {', '.join(df_integrated['ticker'].unique())}\n\n")
        
        f.write("DATA COMPOSITION:\n")
        f.write("-"*40 + "\n")
        f.write(f"Stock columns: {len([c for c in df_integrated.columns if 'news_' not in c and c not in ['date', 'ticker', 'price_change', 'target']])}\n")
        f.write(f"News columns: {len([c for c in df_integrated.columns if 'news_' in c])}\n\n")
        
        f.write("TARGET DISTRIBUTION:\n")
        f.write("-"*40 + "\n")
        up_count = (df_integrated['target'] == 1).sum()
        down_count = (df_integrated['target'] == 0).sum()
        total = len(df_integrated)
        f.write(f"UP (target=1): {up_count} ({up_count/total:.1%})\n")
        f.write(f"DOWN (target=0): {down_count} ({down_count/total:.1%})\n\n")
        
        f.write("NEWS DATA COVERAGE:\n")
        f.write("-"*40 + "\n")
        tickers = df_integrated['ticker'].unique()
        for ticker in tickers:
            ticker_data = df_integrated[df_integrated['ticker'] == ticker]
            count_col = f'news_{ticker}_count'
            
            if count_col in df_integrated.columns:
                news_count = ticker_data[count_col].notna().sum()
                coverage = news_count / len(ticker_data) if len(ticker_data) > 0 else 0
                f.write(f"{ticker}: {news_count}/{len(ticker_data)} records ({coverage:.1%})\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FILES CREATED:\n")
        f.write("-"*40 + "\n")
        f.write(f"1. Full dataset: {full_file}\n")
        f.write(f"   - Records: {len(df_integrated)}\n")
        f.write(f"   - Columns: {len(df_integrated.columns)}\n")
        f.write(f"\n2. Simplified dataset: {simple_file}\n")
        f.write(f"   - Records: {len(df_simplified)}\n")
        f.write(f"   - Columns: {len(df_simplified.columns)}\n")
    
    print(f"📋 Report saved to: {report_file}")
    
    return full_file, simple_file, report_file

def validate_integration(df_integrated):
    """Validate the integration was successful"""
    print("\n" + "="*80)
    print("VALIDATION CHECK")
    print("="*80)
    
    # Check 1: No Reddit columns
    reddit_cols = [col for col in df_integrated.columns if 'reddit_' in col]
    if len(reddit_cols) == 0:
        print("✅ PASS: No Reddit columns found")
    else:
        print(f"❌ FAIL: {len(reddit_cols)} Reddit columns found")
        print(f"   Columns: {reddit_cols}")
    
    # Check 2: Has news columns
    news_cols = [col for col in df_integrated.columns if 'news_' in col]
    if len(news_cols) > 0:
        print(f"✅ PASS: {len(news_cols)} news columns found")
    else:
        print("❌ FAIL: No news columns found")
    
    # Check 3: Has target column
    if 'target' in df_integrated.columns:
        print("✅ PASS: Target column created")
    else:
        print("❌ FAIL: Target column missing")
    
    # Check 4: No NaN in essential columns
    essential_cols = ['date', 'ticker', 'Close', 'target']
    missing_data = {}
    for col in essential_cols:
        if col in df_integrated.columns:
            nan_count = df_integrated[col].isna().sum()
            if nan_count == 0:
                print(f"✅ PASS: No NaN values in {col}")
            else:
                print(f"⚠️ WARNING: {nan_count} NaN values in {col}")
                missing_data[col] = nan_count
    
    # Check 5: Data by ticker
    print(f"\n📈 DATA BY TICKER:")
    for ticker in df_integrated['ticker'].unique():
        ticker_data = df_integrated[df_integrated['ticker'] == ticker]
        news_count_col = f'news_{ticker}_count'
        
        if news_count_col in df_integrated.columns:
            news_records = ticker_data[news_count_col].notna().sum()
            print(f"   {ticker}: {len(ticker_data)} total, {news_records} with news ({news_records/len(ticker_data):.1%})")
        else:
            print(f"   {ticker}: {len(ticker_data)} total, 0 with news (0%)")
    
    return len(reddit_cols) == 0 and len(news_cols) > 0 and 'target' in df_integrated.columns

def main():
    """Main function to integrate stock and news data"""
    print("="*80)
    print("STOCK + NEWS DATA INTEGRATION")
    print("="*80)
    
    # Step 1: Analyze data coverage
    df, news_summary = analyze_data_coverage()
    
    print(f"\n📋 NEWS SUMMARY TABLE:")
    print(news_summary.to_string())
    
    # Step 2: Integrate stock and news only
    df_integrated = integrate_stock_news_only(df)
    
    # Step 3: Validate integration
    is_valid = validate_integration(df_integrated)
    
    if is_valid:
        # Step 4: Save the integrated dataset
        full_file, simple_file, report_file = save_integrated_dataset(df_integrated)
        
        print(f"\n" + "="*80)
        print("🎯 INTEGRATION COMPLETE!")
        print("="*80)
        
        print(f"\n📁 YOUR FILES:")
        print(f"   1. Full integrated dataset: {full_file}")
        print(f"   2. Simplified dataset: {simple_file}")
        print(f"   3. Integration report: {report_file}")
        
        print(f"\n🔍 DATASET READY FOR:")
        print(f"   • Machine learning experiments")
        print(f"   • Predictive modeling")
        print(f"   • Analysis of news impact on stocks")
        print(f"   • Time-series forecasting")
        
        return df_integrated, full_file, simple_file, report_file
    else:
        print(f"\n❌ Integration failed validation. Please check the warnings above.")
        return None

if __name__ == "__main__":
    result = main()
    
    if result:
        df_integrated, full_file, simple_file, report_file = result
        
        # Quick analysis of the final dataset
        print(f"\n📊 FINAL DATASET QUICK STATS:")
        print(f"   Shape: {df_integrated.shape}")
        print(f"   Memory usage: {df_integrated.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Show column types
        print(f"\n📝 COLUMN TYPES:")
        print(f"   Numeric: {sum([pd.api.types.is_numeric_dtype(df_integrated[col]) for col in df_integrated.columns])}")
        print(f"   Object: {sum([pd.api.types.is_string_dtype(df_integrated[col]) for col in df_integrated.columns])}")
        print(f"   Datetime: {sum([pd.api.types.is_datetime64_any_dtype(df_integrated[col]) for col in df_integrated.columns])}")
