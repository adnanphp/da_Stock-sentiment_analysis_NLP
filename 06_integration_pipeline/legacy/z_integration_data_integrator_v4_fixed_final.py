"""
z_integration_debugger.py - Debug why news data isn't merging
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def debug_integration():
    print("="*80)
    print("DEBUGGING INTEGRATION ISSUES")
    print("="*80)
    
    # Load the news data
    print("\n1. LOADING NEWS DATA...")
    news = pd.read_csv("./feature_engineered_data/news_data/feature_engineered_outlier_treated_alpha_vantage_news_expanded.csv")
    print(f"   News shape: {news.shape}")
    
    # Look for AAPL mentions
    print("\n2. SEARCHING FOR AAPL IN NEWS...")
    
    # Check different ways to find AAPL
    if 'title' in news.columns:
        aapl_in_title = news['title'].astype(str).str.upper().str.contains('AAPL', na=False)
        print(f"   AAPL in title: {aapl_in_title.sum()} mentions")
        
        # Show some examples
        print(f"   Sample titles with AAPL:")
        aapl_titles = news[aapl_in_title]['title'].head(3).tolist()
        for i, title in enumerate(aapl_titles, 1):
            print(f"     {i}. {title[:100]}...")
    
    if 'tickers' in news.columns:
        aapl_in_tickers = news['tickers'].astype(str).str.upper().str.contains('AAPL', na=False)
        print(f"   AAPL in tickers column: {aapl_in_tickers.sum()} mentions")
        
        # Show tickers column samples
        print(f"   Sample tickers values:")
        ticker_samples = news[aapl_in_tickers]['tickers'].head(5).tolist()
        for i, tickers in enumerate(ticker_samples, 1):
            print(f"     {i}. {tickers}")
    
    # Process dates
    print("\n3. CHECKING DATES...")
    if 'published_at' in news.columns:
        news['date'] = pd.to_datetime(news['published_at'])
        news['date_only'] = news['date'].dt.date
        
        print(f"   First few dates in news:")
        print(f"     {news['date_only'].head(5).tolist()}")
        print(f"     Date range: {news['date_only'].min()} to {news['date_only'].max()}")
    
    # Load stock data
    print("\n4. LOADING STOCK DATA...")
    stocks = pd.read_csv("./feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv")
    stocks = stocks[stocks['ticker'] == 'AAPL'].copy()
    print(f"   AAPL stock records: {stocks.shape}")
    
    if 'Date' in stocks.columns:
        stocks['date'] = pd.to_datetime(stocks['Date'])
        stocks['date_only'] = stocks['date'].dt.date
        
        print(f"   First few dates in stocks:")
        print(f"     {stocks['date_only'].head(5).tolist()}")
        print(f"     Date range: {stocks['date_only'].min()} to {stocks['date_only'].max()}")
    
    # Create news aggregates for AAPL
    print("\n5. CREATING NEWS AGGREGATES FOR AAPL...")
    
    # Find AAPL mentions
    mask = news['title'].astype(str).str.upper().str.contains('AAPL', na=False)
    aapl_news = news[mask].copy()
    
    if len(aapl_news) > 0:
        print(f"   Found {len(aapl_news)} AAPL mentions")
        
        # Group by date
        daily_counts = aapl_news.groupby('date_only').size().reset_index(name='news_aapl_count')
        print(f"   Created {len(daily_counts)} daily aggregates")
        
        # Show the aggregates
        print(f"\n   DAILY NEWS AGGREGATES:")
        print(daily_counts.head(10).to_string())
        
        # Check if dates match
        print(f"\n6. CHECKING DATE MATCHES...")
        
        # Get stock dates
        stock_dates = set(stocks['date_only'].tolist())
        news_dates = set(daily_counts['date_only'].tolist())
        
        print(f"   Stock dates: {len(stock_dates)} unique dates")
        print(f"   News dates: {len(news_dates)} unique dates")
        
        # Check for overlap
        overlap = stock_dates.intersection(news_dates)
        print(f"   Overlapping dates: {len(overlap)}")
        
        if overlap:
            print(f"   Sample overlapping dates: {list(overlap)[:5]}")
        else:
            print(f"   ⚠️ NO DATE OVERLAP FOUND!")
            
            # Show date ranges
            print(f"   Stock date range: {stocks['date_only'].min()} to {stocks['date_only'].max()}")
            print(f"   News date range: {daily_counts['date_only'].min()} to {daily_counts['date_only'].max()}")
            
            # Check if we can align dates
            print(f"\n7. TRYING TO ALIGN DATES...")
            
            # Find closest dates
            stock_date_list = sorted(list(stock_dates))
            news_date_list = sorted(list(news_dates))
            
            print(f"   First 5 stock dates: {stock_date_list[:5]}")
            print(f"   First 5 news dates: {news_date_list[:5]}")
            
            # Check if it's a format issue
            print(f"\n8. CHECKING DATE FORMATS...")
            sample_stock_date = stocks['Date'].iloc[0]
            sample_news_date = news['published_at'].iloc[0] if len(news) > 0 else 'N/A'
            
            print(f"   Sample stock date (raw): {sample_stock_date}")
            print(f"   Sample news date (raw): {sample_news_date}")
            
            # Try different parsing
            print(f"\n9. TRYING DIFFERENT DATE PARSING...")
            
            # Check news date format
            if 'published_at' in news.columns:
                print(f"   News published_at sample: {news['published_at'].iloc[0]}")
                
                # Try to parse with timezone
                try:
                    news['date_parsed'] = pd.to_datetime(news['published_at'], utc=True)
                    news['date_parsed_only'] = news['date_parsed'].dt.date
                    print(f"   Parsed with UTC: {news['date_parsed_only'].iloc[0]}")
                except:
                    print(f"   Could not parse with UTC")
            
    else:
        print(f"   No AAPL mentions found")
    
    # Check the actual integration attempt
    print("\n" + "="*80)
    print("DEBUGGING THE MERGE ISSUE")
    print("="*80)
    
    # Try a simple merge
    print("\nTrying simple merge...")
    
    # Prepare data
    stocks_sample = stocks[['date_only', 'Open', 'Close', 'Volume']].head(10).copy()
    if len(daily_counts) > 0:
        news_sample = daily_counts.head(10).copy()
        
        print(f"\nStocks sample:")
        print(stocks_sample.to_string())
        
        print(f"\nNews sample:")
        print(news_sample.to_string())
        
        # Try merge
        print(f"\nAttempting merge on 'date_only'...")
        merged = pd.merge(stocks_sample, news_sample, on='date_only', how='left')
        print(f"Merge result shape: {merged.shape}")
        print(f"Merged data:")
        print(merged.to_string())
        
        if 'news_aapl_count' in merged.columns:
            print(f"\n✅ Merge successful! News data added.")
            print(f"News count column present with {merged['news_aapl_count'].notna().sum()} non-NaN values")
        else:
            print(f"\n❌ Merge failed - news column not added")

def fix_and_run_integration():
    """Fix the integration and run it properly"""
    print("\n" + "="*80)
    print("FIXED INTEGRATION - WORKING VERSION")
    print("="*80)
    
    # Load data
    print("\n1. Loading data...")
    stocks = pd.read_csv("./feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv")
    stocks = stocks[stocks['ticker'].isin(['AAPL', 'GOOGL', 'TSLA'])].copy()
    
    news = pd.read_csv("./feature_engineered_data/news_data/feature_engineered_outlier_treated_alpha_vantage_news_expanded.csv")
    
    print(f"   Stocks: {stocks.shape}")
    print(f"   News: {news.shape}")
    
    # Convert dates PROPERLY
    print("\n2. Converting dates...")
    
    # Stocks date
    stocks['date'] = pd.to_datetime(stocks['Date'])
    stocks['merge_date'] = stocks['date'].dt.date  # Date only, no time
    
    # News date - handle timezone
    news['date_raw'] = news['published_at']
    
    # Try different parsing strategies
    try:
        # Try parsing with timezone
        news['date'] = pd.to_datetime(news['published_at'], utc=True)
    except:
        try:
            # Try without timezone
            news['date'] = pd.to_datetime(news['published_at'])
        except:
            # Last resort
            news['date'] = pd.to_datetime(news['published_at'], errors='coerce')
    
    news['merge_date'] = news['date'].dt.date
    
    print(f"   Stock date range: {stocks['merge_date'].min()} to {stocks['merge_date'].max()}")
    print(f"   News date range: {news['merge_date'].min()} to {news['merge_date'].max()}")
    
    # Process each ticker
    print("\n3. Processing tickers...")
    
    all_integrated = []
    
    for ticker in ['AAPL', 'GOOGL', 'TSLA']:
        print(f"\n   Processing {ticker}...")
        
        # Get stock data
        ticker_stocks = stocks[stocks['ticker'] == ticker].copy()
        print(f"     Stock records: {len(ticker_stocks)}")
        
        # Find news for this ticker
        # Search in multiple columns
        mask = (
            news['title'].astype(str).str.upper().str.contains(ticker, na=False) |
            news['tickers'].astype(str).str.upper().str.contains(ticker, na=False) |
            news['summary'].astype(str).str.upper().str.contains(ticker, na=False)
        )
        
        ticker_news = news[mask].copy()
        print(f"     News mentions found: {len(ticker_news)}")
        
        if len(ticker_news) > 0:
            # Create aggregates
            daily_agg = ticker_news.groupby('merge_date').agg({
                'title': 'count',
                'overall_sentiment': 'mean'
            }).reset_index()
            
            daily_agg.columns = ['merge_date', f'news_{ticker}_count', f'news_{ticker}_sentiment']
            
            print(f"     Daily aggregates: {len(daily_agg)}")
            
            # Merge with stocks
            merged = pd.merge(
                ticker_stocks,
                daily_agg,
                on='merge_date',
                how='left'
            )
            
            # Count non-null news values
            news_col = f'news_{ticker}_count'
            if news_col in merged.columns:
                non_null = merged[news_col].notna().sum()
                print(f"     Successfully merged: {non_null} records have news data")
            else:
                print(f"     ⚠️ News column not added")
            
            all_integrated.append(merged)
        else:
            print(f"     No news to merge")
            all_integrated.append(ticker_stocks)
    
    # Combine all tickers
    if all_integrated:
        final_df = pd.concat(all_integrated, ignore_index=True)
        
        print(f"\n✅ FINAL INTEGRATED DATASET:")
        print(f"   Rows: {len(final_df)}")
        print(f"   Columns: {len(final_df.columns)}")
        
        # Count news columns
        news_cols = [col for col in final_df.columns if col.startswith('news_')]
        print(f"   News columns: {len(news_cols)}")
        
        if news_cols:
            print(f"   News columns added: {news_cols}")
            
            # Check if news data is actually in the dataset
            for col in news_cols:
                non_null = final_df[col].notna().sum()
                print(f"     • {col}: {non_null} non-null values")
        
        # Save the dataset
        output_dir = "./integrated_fixed"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"integrated_fixed_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        final_df.to_csv(filepath, index=False)
        
        print(f"\n💾 Saved to: {filepath}")
        
        # Show sample
        print(f"\n👀 SAMPLE DATA WITH NEWS:")
        sample_cols = ['date', 'ticker', 'Open', 'Close']
        if news_cols:
            sample_cols.extend(news_cols[:2])
        
        sample = final_df[sample_cols].head(10)
        print(sample.to_string())
        
        return final_df, filepath
    
    return None, None

if __name__ == "__main__":
    # First debug to understand the issue
    debug_integration()
    
    # Then run the fixed version
    fix_and_run_integration()
