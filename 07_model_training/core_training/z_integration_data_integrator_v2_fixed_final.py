"""
data_integrator_fixed_final.py - Properly integrates ALL data sources
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import re

class FinancialDataIntegratorFinal:
    """
    Final integrator that properly finds and integrates ALL data sources
    """
    
    def __init__(self, base_path="."):
        self.base_path = base_path
        
    def inspect_datasets_detailed(self):
        """Detailed inspection of dataset structures"""
        print("🔍 DETAILED DATASET INSPECTION")
        print("="*60)
        
        # 1. Stock Data
        stock_path = "./feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv"
        if os.path.exists(stock_path):
            print(f"\n📈 STOCK DATA:")
            stocks = pd.read_csv(stock_path, nrows=3)
            print(f"   Shape: {stocks.shape}")
            print(f"   First 10 columns: {list(stocks.columns[:10])}")
            print(f"   Column types:")
            for col in stocks.columns[:15]:
                print(f"     • {col}: {stocks[col].dtype}")
            print(f"   Date columns: {[col for col in stocks.columns if 'date' in col.lower() or 'Date' in col]}")
            print(f"   Ticker columns: {[col for col in stocks.columns if 'ticker' in col.lower() or 'symbol' in col.lower()]}")
        
        # 2. News Data
        news_path = "./feature_engineered_data/news_data/feature_engineered_outlier_treated_alpha_vantage_news_expanded.csv"
        if os.path.exists(news_path):
            print(f"\n📰 NEWS DATA:")
            news = pd.read_csv(news_path, nrows=3)
            print(f"   Shape: {news.shape}")
            print(f"   First 10 columns: {list(news.columns[:10])}")
            print(f"   Date columns: {[col for col in news.columns if 'date' in col.lower() or 'time' in col.lower() or 'publish' in col.lower()]}")
            print(f"   Text columns: {[col for col in news.columns if col in ['title', 'summary', 'content', 'text', 'headline'] or 'sentiment' in col.lower()]}")
            print(f"   Sample title: {news['title'].iloc[0][:100] if 'title' in news.columns else 'N/A'}")
        
        # 3. Reddit Data
        reddit_path = "./feature_engineered_data/reddit_data/feature_engineered_outlier_treated_all_reddit_posts_combined.csv"
        if os.path.exists(reddit_path):
            print(f"\n💬 REDDIT DATA:")
            reddit = pd.read_csv(reddit_path, nrows=3)
            print(f"   Shape: {reddit.shape}")
            print(f"   First 10 columns: {list(reddit.columns[:10])}")
            print(f"   Date columns: {[col for col in reddit.columns if 'date' in col.lower() or 'time' in col.lower() or 'created' in col.lower()]}")
            print(f"   Text columns: {[col for col in reddit.columns if col in ['title', 'content', 'selftext', 'text', 'body']]}")
            print(f"   Sample title: {reddit['title'].iloc[0][:100] if 'title' in reddit.columns else 'N/A'}")
        
        return stocks if 'stocks' in locals() else None
    
    def create_integrated_dataset_final(self, tickers=['AAPL', 'GOOGL', 'TSLA']):
        """
        Create integrated dataset with proper data integration
        """
        print(f"\n🔗 CREATING FINAL INTEGRATED DATASET")
        print("="*60)
        
        # Load datasets
        print("Loading datasets...")
        stocks = pd.read_csv("./feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv")
        news = pd.read_csv("./feature_engineered_data/news_data/feature_engineered_outlier_treated_alpha_vantage_news_expanded.csv")
        reddit = pd.read_csv("./feature_engineered_data/reddit_data/feature_engineered_outlier_treated_all_reddit_posts_combined.csv")
        
        print(f"   Stocks: {stocks.shape}")
        print(f"   News: {news.shape}")
        print(f"   Reddit: {reddit.shape}")
        
        # Filter stocks for specified tickers
        print(f"\n📊 Filtering stocks for {len(tickers)} tickers...")
        stocks_filtered = stocks[stocks['ticker'].isin(tickers)].copy()
        
        # Prepare stock data
        print("\n📈 Preparing stock data...")
        stocks_processed = self._prepare_stock_data(stocks_filtered)
        
        # Process external data sources
        print("\n🔍 Processing external data sources...")
        
        # Process news data
        print("   📰 Processing news data...")
        news_processed = self._prepare_news_data(news, tickers)
        
        # Process reddit data
        print("   💬 Processing reddit data...")
        reddit_processed = self._prepare_reddit_data(reddit, tickers)
        
        # Merge all data
        print("\n🤝 Merging ALL data sources...")
        integrated_df = self._merge_all_data_sources(stocks_processed, news_processed, reddit_processed, tickers)
        
        if integrated_df is None or len(integrated_df) == 0:
            print("   ❌ Failed to create integrated dataset")
            return None
        
        # Save the dataset
        output_file = self._save_final_dataset(integrated_df, tickers)
        
        # Create detailed summary
        self._create_detailed_summary(integrated_df, tickers, stocks_processed, news_processed, reddit_processed)
        
        return integrated_df, output_file
    
    def _prepare_stock_data(self, stocks_df):
        """Prepare stock data with proper date handling"""
        df = stocks_df.copy()
        
        # Convert date
        if 'Date' in df.columns:
            df['date'] = pd.to_datetime(df['Date'])
            df['date_only'] = df['date'].dt.date
        else:
            print("   ⚠️ No Date column in stocks data")
            df['date'] = pd.to_datetime('2023-01-01')
            df['date_only'] = df['date'].dt.date
        
        return df
    
    def _prepare_news_data(self, news_df, tickers):
        """Prepare and process news data"""
        if news_df is None or len(news_df) == 0:
            print("     ⚠️ No news data available")
            return None
        
        df = news_df.copy()
        
        print(f"     Analyzing news data structure...")
        print(f"       Columns: {list(df.columns)[:15]}")
        
        # Find date column
        date_col = None
        for col in df.columns:
            if 'publish' in col.lower() or 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        
        if date_col:
            print(f"       Found date column: {date_col}")
            try:
                df['date'] = pd.to_datetime(df[date_col])
                df['date_only'] = df['date'].dt.date
            except:
                print(f"       ⚠️ Could not parse {date_col} as datetime")
                df['date_only'] = pd.to_datetime('2023-01-01').date()
        else:
            print(f"       ⚠️ No date column found")
            df['date_only'] = pd.to_datetime('2023-01-01').date()
        
        # Find text columns for ticker search
        text_cols = []
        for col in df.columns:
            if col in ['title', 'summary', 'content', 'text', 'headline'] or 'sentiment' in col.lower():
                text_cols.append(col)
        
        print(f"       Text columns for search: {text_cols[:5]}")
        
        # Process each ticker
        ticker_data = {}
        
        for ticker in tickers:
            print(f"       Searching for {ticker}...")
            
            # Create mask for ticker mentions
            mask = pd.Series([False] * len(df))
            
            for col in text_cols:
                if col in df.columns:
                    # Check if column contains strings
                    if df[col].dtype == 'object':
                        mask = mask | df[col].astype(str).str.upper().str.contains(ticker, na=False)
                    else:
                        # Try to convert to string
                        try:
                            mask = mask | df[col].astype(str).str.upper().str.contains(ticker, na=False)
                        except:
                            pass
            
            ticker_news = df[mask].copy()
            
            if len(ticker_news) > 0:
                print(f"         Found {len(ticker_news)} mentions")
                
                # Create daily aggregates
                daily_agg = self._create_daily_aggregates_comprehensive(ticker_news, 'news', ticker)
                if daily_agg is not None:
                    ticker_data[ticker] = daily_agg
            else:
                print(f"         No mentions found")
        
        return ticker_data if ticker_data else None
    
    def _prepare_reddit_data(self, reddit_df, tickers):
        """Prepare and process reddit data"""
        if reddit_df is None or len(reddit_df) == 0:
            print("     ⚠️ No reddit data available")
            return None
        
        df = reddit_df.copy()
        
        print(f"     Analyzing reddit data structure...")
        print(f"       Columns: {list(df.columns)[:15]}")
        
        # Find date column
        date_col = None
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower() or 'created' in col.lower():
                date_col = col
                break
        
        if date_col:
            print(f"       Found date column: {date_col}")
            try:
                df['date'] = pd.to_datetime(df[date_col])
                df['date_only'] = df['date'].dt.date
            except:
                print(f"       ⚠️ Could not parse {date_col} as datetime")
                df['date_only'] = pd.to_datetime('2023-01-01').date()
        else:
            print(f"       ⚠️ No date column found")
            df['date_only'] = pd.to_datetime('2023-01-01').date()
        
        # Find text columns for ticker search
        text_cols = []
        for col in df.columns:
            if col in ['title', 'content', 'selftext', 'text', 'body']:
                text_cols.append(col)
        
        print(f"       Text columns for search: {text_cols}")
        
        # Process each ticker
        ticker_data = {}
        
        for ticker in tickers:
            print(f"       Searching for {ticker}...")
            
            # Create mask for ticker mentions
            mask = pd.Series([False] * len(df))
            
            for col in text_cols:
                if col in df.columns:
                    # Check if column contains strings
                    if df[col].dtype == 'object':
                        mask = mask | df[col].astype(str).str.upper().str.contains(ticker, na=False)
                    else:
                        # Try to convert to string
                        try:
                            mask = mask | df[col].astype(str).str.upper().str.contains(ticker, na=False)
                        except:
                            pass
            
            ticker_reddit = df[mask].copy()
            
            if len(ticker_reddit) > 0:
                print(f"         Found {len(ticker_reddit)} mentions")
                
                # Create daily aggregates
                daily_agg = self._create_daily_aggregates_comprehensive(ticker_reddit, 'reddit', ticker)
                if daily_agg is not None:
                    ticker_data[ticker] = daily_agg
            else:
                print(f"         No mentions found")
        
        return ticker_data if ticker_data else None
    
    def _create_daily_aggregates_comprehensive(self, df, source_type, ticker):
        """Create comprehensive daily aggregates"""
        if df is None or len(df) == 0 or 'date_only' not in df.columns:
            return None
        
        df = df.copy()
        
        # Get numeric columns
        numeric_cols = []
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) and col != 'date_only':
                numeric_cols.append(col)
        
        # Create aggregation dictionary
        agg_dict = {}
        
        # Count of posts
        agg_dict['count'] = 'size'
        
        # Aggregate numeric columns if available
        if numeric_cols:
            for col in numeric_cols[:10]:  # Limit to first 10 numeric columns
                agg_dict[col] = ['mean', 'sum']
        
        # Group by date
        grouped = df.groupby('date_only')
        
        if agg_dict:
            try:
                daily_agg = grouped.agg(agg_dict)
                
                # Flatten multi-level columns
                daily_agg.columns = [f'{source_type}_{ticker}_{col[0]}_{col[1]}' if isinstance(col, tuple) 
                                   else f'{source_type}_{ticker}_{col}' for col in daily_agg.columns]
                
                daily_agg = daily_agg.reset_index()
                daily_agg = daily_agg.rename(columns={'date_only': 'date'})
                
                return daily_agg
            except Exception as e:
                print(f"         ⚠️ Error creating aggregates: {e}")
                # Simple count aggregation
                daily_counts = df.groupby('date_only').size().reset_index(name=f'{source_type}_{ticker}_count')
                daily_counts = daily_counts.rename(columns={'date_only': 'date'})
                return daily_counts
        
        # Simple count if no aggregation possible
        daily_counts = df.groupby('date_only').size().reset_index(name=f'{source_type}_{ticker}_count')
        daily_counts = daily_counts.rename(columns={'date_only': 'date'})
        
        return daily_counts
    
    def _merge_all_data_sources(self, stocks_df, news_data, reddit_data, tickers):
        """Merge all data sources together"""
        print(f"   Merging data for {len(tickers)} tickers...")
        
        integrated_data = []
        
        for ticker in tickers:
            print(f"     Processing {ticker}...")
            
            # Get stock data for this ticker
            ticker_stocks = stocks_df[stocks_df['ticker'] == ticker].copy()
            
            if len(ticker_stocks) == 0:
                print(f"       ⚠️ No stock data for {ticker}")
                continue
            
            # Ensure date_only column exists
            if 'date_only' not in ticker_stocks.columns and 'date' in ticker_stocks.columns:
                ticker_stocks['date_only'] = ticker_stocks['date'].dt.date
            
            # Merge with news if available
            if news_data is not None and ticker in news_data:
                news_df = news_data[ticker]
                # Ensure news has date column (not date_only)
                if 'date' not in news_df.columns and 'date_only' in news_df.columns:
                    news_df = news_df.rename(columns={'date_only': 'date'})
                
                ticker_stocks = pd.merge(
                    ticker_stocks,
                    news_df,
                    on='date',
                    how='left'
                )
                print(f"       ✓ Added {len(news_df.columns)-1} news features")
            
            # Merge with reddit if available
            if reddit_data is not None and ticker in reddit_data:
                reddit_df = reddit_data[ticker]
                # Ensure reddit has date column (not date_only)
                if 'date' not in reddit_df.columns and 'date_only' in reddit_df.columns:
                    reddit_df = reddit_df.rename(columns={'date_only': 'date'})
                
                ticker_stocks = pd.merge(
                    ticker_stocks,
                    reddit_df,
                    on='date',
                    how='left'
                )
                print(f"       ✓ Added {len(reddit_df.columns)-1} reddit features")
            
            # Remove temporary date_only column
            if 'date_only' in ticker_stocks.columns:
                ticker_stocks = ticker_stocks.drop('date_only', axis=1)
            
            integrated_data.append(ticker_stocks)
        
        if integrated_data:
            final_df = pd.concat(integrated_data, ignore_index=True)
            print(f"\n   ✅ Successfully merged data")
            print(f"   Total records: {len(final_df)}")
            print(f"   Total columns: {len(final_df.columns)}")
            return final_df
        else:
            print(f"   ❌ No data to merge")
            return None
    
    def _save_final_dataset(self, df, tickers):
        """Save the final integrated dataset"""
        output_dir = "./integrated_datasets_final"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"integrated_final_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        df.to_csv(filepath, index=False)
        
        print(f"\n💾 Saved final dataset to: {filepath}")
        
        return filepath
    
    def _create_detailed_summary(self, integrated_df, tickers, stocks_df, news_data, reddit_data):
        """Create detailed summary of the integration"""
        output_dir = "./integrated_datasets_final"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(output_dir, f"integration_summary_{timestamp}.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"FINAL INTEGRATED DATASET - DETAILED SUMMARY\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Tickers: {', '.join(tickers)}\n")
            f.write(f"Total rows: {len(integrated_df):,}\n")
            f.write(f"Total columns: {len(integrated_df.columns):,}\n\n")
            
            # Data source statistics
            f.write("DATA SOURCE STATISTICS:\n")
            f.write("="*60 + "\n")
            
            # Stock data
            f.write(f"\n📈 STOCK DATA:\n")
            f.write(f"   Total stock records: {len(stocks_df):,}\n")
            for ticker in tickers:
                ticker_count = len(stocks_df[stocks_df['ticker'] == ticker])
                f.write(f"   • {ticker}: {ticker_count:,} records\n")
            
            # News data
            f.write(f"\n📰 NEWS DATA:\n")
            if news_data:
                for ticker in tickers:
                    if ticker in news_data:
                        news_count = len(news_data[ticker])
                        f.write(f"   • {ticker}: {news_count:,} aggregated daily records\n")
                    else:
                        f.write(f"   • {ticker}: No news data found\n")
            else:
                f.write(f"   • No news data integrated\n")
            
            # Reddit data
            f.write(f"\n💬 REDDIT DATA:\n")
            if reddit_data:
                for ticker in tickers:
                    if ticker in reddit_data:
                        reddit_count = len(reddit_data[ticker])
                        f.write(f"   • {ticker}: {reddit_count:,} aggregated daily records\n")
                    else:
                        f.write(f"   • {ticker}: No reddit data found\n")
            else:
                f.write(f"   • No reddit data integrated\n")
            
            # Column analysis
            f.write(f"\n\nCOLUMN ANALYSIS:\n")
            f.write("="*60 + "\n")
            
            # Count columns by source
            stock_cols = [col for col in integrated_df.columns 
                         if not any(x in col for x in ['_news_', '_reddit_'])]
            news_cols = [col for col in integrated_df.columns if '_news_' in col]
            reddit_cols = [col for col in integrated_df.columns if '_reddit_' in col]
            
            f.write(f"\nColumn Count by Source:\n")
            f.write(f"   • Stock columns: {len(stock_cols):,}\n")
            f.write(f"   • News columns: {len(news_cols):,}\n")
            f.write(f"   • Reddit columns: {len(reddit_cols):,}\n")
            
            # Sample columns
            f.write(f"\nSample Stock Columns (first 20):\n")
            for col in stock_cols[:20]:
                f.write(f"   • {col}\n")
            
            if news_cols:
                f.write(f"\nSample News Columns (first 10):\n")
                for col in news_cols[:10]:
                    f.write(f"   • {col}\n")
            
            if reddit_cols:
                f.write(f"\nSample Reddit Columns (first 10):\n")
                for col in reddit_cols[:10]:
                    f.write(f"   • {col}\n")
            
            # Missing data analysis
            f.write(f"\n\nMISSING DATA ANALYSIS:\n")
            f.write("="*60 + "\n")
            
            # Calculate missing percentages for external data
            if news_cols:
                news_missing = integrated_df[news_cols].isnull().mean().mean() * 100
                f.write(f"\nNews data missing: {news_missing:.1f}%\n")
            
            if reddit_cols:
                reddit_missing = integrated_df[reddit_cols].isnull().mean().mean() * 100
                f.write(f"Reddit data missing: {reddit_missing:.1f}%\n")
            
            # Date range
            if 'date' in integrated_df.columns:
                f.write(f"\nDATE RANGE:\n")
                f.write(f"   Start: {integrated_df['date'].min()}\n")
                f.write(f"   End: {integrated_df['date'].max()}\n")
                f.write(f"   Days: {(integrated_df['date'].max() - integrated_df['date'].min()).days}\n")
        
        print(f"📄 Detailed summary saved to: {summary_file}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("FINAL FINANCIAL DATA INTEGRATOR - WITH PROPER DATA MERGING")
    print("="*80)
    
    integrator = FinancialDataIntegratorFinal()
    
    # First, inspect dataset structures in detail
    stocks_sample = integrator.inspect_datasets_detailed()
    
    print("\n" + "="*80)
    print("STARTING FINAL INTEGRATION PROCESS...")
    print("="*80)
    
    # Create integrated dataset with ALL tickers
    tickers_to_integrate = ['AAPL', 'GOOGL', 'TSLA']
    integrated_df, output_file = integrator.create_integrated_dataset_final(tickers=tickers_to_integrate)
    
    if integrated_df is not None:
        print("\n" + "="*80)
        print("✅ FINAL INTEGRATION COMPLETE!")
        print("="*80)
        
        print(f"\n📊 FINAL DATASET INFORMATION:")
        print(f"   File: {output_file}")
        print(f"   Total rows: {len(integrated_df):,}")
        print(f"   Total columns: {len(integrated_df.columns):,}")
        
        # Show column breakdown
        stock_cols = [col for col in integrated_df.columns 
                     if not any(x in col for x in ['_news_', '_reddit_'])]
        news_cols = [col for col in integrated_df.columns if '_news_' in col]
        reddit_cols = [col for col in integrated_df.columns if '_reddit_' in col]
        
        print(f"\n📈 COLUMN BREAKDOWN:")
        print(f"   • Stock columns: {len(stock_cols):,}")
        print(f"   • News columns: {len(news_cols):,}")
        print(f"   • Reddit columns: {len(reddit_cols):,}")
        
        # Show sample with external data
        print(f"\n👀 SAMPLE DATA WITH EXTERNAL FEATURES:")
        
        # Find a row with external data
        has_external = False
        sample_row = None
        
        for ticker in tickers_to_integrate:
            ticker_data = integrated_df[integrated_df['ticker'] == ticker]
            
            # Look for rows with news or reddit data
            if len(news_cols) > 0:
                rows_with_news = ticker_data[ticker_data[news_cols[0]].notna()]
                if len(rows_with_news) > 0:
                    sample_row = rows_with_news.iloc[0]
                    has_external = True
                    break
            
            if len(reddit_cols) > 0:
                rows_with_reddit = ticker_data[ticker_data[reddit_cols[0]].notna()]
                if len(rows_with_reddit) > 0:
                    sample_row = rows_with_reddit.iloc[0]
                    has_external = True
                    break
        
        if has_external and sample_row is not None:
            # Show basic info and external features
            print(f"   Ticker: {sample_row['ticker']}")
            print(f"   Date: {sample_row['date']}")
            
            if 'Open' in sample_row.index and 'Close' in sample_row.index:
                print(f"   Open: {sample_row['Open']:.2f}, Close: {sample_row['Close']:.2f}")
            
            # Show external features
            external_features = []
            for col in sample_row.index:
                if '_news_' in col or '_reddit_' in col:
                    if pd.notna(sample_row[col]):
                        external_features.append(col)
            
            if external_features:
                print(f"   External features present: {len(external_features)}")
                for feat in external_features[:3]:
                    print(f"     • {feat}: {sample_row[feat]}")
        else:
            print(f"   ⚠️ No external data found in sample")
        
        print(f"\n🎯 READY FOR PERFORMANCE ANALYSIS:")
        print(f"1. Dataset contains integrated stock + news + reddit data")
        print(f"2. Total features: {len(integrated_df.columns):,}")
        print(f"3. All tickers in single file for easy analysis")
    else:
        print("\n❌ Integration failed. Check the detailed inspection above.")
