"""
data_integrator_fixed.py - Creates ONE integrated dataset with ALL tickers
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import glob

class SimpleFinancialDataIntegratorFixed:
    """
    Fixed integrator that creates ONE dataset with ALL tickers
    """
    
    def __init__(self, base_path="."):
        self.base_path = base_path
        
    def load_and_inspect_datasets(self):
        """Load and inspect datasets to understand their structure"""
        print("🔍 INSPECTING DATASET STRUCTURES")
        print("="*60)
        
        datasets = {}
        
        # 1. Stock Data
        stock_path = "./feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv"
        if os.path.exists(stock_path):
            print(f"\n📈 STOCK DATA ({stock_path}):")
            stocks = pd.read_csv(stock_path, nrows=5)  # Just load first few rows
            datasets['stocks'] = stocks
            print(f"   Shape: {stocks.shape}")
            print(f"   Columns: {list(stocks.columns)[:10]}...")
            print(f"   Ticker column exists: {'ticker' in stocks.columns}")
            print(f"   Unique tickers in sample: {stocks['ticker'].unique() if 'ticker' in stocks.columns else 'N/A'}")
        
        # 2. News Data
        news_path = "./feature_engineered_data/news_data/feature_engineered_outlier_treated_alpha_vantage_news_expanded.csv"
        if os.path.exists(news_path):
            print(f"\n📰 NEWS DATA ({news_path}):")
            news = pd.read_csv(news_path, nrows=5)
            datasets['news'] = news
            print(f"   Shape: {news.shape}")
            print(f"   Columns: {list(news.columns)[:10]}...")
        
        # 3. Reddit Data
        reddit_path = "./feature_engineered_data/reddit_data/feature_engineered_outlier_treated_all_reddit_posts_combined.csv"
        if os.path.exists(reddit_path):
            print(f"\n💬 REDDIT DATA ({reddit_path}):")
            reddit = pd.read_csv(reddit_path, nrows=5)
            datasets['reddit'] = reddit
            print(f"   Shape: {reddit.shape}")
            print(f"   Columns: {list(reddit.columns)[:10]}...")
        
        return datasets
    
    def create_integrated_dataset_all_tickers(self, tickers=['AAPL', 'GOOGL', 'TSLA']):
        """
        Create ONE integrated dataset containing ALL specified tickers
        """
        print(f"\n🔗 CREATING INTEGRATED DATASET FOR ALL TICKERS: {tickers}")
        print("="*60)
        
        # Load full datasets
        print("Loading datasets...")
        stocks = pd.read_csv("./feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv")
        news = pd.read_csv("./feature_engineered_data/news_data/feature_engineered_outlier_treated_alpha_vantage_news_expanded.csv")
        reddit = pd.read_csv("./feature_engineered_data/reddit_data/feature_engineered_outlier_treated_all_reddit_posts_combined.csv")
        
        print(f"Loaded: Stocks({stocks.shape}), News({news.shape}), Reddit({reddit.shape})")
        
        # Filter stocks for specified tickers
        print(f"\nFiltering stocks for {len(tickers)} tickers...")
        if 'ticker' in stocks.columns:
            stocks = stocks[stocks['ticker'].isin(tickers)].copy()
            print(f"   Filtered stocks: {stocks.shape}")
            print(f"   Ticker distribution:")
            for ticker in tickers:
                count = len(stocks[stocks['ticker'] == ticker])
                print(f"     • {ticker}: {count} records")
        else:
            print("   ⚠️ No ticker column in stocks data")
            return None
        
        # Prepare stock data
        print("\nPreparing stock data...")
        stocks = self._prepare_stock_data(stocks)
        
        # Process news data for each ticker
        print("\nProcessing news data...")
        news_aggregates = self._process_news_data(news, tickers)
        
        # Process reddit data for each ticker
        print("\nProcessing reddit data...")
        reddit_aggregates = self._process_reddit_data(reddit, tickers)
        
        # Merge all data
        print("\n🤝 Merging ALL datasets...")
        
        integrated_data = []
        
        for ticker in tickers:
            print(f"   Processing {ticker}...")
            
            # Get stock data for this ticker
            ticker_stocks = stocks[stocks['ticker'] == ticker].copy()
            
            if len(ticker_stocks) == 0:
                print(f"     ⚠️ No stock data for {ticker}")
                continue
            
            # Add date column for merging
            ticker_stocks['date_only'] = ticker_stocks['date'].dt.date
            
            # Merge with news if available
            if news_aggregates is not None and ticker in news_aggregates:
                ticker_stocks = pd.merge(
                    ticker_stocks, 
                    news_aggregates[ticker], 
                    on='date_only', 
                    how='left',
                    suffixes=('', '_news')
                )
                print(f"     ✓ Merged news data")
            
            # Merge with reddit if available
            if reddit_aggregates is not None and ticker in reddit_aggregates:
                ticker_stocks = pd.merge(
                    ticker_stocks, 
                    reddit_aggregates[ticker], 
                    on='date_only', 
                    how='left',
                    suffixes=('', '_reddit')
                )
                print(f"     ✓ Merged reddit data")
            
            integrated_data.append(ticker_stocks)
        
        # Combine all tickers
        if integrated_data:
            final_integrated = pd.concat(integrated_data, ignore_index=True)
        else:
            print("   ❌ No data to integrate")
            return None
        
        # Clean up
        if 'date_only' in final_integrated.columns:
            final_integrated = final_integrated.drop('date_only', axis=1)
        
        print(f"\n✅ FINAL INTEGRATED DATASET (ALL TICKERS):")
        print(f"   Rows: {len(final_integrated)}")
        print(f"   Columns: {len(final_integrated.columns)}")
        print(f"   Date range: {final_integrated['date'].min()} to {final_integrated['date'].max()}")
        
        # Ticker distribution
        print(f"\n📊 Ticker distribution in integrated data:")
        for ticker in tickers:
            count = len(final_integrated[final_integrated['ticker'] == ticker])
            print(f"   • {ticker}: {count} records")
        
        # Save the integrated dataset
        output_file = self._save_integrated_dataset_all(final_integrated, tickers)
        
        return final_integrated, output_file
    
    def _prepare_stock_data(self, stocks_df):
        """Prepare stock data for integration"""
        df = stocks_df.copy()
        
        # Find date column
        date_col = self._find_date_column(df, ['Date', 'date', 'timestamp'])
        if date_col:
            df['date'] = pd.to_datetime(df[date_col])
        
        return df
    
    def _process_news_data(self, news_df, tickers):
        """Process news data for multiple tickers"""
        if news_df is None or len(news_df) == 0:
            return None
        
        print(f"   Processing news mentions for {len(tickers)} tickers...")
        
        # Find date column
        date_col = self._find_date_column(news_df, ['timestamp', 'time_published', 'date', 'post_date'])
        
        # Convert date
        news_df_processed = news_df.copy()
        if date_col:
            news_df_processed['date'] = pd.to_datetime(news_df_processed[date_col])
            news_df_processed['date_only'] = news_df_processed['date'].dt.date
        else:
            print("     ⚠️ No date column found in news data")
            return None
        
        # Look for text columns
        text_cols = [col for col in news_df_processed.columns if col in ['title', 'summary', 'content', 'text', 'headline']]
        
        if not text_cols:
            print("     ⚠️ No text columns found in news data")
            return None
        
        # Process each ticker
        news_aggregates = {}
        
        for ticker in tickers:
            print(f"     Looking for {ticker} mentions...")
            
            # Create mask for ticker mentions
            mask = pd.Series([False] * len(news_df_processed))
            for col in text_cols:
                if col in news_df_processed.columns:
                    mask = mask | news_df_processed[col].astype(str).str.contains(ticker, case=False, na=False)
            
            ticker_news = news_df_processed[mask].copy()
            
            if len(ticker_news) > 0:
                print(f"       Found {len(ticker_news)} mentions")
                
                # Create daily aggregates
                daily_agg = self._create_daily_aggregates_enhanced(ticker_news, f'news_{ticker}')
                if daily_agg is not None:
                    news_aggregates[ticker] = daily_agg
            else:
                print(f"       No mentions found")
        
        return news_aggregates if news_aggregates else None
    
    def _process_reddit_data(self, reddit_df, tickers):
        """Process reddit data for multiple tickers"""
        if reddit_df is None or len(reddit_df) == 0:
            return None
        
        print(f"   Processing reddit mentions for {len(tickers)} tickers...")
        
        # Find date column
        date_col = self._find_date_column(reddit_df, ['created_utc', 'created', 'timestamp', 'date', 'post_date', 'post_datetime'])
        
        # Convert date
        reddit_df_processed = reddit_df.copy()
        if date_col:
            reddit_df_processed['date'] = pd.to_datetime(reddit_df_processed[date_col])
            reddit_df_processed['date_only'] = reddit_df_processed['date'].dt.date
        else:
            print("     ⚠️ No date column found in reddit data")
            return None
        
        # Look for text columns
        text_cols = [col for col in reddit_df_processed.columns if col in ['title', 'selftext', 'content', 'text', 'body']]
        
        if not text_cols:
            print("     ⚠️ No text columns found in reddit data")
            return None
        
        # Process each ticker
        reddit_aggregates = {}
        
        for ticker in tickers:
            print(f"     Looking for {ticker} mentions...")
            
            # Create mask for ticker mentions
            mask = pd.Series([False] * len(reddit_df_processed))
            for col in text_cols:
                if col in reddit_df_processed.columns:
                    mask = mask | reddit_df_processed[col].astype(str).str.contains(ticker, case=False, na=False)
            
            ticker_reddit = reddit_df_processed[mask].copy()
            
            if len(ticker_reddit) > 0:
                print(f"       Found {len(ticker_reddit)} mentions")
                
                # Create daily aggregates
                daily_agg = self._create_daily_aggregates_enhanced(ticker_reddit, f'reddit_{ticker}')
                if daily_agg is not None:
                    reddit_aggregates[ticker] = daily_agg
            else:
                print(f"       No mentions found")
        
        return reddit_aggregates if reddit_aggregates else None
    
    def _create_daily_aggregates_enhanced(self, df, source_name):
        """Create enhanced daily aggregates"""
        if df is None or len(df) == 0 or 'date_only' not in df.columns:
            return None
        
        df = df.copy()
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create aggregation dictionary
        agg_dict = {}
        
        # Count of posts/comments
        agg_dict['count'] = 'size'
        
        # Aggregate numeric columns
        for col in numeric_cols:
            if col != 'date_only':
                agg_dict[col] = ['mean', 'sum', 'std']
        
        # Group by date
        grouped = df.groupby('date_only')
        
        if agg_dict:
            daily_agg = grouped.agg(agg_dict)
            
            # Flatten multi-level columns
            daily_agg.columns = [f'{source_name}_{col[0]}_{col[1]}' if isinstance(col, tuple) and len(col) == 2 
                               else f'{source_name}_{col}' for col in daily_agg.columns]
            
            daily_agg = daily_agg.reset_index()
            daily_agg = daily_agg.rename(columns={'date_only': 'date'})
            
            return daily_agg
        
        return None
    
    def _find_date_column(self, df, possible_names):
        """Find date column in dataframe"""
        for col in possible_names:
            if col in df.columns:
                return col
        return None
    
    def _save_integrated_dataset_all(self, df, tickers):
        """Save integrated dataset to file"""
        output_dir = "./integrated_datasets"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"integrated_ALL_TICKERS_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        df.to_csv(filepath, index=False)
        
        print(f"\n💾 Saved integrated dataset to: {filepath}")
        
        # Create summary
        self._create_integration_summary(df, tickers, timestamp)
        
        return filepath
    
    def _create_integration_summary(self, df, tickers, timestamp):
        """Create detailed integration summary"""
        output_dir = "./integrated_datasets"
        summary_file = os.path.join(output_dir, f"integration_summary_ALL_{timestamp}.txt")
        
        with open(summary_file, 'w') as f:
            f.write(f"INTEGRATED DATASET - ALL TICKERS\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Tickers included: {', '.join(tickers)}\n")
            f.write(f"Total rows: {len(df):,}\n")
            f.write(f"Total columns: {len(df.columns):,}\n\n")
            
            # Ticker statistics
            f.write("TICKER STATISTICS:\n")
            f.write("="*50 + "\n")
            for ticker in tickers:
                ticker_data = df[df['ticker'] == ticker]
                f.write(f"\n{ticker}:\n")
                f.write(f"  • Records: {len(ticker_data):,}\n")
                f.write(f"  • Date range: {ticker_data['date'].min()} to {ticker_data['date'].max()}\n")
                
                # Count features by source
                stock_cols = [col for col in ticker_data.columns if not col.endswith('_news') and not col.endswith('_reddit') 
                            and col not in ['date', 'ticker']]
                news_cols = [col for col in ticker_data.columns if col.endswith('_news')]
                reddit_cols = [col for col in ticker_data.columns if col.endswith('_reddit')]
                
                f.write(f"  • Stock features: {len(stock_cols):,}\n")
                f.write(f"  • News features: {len(news_cols):,}\n")
                f.write(f"  • Reddit features: {len(reddit_cols):,}\n")
            
            # Column breakdown
            f.write("\n\nCOLUMN BREAKDOWN:\n")
            f.write("="*50 + "\n")
            
            # Categorize columns
            stock_cols = [col for col in df.columns if not col.endswith('_news') and not col.endswith('_reddit') 
                         and col not in ['date', 'ticker']]
            news_cols = [col for col in df.columns if '_news_' in col]
            reddit_cols = [col for col in df.columns if '_reddit_' in col]
            
            f.write(f"\nStock Columns ({len(stock_cols)}):\n")
            for col in stock_cols[:30]:
                f.write(f"  • {col}\n")
            if len(stock_cols) > 30:
                f.write(f"  • ... and {len(stock_cols) - 30} more\n")
            
            f.write(f"\nNews Columns ({len(news_cols)}):\n")
            for col in news_cols[:15]:
                f.write(f"  • {col}\n")
            
            f.write(f"\nReddit Columns ({len(reddit_cols)}):\n")
            for col in reddit_cols[:15]:
                f.write(f"  • {col}\n")
            
            # Missing data analysis
            f.write("\n\nMISSING DATA ANALYSIS:\n")
            f.write("="*50 + "\n")
            
            missing_stats = df.isnull().sum()
            missing_stats = missing_stats[missing_stats > 0]
            
            if len(missing_stats) > 0:
                f.write(f"\nColumns with missing values:\n")
                for col, count in missing_stats.head(20).items():
                    percentage = (count / len(df)) * 100
                    f.write(f"  • {col}: {count:,} ({percentage:.1f}%)\n")
            else:
                f.write("\nNo missing values found!\n")
        
        print(f"📄 Summary saved to: {summary_file}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("FIXED FINANCIAL DATA INTEGRATOR - ALL TICKERS")
    print("="*80)
    
    integrator = SimpleFinancialDataIntegratorFixed()
    
    # First, inspect dataset structures
    datasets = integrator.load_and_inspect_datasets()
    
    print("\n" + "="*80)
    print("STARTING INTEGRATION FOR ALL TICKERS...")
    print("="*80)
    
    # Create integrated dataset with ALL tickers
    tickers_to_integrate = ['AAPL', 'GOOGL', 'TSLA']
    integrated_df, output_file = integrator.create_integrated_dataset_all_tickers(tickers=tickers_to_integrate)
    
    if integrated_df is not None:
        print("\n" + "="*80)
        print("✅ INTEGRATION SUCCESSFUL!")
        print("="*80)
        
        print(f"\n📊 FINAL DATASET INFO:")
        print(f"   File: {output_file}")
        print(f"   Total rows: {len(integrated_df):,}")
        print(f"   Total columns: {len(integrated_df.columns):,}")
        print(f"   Tickers included: {tickers_to_integrate}")
        
        # Show sample
        print(f"\n👀 SAMPLE DATA (first ticker):")
        first_ticker = tickers_to_integrate[0]
        sample = integrated_df[integrated_df['ticker'] == first_ticker].head(3)
        print(sample[['date', 'ticker', 'Open', 'Close']])
        
        print("\n🎯 READY FOR PERFORMANCE COMPARISON:")
        print("1. Single dataset contains ALL tickers")
        print("2. Each ticker has stock + news + reddit data")
        print("3. Can run comparison analysis on entire dataset")
    else:
        print("\n❌ Integration failed. Check dataset structures above.")
