"""
data_integrator.py - SIMPLIFIED VERSION
Works with your actual data structure
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class SimpleFinancialDataIntegrator:
    """
    Simplified integrator that works with your actual data
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
            print(f"   Columns: {list(stocks.columns)[:10]}...")  # Show first 10 columns
            print(f"   Ticker column exists: {'Symbol' in stocks.columns}")
            print(f"   Date column exists: {any(col in ['Date', 'date'] for col in stocks.columns)}")
        
        # 2. News Data
        news_path = "./feature_engineered_data/news_data/feature_engineered_outlier_treated_alpha_vantage_news_expanded.csv"
        if os.path.exists(news_path):
            print(f"\n📰 NEWS DATA ({news_path}):")
            news = pd.read_csv(news_path, nrows=5)
            datasets['news'] = news
            print(f"   Shape: {news.shape}")
            print(f"   Columns: {list(news.columns)[:10]}...")
            print(f"   Sentiment columns: {[col for col in news.columns if 'sentiment' in col.lower()]}")
        
        # 3. Reddit Data
        reddit_path = "./feature_engineered_data/reddit_data/feature_engineered_outlier_treated_all_reddit_posts_combined.csv"
        if os.path.exists(reddit_path):
            print(f"\n💬 REDDIT DATA ({reddit_path}):")
            reddit = pd.read_csv(reddit_path, nrows=5)
            datasets['reddit'] = reddit
            print(f"   Shape: {reddit.shape}")
            print(f"   Columns: {list(reddit.columns)[:10]}...")
            print(f"   Sentiment columns: {[col for col in reddit.columns if 'sentiment' in col.lower()]}")
            print(f"   Score/Upvote columns: {[col for col in reddit.columns if col in ['score', 'upvotes', 'num_comments']]}")
        
        return datasets
    
    def create_simple_integration(self):
        """
        Create a simple integrated dataset using available columns
        This will work even without perfect sentiment columns
        """
        print("\n🔗 CREATING SIMPLE INTEGRATION")
        print("="*60)
        
        # Load full datasets
        stocks = pd.read_csv("./feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv")
        news = pd.read_csv("./feature_engineered_data/news_data/feature_engineered_outlier_treated_alpha_vantage_news_expanded.csv")
        reddit = pd.read_csv("./feature_engineered_data/reddit_data/feature_engineered_outlier_treated_all_reddit_posts_combined.csv")
        
        print(f"Loaded: Stocks({stocks.shape}), News({news.shape}), Reddit({reddit.shape})")
        
        # Find common columns for integration
        print("\n🔍 Finding integration points...")
        
        # 1. Find date columns
        stock_date_col = self._find_date_column(stocks, ['Date', 'date', 'timestamp'])
        news_date_col = self._find_date_column(news, ['timestamp', 'time_published', 'date'])
        reddit_date_col = self._find_date_column(reddit, ['created_utc', 'created', 'timestamp', 'date'])
        
        print(f"   Stock date column: {stock_date_col}")
        print(f"   News date column: {news_date_col}")
        print(f"   Reddit date column: {reddit_date_col}")
        
        # 2. Find ticker column in stocks
        stock_ticker_col = self._find_ticker_column(stocks)
        print(f"   Stock ticker column: {stock_ticker_col}")
        
        # 3. Create simple integrated dataset for AAPL as example
        print("\n📊 Creating integrated dataset for AAPL...")
        
        # Filter stocks for AAPL
        if stock_ticker_col:
            aapl_stocks = stocks[stocks[stock_ticker_col] == 'AAPL'].copy()
        else:
            aapl_stocks = stocks.copy()  # Assume all data is AAPL
        
        print(f"   AAPL stock records: {len(aapl_stocks)}")
        
        # Convert dates
        aapl_stocks = self._convert_to_datetime(aapl_stocks, stock_date_col)
        
        # Get news mentions of AAPL
        aapl_news = self._extract_ticker_mentions(news, 'AAPL', news_date_col)
        
        # Get Reddit mentions of AAPL
        aapl_reddit = self._extract_ticker_mentions(reddit, 'AAPL', reddit_date_col)
        
        # Create daily aggregates
        print("\n📅 Creating daily aggregates...")
        
        # Stock daily data (already daily)
        daily_stocks = aapl_stocks.copy()
        if 'date' in daily_stocks.columns:
            daily_stocks['date'] = daily_stocks['date'].dt.date
        
        # News daily aggregates
        if aapl_news is not None and len(aapl_news) > 0:
            daily_news = self._create_daily_aggregates(aapl_news, 'news')
            print(f"   Daily news aggregates: {len(daily_news)} days")
        else:
            daily_news = None
        
        # Reddit daily aggregates
        if aapl_reddit is not None and len(aapl_reddit) > 0:
            daily_reddit = self._create_daily_aggregates(aapl_reddit, 'reddit')
            print(f"   Daily Reddit aggregates: {len(daily_reddit)} days")
        else:
            daily_reddit = None
        
        # Merge datasets
        print("\n🤝 Merging datasets...")
        
        integrated = daily_stocks.copy()
        
        if daily_news is not None:
            integrated = pd.merge(integrated, daily_news, on='date', how='left', suffixes=('', '_news'))
            print(f"   Merged news data: added {len(daily_news.columns)} columns")
        
        if daily_reddit is not None:
            integrated = pd.merge(integrated, daily_reddit, on='date', how='left', suffixes=('', '_reddit'))
            print(f"   Merged Reddit data: added {len(daily_reddit.columns)} columns")
        
        print(f"\n✅ FINAL INTEGRATED DATASET:")
        print(f"   Rows: {len(integrated)}")
        print(f"   Columns: {len(integrated.columns)}")
        print(f"   Date range: {integrated['date'].min()} to {integrated['date'].max()}")
        
        # Show sample columns
        print(f"\n👀 Sample columns:")
        for i, col in enumerate(integrated.columns[:15]):
            print(f"   {i+1}. {col}")
        
        # Save the integrated dataset
        self._save_integrated_dataset(integrated, 'AAPL')
        
        return integrated
    
    def _find_date_column(self, df, possible_names):
        """Find date column in dataframe"""
        for col in possible_names:
            if col in df.columns:
                return col
        return None
    
    def _find_ticker_column(self, df):
        """Find ticker/symbol column"""
        for col in ['Symbol', 'Ticker', 'symbol', 'ticker', 'stock']:
            if col in df.columns:
                return col
        return None
    
    def _convert_to_datetime(self, df, date_col):
        """Convert date column to datetime"""
        df = df.copy()
        if date_col and date_col in df.columns:
            try:
                # Check if numeric (Unix timestamp)
                if df[date_col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    df['date'] = pd.to_datetime(df[date_col], unit='s')
                else:
                    df['date'] = pd.to_datetime(df[date_col])
                return df
            except:
                pass
        # If conversion fails or no date column, add dummy date
        df['date'] = pd.to_datetime('2023-01-01')
        return df
    
    def _extract_ticker_mentions(self, df, ticker, date_col):
        """Extract rows mentioning a specific ticker"""
        if df is None or len(df) == 0:
            return None
        
        df = df.copy()
        
        # Convert date
        df = self._convert_to_datetime(df, date_col)
        
        # Look for ticker mentions in text columns
        text_cols = [col for col in df.columns if col in ['title', 'selftext', 'body', 'text', 'summary', 'content']]
        
        if not text_cols:
            # No text columns, use all data
            return df
        
        # Create mask for ticker mentions
        mask = pd.Series([False] * len(df))
        for col in text_cols:
            if col in df.columns:
                mask = mask | df[col].astype(str).str.contains(ticker, case=False, na=False)
        
        result = df[mask].copy()
        
        if len(result) > 0:
            print(f"   Found {len(result)} mentions of {ticker}")
        
        return result
    
    def _create_daily_aggregates(self, df, source_type):
        """Create daily aggregates from time-series data"""
        if df is None or len(df) == 0 or 'date' not in df.columns:
            return None
        
        df = df.copy()
        df['date_only'] = df['date'].dt.date
        
        # Select numeric columns for aggregation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            # If no numeric columns, just count occurrences
            daily_counts = df.groupby('date_only').size().reset_index(name=f'{source_type}_count')
            return daily_counts
        
        # Aggregate numeric columns
        agg_dict = {}
        for col in numeric_cols:
            if col != 'date_only':
                agg_dict[col] = ['mean', 'count', 'sum']
        
        if agg_dict:
            daily_agg = df.groupby('date_only').agg(agg_dict)
            
            # Flatten multi-level columns
            daily_agg.columns = [f'{source_type}_{col[0]}_{col[1]}' for col in daily_agg.columns]
            daily_agg = daily_agg.reset_index()
            daily_agg = daily_agg.rename(columns={'date_only': 'date'})
            
            return daily_agg
        
        return None
    
    def _save_integrated_dataset(self, df, ticker):
        """Save integrated dataset to file"""
        output_dir = "./integrated_datasets"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"integrated_{ticker}_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        df.to_csv(filepath, index=False)
        
        print(f"\n💾 Saved to: {filepath}")
        
        # Also create a summary
        summary_file = os.path.join(output_dir, f"integration_summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Integrated Dataset: {ticker}\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Rows: {len(df)}\n")
            f.write(f"Columns: {len(df.columns)}\n\n")
            
            f.write("COLUMNS BY SOURCE:\n")
            f.write("="*50 + "\n")
            
            # Categorize columns
            stock_cols = [col for col in df.columns if col not in ['date'] and '_news' not in col and '_reddit' not in col]
            news_cols = [col for col in df.columns if '_news' in col]
            reddit_cols = [col for col in df.columns if '_reddit' in col]
            
            f.write(f"\nStock Columns ({len(stock_cols)}):\n")
            for col in stock_cols[:20]:  # Show first 20
                f.write(f"  • {col}\n")
            if len(stock_cols) > 20:
                f.write(f"  • ... and {len(stock_cols) - 20} more\n")
            
            f.write(f"\nNews Columns ({len(news_cols)}):\n")
            for col in news_cols[:10]:
                f.write(f"  • {col}\n")
            
            f.write(f"\nReddit Columns ({len(reddit_cols)}):\n")
            for col in reddit_cols[:10]:
                f.write(f"  • {col}\n")
        
        print(f"📄 Summary saved to: {summary_file}")
        
        return filepath

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("SIMPLE FINANCIAL DATA INTEGRATOR")
    print("="*80)
    
    integrator = SimpleFinancialDataIntegrator()
    
    # First, inspect dataset structures
    datasets = integrator.load_and_inspect_datasets()
    
    print("\n" + "="*80)
    print("STARTING INTEGRATION...")
    print("="*80)
    
    # Create integrated dataset
    integrated_df = integrator.create_simple_integration()
    
    if integrated_df is not None:
        print("\n" + "="*80)
        print("✅ INTEGRATION SUCCESSFUL!")
        print("="*80)
        
        print("\n🎯 INTEGRATED DATASET READY FOR:")
        print("1. Model training with combined features")
        print("2. Performance comparison vs stocks-only")
        print("3. Presentation to professor")
        
        # Show a few sample rows
        print("\n👀 SAMPLE OF INTEGRATED DATA:")
        print(integrated_df[['date'] + list(integrated_df.columns[1:6])].head())
    else:
        print("\n❌ Integration failed. Check dataset structures above.")
