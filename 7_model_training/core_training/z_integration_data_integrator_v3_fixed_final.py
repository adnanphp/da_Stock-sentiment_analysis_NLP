"""
data_integrator_fixed_final_robust.py - Robust integrator with error handling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import re

class FinancialDataIntegratorRobust:
    """
    Robust integrator with proper error handling
    """
    
    def __init__(self, base_path="."):
        self.base_path = base_path
        
    def inspect_datasets_detailed_safe(self):
        """Safe detailed inspection of dataset structures"""
        print("🔍 DETAILED DATASET INSPECTION")
        print("="*60)
        
        # 1. Stock Data
        stock_path = "./feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv"
        if os.path.exists(stock_path):
            print(f"\n📈 STOCK DATA:")
            try:
                stocks = pd.read_csv(stock_path, nrows=3)
                print(f"   Shape: {stocks.shape}")
                print(f"   First 10 columns: {list(stocks.columns[:10])}")
                print(f"   Date columns: {[col for col in stocks.columns if 'date' in col.lower() or 'Date' in col][:5]}")
                print(f"   Ticker columns: {[col for col in stocks.columns if 'ticker' in col.lower() or 'symbol' in col.lower()]}")
            except Exception as e:
                print(f"   Error loading stock data: {e}")
        
        # 2. News Data
        news_path = "./feature_engineered_data/news_data/feature_engineered_outlier_treated_alpha_vantage_news_expanded.csv"
        if os.path.exists(news_path):
            print(f"\n📰 NEWS DATA:")
            try:
                news = pd.read_csv(news_path, nrows=3)
                print(f"   Shape: {news.shape}")
                print(f"   First 10 columns: {list(news.columns[:10])}")
                
                # Find date columns
                date_cols = []
                for col in news.columns:
                    col_lower = str(col).lower()
                    if 'date' in col_lower or 'time' in col_lower or 'publish' in col_lower:
                        date_cols.append(col)
                print(f"   Date columns: {date_cols[:5]}")
                
                # Find text columns
                text_cols = []
                for col in news.columns:
                    if col in ['title', 'summary', 'content', 'text', 'headline'] or 'sentiment' in str(col).lower():
                        text_cols.append(col)
                print(f"   Text columns: {text_cols[:5]}")
                
                # Sample title safely
                if 'title' in news.columns and len(news) > 0:
                    title_val = news['title'].iloc[0]
                    if pd.notna(title_val) and isinstance(title_val, str):
                        print(f"   Sample title: {title_val[:100]}...")
                    else:
                        print(f"   Sample title: N/A (null or not string)")
            except Exception as e:
                print(f"   Error loading news data: {e}")
        
        # 3. Reddit Data
        reddit_path = "./feature_engineered_data/reddit_data/feature_engineered_outlier_treated_all_reddit_posts_combined.csv"
        if os.path.exists(reddit_path):
            print(f"\n💬 REDDIT DATA:")
            try:
                reddit = pd.read_csv(reddit_path, nrows=3)
                print(f"   Shape: {reddit.shape}")
                print(f"   First 10 columns: {list(reddit.columns[:10])}")
                
                # Find date columns
                date_cols = []
                for col in reddit.columns:
                    col_lower = str(col).lower()
                    if 'date' in col_lower or 'time' in col_lower or 'created' in col_lower:
                        date_cols.append(col)
                print(f"   Date columns: {date_cols[:5]}")
                
                # Find text columns
                text_cols = []
                for col in reddit.columns:
                    if col in ['title', 'content', 'selftext', 'text', 'body']:
                        text_cols.append(col)
                print(f"   Text columns: {text_cols[:5]}")
                
                # Sample title safely
                if 'title' in reddit.columns and len(reddit) > 0:
                    title_val = reddit['title'].iloc[0]
                    if pd.notna(title_val) and isinstance(title_val, str):
                        print(f"   Sample title: {title_val[:100]}...")
                    else:
                        print(f"   Sample title: N/A (null or not string)")
            except Exception as e:
                print(f"   Error loading reddit data: {e}")
    
    def create_integrated_dataset_robust(self, tickers=['AAPL', 'GOOGL', 'TSLA']):
        """
        Create integrated dataset with robust error handling
        """
        print(f"\n🔗 CREATING ROBUST INTEGRATED DATASET")
        print("="*60)
        
        try:
            # Load datasets with error handling
            print("Loading datasets...")
            
            stocks = self._load_dataset_safe(
                "./feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv",
                "stocks"
            )
            
            news = self._load_dataset_safe(
                "./feature_engineered_data/news_data/feature_engineered_outlier_treated_alpha_vantage_news_expanded.csv",
                "news"
            )
            
            reddit = self._load_dataset_safe(
                "./feature_engineered_data/reddit_data/feature_engineered_outlier_treated_all_reddit_posts_combined.csv",
                "reddit"
            )
            
            if stocks is None:
                print("❌ Stock data is required but not available")
                return None, None
            
            print(f"   Stocks: {stocks.shape if stocks is not None else 'N/A'}")
            print(f"   News: {news.shape if news is not None else 'N/A'}")
            print(f"   Reddit: {reddit.shape if reddit is not None else 'N/A'}")
            
            # Filter stocks for specified tickers
            print(f"\n📊 Filtering stocks for {len(tickers)} tickers...")
            if 'ticker' in stocks.columns:
                stocks_filtered = stocks[stocks['ticker'].isin(tickers)].copy()
                print(f"   Filtered to {len(stocks_filtered)} records")
                
                # Show ticker distribution
                for ticker in tickers:
                    count = len(stocks_filtered[stocks_filtered['ticker'] == ticker])
                    print(f"     • {ticker}: {count} records")
            else:
                print("   ⚠️ No ticker column in stocks data")
                stocks_filtered = stocks.copy()
            
            # Prepare stock data
            print("\n📈 Preparing stock data...")
            stocks_processed = self._prepare_stock_data_robust(stocks_filtered)
            
            # Process external data sources
            print("\n🔍 Processing external data sources...")
            
            # Process news data
            news_processed = None
            if news is not None and len(news) > 0:
                print("   📰 Processing news data...")
                news_processed = self._process_external_data_robust(news, tickers, 'news')
            else:
                print("   📰 Skipping news data (not available)")
            
            # Process reddit data
            reddit_processed = None
            if reddit is not None and len(reddit) > 0:
                print("   💬 Processing reddit data...")
                reddit_processed = self._process_external_data_robust(reddit, tickers, 'reddit')
            else:
                print("   💬 Skipping reddit data (not available)")
            
            # Merge all data
            print("\n🤝 Merging ALL data sources...")
            integrated_df = self._merge_data_sources_robust(stocks_processed, news_processed, reddit_processed, tickers)
            
            if integrated_df is None or len(integrated_df) == 0:
                print("   ⚠️ Failed to create integrated dataset")
                return None, None
            
            # Save the dataset
            output_file = self._save_dataset_robust(integrated_df, tickers)
            
            # Create summary
            self._create_integration_summary_robust(integrated_df, tickers, output_file)
            
            return integrated_df, output_file
            
        except Exception as e:
            print(f"\n❌ Error in integration process: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _load_dataset_safe(self, filepath, dataset_name):
        """Load dataset safely with error handling"""
        if not os.path.exists(filepath):
            print(f"   ⚠️ {dataset_name} file not found: {filepath}")
            return None
        
        try:
            print(f"   Loading {dataset_name}...")
            df = pd.read_csv(filepath)
            print(f"     Success: {df.shape}")
            return df
        except Exception as e:
            print(f"   ❌ Error loading {dataset_name}: {e}")
            return None
    
    def _prepare_stock_data_robust(self, stocks_df):
        """Prepare stock data robustly"""
        df = stocks_df.copy()
        
        # Convert date - try different column names
        date_converted = False
        date_columns_to_try = ['Date', 'date', 'timestamp', 'time', 'datetime']
        
        for date_col in date_columns_to_try:
            if date_col in df.columns:
                try:
                    df['date'] = pd.to_datetime(df[date_col])
                    df['date_only'] = df['date'].dt.date
                    print(f"   ✓ Using date column: {date_col}")
                    date_converted = True
                    break
                except:
                    continue
        
        if not date_converted:
            print("   ⚠️ Could not find or parse date column, using default")
            df['date'] = pd.to_datetime('2023-01-01')
            df['date_only'] = df['date'].dt.date
        
        return df
    
    def _process_external_data_robust(self, df, tickers, source_type):
        """Process external data (news/reddit) robustly"""
        df_processed = df.copy()
        
        # Convert date
        date_converted = False
        date_columns_to_try = ['published_at', 'timestamp', 'time_published', 'date', 
                              'post_date', 'post_datetime', 'created_utc', 'created']
        
        for date_col in date_columns_to_try:
            if date_col in df_processed.columns:
                try:
                    df_processed['date'] = pd.to_datetime(df_processed[date_col])
                    df_processed['date_only'] = df_processed['date'].dt.date
                    print(f"     ✓ Using date column: {date_col}")
                    date_converted = True
                    break
                except:
                    continue
        
        if not date_converted:
            print(f"     ⚠️ No valid date column found in {source_type} data")
            return None
        
        # Find text columns for ticker search
        text_cols = []
        common_text_cols = ['title', 'summary', 'content', 'text', 'headline', 'selftext', 'body']
        
        for col in common_text_cols:
            if col in df_processed.columns:
                text_cols.append(col)
        
        if not text_cols:
            # Try to find any string columns
            for col in df_processed.columns:
                if df_processed[col].dtype == 'object':
                    sample_val = df_processed[col].iloc[0] if len(df_processed) > 0 else None
                    if isinstance(sample_val, str) and len(sample_val) > 10:
                        text_cols.append(col)
                        if len(text_cols) >= 3:  # Limit to 3 columns
                            break
        
        print(f"     Text columns for search: {text_cols[:3]}")
        
        # Process each ticker
        ticker_data = {}
        
        for ticker in tickers:
            print(f"     Searching for {ticker}...")
            
            # Create mask for ticker mentions
            mask = pd.Series([False] * len(df_processed))
            mentions_found = 0
            
            for col in text_cols:
                if col in df_processed.columns:
                    try:
                        # Convert to string and search
                        col_data = df_processed[col].astype(str).fillna('')
                        col_mask = col_data.str.upper().str.contains(ticker, na=False)
                        mask = mask | col_mask
                        mentions_found += col_mask.sum()
                    except:
                        pass
            
            if mentions_found > 0:
                print(f"       Found {mentions_found} potential mentions")
                ticker_external = df_processed[mask].copy()
                
                # Create daily aggregates
                daily_agg = self._create_daily_aggregates_simple(ticker_external, source_type, ticker)
                if daily_agg is not None and len(daily_agg) > 0:
                    ticker_data[ticker] = daily_agg
                    print(f"       Created {len(daily_agg)} daily aggregates")
                else:
                    print(f"       Could not create aggregates")
            else:
                print(f"       No mentions found")
        
        return ticker_data if ticker_data else None
    
    def _create_daily_aggregates_simple(self, df, source_type, ticker):
        """Create simple daily aggregates"""
        if df is None or len(df) == 0 or 'date_only' not in df.columns:
            return None
        
        try:
            # Simple count aggregation
            daily_counts = df.groupby('date_only').size().reset_index(name=f'{source_type}_{ticker}_count')
            
            # Try to add some numeric aggregates if available
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                # Take first few numeric columns
                for col in numeric_cols[:3]:  # Limit to 3 columns
                    if col != 'date_only':
                        # Calculate mean for this column
                        col_mean = df.groupby('date_only')[col].mean().reset_index(name=f'{source_type}_{ticker}_{col}_mean')
                        daily_counts = pd.merge(daily_counts, col_mean, on='date_only', how='left')
            
            daily_counts = daily_counts.rename(columns={'date_only': 'date'})
            return daily_counts
            
        except Exception as e:
            print(f"       Error creating aggregates: {e}")
            # Return just counts
            try:
                daily_counts = df.groupby('date_only').size().reset_index(name=f'{source_type}_{ticker}_count')
                daily_counts = daily_counts.rename(columns={'date_only': 'date'})
                return daily_counts
            except:
                return None
    
    def _merge_data_sources_robust(self, stocks_df, news_data, reddit_data, tickers):
        """Merge data sources robustly"""
        integrated_records = []
        
        for ticker in tickers:
            print(f"     Processing {ticker}...")
            
            # Get stock data for this ticker
            ticker_stocks = stocks_df[stocks_df['ticker'] == ticker].copy()
            
            if len(ticker_stocks) == 0:
                print(f"       ⚠️ No stock data for {ticker}")
                continue
            
            # Ensure we have date column
            if 'date' not in ticker_stocks.columns and 'date_only' in ticker_stocks.columns:
                ticker_stocks = ticker_stocks.rename(columns={'date_only': 'date'})
            
            # Merge with news data
            if news_data is not None and ticker in news_data:
                news_df = news_data[ticker]
                # Ensure consistent date format
                if 'date' in news_df.columns:
                    news_df['date'] = pd.to_datetime(news_df['date'])
                    ticker_stocks['date'] = pd.to_datetime(ticker_stocks['date'])
                    
                    ticker_stocks = pd.merge(
                        ticker_stocks,
                        news_df,
                        on='date',
                        how='left'
                    )
                    print(f"       ✓ Added news features")
            
            # Merge with reddit data
            if reddit_data is not None and ticker in reddit_data:
                reddit_df = reddit_data[ticker]
                # Ensure consistent date format
                if 'date' in reddit_df.columns:
                    reddit_df['date'] = pd.to_datetime(reddit_df['date'])
                    if 'date' not in ticker_stocks.columns:
                        ticker_stocks['date'] = pd.to_datetime(ticker_stocks.index)
                    
                    ticker_stocks = pd.merge(
                        ticker_stocks,
                        reddit_df,
                        on='date',
                        how='left'
                    )
                    print(f"       ✓ Added reddit features")
            
            integrated_records.append(ticker_stocks)
        
        if integrated_records:
            try:
                final_df = pd.concat(integrated_records, ignore_index=True)
                print(f"\n   ✅ Successfully merged {len(integrated_records)} tickers")
                print(f"   Total records: {len(final_df)}")
                print(f"   Total columns: {len(final_df.columns)}")
                return final_df
            except Exception as e:
                print(f"   ❌ Error concatenating data: {e}")
                return None
        else:
            print(f"   ❌ No data to merge")
            return None
    
    def _save_dataset_robust(self, df, tickers):
        """Save dataset robustly"""
        output_dir = "./integrated_datasets_final"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"integrated_robust_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        try:
            df.to_csv(filepath, index=False)
            print(f"\n💾 Saved dataset to: {filepath}")
            return filepath
        except Exception as e:
            print(f"   ❌ Error saving dataset: {e}")
            return None
    
    def _create_integration_summary_robust(self, df, tickers, output_file):
        """Create integration summary"""
        if df is None:
            return
        
        output_dir = "./integrated_datasets_final"
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = os.path.join(output_dir, f"integration_summary_{timestamp}.txt")
        
        try:
            with open(summary_file, 'w') as f:
                f.write(f"ROBUST INTEGRATED DATASET\n")
                f.write(f"Generated: {datetime.now()}\n")
                f.write(f"Tickers: {', '.join(tickers)}\n")
                f.write(f"File: {output_file}\n")
                f.write(f"Total rows: {len(df):,}\n")
                f.write(f"Total columns: {len(df.columns):,}\n\n")
                
                # Column analysis
                f.write("COLUMN ANALYSIS:\n")
                f.write("="*60 + "\n")
                
                # Count by source
                stock_cols = [col for col in df.columns 
                             if not any(x in col for x in ['_news_', '_reddit_'])]
                news_cols = [col for col in df.columns if '_news_' in col]
                reddit_cols = [col for col in df.columns if '_reddit_' in col]
                
                f.write(f"\nStock columns: {len(stock_cols):,}\n")
                f.write(f"News columns: {len(news_cols):,}\n")
                f.write(f"Reddit columns: {len(reddit_cols):,}\n")
                
                # Sample data
                f.write(f"\n\nSAMPLE DATA:\n")
                f.write("="*60 + "\n")
                
                for ticker in tickers:
                    ticker_data = df[df['ticker'] == ticker]
                    if len(ticker_data) > 0:
                        f.write(f"\n{ticker} (first record):\n")
                        sample = ticker_data.iloc[0]
                        
                        # Basic info
                        if 'date' in sample.index:
                            f.write(f"  Date: {sample['date']}\n")
                        
                        if 'Open' in sample.index and 'Close' in sample.index:
                            f.write(f"  Open: {sample['Open']:.2f}, Close: {sample['Close']:.2f}\n")
                        
                        # External features
                        ext_features = []
                        for col in sample.index:
                            if '_news_' in col or '_reddit_' in col:
                                if pd.notna(sample[col]):
                                    ext_features.append(col)
                        
                        if ext_features:
                            f.write(f"  External features: {len(ext_features)}\n")
                            for feat in ext_features[:5]:
                                f.write(f"    • {feat}: {sample[feat]}\n")
                        else:
                            f.write(f"  No external features\n")
                
            print(f"📄 Summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"   ⚠️ Error creating summary: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("ROBUST FINANCIAL DATA INTEGRATOR")
    print("="*80)
    
    integrator = FinancialDataIntegratorRobust()
    
    # First, inspect dataset structures safely
    integrator.inspect_datasets_detailed_safe()
    
    print("\n" + "="*80)
    print("STARTING ROBUST INTEGRATION...")
    print("="*80)
    
    # Create integrated dataset
    tickers_to_integrate = ['AAPL', 'GOOGL', 'TSLA']
    integrated_df, output_file = integrator.create_integrated_dataset_robust(tickers=tickers_to_integrate)
    
    if integrated_df is not None:
        print("\n" + "="*80)
        print("✅ INTEGRATION SUCCESSFUL!")
        print("="*80)
        
        print(f"\n📊 FINAL DATASET:")
        print(f"   File: {output_file}")
        print(f"   Rows: {len(integrated_df):,}")
        print(f"   Columns: {len(integrated_df.columns):,}")
        
        # Quick analysis
        print(f"\n📈 QUICK ANALYSIS:")
        
        # Column breakdown
        stock_cols = [col for col in integrated_df.columns 
                     if not any(x in col for x in ['_news_', '_reddit_'])]
        news_cols = [col for col in integrated_df.columns if '_news_' in col]
        reddit_cols = [col for col in integrated_df.columns if '_reddit_' in col]
        
        print(f"   • Stock features: {len(stock_cols)}")
        print(f"   • News features: {len(news_cols)}")
        print(f"   • Reddit features: {len(reddit_cols)}")
        
        # Check for external data
        has_external = False
        for ticker in tickers_to_integrate:
            ticker_data = integrated_df[integrated_df['ticker'] == ticker]
            if len(news_cols) > 0:
                has_news = ticker_data[news_cols].notna().any().any()
                if has_news:
                    print(f"   • {ticker} has news data")
                    has_external = True
            if len(reddit_cols) > 0:
                has_reddit = ticker_data[reddit_cols].notna().any().any()
                if has_reddit:
                    print(f"   • {ticker} has reddit data")
                    has_external = True
        
        if not has_external:
            print(f"   ⚠️ No external data integrated")
        
        print(f"\n🎯 READY FOR PERFORMANCE ANALYSIS:")
    else:
        print("\n❌ Integration failed.")
