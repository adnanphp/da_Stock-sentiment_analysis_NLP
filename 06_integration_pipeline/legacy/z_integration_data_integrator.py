"""
data_integrator.py
Integrates stocks, news, and Reddit data for financial prediction
Creates unified dataset with stock prices + news sentiment + Reddit sentiment
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class FinancialDataIntegrator:
    """
    Main integration class - combines all data sources
    """
    
    def __init__(self, base_path="."):
        self.base_path = base_path
        self.integrated_data = {}
        
    def load_all_datasets(self):
        """Load all cleaned/feature-engineered datasets"""
        print("📁 LOADING DATASETS FOR INTEGRATION")
        print("=" * 60)
        
        datasets = {}
        
        # 1. Stock Data (Primary)
        stock_paths = [
            "./feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv",
            "./cleaned_financial_data/stock_data/cleaned_all_stocks_combined.csv",
            "./financial_data_large/stock_data/all_stocks_combined.csv"
        ]
        
        for path in stock_paths:
            if os.path.exists(path):
                print(f"   📈 Loading stock data: {path}")
                datasets['stocks'] = pd.read_csv(path)
                print(f"      ✓ Loaded: {datasets['stocks'].shape[0]:,} rows, {datasets['stocks'].shape[1]} columns")
                break
        
        # 2. News Data
        news_paths = [
            "./feature_engineered_data/news_data/feature_engineered_outlier_treated_alpha_vantage_news_expanded.csv",
            "./cleaned_financial_data/news_data/cleaned_alpha_vantage_news_expanded.csv",
            "./financial_data_large/news_data/alpha_vantage_news_expanded.csv"
        ]
        
        for path in news_paths:
            if os.path.exists(path):
                print(f"   📰 Loading news data: {path}")
                datasets['news'] = pd.read_csv(path)
                print(f"      ✓ Loaded: {datasets['news'].shape[0]:,} rows, {datasets['news'].shape[1]} columns")
                break
        
        # 3. Reddit Data
        reddit_paths = [
            "./feature_engineered_data/reddit_data/feature_engineered_outlier_treated_all_reddit_posts_combined.csv",
            "./cleaned_financial_data/reddit_data/cleaned_all_reddit_posts_combined.csv",
            "./financial_data_large/reddit_data/all_reddit_posts_combined.csv"
        ]
        
        for path in reddit_paths:
            if os.path.exists(path):
                print(f"   💬 Loading Reddit data: {path}")
                datasets['reddit'] = pd.read_csv(path)
                print(f"      ✓ Loaded: {datasets['reddit'].shape[0]:,} rows, {datasets['reddit'].shape[1]} columns")
                break
        
        # 4. Market Data (Optional)
        market_path = "./financial_data_large/market_data/market_indicators.csv"
        if os.path.exists(market_path):
            print(f"   📊 Loading market data: {market_path}")
            datasets['market'] = pd.read_csv(market_path)
            print(f"      ✓ Loaded: {datasets['market'].shape[0]:,} rows")
        
        print(f"\n✅ All datasets loaded successfully!")
        return datasets
    
    def create_integrated_dataset(self, tickers=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'], 
                                  start_date='2020-01-01', end_date='2023-12-31'):
        """
        Create integrated dataset for multiple tickers
        
        Parameters:
        -----------
        tickers : list
            List of stock tickers to integrate
        start_date, end_date : str
            Date range for integration
        
        Returns:
        --------
        dict: Integrated datasets for each ticker
        """
        print(f"\n🔗 CREATING INTEGRATED DATASET")
        print(f"   Tickers: {', '.join(tickers)}")
        print(f"   Date Range: {start_date} to {end_date}")
        print("=" * 60)
        
        # Load all datasets
        datasets = self.load_all_datasets()
        
        if 'stocks' not in datasets:
            print("❌ No stock data found!")
            return {}
        
        integrated_results = {}
        
        for ticker in tickers[:3]:  # Process first 3 for demo
            print(f"\n📊 Processing {ticker}...")
            
            # 1. Get stock data for this ticker
            ticker_stocks = self._get_ticker_data(datasets['stocks'], ticker)
            if ticker_stocks is None or len(ticker_stocks) == 0:
                print(f"   ⚠️ No stock data for {ticker}, skipping")
                continue
            
            # 2. Process news data
            ticker_news = self._process_news_data(datasets.get('news'), ticker)
            
            # 3. Process Reddit data
            ticker_reddit = self._process_reddit_data(datasets.get('reddit'), ticker)
            
            # 4. Process market data
            ticker_market = datasets.get('market', pd.DataFrame())
            
            # 5. Integrate all data
            integrated_df = self._integrate_all_sources(
                ticker_stocks, ticker_news, ticker_reddit, ticker_market,
                start_date, end_date
            )
            
            if integrated_df is not None and len(integrated_df) > 0:
                integrated_results[ticker] = integrated_df
                print(f"   ✅ Integrated: {len(integrated_df)} records")
                print(f"   📊 Features: {len(integrated_df.columns)} columns")
                
                # Show sample
                sample_cols = [col for col in integrated_df.columns if any(x in col for x in ['Close', 'sentiment', 'volume', 'return'])]
                if len(sample_cols) > 5:
                    sample_cols = sample_cols[:5]
                print(f"   👀 Sample columns: {', '.join(sample_cols)}")
        
        self.integrated_data = integrated_results
        return integrated_results
    
    def _get_ticker_data(self, stocks_df, ticker):
        """Extract data for specific ticker"""
        # Try different column names for ticker
        ticker_cols = ['Symbol', 'Ticker', 'symbol', 'ticker', 'Stock']
        ticker_col = None
        
        for col in ticker_cols:
            if col in stocks_df.columns:
                ticker_col = col
                break
        
        if ticker_col:
            ticker_data = stocks_df[stocks_df[ticker_col] == ticker].copy()
        else:
            # If no ticker column, assume all data is for one stock
            ticker_data = stocks_df.copy()
            ticker_data['Symbol'] = ticker
        
        # Convert date
        ticker_data = self._standardize_dates(ticker_data, 'stock')
        
        return ticker_data
    
    def _process_news_data(self, news_df, ticker):
        """Process news data for a specific ticker"""
        if news_df is None or len(news_df) == 0:
            return None
        
        # Convert date
        news_df = self._standardize_dates(news_df.copy(), 'news')
        
        # Find sentiment column
        sentiment_cols = ['sentiment_score', 'sentiment', 'overall_sentiment', 'compound', 'score']
        sentiment_col = None
        for col in sentiment_cols:
            if col in news_df.columns:
                sentiment_col = col
                break
        
        if sentiment_col is None:
            print(f"   ⚠️ No sentiment column found in news data for {ticker}")
            return None
        
        # Filter for ticker mentions if possible
        if 'tickers' in news_df.columns or 'ticker' in news_df.columns:
            ticker_col = 'tickers' if 'tickers' in news_df.columns else 'ticker'
            ticker_mask = news_df[ticker_col].astype(str).str.contains(ticker, case=False, na=False)
            ticker_news = news_df[ticker_mask]
        else:
            ticker_news = news_df
        
        if len(ticker_news) == 0:
            print(f"   ⚠️ No news mentions found for {ticker}")
            return None
        
        # Aggregate by date
        news_agg = ticker_news.groupby('date').agg({
            sentiment_col: ['mean', 'count', 'std', 'min', 'max']
        }).round(4)
        
        # Flatten column names
        news_agg.columns = [f'news_{col[0]}_{col[1]}' for col in news_agg.columns]
        news_agg = news_agg.reset_index()
        
        # Rename for clarity
        rename_dict = {}
        for col in news_agg.columns:
            if 'mean' in col:
                rename_dict[col] = col.replace('_mean', '_avg')
        
        news_agg = news_agg.rename(columns=rename_dict)
        
        print(f"   📰 News: {len(ticker_news)} articles → {len(news_agg)} daily aggregates")
        return news_agg
    
    def _process_reddit_data(self, reddit_df, ticker):
        """Process Reddit data for a specific ticker"""
        if reddit_df is None or len(reddit_df) == 0:
            return None
        
        # Convert date
        reddit_df = self._standardize_dates(reddit_df.copy(), 'reddit')
        
        # Find sentiment column
        sentiment_cols = ['sentiment', 'sentiment_score', 'compound', 'vader_compound', 'score']
        sentiment_col = None
        for col in sentiment_cols:
            if col in reddit_df.columns:
                sentiment_col = col
                break
        
        if sentiment_col is None:
            # Check if sentiment can be calculated from other columns
            if 'upvotes' in reddit_df.columns and 'downvotes' in reddit_df.columns:
                reddit_df['sentiment_score'] = (reddit_df['upvotes'] - reddit_df['downvotes']) / (reddit_df['upvotes'] + reddit_df['downvotes'] + 1)
                sentiment_col = 'sentiment_score'
            elif 'score' in reddit_df.columns:
                # Normalize score as sentiment proxy
                reddit_df['sentiment_score'] = reddit_df['score'] / (reddit_df['score'].abs().max() + 1)
                sentiment_col = 'sentiment_score'
            else:
                print(f"   ⚠️ No sentiment column found in Reddit data for {ticker}")
                return None
        
        # Filter for ticker mentions
        text_cols = ['title', 'selftext', 'body', 'text', 'content']
        ticker_mask = pd.Series([False] * len(reddit_df))
        
        for col in text_cols:
            if col in reddit_df.columns:
                # Look for $TICKER or TICKER mentions
                ticker_mask = ticker_mask | reddit_df[col].astype(str).str.contains(
                    f'\\${ticker}\\b|\\b{ticker}\\b', case=False, na=False
                )
        
        ticker_reddit = reddit_df[ticker_mask]
        
        if len(ticker_reddit) == 0:
            print(f"   ⚠️ No Reddit mentions found for {ticker}")
            return None
        
        # Aggregate by date
        agg_dict = {
            sentiment_col: ['mean', 'count', 'std']
        }
        
        # Add additional metrics if available
        if 'score' in ticker_reddit.columns:
            agg_dict['score'] = ['mean', 'sum']
        if 'num_comments' in ticker_reddit.columns:
            agg_dict['num_comments'] = ['mean', 'sum']
        
        reddit_agg = ticker_reddit.groupby('date').agg(agg_dict).round(4)
        
        # Flatten column names
        reddit_agg.columns = [f'reddit_{col[0]}_{col[1]}' if isinstance(col, tuple) else f'reddit_{col}' 
                             for col in reddit_agg.columns]
        reddit_agg = reddit_agg.reset_index()
        
        print(f"   💬 Reddit: {len(ticker_reddit)} posts → {len(reddit_agg)} daily aggregates")
        return reddit_agg
    
    def _integrate_all_sources(self, stocks_df, news_df, reddit_df, market_df, 
                               start_date, end_date):
        """Integrate all data sources into one dataset"""
        
        # Start with stock data as base
        integrated = stocks_df.copy()
        
        # Filter by date range
        integrated = integrated[
            (integrated['date'] >= start_date) & 
            (integrated['date'] <= end_date)
        ].sort_values('date')
        
        if len(integrated) == 0:
            return None
        
        # Merge news data
        if news_df is not None and len(news_df) > 0:
            integrated = pd.merge(integrated, news_df, on='date', how='left')
        
        # Merge Reddit data
        if reddit_df is not None and len(reddit_df) > 0:
            integrated = pd.merge(integrated, reddit_df, on='date', how='left')
        
        # Merge market data (if available)
        if market_df is not None and len(market_df) > 0:
            market_df = self._standardize_dates(market_df.copy(), 'market')
            integrated = pd.merge(integrated, market_df, on='date', how='left')
        
        # Fill missing sentiment with 0 (neutral)
        sentiment_cols = [col for col in integrated.columns if 'sentiment' in col]
        for col in sentiment_cols:
            if col in integrated.columns:
                integrated[col] = integrated[col].fillna(0)
        
        # Create combined features
        integrated = self._create_combined_features(integrated)
        
        # Calculate target variables
        integrated = self._create_target_variables(integrated)
        
        return integrated
    
    def _standardize_dates(self, df, source_type):
        """Standardize date columns across different datasets"""
        df = df.copy()
        
        date_patterns = {
            'stock': ['Date', 'date', 'timestamp', 'time', 'datetime'],
            'news': ['timestamp', 'published_at', 'time_published', 'date', 'datetime'],
            'reddit': ['created_utc', 'timestamp', 'created', 'date'],
            'market': ['Date', 'date', 'timestamp']
        }
        
        date_cols = date_patterns.get(source_type, ['date', 'Date', 'timestamp'])
        
        for col in date_cols:
            if col in df.columns:
                try:
                    if df[col].dtype in ['int64', 'float64']:
                        # Unix timestamp
                        df['date'] = pd.to_datetime(df[col], unit='s')
                    else:
                        df['date'] = pd.to_datetime(df[col])
                    
                    df['date'] = df['date'].dt.normalize()  # Keep only date part
                    break
                except:
                    continue
        
        # If no date column found, create one from index
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime('today').normalize()
        
        return df
    
    def _create_combined_features(self, df):
        """Create combined features from integrated data"""
        df = df.copy()
        
        # 1. Combined sentiment score (weighted average)
        if 'news_sentiment_avg' in df.columns and 'reddit_sentiment_avg' in df.columns:
            df['combined_sentiment'] = (
                df['news_sentiment_avg'].fillna(0) * 0.6 + 
                df['reddit_sentiment_avg'].fillna(0) * 0.4
            )
        elif 'news_sentiment_avg' in df.columns:
            df['combined_sentiment'] = df['news_sentiment_avg']
        elif 'reddit_sentiment_avg' in df.columns:
            df['combined_sentiment'] = df['reddit_sentiment_avg']
        
        # 2. Sentiment momentum indicators
        sentiment_cols = [col for col in df.columns if 'sentiment' in col and 'avg' in col]
        for col in sentiment_cols:
            df[f'{col}_3d_ma'] = df[col].rolling(window=3, min_periods=1).mean()
            df[f'{col}_7d_ma'] = df[col].rolling(window=7, min_periods=1).mean()
            df[f'{col}_change'] = df[col].pct_change().fillna(0)
        
        # 3. Volume-sentiment interaction (if volume exists)
        if 'Volume' in df.columns and 'combined_sentiment' in df.columns:
            df['volume_sentiment'] = df['Volume'] * df['combined_sentiment'].abs()
            df['volume_sentiment_ratio'] = df['volume_sentiment'] / (df['Volume'].mean() + 1)
        
        # 4. Social media activity indicators
        activity_cols = []
        if 'news_article_count' in df.columns:
            activity_cols.append('news_article_count')
        if 'reddit_post_count' in df.columns:
            activity_cols.append('reddit_post_count')
        
        for col in activity_cols:
            df[f'{col}_normalized'] = df[col] / (df[col].max() + 1)
            df[f'{col}_spike'] = (df[col] > df[col].rolling(window=7).mean() * 1.5).astype(int)
        
        # 5. Market-social divergence
        if 'Close' in df.columns and 'combined_sentiment' in df.columns:
            price_change = df['Close'].pct_change().fillna(0)
            sentiment_change = df['combined_sentiment'].diff().fillna(0)
            df['price_sentiment_divergence'] = price_change - sentiment_change
            df['divergence_signal'] = np.where(df['price_sentiment_divergence'] > 0.02, 1, 
                                              np.where(df['price_sentiment_divergence'] < -0.02, -1, 0))
        
        # 6. Composite score
        score_components = []
        if 'combined_sentiment' in df.columns:
            score_components.append(df['combined_sentiment'])
        if 'Close' in df.columns:
            price_score = df['Close'].pct_change().rolling(window=5).mean().fillna(0)
            score_components.append(price_score)
        
        if score_components:
            df['composite_score'] = sum(score_components) / len(score_components)
        
        return df
    
    def _create_target_variables(self, df):
        """Create target variables for prediction"""
        if 'Close' not in df.columns:
            return df
        
        # 1. Next day return (regression target)
        df['target_next_day_return'] = df['Close'].shift(-1).pct_change().shift(1)
        
        # 2. Binary classification: Will price go up tomorrow?
        df['target_price_up'] = (df['target_next_day_return'] > 0).astype(int)
        
        # 3. Multi-class: Strong up, slight up, down, etc.
        df['target_price_movement'] = pd.cut(
            df['target_next_day_return'],
            bins=[-np.inf, -0.02, 0, 0.02, np.inf],
            labels=['strong_down', 'slight_down', 'slight_up', 'strong_up']
        )
        
        # 4. Volatility prediction
        df['target_volatility'] = df['Close'].pct_change().rolling(window=5).std().shift(-1)
        
        return df
    
    def save_integrated_datasets(self, output_dir="./integrated_datasets"):
        """Save integrated datasets to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n💾 SAVING INTEGRATED DATASETS")
        print("=" * 60)
        
        saved_files = []
        
        for ticker, df in self.integrated_data.items():
            if df is not None and len(df) > 0:
                filename = f"integrated_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv"
                filepath = os.path.join(output_dir, filename)
                
                df.to_csv(filepath, index=False)
                saved_files.append(filepath)
                
                print(f"   ✅ {ticker}: {len(df):,} rows, {len(df.columns)} columns")
                print(f"      Saved to: {filename}")
                
                # Also save a summary file
                summary_file = os.path.join(output_dir, f"integrated_{ticker}_summary.txt")
                with open(summary_file, 'w') as f:
                    f.write(f"Integrated Dataset: {ticker}\n")
                    f.write(f"Date Range: {df['date'].min()} to {df['date'].max()}\n")
                    f.write(f"Total Records: {len(df)}\n")
                    f.write(f"Total Features: {len(df.columns)}\n\n")
                    
                    f.write("Feature Categories:\n")
                    f.write("-" * 40 + "\n")
                    
                    # Categorize features
                    categories = {
                        'Price Features': [col for col in df.columns if any(x in col.lower() for x in ['close', 'open', 'high', 'low', 'price'])],
                        'Volume Features': [col for col in df.columns if 'volume' in col.lower()],
                        'News Features': [col for col in df.columns if 'news' in col.lower()],
                        'Reddit Features': [col for col in df.columns if 'reddit' in col.lower()],
                        'Sentiment Features': [col for col in df.columns if 'sentiment' in col.lower()],
                        'Technical Features': [col for col in df.columns if any(x in col.lower() for x in ['ma', 'rsi', 'macd', 'bollinger'])],
                        'Target Variables': [col for col in df.columns if 'target' in col.lower()]
                    }
                    
                    for category, features in categories.items():
                        if features:
                            f.write(f"\n{category} ({len(features)}):\n")
                            for feature in features[:10]:  # Show first 10
                                f.write(f"  • {feature}\n")
                            if len(features) > 10:
                                f.write(f"  • ... and {len(features) - 10} more\n")
        
        print(f"\n📁 All datasets saved to: {output_dir}/")
        
        # Create a combined dataset if multiple tickers
        if len(self.integrated_data) > 1:
            combined_df = pd.concat(self.integrated_data.values(), ignore_index=True)
            combined_path = os.path.join(output_dir, f"integrated_ALL_{datetime.now().strftime('%Y%m%d')}.csv")
            combined_df.to_csv(combined_path, index=False)
            print(f"📊 Combined dataset: {combined_path} ({len(combined_df):,} rows)")
        
        return saved_files

# ============================================================================
# QUICK DEMO FUNCTION
# ============================================================================

def run_integration_demo():
    """Quick demo of data integration"""
    print("="*80)
    print("FINANCIAL DATA INTEGRATION DEMO")
    print("="*80)
    
    integrator = FinancialDataIntegrator()
    
    # Create integrated datasets
    integrated_data = integrator.create_integrated_dataset(
        tickers=['AAPL', 'GOOGL', 'TSLA'],
        start_date='2022-01-01',
        end_date='2023-12-31'
    )
    
    # Save results
    if integrated_data:
        integrator.save_integrated_datasets()
        
        # Show statistics
        print(f"\n📈 INTEGRATION STATISTICS:")
        print("-" * 40)
        
        for ticker, df in integrated_data.items():
            print(f"\n{ticker}:")
            print(f"  • Records: {len(df):,}")
            print(f"  • Features: {len(df.columns)}")
            print(f"  • Date Range: {df['date'].min().date()} to {df['date'].max().date()}")
            
            # Count features by type
            price_features = len([col for col in df.columns if 'close' in col.lower() or 'price' in col.lower()])
            sentiment_features = len([col for col in df.columns if 'sentiment' in col.lower()])
            news_features = len([col for col in df.columns if 'news' in col.lower()])
            reddit_features = len([col for col in df.columns if 'reddit' in col.lower()])
            
            print(f"  • Price Features: {price_features}")
            print(f"  • Sentiment Features: {sentiment_features}")
            print(f"  • News Features: {news_features}")
            print(f"  • Reddit Features: {reddit_features}")
            
            # Show sample of integrated features
            sample_features = [col for col in df.columns if any(x in col for x in ['Close', 'sentiment', 'target'])]
            if sample_features:
                print(f"  • Sample Features: {', '.join(sample_features[:5])}")
    
    return integrator

if __name__ == "__main__":
    # Run the integration demo
    integrator = run_integration_demo()
    
    print(f"\n{'='*80}")
    print("✅ INTEGRATION COMPLETE!")
    print(f"{'='*80}")
    print("\n🎯 NEXT STEPS:")
    print("1. Integrated datasets saved to './integrated_datasets/'")
    print("2. Use these for model training with integrated features")
    print("3. Compare performance with vs without integration")
    print("4. Present integration results to professor")
