"""
z_fix_integration_properly.py - Fix all integration issues properly
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import re

def fix_and_integrate_all_data():
    """Fix ALL data issues and create proper integrated dataset"""
    print("="*80)
    print("PROPER DATA INTEGRATION FIX")
    print("="*80)
    
    # Load all data
    print("\n1. LOADING ALL DATA...")
    
    # Stocks
    stocks_path = "./feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv"
    stocks = pd.read_csv(stocks_path)
    print(f"   📈 Stocks: {stocks.shape}")
    
    # News
    news_path = "./feature_engineered_data/news_data/feature_engineered_outlier_treated_alpha_vantage_news_expanded.csv"
    news = pd.read_csv(news_path)
    print(f"   📰 News: {news.shape}")
    
    # Reddit
    reddit_path = "./feature_engineered_data/reddit_data/feature_engineered_outlier_treated_all_reddit_posts_combined.csv"
    reddit = pd.read_csv(reddit_path)
    print(f"   💬 Reddit: {reddit.shape}")
    
    # Filter stocks for our tickers
    tickers = ['AAPL', 'GOOGL', 'TSLA']
    stocks = stocks[stocks['ticker'].isin(tickers)].copy()
    print(f"\n   Filtered stocks: {stocks.shape}")
    
    # Fix reddit data dates
    print("\n2. FIXING REDDIT DATE PARSING...")
    
    # Look at post_datetime column - it has format like "20210201t083240" with extra digit
    if 'post_datetime' in reddit.columns:
        print(f"   Found post_datetime column")
        
        # Show raw samples
        samples = reddit['post_datetime'].head(5).tolist()
        print(f"   Raw samples: {samples}")
        
        # Fix the date format - remove extra trailing digit
        def fix_reddit_date(date_str):
            if pd.isna(date_str):
                return pd.NaT
            
            # Convert to string
            date_str = str(date_str)
            
            # Check if it has format like 20210201t083240 (15 chars) or 20210201t0832400 (16 chars)
            if len(date_str) == 16 and 't' in date_str.lower():
                # Remove last digit if it's 16 chars
                date_str = date_str[:-1]
            
            # Try to parse
            try:
                # Format: YYYYMMDDtHHMMSS
                return pd.to_datetime(date_str, format='%Y%m%dt%H%M%S', errors='coerce')
            except:
                try:
                    # Try without format
                    return pd.to_datetime(date_str, errors='coerce')
                except:
                    return pd.NaT
        
        reddit['date_fixed'] = reddit['post_datetime'].apply(fix_reddit_date)
        
        # Check results
        valid_dates = reddit['date_fixed'].notna().sum()
        print(f"   Successfully parsed {valid_dates}/{len(reddit)} dates")
        
        if valid_dates > 0:
            print(f"   Date range: {reddit['date_fixed'].min()} to {reddit['date_fixed'].max()}")
            reddit['reddit_date'] = reddit['date_fixed'].dt.date
        else:
            print(f"   ⚠️ Could not parse any dates, checking other columns...")
            
            # Check for post_date_year, post_date_month, post_date_day columns
            if all(col in reddit.columns for col in ['post_date_year', 'post_date_month', 'post_date_day']):
                print(f"   Found year/month/day columns")
                # Create date from components
                reddit['reddit_date'] = pd.to_datetime(
                    reddit[['post_date_year', 'post_date_month', 'post_date_day']]
                ).dt.date
                print(f"   Created dates from components")
    
    # Check reddit text content
    print("\n3. CHECKING REDDIT CONTENT...")
    
    # Look for text columns that actually have data
    text_cols_to_check = ['title', 'content', 'selftext', 'text', 'body']
    
    for col in text_cols_to_check:
        if col in reddit.columns:
            non_null = reddit[col].notna().sum()
            print(f"   {col}: {non_null} non-null values")
            
            if non_null > 0:
                # Show samples
                samples = reddit[col].dropna().head(3).tolist()
                for i, sample in enumerate(samples):
                    print(f"     Sample {i+1}: {str(sample)[:100]}...")
    
    # If no text columns have data, check for any columns with text
    if 'reddit_date' not in reddit.columns:
        # Last resort: look for any column that might contain dates
        print(f"\n   Searching for any date-like columns...")
        for col in reddit.columns[:50]:  # Check first 50 columns
            sample = reddit[col].iloc[0] if len(reddit) > 0 else None
            if isinstance(sample, str) and any(x in sample for x in ['202', '2023', '2024', '2025']):
                print(f"   Found potential date in {col}: {sample}")
                try:
                    reddit['reddit_date'] = pd.to_datetime(reddit[col]).dt.date
                    print(f"   Parsed dates from {col}")
                    break
                except:
                    pass
    
    # Prepare stock dates
    print("\n4. PREPARING STOCK DATES...")
    stocks['stock_date'] = pd.to_datetime(stocks['Date']).dt.date
    print(f"   Stock date range: {stocks['stock_date'].min()} to {stocks['stock_date'].max()}")
    
    # Prepare news dates
    print("\n5. PREPARING NEWS DATES...")
    if 'published_at' in news.columns:
        news['news_date'] = pd.to_datetime(news['published_at']).dt.date
        print(f"   News date range: {news['news_date'].min()} to {news['news_date'].max()}")
    
    # Create integrated dataset
    print("\n6. CREATING INTEGRATED DATASET...")
    
    all_integrated = []
    
    for ticker in tickers:
        print(f"\n   Processing {ticker}...")
        
        # Get stock data
        ticker_stocks = stocks[stocks['ticker'] == ticker].copy()
        ticker_stocks = ticker_stocks.sort_values('stock_date')
        ticker_stocks['date'] = ticker_stocks['stock_date']
        
        print(f"     • Stock records: {len(ticker_stocks)}")
        
        # --- NEWS INTEGRATION ---
        if 'news_date' in news.columns:
            # Search for ticker mentions in news
            news_mask = pd.Series([False] * len(news))
            
            # Search in tickers column (contains list of tickers)
            if 'tickers' in news.columns:
                news_mask = news_mask | news['tickers'].astype(str).str.contains(ticker, na=False)
            
            # Search in title
            if 'title' in news.columns:
                news_mask = news_mask | news['title'].astype(str).str.contains(ticker, case=False, na=False)
            
            # Search in summary
            if 'summary' in news.columns:
                news_mask = news_mask | news['summary'].astype(str).str.contains(ticker, case=False, na=False)
            
            ticker_news = news[news_mask].copy()
            
            if len(ticker_news) > 0:
                print(f"     • News mentions found: {len(ticker_news)}")
                
                # Create daily news aggregates
                news_agg = ticker_news.groupby('news_date').agg({
                    'title': 'count',
                    'overall_sentiment': 'mean'
                }).reset_index()
                
                news_agg.columns = ['date', f'news_{ticker}_count', f'news_{ticker}_sentiment']
                
                # Merge with stocks
                ticker_stocks = pd.merge(
                    ticker_stocks,
                    news_agg,
                    on='date',
                    how='left'
                )
                
                news_records = ticker_stocks[f'news_{ticker}_count'].notna().sum()
                print(f"     • News integrated: {news_records} records")
            else:
                print(f"     • No news mentions found")
        else:
            print(f"     • No news dates available")
        
        # --- REDDIT INTEGRATION ---
        if 'reddit_date' in reddit.columns:
            # Search for company mentions (not just ticker)
            company_terms = {
                'AAPL': ['APPLE', 'IPHONE', 'MAC', 'IPAD'],
                'GOOGL': ['GOOGLE', 'ALPHABET', 'ANDROID'],
                'TSLA': ['TESLA', 'ELON MUSK', 'CYBERTRUCK']
            }
            
            reddit_mask = pd.Series([False] * len(reddit))
            
            # Search in available text columns
            text_cols = []
            for col in ['title', 'content', 'selftext']:
                if col in reddit.columns and reddit[col].notna().any():
                    text_cols.append(col)
            
            if text_cols:
                print(f"     • Searching reddit in columns: {text_cols}")
                
                for col in text_cols:
                    for term in company_terms[ticker]:
                        try:
                            col_data = reddit[col].fillna('').astype(str)
                            term_mask = col_data.str.upper().str.contains(term, na=False)
                            reddit_mask = reddit_mask | term_mask
                        except:
                            pass
                
                ticker_reddit = reddit[reddit_mask].copy()
                
                if len(ticker_reddit) > 0:
                    print(f"     • Reddit mentions found: {len(ticker_reddit)}")
                    
                    # Create daily reddit aggregates
                    reddit_agg = ticker_reddit.groupby('reddit_date').agg({
                        'reddit_date': 'count'
                    }).reset_index()
                    
                    reddit_agg.columns = ['date', f'reddit_{ticker}_count']
                    
                    # Add upvotes if available
                    if 'upvotes' in ticker_reddit.columns:
                        upvotes_agg = ticker_reddit.groupby('reddit_date')['upvotes'].agg(['mean', 'sum']).reset_index()
                        upvotes_agg.columns = ['date', f'reddit_{ticker}_upvotes_mean', f'reddit_{ticker}_upvotes_sum']
                        reddit_agg = pd.merge(reddit_agg, upvotes_agg, on='date', how='left')
                    
                    # Merge with stocks
                    ticker_stocks = pd.merge(
                        ticker_stocks,
                        reddit_agg,
                        on='date',
                        how='left'
                    )
                    
                    reddit_col = f'reddit_{ticker}_count'
                    if reddit_col in ticker_stocks.columns:
                        reddit_records = ticker_stocks[reddit_col].notna().sum()
                        print(f"     • Reddit integrated: {reddit_records} records")
                else:
                    print(f"     • No reddit mentions found")
            else:
                print(f"     • No text columns with data in reddit")
        else:
            print(f"     • No reddit dates available")
        
        # Clean up
        cols_to_remove = ['stock_date', 'Date', 'news_date', 'reddit_date', 'date_fixed']
        for col in cols_to_remove:
            if col in ticker_stocks.columns:
                ticker_stocks = ticker_stocks.drop(col, axis=1)
        
        all_integrated.append(ticker_stocks)
    
    # Combine all tickers
    if all_integrated:
        final_df = pd.concat(all_integrated, ignore_index=True)
        
        print(f"\n✅ FINAL INTEGRATED DATASET:")
        print(f"   • Total records: {len(final_df)}")
        print(f"   • Total columns: {len(final_df.columns)}")
        
        # Count external data columns
        news_cols = [col for col in final_df.columns if 'news_' in col]
        reddit_cols = [col for col in final_df.columns if 'reddit_' in col]
        
        print(f"   • News columns: {len(news_cols)}")
        print(f"   • Reddit columns: {len(reddit_cols)}")
        
        # Show integration stats
        print(f"\n📊 INTEGRATION STATISTICS:")
        
        for ticker in tickers:
            ticker_data = final_df[final_df['ticker'] == ticker]
            total = len(ticker_data)
            
            # News
            news_col = f'news_{ticker}_count'
            if news_col in final_df.columns:
                news_with_data = ticker_data[news_col].notna().sum()
                if news_with_data > 0:
                    print(f"   {ticker} - News: {news_with_data}/{total} records ({news_with_data/total*100:.1f}%)")
            
            # Reddit
            reddit_col = f'reddit_{ticker}_count'
            if reddit_col in final_df.columns:
                reddit_with_data = ticker_data[reddit_col].notna().sum()
                if reddit_with_data > 0:
                    print(f"   {ticker} - Reddit: {reddit_with_data}/{total} records ({reddit_with_data/total*100:.1f}%)")
        
        # Save the dataset
        output_dir = "./properly_integrated"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"properly_integrated_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        final_df.to_csv(filepath, index=False)
        
        print(f"\n💾 Saved to: {filepath}")
        
        # Show sample with external data
        print(f"\n👀 SAMPLE DATA WITH EXTERNAL FEATURES:")
        
        sample_cols = ['date', 'ticker', 'Open', 'Close']
        
        # Add external columns that have data
        external_cols = []
        for ticker in tickers[:1]:  # Just show for first ticker
            ticker_external = [col for col in final_df.columns if ticker in col]
            
            # Check which ones have data
            for col in ticker_external:
                if final_df[col].notna().any():
                    external_cols.append(col)
                    if len(external_cols) >= 3:  # Show up to 3
                        break
        
        sample_df = final_df[sample_cols + external_cols].head(10)
        print(sample_df.to_string())
        
        # Check if we actually have any external data
        has_external = False
        for col in external_cols:
            if sample_df[col].notna().any():
                has_external = True
                break
        
        if not has_external:
            print(f"\n⚠️ WARNING: No external data found in sample!")
            print(f"   This means integration happened but dates don't match")
            
            # Show date ranges to debug
            print(f"\n📅 DATE RANGES:")
            print(f"   Stocks: {stocks['stock_date'].min()} to {stocks['stock_date'].max()}")
            
            if 'news_date' in news.columns:
                print(f"   News: {news['news_date'].min()} to {news['news_date'].max()}")
            
            if 'reddit_date' in reddit.columns:
                print(f"   Reddit: {reddit['reddit_date'].min()} to {reddit['reddit_date'].max()}")
        
        return final_df, filepath
    
    return None, None

def create_complete_performance_analysis(integrated_file):
    """Create complete performance analysis"""
    print("\n" + "="*80)
    print("COMPLETE PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Load integrated data
    print(f"\n📂 Loading: {integrated_file}")
    df = pd.read_csv(integrated_file)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    print(f"   Loaded: {df.shape}")
    
    # Analyze each ticker
    tickers = ['AAPL', 'GOOGL', 'TSLA']
    
    results = []
    
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {ticker}")
        print(f"{'='*60}")
        
        ticker_data = df[df['ticker'] == ticker].copy()
        
        if len(ticker_data) < 50:
            print(f"   ⚠️ Not enough data")
            continue
        
        print(f"   Records: {len(ticker_data)}")
        
        # Get all numeric columns as features
        numeric_cols = [col for col in ticker_data.columns 
                       if col not in ['date', 'ticker'] 
                       and pd.api.types.is_numeric_dtype(ticker_data[col])]
        
        # Separate stock-only and external features
        stock_features = [col for col in numeric_cols 
                         if not any(x in col for x in ['news_', 'reddit_'])]
        
        external_features = [col for col in numeric_cols 
                           if any(x in col for x in ['news_', 'reddit_'])]
        
        print(f"   • Stock features: {len(stock_features)}")
        print(f"   • External features: {len(external_features)}")
        
        if external_features:
            print(f"   • External features present!")
        
        # Run models
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        
        # Prepare data for prediction
        ticker_data = ticker_data.sort_values('date')
        ticker_data['target'] = ticker_data['Close'].pct_change().shift(-1)
        ticker_data = ticker_data.dropna(subset=['target'])
        
        if len(ticker_data) < 30:
            print(f"   ⚠️ Not enough data for prediction")
            continue
        
        # Stock-only model
        X_stock = ticker_data[stock_features].fillna(0).values
        y = ticker_data['target'].values
        
        # Integrated model (use all features)
        X_integrated = ticker_data[numeric_cols].fillna(0).values
        
        # Time-series split
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        stock_mse_scores = []
        stock_r2_scores = []
        integrated_mse_scores = []
        integrated_r2_scores = []
        
        for train_idx, test_idx in tscv.split(X_stock):
            # Stock-only
            X_train_stock, X_test_stock = X_stock[train_idx], X_stock[test_idx]
            X_train_int, X_test_int = X_integrated[train_idx], X_integrated[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train models
            model_stock = RandomForestRegressor(n_estimators=50, random_state=42)
            model_stock.fit(X_train_stock, y_train)
            
            model_int = RandomForestRegressor(n_estimators=50, random_state=42)
            model_int.fit(X_train_int, y_train)
            
            # Predict and evaluate
            y_pred_stock = model_stock.predict(X_test_stock)
            y_pred_int = model_int.predict(X_test_int)
            
            stock_mse_scores.append(mean_squared_error(y_test, y_pred_stock))
            stock_r2_scores.append(r2_score(y_test, y_pred_stock))
            
            integrated_mse_scores.append(mean_squared_error(y_test, y_pred_int))
            integrated_r2_scores.append(r2_score(y_test, y_pred_int))
        
        # Calculate averages
        stock_mse = np.mean(stock_mse_scores)
        stock_r2 = np.mean(stock_r2_scores)
        integrated_mse = np.mean(integrated_mse_scores)
        integrated_r2 = np.mean(integrated_r2_scores)
        
        # Calculate improvements
        mse_improvement = ((stock_mse - integrated_mse) / stock_mse) * 100 if stock_mse != 0 else 0
        r2_improvement = ((integrated_r2 - stock_r2) / abs(stock_r2 + 1e-10)) * 100
        
        print(f"\n   📊 RESULTS:")
        print(f"     {'Metric':<10} {'Stocks-Only':<12} {'Integrated':<12} {'Δ':<10}")
        print(f"     {'-'*45}")
        
        print(f"     {'MSE':<10} {stock_mse:<12.6f} {integrated_mse:<12.6f} {mse_improvement:+.1f}%")
        print(f"     {'R²':<10} {stock_r2:<12.4f} {integrated_r2:<12.4f} {r2_improvement:+.1f}%")
        
        results.append({
            'ticker': ticker,
            'stock_mse': stock_mse,
            'integrated_mse': integrated_mse,
            'mse_improvement_pct': mse_improvement,
            'stock_r2': stock_r2,
            'integrated_r2': integrated_r2,
            'r2_improvement_pct': r2_improvement,
            'stock_features': len(stock_features),
            'external_features': len(external_features),
            'total_records': len(ticker_data)
        })
    
    # Create summary
    if results:
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        
        results_df = pd.DataFrame(results)
        
        # Save results
        output_dir = "./performance_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f"performance_analysis_{timestamp}.csv")
        results_df.to_csv(results_file, index=False)
        
        print(f"\n📄 Results saved to: {results_file}")
        
        # Calculate overall statistics
        avg_mse_imp = results_df['mse_improvement_pct'].mean()
        avg_r2_imp = results_df['r2_improvement_pct'].mean()
        
        print(f"\n📊 OVERALL PERFORMANCE:")
        print(f"   • Average MSE improvement: {avg_mse_imp:+.1f}%")
        print(f"   • Average R² improvement: {avg_r2_imp:+.1f}%")
        
        improved_tickers = results_df[results_df['mse_improvement_pct'] > 0]
        print(f"   • Tickers with improved MSE: {len(improved_tickers)}/{len(results_df)}")
        
        print(f"\n💡 CONCLUSION:")
        if avg_mse_imp > 5:
            print(f"   ✅ Integration provides significant improvement")
        elif avg_mse_imp > 0:
            print(f"   ⚠️ Integration provides modest improvement")
        else:
            print(f"   ❌ Integration does not improve predictions")
            print(f"   Reason: Limited external data or date mismatches")

if __name__ == "__main__":
    # Fix and integrate all data
    integrated_df, output_file = fix_and_integrate_all_data()
    
    # Run performance analysis
    if integrated_df is not None:
        create_complete_performance_analysis(output_file)
        
        print("\n" + "="*80)
        print("✅ INTEGRATION COMPLETE!")
        print("="*80)
        print(f"\n📁 Files created:")
        print(f"   1. {output_file} - Properly integrated dataset")
        print(f"   2. ./performance_analysis/ - Performance results")
