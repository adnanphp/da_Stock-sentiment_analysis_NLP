"""
z_fix_reddit_integration.py - Fix reddit data integration
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def debug_reddit_data():
    """Debug why reddit data isn't working"""
    print("="*80)
    print("DEBUGGING REDDIT DATA INTEGRATION")
    print("="*80)
    
    # Load reddit data
    print("\n1. LOADING REDDIT DATA...")
    reddit = pd.read_csv("./feature_engineered_data/reddit_data/feature_engineered_outlier_treated_all_reddit_posts_combined.csv")
    print(f"   Reddit shape: {reddit.shape}")
    print(f"   Columns: {list(reddit.columns)[:20]}")
    
    print("\n2. CHECKING DATE COLUMNS...")
    # Check all date-related columns
    date_cols = []
    for col in reddit.columns:
        if 'date' in col.lower() or 'time' in col.lower() or 'created' in col.lower():
            date_cols.append(col)
    
    print(f"   Date-related columns: {date_cols}")
    
    # Try to parse each date column
    for col in date_cols[:5]:  # Check first 5
        print(f"\n   Testing column: {col}")
        print(f"   Sample values: {reddit[col].head(3).tolist()}")
        
        # Try to parse as datetime
        try:
            parsed = pd.to_datetime(reddit[col].head(3))
            print(f"   Parsed as: {parsed.tolist()}")
        except Exception as e:
            print(f"   Error parsing: {e}")
    
    print("\n3. CHECKING TEXT CONTENT...")
    # Check text columns
    text_cols = ['title', 'content', 'selftext', 'text', 'body']
    available_text_cols = [col for col in text_cols if col in reddit.columns]
    
    print(f"   Available text columns: {available_text_cols}")
    
    if 'title' in reddit.columns:
        print(f"\n   Sample titles:")
        for i in range(3):
            title = reddit['title'].iloc[i]
            print(f"   {i+1}. {str(title)[:100]}...")
    
    if 'content' in reddit.columns:
        print(f"\n   Sample content (first 100 chars):")
        for i in range(3):
            content = reddit['content'].iloc[i]
            print(f"   {i+1}. {str(content)[:100]}...")
    
    print("\n4. SEARCHING FOR COMPANY MENTIONS...")
    # Search for company names (not tickers)
    company_terms = {
        'AAPL': ['APPLE', 'IPHONE', 'MAC', 'IPAD', 'IOS', 'APPLE STOCK', 'AAPL'],
        'GOOGL': ['GOOGLE', 'ALPHABET', 'ANDROID', 'YOUTUBE', 'GOOGLE STOCK', 'GOOGLE INC', 'GOOGL'],
        'TSLA': ['TESLA', 'ELON MUSK', 'TESLA STOCK', 'CYBERTRUCK', 'MODEL', 'EV', 'TSLA']
    }
    
    for ticker, terms in company_terms.items():
        print(f"\n   Searching for {ticker} terms: {terms}")
        
        for col in available_text_cols:
            if col in reddit.columns:
                mask = pd.Series([False] * len(reddit))
                
                for term in terms:
                    mask = mask | reddit[col].astype(str).str.upper().str.contains(term, na=False)
                
                count = mask.sum()
                if count > 0:
                    print(f"     • Found {count} mentions in {col}")
                    
                    # Show samples
                    samples = reddit[mask][col].head(3).tolist()
                    for i, sample in enumerate(samples):
                        print(f"       {i+1}. {str(sample)[:100]}...")

def fix_and_integrate_reddit():
    """Fix reddit integration and create final dataset"""
    print("\n" + "="*80)
    print("FIXING REDDIT INTEGRATION")
    print("="*80)
    
    # Load all data
    print("\n1. Loading all datasets...")
    stocks = pd.read_csv("./feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv")
    news = pd.read_csv("./feature_engineered_data/news_data/feature_engineered_outlier_treated_alpha_vantage_news_expanded.csv")
    reddit = pd.read_csv("./feature_engineered_data/reddit_data/feature_engineered_outlier_treated_all_reddit_posts_combined.csv")
    
    # Filter stocks
    tickers = ['AAPL', 'GOOGL', 'TSLA']
    stocks = stocks[stocks['ticker'].isin(tickers)].copy()
    
    print(f"   • Stocks: {stocks.shape}")
    print(f"   • News: {news.shape}")
    print(f"   • Reddit: {reddit.shape}")
    
    print("\n2. Fixing reddit date parsing...")
    # Try to find and parse date column
    reddit_date_col = None
    
    # Check common date column names
    possible_date_cols = ['post_date', 'post_datetime', 'created_utc', 'created', 'timestamp', 'date']
    
    for col in possible_date_cols:
        if col in reddit.columns:
            try:
                reddit['date'] = pd.to_datetime(reddit[col])
                reddit_date_col = col
                print(f"   ✓ Using date column: {col}")
                print(f"     Sample: {reddit['date'].iloc[0]}")
                break
            except:
                continue
    
    if reddit_date_col is None:
        print(f"   ⚠️ Could not find valid date column, checking all columns...")
        
        # Check first few rows of all columns
        for col in reddit.columns[:20]:  # Check first 20 columns
            sample = reddit[col].iloc[0] if len(reddit) > 0 else None
            print(f"   {col}: {sample}")
            
            # Check if it looks like a date
            if isinstance(sample, str) and ('202' in sample or '202' in str(sample)):
                try:
                    reddit['date'] = pd.to_datetime(reddit[col])
                    reddit_date_col = col
                    print(f"   ✓ Found date column: {col}")
                    break
                except:
                    pass
    
    if reddit_date_col:
        reddit['date_only'] = reddit['date'].dt.date
        print(f"   Date range: {reddit['date_only'].min()} to {reddit['date_only'].max()}")
    else:
        print(f"   ❌ Could not parse reddit dates, using placeholder")
        reddit['date_only'] = pd.to_datetime('2023-01-01').date()
    
    print("\n3. Processing reddit mentions...")
    
    # Company search terms
    company_search_terms = {
        'AAPL': ['APPLE', 'IPHONE', 'MAC', 'IPAD', 'AAPL', 'APPLE STOCK'],
        'GOOGL': ['GOOGLE', 'ALPHABET', 'GOOGL', 'GOOGLE STOCK'],
        'TSLA': ['TESLA', 'ELON MUSK', 'TSLA', 'TESLA STOCK', 'CYBERTRUCK']
    }
    
    reddit_results = {}
    
    for ticker, terms in company_search_terms.items():
        print(f"\n   Searching reddit for {ticker}...")
        
        # Create search mask
        mask = pd.Series([False] * len(reddit))
        
        # Search in available text columns
        text_cols_to_search = []
        for col in ['title', 'content', 'selftext', 'text']:
            if col in reddit.columns:
                text_cols_to_search.append(col)
        
        if not text_cols_to_search:
            print(f"     ⚠️ No text columns found for search")
            continue
        
        for col in text_cols_to_search:
            for term in terms:
                try:
                    col_mask = reddit[col].astype(str).str.upper().str.contains(term, na=False)
                    mask = mask | col_mask
                except:
                    pass
        
        ticker_reddit = reddit[mask].copy()
        
        if len(ticker_reddit) > 0:
            print(f"     ✓ Found {len(ticker_reddit)} mentions")
            
            # Create daily aggregates
            if 'date_only' in ticker_reddit.columns:
                daily_agg = ticker_reddit.groupby('date_only').agg({
                    'title': 'count'
                }).reset_index()
                
                daily_agg.columns = ['date', f'reddit_{ticker}_count']
                
                # Add some numeric aggregates if available
                numeric_cols = []
                for col in ['upvotes', 'comments', 'awards']:
                    if col in ticker_reddit.columns and pd.api.types.is_numeric_dtype(ticker_reddit[col]):
                        numeric_cols.append(col)
                
                for col in numeric_cols[:2]:  # Limit to 2 columns
                    col_stats = ticker_reddit.groupby('date_only')[col].agg(['mean', 'sum']).reset_index()
                    col_stats.columns = ['date', f'reddit_{ticker}_{col}_mean', f'reddit_{ticker}_{col}_sum']
                    daily_agg = pd.merge(daily_agg, col_stats, on='date', how='left')
                
                reddit_results[ticker] = daily_agg
                print(f"     Created {len(daily_agg)} daily aggregates")
                
                # Show sample
                print(f"     Sample dates: {daily_agg['date'].head(3).tolist()}")
        else:
            print(f"     No mentions found")
    
    print("\n4. Creating final integrated dataset...")
    
    # Prepare stock data
    stocks['stock_date'] = pd.to_datetime(stocks['Date']).dt.date
    
    # Prepare news data
    news['news_date'] = pd.to_datetime(news['published_at']).dt.date
    
    all_integrated = []
    
    for ticker in tickers:
        print(f"\n   Processing {ticker}...")
        
        # Get stock data
        ticker_stocks = stocks[stocks['ticker'] == ticker].copy()
        ticker_stocks = ticker_stocks.sort_values('stock_date')
        
        # Add date column
        ticker_stocks['date'] = ticker_stocks['stock_date']
        
        print(f"     • Stock records: {len(ticker_stocks)}")
        
        # Add news data
        news_mask = (
            news['tickers'].astype(str).str.upper().str.contains(ticker, na=False) |
            news['title'].astype(str).str.upper().str.contains(ticker, na=False)
        )
        
        ticker_news = news[news_mask].copy()
        
        if len(ticker_news) > 0:
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
            print(f"     • News integrated: 0 records")
        
        # Add reddit data
        if ticker in reddit_results:
            reddit_agg = reddit_results[ticker]
            
            # Ensure date is date type
            reddit_agg['date'] = pd.to_datetime(reddit_agg['date']).dt.date
            
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
                print(f"     • Reddit integrated: 0 records")
        else:
            print(f"     • Reddit integrated: 0 records")
        
        # Clean up
        cols_to_drop = ['stock_date', 'Date', 'news_date']
        for col in cols_to_drop:
            if col in ticker_stocks.columns:
                ticker_stocks = ticker_stocks.drop(col, axis=1)
        
        all_integrated.append(ticker_stocks)
    
    # Combine all tickers
    if all_integrated:
        final_df = pd.concat(all_integrated, ignore_index=True)
        
        print(f"\n✅ FINAL INTEGRATED DATASET CREATED")
        print(f"   • Total records: {len(final_df)}")
        print(f"   • Total columns: {len(final_df.columns)}")
        
        # Count external columns
        news_cols = [col for col in final_df.columns if 'news_' in col]
        reddit_cols = [col for col in final_df.columns if 'reddit_' in col]
        
        print(f"   • News columns: {len(news_cols)}")
        print(f"   • Reddit columns: {len(reddit_cols)}")
        
        # Save the dataset
        output_dir = "./final_integrated_dataset"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"final_integrated_{timestamp}.csv"
        filepath = os.path.join(output_dir, filename)
        
        final_df.to_csv(filepath, index=False)
        
        print(f"\n💾 Dataset saved to: {filepath}")
        
        # Show integration statistics
        print(f"\n📊 INTEGRATION STATISTICS:")
        
        for ticker in tickers:
            ticker_data = final_df[final_df['ticker'] == ticker]
            total = len(ticker_data)
            
            # News stats
            news_col = f'news_{ticker}_count'
            if news_col in final_df.columns:
                news_with_data = ticker_data[news_col].notna().sum()
                print(f"   {ticker}:")
                print(f"     • News: {news_with_data}/{total} records ({news_with_data/total*100:.1f}%)")
            
            # Reddit stats
            reddit_col = f'reddit_{ticker}_count'
            if reddit_col in final_df.columns:
                reddit_with_data = ticker_data[reddit_col].notna().sum()
                print(f"     • Reddit: {reddit_with_data}/{total} records ({reddit_with_data/total*100:.1f}%)")
        
        # Show sample
        print(f"\n👀 SAMPLE OF INTEGRATED DATA:")
        sample_cols = ['date', 'ticker', 'Open', 'Close']
        
        # Add some external columns
        for ticker in tickers[:1]:  # Just show for first ticker
            external_cols = [col for col in final_df.columns if ticker in col]
            sample_cols.extend(external_cols[:3])
        
        sample = final_df[sample_cols].head(10)
        print(sample.to_string())
        
        return final_df, filepath
    
    return None, None

def create_performance_comparison():
    """Create performance comparison for presentation"""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    
    # Load the latest integrated dataset
    integrated_dir = "./final_integrated_dataset"
    
    if not os.path.exists(integrated_dir):
        print(f"❌ Directory not found: {integrated_dir}")
        return
    
    # Find the latest file
    integrated_files = [f for f in os.listdir(integrated_dir) 
                      if f.startswith('final_integrated_') and f.endswith('.csv')]
    
    if not integrated_files:
        print(f"❌ No integrated files found")
        return
    
    latest_file = sorted(integrated_files)[-1]
    filepath = os.path.join(integrated_dir, latest_file)
    
    print(f"\n📂 Loading: {latest_file}")
    df = pd.read_csv(filepath)
    
    # Convert date
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    print(f"   Loaded: {df.shape}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Run analysis for each ticker
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
        
        # Prepare feature sets
        all_features = [col for col in ticker_data.columns 
                       if col not in ['date', 'ticker'] 
                       and pd.api.types.is_numeric_dtype(ticker_data[col])]
        
        # Stock-only features (exclude external data)
        stock_features = [col for col in all_features 
                         if not any(x in col for x in ['news_', 'reddit_'])]
        
        # Integrated features (all features)
        integrated_features = all_features
        
        # External features
        external_features = [col for col in all_features 
                           if any(x in col for x in ['news_', 'reddit_'])]
        
        print(f"   Features:")
        print(f"     • Stock-only: {len(stock_features)}")
        print(f"     • Integrated: {len(integrated_features)}")
        print(f"     • External: {len(external_features)}")
        
        if external_features:
            print(f"     • External features: {external_features[:3]}")
            if len(external_features) > 3:
                print(f"       ... and {len(external_features) - 3} more")
        
        # Run models
        stocks_result = run_prediction_model(ticker_data, stock_features, f"{ticker} Stocks-Only")
        integrated_result = run_prediction_model(ticker_data, integrated_features, f"{ticker} Integrated")
        
        if stocks_result and integrated_result:
            # Calculate improvements
            mse_imp = ((stocks_result['mse'] - integrated_result['mse']) / stocks_result['mse']) * 100
            r2_imp = ((integrated_result['r2'] - stocks_result['r2']) / abs(stocks_result['r2'] + 1e-10)) * 100
            
            results.append({
                'ticker': ticker,
                'stocks_mse': stocks_result['mse'],
                'integrated_mse': integrated_result['mse'],
                'mse_improvement_pct': mse_imp,
                'stocks_r2': stocks_result['r2'],
                'integrated_r2': integrated_result['r2'],
                'r2_improvement_pct': r2_imp,
                'stocks_mae': stocks_result.get('mae', 0),
                'integrated_mae': integrated_result.get('mae', 0),
                'stock_features': len(stock_features),
                'integrated_features': len(integrated_features),
                'external_features': len(external_features)
            })
            
            print(f"\n   📊 RESULTS:")
            print(f"     {'Metric':<10} {'Stocks-Only':<12} {'Integrated':<12} {'Improvement':<10}")
            print(f"     {'-'*45}")
            
            metrics = [
                ('MSE', stocks_result['mse'], integrated_result['mse'], mse_imp, True),
                ('R²', stocks_result['r2'], integrated_result['r2'], r2_imp, False)
            ]
            
            for name, s_val, i_val, imp, lower_is_better in metrics:
                arrow = "↓" if (i_val < s_val if lower_is_better else i_val > s_val) else "↑"
                print(f"     {name:<10} {s_val:<12.6f} {i_val:<12.6f} {arrow} {abs(imp):.1f}%")
    
    # Create final summary
    if results:
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        
        results_df = pd.DataFrame(results)
        
        # Save results
        results_dir = "./performance_results"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(results_dir, f"performance_comparison_{timestamp}.csv")
        results_df.to_csv(results_file, index=False)
        
        print(f"\n📄 Results saved to: {results_file}")
        
        # Calculate averages
        avg_mse_imp = results_df['mse_improvement_pct'].mean()
        avg_r2_imp = results_df['r2_improvement_pct'].mean()
        
        print(f"\n📊 AVERAGE IMPROVEMENT:")
        print(f"   • MSE: {avg_mse_imp:+.1f}%")
        print(f"   • R²:  {avg_r2_imp:+.1f}%")
        
        improved_tickers = results_df[results_df['mse_improvement_pct'] > 0]['ticker'].tolist()
        print(f"   • Tickers improved: {len(improved_tickers)}/{len(results_df)}")
        
        print(f"\n💡 CONCLUSION:")
        if avg_mse_imp > 5 or avg_r2_imp > 5:
            print("   ✅ Data integration provides SIGNIFICANT improvement")
        elif avg_mse_imp > 0 or avg_r2_imp > 0:
            print("   ⚠️ Data integration provides MODEST improvement")
        else:
            print("   ❌ Data integration does NOT improve predictions")
        
        print(f"\n🎯 FOR YOUR PRESENTATION:")
        print(f"   1. Show successful integration of stock + news + reddit data")
        print(f"   2. Present performance comparison results")
        print(f"   3. Explain data limitations (short news/reddit timeframe)")
        print(f"   4. Demonstrate methodology works for larger datasets")

def run_prediction_model(df, feature_cols, model_name):
    """Run prediction model and return metrics"""
    try:
        # Create target: next day return
        df_temp = df.copy()
        df_temp['target'] = df_temp['Close'].pct_change().shift(-1)
        df_temp = df_temp.dropna(subset=['target'])
        
        if len(df_temp) < 30:
            print(f"       ⚠️ Not enough data for {model_name}")
            return None
        
        # Prepare features
        X = df_temp[feature_cols].fillna(0).values
        y = df_temp['target'].values
        
        # Simple time-based split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train model
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"       {model_name}:")
        print(f"         • MSE: {mse:.6f}")
        print(f"         • MAE: {mae:.6f}")
        print(f"         • R²:  {r2:.4f}")
        print(f"         • Features: {len(feature_cols)}")
        
        return {'mse': mse, 'mae': mae, 'r2': r2}
        
    except Exception as e:
        print(f"       ❌ Error in {model_name}: {e}")
        return None

if __name__ == "__main__":
    # First debug the reddit data
    debug_reddit_data()
    
    # Then fix and create integrated dataset
    final_df, filepath = fix_and_integrate_reddit()
    
    # Finally run performance comparison
    if final_df is not None:
        create_performance_comparison()
        
        print("\n" + "="*80)
        print("✅ ALL DONE! READY FOR PRESENTATION")
        print("="*80)
        print("\n📁 Your files are in:")
        print("   • ./final_integrated_dataset/ - Final integrated dataset")
        print("   • ./performance_results/ - Performance comparison results")
