"""
z_create_final_integrated_dataset.py - Create final properly integrated dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def create_final_dataset():
    """Create the final integrated dataset with proper features"""
    print("="*80)
    print("CREATING FINAL INTEGRATED DATASET")
    print("="*80)
    
    # Load the integrated dataset we just created
    integrated_file = "./fully_integrated/fully_integrated_20251202_221327.csv"
    df = pd.read_csv(integrated_file)
    
    print(f"📂 Loaded: {df.shape}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Tickers: {df['ticker'].unique().tolist()}")
    
    # Create better features from external data
    print("\n📊 CREATING ENHANCED FEATURES...")
    
    # For each ticker, create ticker-specific external features
    tickers = ['AAPL', 'GOOGL', 'TSLA']
    
    for ticker in tickers:
        print(f"   Enhancing {ticker} features...")
        
        ticker_data = df[df['ticker'] == ticker].copy()
        
        # Create combined external activity score
        external_features = []
        
        # News features
        news_count_col = f'news_{ticker}_count'
        news_sentiment_col = f'news_{ticker}_sentiment'
        
        if news_count_col in df.columns:
            # Normalize news count
            df[f'{ticker}_news_activity'] = df[news_count_col].fillna(0)
            external_features.append(f'{ticker}_news_activity')
            
            # Use sentiment if available
            if news_sentiment_col in df.columns:
                df[f'{ticker}_news_sentiment'] = df[news_sentiment_col].fillna(0)
                external_features.append(f'{ticker}_news_sentiment')
        
        # Reddit features - make them ticker-specific
        reddit_count_col = f'reddit_{ticker}_post_count'
        if reddit_count_col in df.columns:
            # Normalize reddit activity
            max_reddit = df[reddit_count_col].max()
            if max_reddit > 0:
                df[f'{ticker}_reddit_activity'] = df[reddit_count_col].fillna(0) / max_reddit
            else:
                df[f'{ticker}_reddit_activity'] = df[reddit_count_col].fillna(0)
            
            external_features.append(f'{ticker}_reddit_activity')
        
        print(f"     Created {len(external_features)} enhanced features")
    
    # Create target variable for prediction
    print("\n🎯 CREATING PREDICTION TARGET...")
    df = df.sort_values(['ticker', 'date'])
    
    # Create next day return as target
    df['target'] = df.groupby('ticker')['Close'].transform(lambda x: x.pct_change().shift(-1))
    
    # Remove rows without target
    df = df.dropna(subset=['target'])
    
    print(f"   Target created: {df['target'].notna().sum()} records with target")
    
    # Save final dataset
    output_dir = "./final_dataset"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_file = os.path.join(output_dir, f"final_integrated_dataset_{timestamp}.csv")
    df.to_csv(final_file, index=False)
    
    print(f"\n💾 FINAL DATASET SAVED TO: {final_file}")
    print(f"   Records: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    
    # Create summary
    print(f"\n📊 DATASET SUMMARY:")
    print(f"   Stock features: ~2,671 columns")
    print(f"   External features: Created enhanced ticker-specific features")
    print(f"   Target variable: 'target' (next day return)")
    
    # Show sample with new features
    print(f"\n👀 SAMPLE WITH ENHANCED FEATURES:")
    sample_cols = ['date', 'ticker', 'Open', 'Close', 'target']
    
    # Add some enhanced features
    for ticker in tickers[:1]:  # Show for AAPL
        enhanced_cols = [col for col in df.columns if ticker in col and ('news' in col or 'reddit' in col)]
        sample_cols.extend(enhanced_cols[:3])
    
    sample = df[sample_cols].head(10)
    print(sample.to_string())
    
    return df, final_file

def run_final_analysis_with_better_features(df_file):
    """Run analysis with better feature engineering"""
    print("\n" + "="*80)
    print("FINAL ANALYSIS WITH ENHANCED FEATURES")
    print("="*80)
    
    df = pd.read_csv(df_file)
    
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    
    tickers = ['AAPL', 'GOOGL', 'TSLA']
    results = []
    
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"ANALYZING {ticker} WITH ENHANCED FEATURES")
        print(f"{'='*60}")
        
        ticker_data = df[df['ticker'] == ticker].copy()
        
        if len(ticker_data) < 50:
            print(f"   ⚠️ Not enough data")
            continue
        
        # Select features
        # Stock features (basic price/volume)
        basic_stock_features = ['Open', 'High', 'Low', 'Close', 'Volume']
        basic_stock_features = [col for col in basic_stock_features if col in ticker_data.columns]
        
        # Enhanced stock features (select top correlated)
        all_stock_features = [col for col in ticker_data.columns 
                            if col not in ['date', 'ticker', 'target'] 
                            and pd.api.types.is_numeric_dtype(ticker_data[col])
                            and not any(x in col for x in ['news_', 'reddit_', 'AAPL', 'GOOGL', 'TSLA'])]
        
        # Select top 20 stock features by correlation with target
        if len(all_stock_features) > 20:
            correlations = ticker_data[all_stock_features].corrwith(ticker_data['target']).abs()
            top_stock_features = correlations.nlargest(20).index.tolist()
        else:
            top_stock_features = all_stock_features
        
        # External features (our enhanced features)
        external_features = [col for col in ticker_data.columns 
                           if any(x in col for x in [f'{ticker}_news', f'{ticker}_reddit'])]
        
        # Feature sets
        stocks_only_features = basic_stock_features + top_stock_features
        integrated_features = stocks_only_features + external_features
        
        print(f"   Features:")
        print(f"     • Basic stock: {len(basic_stock_features)}")
        print(f"     • Enhanced stock: {len(top_stock_features)}")
        print(f"     • External: {len(external_features)}")
        print(f"     • Total integrated: {len(integrated_features)}")
        
        # Prepare data
        X_stock = ticker_data[stocks_only_features].fillna(0).values
        X_int = ticker_data[integrated_features].fillna(0).values
        y = ticker_data['target'].values
        
        # Time-based split
        split = int(len(X_stock) * 0.8)
        
        X_stock_train, X_stock_test = X_stock[:split], X_stock[split:]
        X_int_train, X_int_test = X_int[:split], X_int[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Train models
        model_stock = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model_stock.fit(X_stock_train, y_train)
        
        model_int = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model_int.fit(X_int_train, y_train)
        
        # Predict and evaluate
        y_pred_stock = model_stock.predict(X_stock_test)
        y_pred_int = model_int.predict(X_int_test)
        
        mse_stock = mean_squared_error(y_test, y_pred_stock)
        mse_int = mean_squared_error(y_test, y_pred_int)
        
        r2_stock = r2_score(y_test, y_pred_stock)
        r2_int = r2_score(y_test, y_pred_int)
        
        # Calculate improvements
        mse_imp = ((mse_stock - mse_int) / mse_stock) * 100 if mse_stock != 0 else 0
        r2_imp = ((r2_int - r2_stock) / abs(r2_stock + 1e-10)) * 100
        
        print(f"\n   📊 RESULTS:")
        print(f"     {'Model':<15} {'MSE':<15} {'R²':<15}")
        print(f"     {'-'*45}")
        print(f"     {'Stocks-only':<15} {mse_stock:<15.6f} {r2_stock:<15.4f}")
        print(f"     {'Integrated':<15} {mse_int:<15.6f} {r2_int:<15.4f}")
        print(f"\n     Improvement:      {mse_imp:+.1f}%           {r2_imp:+.1f}%")
        
        results.append({
            'ticker': ticker,
            'stocks_mse': mse_stock,
            'integrated_mse': mse_int,
            'mse_improvement': mse_imp,
            'stocks_r2': r2_stock,
            'integrated_r2': r2_int,
            'r2_improvement': r2_imp,
            'external_features': len(external_features),
            'has_external_data': len(external_features) > 0
        })
    
    # Final report
    if results:
        print(f"\n{'='*80}")
        print("FINAL REPORT")
        print(f"{'='*80}")
        
        results_df = pd.DataFrame(results)
        
        # Save report
        report_dir = "./final_report"
        os.makedirs(report_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(report_dir, f"final_analysis_report_{timestamp}.csv")
        results_df.to_csv(report_file, index=False)
        
        print(f"\n📄 Report saved to: {report_file}")
        
        # Summary
        avg_mse_imp = results_df['mse_improvement'].mean()
        avg_r2_imp = results_df['r2_improvement'].mean()
        
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   • Average MSE improvement: {avg_mse_imp:+.1f}%")
        print(f"   • Average R² improvement: {avg_r2_imp:+.1f}%")
        
        tickers_improved = results_df[results_df['mse_improvement'] > 0]['ticker'].tolist()
        print(f"   • Tickers with improved MSE: {len(tickers_improved)}/{len(results_df)}")
        if tickers_improved:
            print(f"     {', '.join(tickers_improved)}")
        
        print(f"\n✅ CONCLUSION:")
        print(f"   You have successfully created an integrated dataset with:")
        print(f"   1. Stock data for AAPL, GOOGL, TSLA")
        print(f"   2. News sentiment and volume data")
        print(f"   3. Reddit activity metrics")
        print(f"   4. Proper target variable for prediction")
        
        print(f"\n📁 Your final files are in:")
        print(f"   • ./final_dataset/ - Final integrated dataset")
        print(f"   • ./final_report/ - Analysis report")

if __name__ == "__main__":
    # Create final enhanced dataset
    df, final_file = create_final_dataset()
    
    # Run final analysis
    run_final_analysis_with_better_features(final_file)
    
   
