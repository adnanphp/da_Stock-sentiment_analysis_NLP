import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

class PerformanceAnalyzerV2:
    """Performance analyzer for integrated dataset with ALL tickers"""
    
    def __init__(self):
        self.results = {}
    
    def load_integrated_dataset_all_tickers(self):
        """Load the integrated dataset with ALL tickers"""
        integrated_dir = "./integrated_datasets"
        
        if not os.path.exists(integrated_dir):
            return None
        
        # Find the ALL_TICKERS integrated file
        integrated_files = [f for f in os.listdir(integrated_dir) 
                          if f.startswith('integrated_ALL_TICKERS_') and f.endswith('.csv')]
        
        if not integrated_files:
            print("   ⚠️ No integrated dataset found. Looking for any integrated file...")
            integrated_files = [f for f in os.listdir(integrated_dir) 
                              if f.startswith('integrated_') and f.endswith('.csv')]
        
        if not integrated_files:
            return None
        
        # Use the most recent file
        latest_file = sorted(integrated_files)[-1]
        filepath = os.path.join(integrated_dir, latest_file)
        
        print(f"   Loading: {latest_file}")
        df = pd.read_csv(filepath)
        
        # Convert date if needed
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def analyze_all_tickers_from_single_dataset(self, tickers=['AAPL', 'GOOGL', 'TSLA']):
        """Analyze all tickers from a single integrated dataset"""
        print(f"\n{'='*60}")
        print(f"ANALYZING ALL TICKERS FROM INTEGRATED DATASET")
        print(f"{'='*60}")
        
        # Load the integrated dataset
        print(f"\n1. Loading integrated dataset...")
        integrated_df = self.load_integrated_dataset_all_tickers()
        
        if integrated_df is None or len(integrated_df) == 0:
            print("   ❌ Could not load integrated dataset")
            return None
        
        print(f"   Dataset shape: {integrated_df.shape}")
        print(f"   Columns: {len(integrated_df.columns)}")
        
        if 'ticker' in integrated_df.columns:
            print(f"   Tickers found: {integrated_df['ticker'].unique()}")
        else:
            print("   ⚠️ No ticker column found")
            return None
        
        # Analyze each ticker
        for ticker in tickers:
            print(f"\n{'='*60}")
            print(f"ANALYZING: {ticker}")
            print(f"{'='*60}")
            
            # Filter data for this ticker
            ticker_data = integrated_df[integrated_df['ticker'] == ticker].copy()
            
            if len(ticker_data) == 0:
                print(f"   ❌ No data for {ticker} in integrated dataset")
                continue
            
            print(f"   Records for {ticker}: {len(ticker_data)}")
            
            # Split into stocks-only and integrated data
            stocks_only_result = self.analyze_stocks_only(ticker_data, ticker)
            integrated_result = self.analyze_integrated(ticker_data, ticker)
            
            if stocks_only_result and integrated_result:
                # Calculate improvements
                improvement = self.calculate_improvements(stocks_only_result, integrated_result)
                
                # Store results
                self.results[ticker] = {
                    'stocks_only': stocks_only_result,
                    'integrated': integrated_result,
                    'improvement': improvement,
                    'data_info': {
                        'samples': len(ticker_data),
                        'date_range': f"{ticker_data['date'].min()} to {ticker_data['date'].max()}",
                        'stocks_features': stocks_only_result.get('feature_count', 0),
                        'integrated_features': integrated_result.get('feature_count', 0)
                    }
                }
                
                # Print results for this ticker
                self.print_ticker_results(ticker)
        
        return self.results
    
    def analyze_stocks_only(self, ticker_data, ticker_name):
        """Analyze stocks-only performance"""
        print(f"\n   📈 Stocks-only analysis for {ticker_name}...")
        
        # Create stocks-only dataset (remove news/reddit columns)
        stocks_cols = ['date', 'ticker'] + [col for col in ticker_data.columns 
                                          if not col.endswith('_news') and not col.endswith('_reddit')]
        stocks_data = ticker_data[stocks_cols].copy()
        
        # Prepare features
        X, y, features_used = self.prepare_features_for_analysis(stocks_data, ticker_name, max_features=50)
        
        if X is None or y is None:
            return None
        
        print(f"     Using {len(features_used)} stock features")
        
        # Train and evaluate
        metrics = self.train_and_evaluate_model(X, y, f"Stocks-Only {ticker_name}")
        
        if metrics:
            metrics['feature_count'] = len(features_used)
            return metrics
        
        return None
    
    def analyze_integrated(self, ticker_data, ticker_name):
        """Analyze integrated performance (stocks + news + reddit)"""
        print(f"\n   🔗 Integrated analysis for {ticker_name}...")
        
        # Prepare features from ALL data
        X, y, features_used = self.prepare_features_for_analysis(ticker_data, ticker_name, max_features=50)
        
        if X is None or y is None:
            return None
        
        # Count feature types
        stock_features = [f for f in features_used if not f.endswith('_news') and not f.endswith('_reddit')]
        news_features = [f for f in features_used if '_news_' in f]
        reddit_features = [f for f in features_used if '_reddit_' in f]
        
        print(f"     Using {len(features_used)} total features:")
        print(f"       • Stock: {len(stock_features)}")
        print(f"       • News: {len(news_features)}")
        print(f"       • Reddit: {len(reddit_features)}")
        
        # Train and evaluate
        metrics = self.train_and_evaluate_model(X, y, f"Integrated {ticker_name}")
        
        if metrics:
            metrics['feature_count'] = len(features_used)
            metrics['feature_breakdown'] = {
                'stock': len(stock_features),
                'news': len(news_features),
                'reddit': len(reddit_features)
            }
            return metrics
        
        return None
    
    def prepare_features_for_analysis(self, df, ticker_name, max_features=50):
        """Prepare features for analysis with intelligent selection"""
        df = df.copy()
        
        # Create target: next day return
        close_col = self.find_close_column(df)
        if not close_col:
            print(f"     ⚠️ Could not find Close price column for {ticker_name}")
            return None, None, None
        
        df['target'] = df[close_col].pct_change().shift(-1)
        
        # Remove rows with NaN target
        df = df.dropna(subset=['target'])
        
        if len(df) < 50:
            print(f"     ⚠️ Not enough data for {ticker_name}: {len(df)} rows")
            return None, None, None
        
        # Exclude non-feature columns
        exclude_cols = ['date', 'Date', 'target', 'ticker', 'company_name', 
                       'sector', 'industry', 'interval', 'Dividends', 'Stock Splits']
        
        # Add close column to exclude
        exclude_cols.append(close_col)
        
        # Get numeric features
        numeric_cols = []
        for col in df.columns:
            if col not in exclude_cols:
                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if pd.api.types.is_numeric_dtype(df[col]):
                        numeric_cols.append(col)
                except:
                    pass
        
        # Fill missing values
        X = df[numeric_cols].fillna(0)
        y = df['target']
        
        # Remove zero variance columns
        variance = X.var()
        zero_var_cols = variance[variance == 0].index.tolist()
        if zero_var_cols:
            X = X.drop(columns=zero_var_cols)
            numeric_cols = [col for col in numeric_cols if col not in zero_var_cols]
        
        # Feature selection using correlation with target
        if len(numeric_cols) > max_features:
            print(f"     Selecting top {max_features} features from {len(numeric_cols)}...")
            
            # Calculate correlation with target
            correlations = []
            for col in numeric_cols:
                if col in X.columns:
                    try:
                        corr = np.corrcoef(X[col], y.fillna(0))[0, 1]
                        correlations.append((col, abs(corr) if not np.isnan(corr) else 0))
                    except:
                        correlations.append((col, 0))
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: x[1], reverse=True)
            
            # Select top features
            selected_cols = [col for col, _ in correlations[:max_features]]
            X = X[selected_cols]
            numeric_cols = selected_cols
        
        return X, y, numeric_cols
    
    def find_close_column(self, df):
        """Find the Close price column"""
        close_candidates = ['Close', 'close', 'Adj Close', 'adj_close', 'price']
        for col in close_candidates:
            if col in df.columns:
                return col
        return None
    
    def train_and_evaluate_model(self, X, y, model_name):
        """Train and evaluate with time-series cross-validation"""
        if X is None or y is None or len(X) == 0:
            return None
        
        print(f"     Training {model_name}...")
        print(f"       Samples: {X.shape[0]}, Features: {X.shape[1]}")
        
        # Use time-series split for financial data
        tscv = TimeSeriesSplit(n_splits=5)
        
        mse_scores = []
        mae_scores = []
        r2_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse_scores.append(mean_squared_error(y_test, y_pred))
            mae_scores.append(mean_absolute_error(y_test, y_pred))
            r2_scores.append(r2_score(y_test, y_pred))
        
        # Average metrics across folds
        metrics = {
            'mse': np.mean(mse_scores),
            'mae': np.mean(mae_scores),
            'r2': np.mean(r2_scores),
            'mse_std': np.std(mse_scores),
            'mae_std': np.std(mae_scores),
            'r2_std': np.std(r2_scores)
        }
        
        print(f"       MSE: {metrics['mse']:.6f} (±{metrics['mse_std']:.4f})")
        print(f"       MAE: {metrics['mae']:.6f} (±{metrics['mae_std']:.4f})")
        print(f"       R²:  {metrics['r2']:.4f} (±{metrics['r2_std']:.4f})")
        
        return metrics
    
    def calculate_improvements(self, stocks_result, integrated_result):
        """Calculate improvement metrics"""
        improvement = {}
        
        for metric in ['mse', 'mae', 'r2']:
            s_val = stocks_result[metric]
            i_val = integrated_result[metric]
            
            if metric == 'r2':
                # Higher R² is better
                improvement[metric] = i_val - s_val
                improvement[f'{metric}_pct'] = ((i_val - s_val) / (abs(s_val) + 1e-10)) * 100
            else:
                # Lower MSE/MAE is better
                improvement[metric] = s_val - i_val
                improvement[f'{metric}_pct'] = ((s_val - i_val) / (s_val + 1e-10)) * 100
        
        return improvement
    
    def print_ticker_results(self, ticker):
        """Print results for a single ticker"""
        if ticker not in self.results:
            return
        
        results = self.results[ticker]
        
        print(f"\n   📊 RESULTS: {ticker}")
        print(f"   {'='*50}")
        
        # Performance comparison
        print(f"   {'Metric':<15} {'Stocks Only':<15} {'Integrated':<15} {'Improvement':<15}")
        print(f"   {'-'*60}")
        
        metrics = [
            ('MSE', 'mse', True),
            ('MAE', 'mae', True),
            ('R²', 'r2', False)
        ]
        
        for display_name, metric_name, lower_is_better in metrics:
            stocks_val = results['stocks_only'][metric_name]
            int_val = results['integrated'][metric_name]
            imp_pct = results['improvement'][f'{metric_name}_pct']
            
            if lower_is_better:
                arrow = "↓" if int_val < stocks_val else "↑"
                imp_display = f"{arrow} {abs(imp_pct):.2f}%"
            else:
                arrow = "↑" if int_val > stocks_val else "↓"
                imp_display = f"{arrow} {abs(imp_pct):.2f}%"
            
            print(f"   {display_name:<15} {stocks_val:<15.6f} {int_val:<15.6f} {imp_display:<15}")
        
        # Feature information
        if 'feature_breakdown' in results['integrated']:
            breakdown = results['integrated']['feature_breakdown']
            print(f"\n   📈 Feature breakdown:")
            print(f"     • Stock features: {results['stocks_only']['feature_count']}")
            print(f"     • News features: {breakdown.get('news', 0)}")
            print(f"     • Reddit features: {breakdown.get('reddit', 0)}")
    
    def create_comprehensive_report(self):
        """Create comprehensive report and visualizations"""
        if not self.results:
            print("⚠️ No results to report")
            return
        
        output_dir = "./integration_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        tickers = list(self.results.keys())
        
        # Create visualizations
        self.create_performance_charts(tickers, output_dir)
        
        # Create detailed report
        self.create_detailed_report(tickers, output_dir)
        
        # Print summary
        self.print_overall_summary(tickers)
    
    def create_performance_charts(self, tickers, output_dir):
        """Create performance comparison charts"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Integration Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. MSE Comparison
        ax1 = axes[0, 0]
        mse_stocks = [self.results[t]['stocks_only']['mse'] for t in tickers]
        mse_integrated = [self.results[t]['integrated']['mse'] for t in tickers]
        
        x = np.arange(len(tickers))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, mse_stocks, width, label='Stocks Only', alpha=0.8, color='skyblue')
        bars2 = ax1.bar(x + width/2, mse_integrated, width, label='Integrated', alpha=0.8, color='lightcoral')
        
        ax1.set_xlabel('Ticker')
        ax1.set_ylabel('Mean Squared Error')
        ax1.set_title('MSE Comparison (Lower is Better)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(tickers)
        ax1.legend()
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 2. R² Comparison
        ax2 = axes[0, 1]
        r2_stocks = [self.results[t]['stocks_only']['r2'] for t in tickers]
        r2_integrated = [self.results[t]['integrated']['r2'] for t in tickers]
        
        bars3 = ax2.bar(x - width/2, r2_stocks, width, label='Stocks Only', alpha=0.8, color='skyblue')
        bars4 = ax2.bar(x + width/2, r2_integrated, width, label='Integrated', alpha=0.8, color='lightcoral')
        
        ax2.set_xlabel('Ticker')
        ax2.set_ylabel('R² Score')
        ax2.set_title('R² Score Comparison (Higher is Better)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(tickers)
        ax2.legend()
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # 3. Improvement Percentage
        ax3 = axes[1, 0]
        mse_improvements = [self.results[t]['improvement']['mse_pct'] for t in tickers]
        r2_improvements = [self.results[t]['improvement']['r2_pct'] for t in tickers]
        
        x_axis = np.arange(len(tickers))
        ax3.bar(x_axis - 0.2, mse_improvements, 0.4, label='MSE Improvement', color='green', alpha=0.7)
        ax3.bar(x_axis + 0.2, r2_improvements, 0.4, label='R² Improvement', color='blue', alpha=0.7)
        
        ax3.set_xlabel('Ticker')
        ax3.set_ylabel('Improvement (%)')
        ax3.set_title('Improvement with Integration')
        ax3.set_xticks(x_axis)
        ax3.set_xticklabels(tickers)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.legend()
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # 4. Feature Comparison
        ax4 = axes[1, 1]
        stocks_features = [self.results[t]['stocks_only']['feature_count'] for t in tickers]
        integrated_features = [self.results[t]['integrated']['feature_count'] for t in tickers]
        
        bars5 = ax4.bar(x - width/2, stocks_features, width, label='Stocks Only', alpha=0.8, color='lightblue')
        bars6 = ax4.bar(x + width/2, integrated_features, width, label='Integrated', alpha=0.8, color='lightgreen')
        
        ax4.set_xlabel('Ticker')
        ax4.set_ylabel('Number of Features')
        ax4.set_title('Feature Count Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(tickers)
        ax4.legend()
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, "performance_comparison_all_tickers.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Charts saved to: {plot_path}")
    
    def create_detailed_report(self, tickers, output_dir):
        """Create detailed CSV report"""
        report_data = []
        
        for ticker in tickers:
            results = self.results[ticker]
            
            # Calculate if integration improved performance
            mse_improved = results['improvement']['mse_pct'] > 0
            r2_improved = results['improvement']['r2_pct'] > 0
            
            report_data.append({
                'ticker': ticker,
                'stocks_mse': results['stocks_only']['mse'],
                'integrated_mse': results['integrated']['mse'],
                'mse_improvement_pct': results['improvement']['mse_pct'],
                'mse_improved': mse_improved,
                'stocks_r2': results['stocks_only']['r2'],
                'integrated_r2': results['integrated']['r2'],
                'r2_improvement_pct': results['improvement']['r2_pct'],
                'r2_improved': r2_improved,
                'stocks_mae': results['stocks_only']['mae'],
                'integrated_mae': results['integrated']['mae'],
                'stocks_features': results['stocks_only']['feature_count'],
                'integrated_features': results['integrated']['feature_count'],
                'samples': results['data_info']['samples']
            })
        
        report_df = pd.DataFrame(report_data)
        report_path = os.path.join(output_dir, "detailed_performance_report_all_tickers.csv")
        report_df.to_csv(report_path, index=False)
        
        print(f"📄 Detailed report saved to: {report_path}")
        
        return report_df
    
    def print_overall_summary(self, tickers):
        """Print overall summary of results"""
        print(f"\n{'='*80}")
        print(f"🎯 OVERALL SUMMARY")
        print(f"{'='*80}")
        
        # Calculate statistics
        mse_improvements = []
        r2_improvements = []
        mse_improved_count = 0
        r2_improved_count = 0
        
        for ticker in tickers:
            if ticker in self.results:
                mse_imp = self.results[ticker]['improvement']['mse_pct']
                r2_imp = self.results[ticker]['improvement']['r2_pct']
                
                mse_improvements.append(mse_imp)
                r2_improvements.append(r2_imp)
                
                if mse_imp > 0:
                    mse_improved_count += 1
                if r2_imp > 0:
                    r2_improved_count += 1
        
        if mse_improvements:
            avg_mse_imp = np.mean(mse_improvements)
            avg_r2_imp = np.mean(r2_improvements)
            
            print(f"\n📈 AVERAGE IMPROVEMENT:")
            print(f"   • MSE: {avg_mse_imp:+.2f}%")
            print(f"   • R²:  {avg_r2_imp:+.2f}%")
            
            print(f"\n✅ SUCCESS RATE:")
            print(f"   • MSE improved for {mse_improved_count}/{len(tickers)} tickers")
            print(f"   • R² improved for {r2_improved_count}/{len(tickers)} tickers")
            
            # Conclusion
            if avg_mse_imp > 5 or avg_r2_imp > 5:
                print(f"\n🎉 CONCLUSION: Data integration shows STRONG improvement")
                print(f"   Average MSE reduction: {abs(avg_mse_imp):.1f}%")
                print(f"   Average R² increase: {abs(avg_r2_imp):.1f}%")
            elif avg_mse_imp > 0 or avg_r2_imp > 0:
                print(f"\n👍 CONCLUSION: Data integration shows MODEST improvement")
                print(f"   Further optimization could enhance results")
            else:
                print(f"\n⚠️  CONCLUSION: Integration needs optimization")
                print(f"   Consider: Better feature engineering, different models,")
                print(f"             or alternative data sources")

def main():
    print("="*80)
    print("PERFORMANCE ANALYSIS - INTEGRATED DATASET (ALL TICKERS)")
    print("="*80)
    
    analyzer = PerformanceAnalyzerV2()
    
    # Analyze all tickers from the integrated dataset
    tickers = ['AAPL', 'GOOGL', 'TSLA']
    results = analyzer.analyze_all_tickers_from_single_dataset(tickers=tickers)
    
    if results:
        print(f"\n{'='*80}")
        print(f"📊 CREATING COMPREHENSIVE REPORT")
        print(f"{'='*80}")
        
        analyzer.create_comprehensive_report()
        
        print(f"\n{'='*80}")
        print(f"✅ ANALYSIS COMPLETE")
        print(f"{'='*80}")
        
        print(f"\n📁 Output files in: ./integration_analysis/")
        print("   1. performance_comparison_all_tickers.png")
        print("   2. detailed_performance_report_all_tickers.csv")
    else:
        print(f"\n❌ No results generated. Check if integrated dataset exists.")

if __name__ == "__main__":
    main()
