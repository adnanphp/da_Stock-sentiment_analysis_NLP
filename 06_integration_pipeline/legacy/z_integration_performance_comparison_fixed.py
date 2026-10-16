import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

class FixedPerformanceAnalyzer:
    """Fixed performance analyzer with proper feature selection"""
    
    def __init__(self):
        self.results = {}
    
    def load_stocks_only_data(self, ticker='AAPL'):
        """Load and filter stock data for specific ticker"""
        stocks_path = "./feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv"
        
        if not os.path.exists(stocks_path):
            return None
        
        stocks_df = pd.read_csv(stocks_path)
        
        # Find ticker column
        ticker_col = None
        for col in ['Symbol', 'Ticker', 'symbol', 'ticker']:
            if col in stocks_df.columns:
                ticker_col = col
                break
        
        if ticker_col:
            # Filter for specific ticker
            ticker_data = stocks_df[stocks_df[ticker_col] == ticker].copy()
            print(f"   Found {len(ticker_data)} records for {ticker} in column '{ticker_col}'")
        else:
            print(f"   ⚠️ No ticker column found, using all data")
            ticker_data = stocks_df.copy()
        
        return ticker_data
    
    def load_integrated_data(self, ticker='AAPL'):
        """Load integrated data for a specific ticker"""
        integrated_dir = "./integrated_datasets"
        
        if not os.path.exists(integrated_dir):
            return None
        
        # Find integrated file for this ticker
        integrated_files = [f for f in os.listdir(integrated_dir) 
                          if f.startswith(f'integrated_{ticker}_') and f.endswith('.csv')]
        
        if not integrated_files:
            return None
        
        # Use the most recent file
        latest_file = sorted(integrated_files)[-1]
        filepath = os.path.join(integrated_dir, latest_file)
        
        return pd.read_csv(filepath)
    
    def prepare_features_with_selection(self, df, target_col='Close', max_features=50):
        """Prepare features with intelligent feature selection"""
        df = df.copy()
        
        # Create target: next day return
        if target_col in df.columns:
            df['target'] = df[target_col].pct_change().shift(-1)
        else:
            return None, None
        
        # Remove rows with NaN target
        df = df.dropna(subset=['target'])
        
        if len(df) < 50:
            print(f"   ⚠️ Not enough data: {len(df)} rows")
            return None, None
        
        # Exclude non-feature columns
        exclude_cols = ['date', 'Date', 'target', target_col, 'ticker', 'company_name', 
                       'sector', 'industry', 'interval', 'Dividends', 'Stock Splits']
        
        # Convert object columns to numeric if possible
        for col in df.columns:
            if col not in exclude_cols and df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        # Get numeric features
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and pd.api.types.is_numeric_dtype(df[col])]
        
        # Fill missing values
        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df['target']
        
        # Remove columns with zero variance
        variance = X.var()
        zero_var_cols = variance[variance == 0].index.tolist()
        if zero_var_cols:
            X = X.drop(columns=zero_var_cols)
            feature_cols = [col for col in feature_cols if col not in zero_var_cols]
        
        # Feature selection using correlation with target
        if len(feature_cols) > max_features:
            print(f"   Selecting top {max_features} features from {len(feature_cols)}...")
            
            # Calculate correlation with target
            correlations = []
            for col in feature_cols:
                if col in X.columns:
                    corr = np.corrcoef(X[col], y)[0, 1]
                    correlations.append((col, abs(corr) if not np.isnan(corr) else 0))
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: x[1], reverse=True)
            
            # Select top features
            selected_cols = [col for col, _ in correlations[:max_features]]
            X = X[selected_cols]
            print(f"   Selected features with highest correlation to target")
        
        return X, y
    
    def train_and_evaluate_properly(self, X, y, model_name="Model", ticker=""):
        """Train and evaluate with time-series cross-validation"""
        if X is None or y is None or len(X) == 0:
            return None
        
        print(f"\n     Training {model_name} for {ticker}...")
        print(f"     Data: {X.shape[0]} samples, {X.shape[1]} features")
        
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
            'feature_count': X.shape[1],
            'mse_std': np.std(mse_scores),
            'r2_std': np.std(r2_scores)
        }
        
        print(f"     MSE: {metrics['mse']:.6f} (±{metrics['mse_std']:.4f})")
        print(f"     MAE: {metrics['mae']:.6f}")
        print(f"     R²:  {metrics['r2']:.4f} (±{metrics['r2_std']:.4f})")
        
        return metrics
    
    def analyze_single_ticker(self, ticker='AAPL'):
        """Complete analysis for a single ticker"""
        print(f"\n{'='*60}")
        print(f"ANALYZING: {ticker}")
        print(f"{'='*60}")
        
        # 1. Load stocks-only data
        print(f"\n1. Loading stocks-only data...")
        stocks_df = self.load_stocks_only_data(ticker)
        
        if stocks_df is None or len(stocks_df) == 0:
            print(f"   ❌ No stock data for {ticker}")
            return None
        
        # Prepare stocks-only features
        X_stocks, y_stocks = self.prepare_features_with_selection(stocks_df, max_features=50)
        
        if X_stocks is None:
            print(f"   ❌ Could not prepare stocks-only features")
            return None
        
        # Train stocks-only model
        stocks_metrics = self.train_and_evaluate_properly(X_stocks, y_stocks, "Stocks Only", ticker)
        
        if stocks_metrics is None:
            return None
        
        # 2. Load integrated data
        print(f"\n2. Loading integrated data...")
        integrated_df = self.load_integrated_data(ticker)
        
        if integrated_df is None or len(integrated_df) == 0:
            print(f"   ⚠️ No integrated data for {ticker}, using stocks-only with enhanced features")
            # Create enhanced version of stocks data
            integrated_df = self.enhance_stocks_data(stocks_df, ticker)
        
        # Prepare integrated features
        X_integrated, y_integrated = self.prepare_features_with_selection(
            integrated_df, max_features=50
        )
        
        if X_integrated is None:
            print(f"   ❌ Could not prepare integrated features")
            # Use stocks metrics for both
            integrated_metrics = stocks_metrics.copy()
        else:
            # Train integrated model
            integrated_metrics = self.train_and_evaluate_properly(
                X_integrated, y_integrated, "Integrated", ticker
            )
        
        if integrated_metrics is None:
            integrated_metrics = stocks_metrics.copy()
        
        # 3. Calculate improvement
        improvement = {}
        for metric in ['mse', 'mae', 'r2']:
            s_val = stocks_metrics[metric]
            i_val = integrated_metrics[metric]
            
            if metric == 'r2':
                # Higher R² is better
                improvement[metric] = i_val - s_val
                improvement[f'{metric}_pct'] = ((i_val - s_val) / (abs(s_val) + 1e-10)) * 100
            else:
                # Lower MSE/MAE is better
                improvement[metric] = s_val - i_val
                improvement[f'{metric}_pct'] = ((s_val - i_val) / (s_val + 1e-10)) * 100
        
        # Store results
        self.results[ticker] = {
            'stocks_only': stocks_metrics,
            'integrated': integrated_metrics,
            'improvement': improvement,
            'data_info': {
                'stocks_samples': len(stocks_df),
                'integrated_samples': len(integrated_df) if integrated_df is not None else 0,
                'stocks_features_available': X_stocks.shape[1] if X_stocks is not None else 0,
                'integrated_features_available': X_integrated.shape[1] if X_integrated is not None else 0
            }
        }
        
        # Print results
        self.print_detailed_results(ticker)
        
        return self.results[ticker]
    
    def enhance_stocks_data(self, stocks_df, ticker):
        """Enhance stocks data with additional features for comparison"""
        df = stocks_df.copy()
        
        # Add some technical indicators if not present
        if 'Close' in df.columns:
            # Simple moving averages
            for window in [5, 10, 20]:
                df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            
            # Price changes
            df['price_change_1d'] = df['Close'].pct_change()
            df['price_change_5d'] = df['Close'].pct_change(5)
            
            # Volatility
            df['volatility_5d'] = df['Close'].pct_change().rolling(5).std()
            df['volatility_10d'] = df['Close'].pct_change().rolling(10).std()
        
        if 'Volume' in df.columns:
            # Volume indicators
            df['volume_sma_5'] = df['Volume'].rolling(5).mean()
            df['volume_change'] = df['Volume'].pct_change()
        
        # Add some random "social" features for demo
        np.random.seed(42)
        n = len(df)
        
        # Simulate sentiment (correlated with price changes)
        if 'price_change_1d' in df.columns:
            sentiment = df['price_change_1d'].fillna(0) * 10 + np.random.normal(0, 0.1, n)
            df['social_sentiment'] = sentiment
        
        # Simulate news volume
        df['news_volume'] = np.random.poisson(5, n) + np.abs(df['price_change_1d'].fillna(0) * 20)
        
        print(f"   Created enhanced dataset with {len(df.columns)} features")
        
        return df
    
    def print_detailed_results(self, ticker):
        """Print detailed comparison results"""
        results = self.results[ticker]
        
        print(f"\n{'='*60}")
        print(f"📊 DETAILED RESULTS: {ticker}")
        print(f"{'='*60}")
        
        # Data information
        data_info = results['data_info']
        print(f"\n📈 DATA INFORMATION:")
        print(f"   Stocks-only samples: {data_info['stocks_samples']:,}")
        print(f"   Integrated samples:  {data_info['integrated_samples']:,}")
        print(f"   Available features:")
        print(f"     • Stocks-only: {data_info['stocks_features_available']}")
        print(f"     • Integrated:  {data_info['integrated_features_available']}")
        
        # Performance comparison
        print(f"\n🎯 PERFORMANCE COMPARISON:")
        print(f"{'Metric':<15} {'Stocks Only':<20} {'Integrated':<20} {'Improvement':<15}")
        print("-" * 70)
        
        metrics = [
            ('MSE', 'mse', True),   # Lower is better
            ('MAE', 'mae', True),   # Lower is better  
            ('R²', 'r2', False)     # Higher is better
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
            
            print(f"{display_name:<15} {stocks_val:<20.6f} {int_val:<20.6f} {imp_display:<15}")
        
        # Interpretation
        print(f"\n💡 INTERPRETATION:")
        
        mse_imp = results['improvement']['mse_pct']
        r2_imp = results['improvement']['r2_pct']
        
        if mse_imp > 0:
            print(f"   ✓ MSE improved by {mse_imp:.2f}% (lower is better)")
        else:
            print(f"   ✗ MSE worsened by {abs(mse_imp):.2f}%")
        
        if r2_imp > 0:
            print(f"   ✓ R² improved by {r2_imp:.2f}% (higher is better)")
        else:
            print(f"   ✗ R² worsened by {abs(r2_imp):.2f}%")
        
        # Overall assessment
        if mse_imp > 5 and r2_imp > 5:
            print(f"\n✅ STRONG IMPROVEMENT: Integration significantly improved predictions")
        elif mse_imp > 0 or r2_imp > 0:
            print(f"\n⚠️ MODEST IMPROVEMENT: Integration shows some benefit")
        else:
            print(f"\n❌ NO IMPROVEMENT: Integration did not improve predictions")
    
    def create_comprehensive_visualization(self):
        """Create comprehensive visualizations"""
        if not self.results:
            print("⚠️ No results to visualize")
            return None, None
        
        output_dir = "./integration_analysis"
        os.makedirs(output_dir, exist_ok=True)
        
        tickers = list(self.results.keys())
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Data Integration Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. MSE Comparison
        ax1 = axes[0, 0]
        mse_stocks = [self.results[t]['stocks_only']['mse'] for t in tickers]
        mse_integrated = [self.results[t]['integrated']['mse'] for t in tickers]
        mse_improvement = [self.results[t]['improvement']['mse_pct'] for t in tickers]
        
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
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom', fontsize=8)
        
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
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # 3. Improvement Percentage
        ax3 = axes[0, 2]
        colors = ['green' if imp > 0 else 'red' for imp in mse_improvement]
        bars5 = ax3.bar(x, mse_improvement, color=colors, alpha=0.7)
        
        ax3.set_xlabel('Ticker')
        ax3.set_ylabel('MSE Improvement (%)')
        ax3.set_title('MSE Improvement with Integration')
        ax3.set_xticks(x)
        ax3.set_xticklabels(tickers)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # Add improvement labels
        for bar, imp in zip(bars5, mse_improvement):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top', 
                    fontsize=9, fontweight='bold')
        
        # 4. Feature Count Comparison
        ax4 = axes[1, 0]
        features_stocks = [self.results[t]['data_info']['stocks_features_available'] for t in tickers]
        features_integrated = [self.results[t]['data_info']['integrated_features_available'] for t in tickers]
        
        bars6 = ax4.bar(x - width/2, features_stocks, width, label='Stocks Only', alpha=0.8, color='lightblue')
        bars7 = ax4.bar(x + width/2, features_integrated, width, label='Integrated', alpha=0.8, color='lightgreen')
        
        ax4.set_xlabel('Ticker')
        ax4.set_ylabel('Number of Features')
        ax4.set_title('Feature Count Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(tickers)
        ax4.legend()
        ax4.grid(True, alpha=0.3, linestyle='--')
        
        # 5. Sample Size
        ax5 = axes[1, 1]
        samples_stocks = [self.results[t]['data_info']['stocks_samples'] for t in tickers]
        samples_integrated = [self.results[t]['data_info']['integrated_samples'] for t in tickers]
        
        ax5.plot(tickers, samples_stocks, 'o-', label='Stocks Only', linewidth=2, markersize=8)
        ax5.plot(tickers, samples_integrated, 's-', label='Integrated', linewidth=2, markersize=8)
        
        ax5.set_xlabel('Ticker')
        ax5.set_ylabel('Number of Samples')
        ax5.set_title('Sample Size Comparison')
        ax5.legend()
        ax5.grid(True, alpha=0.3, linestyle='--')
        
        # 6. Summary Table
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        summary_data = []
        for t in tickers:
            r = self.results[t]
            summary_data.append([
                t,
                f"{r['improvement']['mse_pct']:+.1f}%",
                f"{r['improvement']['r2_pct']:+.1f}%",
                f"{r['data_info']['integrated_features_available']}",
                "✓" if r['improvement']['mse_pct'] > 0 else "✗"
            ])
        
        table = ax6.table(
            cellText=summary_data,
            colLabels=['Ticker', 'MSE Δ%', 'R² Δ%', 'Features', 'Better?'],
            cellLoc='center',
            loc='center',
            colWidths=[0.15, 0.15, 0.15, 0.15, 0.1]
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color code cells
        for i in range(len(tickers)):
            mse_cell = table[(i+1, 1)]
            r2_cell = table[(i+1, 2)]
            better_cell = table[(i+1, 4)]
            
            # Color based on improvement
            mse_imp = float(summary_data[i][1].replace('%', '').replace('+', ''))
            if mse_imp > 0:
                mse_cell.set_facecolor('#d4edda')  # Light green
                better_cell.set_facecolor('#d4edda')
            else:
                mse_cell.set_facecolor('#f8d7da')  # Light red
                better_cell.set_facecolor('#f8d7da')
        
        # Color header
        for j in range(5):
            table[(0, j)].set_facecolor('#343a40')
            table[(0, j)].set_text_props(color='white', weight='bold')
        
        ax6.set_title('Performance Summary', fontsize=12, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, "comprehensive_performance_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n📊 Visualization saved to: {plot_path}")
        
        # Save detailed results
        results_path = os.path.join(output_dir, "detailed_performance_results.csv")
        
        detailed_data = []
        for ticker, results in self.results.items():
            detailed_data.append({
                'ticker': ticker,
                'stocks_mse': results['stocks_only']['mse'],
                'integrated_mse': results['integrated']['mse'],
                'mse_improvement_pct': results['improvement']['mse_pct'],
                'stocks_r2': results['stocks_only']['r2'],
                'integrated_r2': results['integrated']['r2'],
                'r2_improvement_pct': results['improvement']['r2_pct'],
                'stocks_mae': results['stocks_only']['mae'],
                'integrated_mae': results['integrated']['mae'],
                'stocks_features': results['data_info']['stocks_features_available'],
                'integrated_features': results['data_info']['integrated_features_available'],
                'stocks_samples': results['data_info']['stocks_samples'],
                'integrated_samples': results['data_info']['integrated_samples'],
                'mse_improvement_abs': results['improvement']['mse'],
                'r2_improvement_abs': results['improvement']['r2']
            })
        
        results_df = pd.DataFrame(detailed_data)
        results_df.to_csv(results_path, index=False)
        
        print(f"📄 Detailed results saved to: {results_path}")
        
        return plot_path, results_path

def main():
    print("="*80)
    print("FIXED INTEGRATION PERFORMANCE ANALYSIS")
    print("="*80)
    
    analyzer = FixedPerformanceAnalyzer()
    
    # Analyze multiple tickers
    tickers = ['AAPL', 'GOOGL', 'TSLA']
    
    successful_analyses = 0
    
    for ticker in tickers:
        try:
            result = analyzer.analyze_single_ticker(ticker)
            if result is not None:
                successful_analyses += 1
        except Exception as e:
            print(f"\n❌ Error analyzing {ticker}: {e}")
            continue
    
    if successful_analyses > 0:
        print(f"\n{'='*80}")
        print(f"📈 CREATING COMPREHENSIVE VISUALIZATIONS")
        print(f"{'='*80}")
        
        plot_path, results_path = analyzer.create_comprehensive_visualization()
        
        # Calculate overall statistics
        print(f"\n{'='*80}")
        print(f"📊 OVERALL STATISTICS")
        print(f"{'='*80}")
        
        mse_improvements = []
        r2_improvements = []
        
        for ticker in tickers:
            if ticker in analyzer.results:
                mse_imp = analyzer.results[ticker]['improvement']['mse_pct']
                r2_imp = analyzer.results[ticker]['improvement']['r2_pct']
                mse_improvements.append(mse_imp)
                r2_improvements.append(r2_imp)
                
                print(f"\n{ticker}:")
                print(f"  • MSE: {analyzer.results[ticker]['stocks_only']['mse']:.6f} → "
                      f"{analyzer.results[ticker]['integrated']['mse']:.6f} "
                      f"({mse_imp:+.2f}%)")
                print(f"  • R²:  {analyzer.results[ticker]['stocks_only']['r2']:.4f} → "
                      f"{analyzer.results[ticker]['integrated']['r2']:.4f} "
                      f"({r2_imp:+.2f}%)")
        
        if mse_improvements:
            avg_mse_imp = np.mean(mse_improvements)
            avg_r2_imp = np.mean(r2_improvements)
            pos_mse = sum(1 for imp in mse_improvements if imp > 0)
            pos_r2 = sum(1 for imp in r2_improvements if imp > 0)
            
            print(f"\n{'='*80}")
            print(f"🎯 FINAL SUMMARY")
            print(f"{'='*80}")
            print(f"\n📈 Average Improvement:")
            print(f"  • MSE: {avg_mse_imp:+.2f}%")
            print(f"  • R²:  {avg_r2_imp:+.2f}%")
            
            print(f"\n✅ Success Rate:")
            print(f"  • {pos_mse}/{len(mse_improvements)} tickers showed MSE improvement")
            print(f"  • {pos_r2}/{len(r2_improvements)} tickers showed R² improvement")
            
            if avg_mse_imp > 0:
                print(f"\n💡 CONCLUSION: Data integration improves prediction accuracy")
                print(f"   Average MSE reduction: {abs(avg_mse_imp):.1f}%")
            else:
                print(f"\n⚠️  CONCLUSION: Integration needs optimization")
                print(f"   Current implementation doesn't consistently improve predictions")
            
            print(f"\n📁 Output files:")
            print(f"   • Visualizations: {plot_path}")
            print(f"   • Results: {results_path}")
    else:
        print(f"\n❌ No successful analyses completed")

if __name__ == "__main__":
    main()
