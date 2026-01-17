import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import fft, fftfreq
import warnings
import os
warnings.filterwarnings('ignore')

class ProfessionalStockVisualizer:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.dfs = {}
        self.load_data()
        
    def load_data(self):
        """Load all CSV files"""
        for file_name, file_path in self.file_paths.items():
            try:
                print(f"Loading {file_name}...")
                self.dfs[file_name] = pd.read_csv(file_path)
                
                if 'Date' in self.dfs[file_name].columns:
                    self.dfs[file_name]['Date'] = pd.to_datetime(self.dfs[file_name]['Date'])
                    self.dfs[file_name] = self.dfs[file_name].sort_values('Date')
                    
                print(f"✓ Loaded {len(self.dfs[file_name]):,} rows")
                
            except Exception as e:
                print(f"✗ Error loading {file_name}: {e}")

    def create_fourier_analysis(self):
        """Create Fourier transform analysis to identify cycles and frequencies"""
        print("Creating Fourier analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Fourier Analysis - Identifying Price Cycles and Frequencies', fontsize=16)
        
        plot_count = 0
        for i, (file_name, df) in enumerate(self.dfs.items()):
            if i >= 4:
                break
                
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if 'Close' in df.columns and len(df) > 50:
                prices = df['Close'].values
                
                # Perform FFT
                yf = fft(prices - np.mean(prices))
                xf = fftfreq(len(prices), 1)
                
                # Plot power spectrum
                power_spectrum = np.abs(yf) ** 2
                positive_freq = xf > 0
                
                ax.semilogy(1/xf[positive_freq], power_spectrum[positive_freq])
                ax.set_xlabel('Period (days)')
                ax.set_ylabel('Power Spectrum')
                ax.set_title(f'{file_name.split(".")[0]}\nFourier Analysis')
                ax.grid(True, alpha=0.3)
                
                # Mark dominant frequencies
                dominant_idx = np.argsort(power_spectrum[positive_freq])[-3:]
                dominant_periods = 1/xf[positive_freq][dominant_idx]
                
                for period in dominant_periods:
                    if period < len(prices)/2 and period > 0:  # Reasonable periods
                        ax.axvline(period, color='red', linestyle='--', alpha=0.7)
                        ax.text(period, np.max(power_spectrum[positive_freq])/2, 
                               f'{period:.1f}d', rotation=90, ha='right')
                plot_count += 1
        
        if plot_count > 0:
            plt.tight_layout()
            plt.savefig('fourier_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("✓ Fourier analysis saved as 'fourier_analysis.png'")
        else:
            print("ℹ No data for Fourier analysis")

    def create_market_regime_analysis(self):
        """Identify different market regimes using clustering"""
        print("Creating market regime analysis...")
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            for file_name, df in self.dfs.items():
                if 'Close' in df.columns and len(df) > 100:
                    # Calculate features for regime detection
                    returns = df['Close'].pct_change().dropna()
                    volatility = returns.rolling(window=20).std()
                    momentum = df['Close'] / df['Close'].shift(20) - 1
                    
                    # Combine features
                    features = pd.DataFrame({
                        'returns': returns,
                        'volatility': volatility,
                        'momentum': momentum
                    }).dropna()
                    
                    if len(features) > 50:
                        # Scale features
                        scaler = StandardScaler()
                        scaled_features = scaler.fit_transform(features)
                        
                        # Cluster into regimes
                        kmeans = KMeans(n_clusters=3, random_state=42)
                        regimes = kmeans.fit_predict(scaled_features)
                        
                        # Plot
                        plt.figure(figsize=(15, 10))
                        
                        # Price with regime colors
                        plt.subplot(2, 1, 1)
                        colors = ['green', 'orange', 'red']
                        for i in range(3):
                            mask = regimes == i
                            regime_dates = features.index[mask]
                            plt.scatter(regime_dates, df.loc[regime_dates, 'Close'], 
                                      c=colors[i], label=f'Regime {i+1}', alpha=0.6, s=10)
                        
                        plt.plot(df['Date'], df['Close'], 'k-', alpha=0.3, linewidth=0.5)
                        plt.title(f'{file_name.split(".")[0]} - Market Regime Detection')
                        plt.ylabel('Price')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        # Feature space
                        plt.subplot(2, 1, 2)
                        scatter = plt.scatter(features['returns'], features['volatility'], 
                                            c=regimes, cmap='viridis', alpha=0.6)
                        plt.xlabel('Returns')
                        plt.ylabel('Volatility')
                        plt.title('Regime Clustering in Feature Space')
                        plt.colorbar(scatter, label='Regime')
                        plt.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        filename = f'market_regime_{file_name.split(".")[0]}.png'
                        plt.savefig(filename, dpi=300, bbox_inches='tight')
                        plt.show()
                        print(f"✓ Market regime analysis saved as '{filename}'")
                        
        except Exception as e:
            print(f"✗ Market regime analysis failed: {e}")

    def create_monte_carlo_simulation(self):
        """Create Monte Carlo simulations for future price paths"""
        print("Creating Monte Carlo simulations...")
        
        for file_name, df in self.dfs.items():
            if 'Close' in df.columns and len(df) > 100:
                # Calculate parameters from historical data
                returns = df['Close'].pct_change().dropna()
                mu = returns.mean()
                sigma = returns.std()
                
                # Monte Carlo parameters
                days = 252  # 1 year
                simulations = 100
                last_price = df['Close'].iloc[-1]
                
                # Generate simulations
                plt.figure(figsize=(15, 10))
                
                for i in range(simulations):
                    # Random walk using geometric brownian motion
                    price_path = [last_price]
                    for _ in range(days):
                        drift = mu - 0.5 * sigma**2
                        shock = sigma * np.random.normal()
                        new_price = price_path[-1] * np.exp(drift + shock)
                        price_path.append(new_price)
                    
                    plt.plot(range(len(price_path)), price_path, alpha=0.1, color='blue')
                
                # Calculate confidence intervals
                all_simulations = []
                for _ in range(1000):  # More simulations for CI
                    price_path = [last_price]
                    for _ in range(days):
                        drift = mu - 0.5 * sigma**2
                        shock = sigma * np.random.normal()
                        new_price = price_path[-1] * np.exp(drift + shock)
                        price_path.append(new_price)
                    all_simulations.append(price_path)
                
                all_simulations = np.array(all_simulations)
                ci_lower = np.percentile(all_simulations, 5, axis=0)
                ci_upper = np.percentile(all_simulations, 95, axis=0)
                
                plt.plot(range(days+1), ci_lower, 'r--', linewidth=2, label='5th Percentile')
                plt.plot(range(days+1), ci_upper, 'r--', linewidth=2, label='95th Percentile')
                plt.axhline(last_price, color='black', linestyle='-', linewidth=2, label='Current Price')
                
                plt.title(f'{file_name.split(".")[0]} - Monte Carlo Simulation (1 Year Forecast)')
                plt.xlabel('Trading Days Ahead')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                filename = f'monte_carlo_{file_name.split(".")[0]}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.show()
                print(f"✓ Monte Carlo simulation saved as '{filename}'")

    def create_fixed_volatility_analysis(self):
        """Fixed volatility analysis with proper data alignment"""
        print("Creating advanced volatility analysis...")
        
        for file_name, df in self.dfs.items():
            if 'Close' in df.columns and len(df) > 100:
                returns = df['Close'].pct_change().dropna()
                
                # Calculate volatility at different horizons
                horizons = [5, 10, 20, 50, 100]
                
                plt.figure(figsize=(15, 10))
                
                for horizon in horizons:
                    rolling_vol = returns.rolling(window=horizon).std() * np.sqrt(252)
                    
                    # Align dates with volatility data (remove NaN from rolling)
                    valid_data = rolling_vol.dropna()
                    valid_dates = df['Date'].iloc[valid_data.index]
                    
                    plt.plot(valid_dates, valid_data, label=f'{horizon}-day Volatility', alpha=0.8, linewidth=2)
                
                plt.title(f'{file_name.split(".")[0]} - Multi-Horizon Volatility Analysis')
                plt.xlabel('Date')
                plt.ylabel('Annualized Volatility')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                filename = f'volatility_analysis_{file_name.split(".")[0]}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                plt.show()
                print(f"✓ Volatility analysis saved as '{filename}'")

    def create_risk_metrics_dashboard(self):
        """Create comprehensive risk metrics dashboard"""
        print("Creating risk metrics dashboard...")
        
        risk_metrics = {}
        
        for file_name, df in self.dfs.items():
            if 'Close' in df.columns and len(df) > 100:
                returns = df['Close'].pct_change().dropna()
                
                if len(returns) > 0:
                    # Calculate various risk metrics
                    metrics = {
                        'Volatility (Annual)': returns.std() * np.sqrt(252),
                        'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                        'Max Drawdown': (df['Close'] / df['Close'].expanding().max() - 1).min(),
                        'VaR (95%)': np.percentile(returns, 5),
                        'CVaR (95%)': returns[returns <= np.percentile(returns, 5)].mean() if len(returns[returns <= np.percentile(returns, 5)]) > 0 else 0,
                        'Skewness': returns.skew(),
                        'Kurtosis': returns.kurtosis(),
                        'Auto-correlation (lag1)': returns.autocorr(lag=1)
                    }
                    
                    risk_metrics[file_name.split(".")[0]] = metrics
        
        if risk_metrics:
            # Create visualization
            metrics_df = pd.DataFrame(risk_metrics).T
            
            # Determine number of subplots needed
            n_metrics = len(metrics_df.columns)
            n_cols = 3
            n_rows = (n_metrics + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
            fig.suptitle('Comprehensive Risk Metrics Dashboard', fontsize=16)
            
            # Flatten axes array for easy indexing
            if n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, metric in enumerate(metrics_df.columns):
                if i < len(axes):
                    axes[i].bar(metrics_df.index, metrics_df[metric])
                    axes[i].set_title(metric)
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].grid(True, alpha=0.3)
            
            # Hide empty subplots
            for i in range(len(metrics_df.columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig('risk_metrics_dashboard.png', dpi=300, bbox_inches='tight')
            plt.show()
            print("✓ Risk metrics dashboard saved as 'risk_metrics_dashboard.png'")
            
            # Print table
            print("\nRisk Metrics Summary:")
            print("="*80)
            print(metrics_df.round(4))
        else:
            print("ℹ No data for risk metrics dashboard")

    def create_correlation_heatmap(self):
        """Create correlation heatmap between different stocks"""
        print("Creating correlation heatmap...")
        
        try:
            # Extract close prices from all datasets
            close_data = {}
            for file_name, df in self.dfs.items():
                if 'Close' in df.columns:
                    close_data[file_name.split(".")[0]] = df.set_index('Date')['Close']
            
            if len(close_data) >= 2:
                # Combine data and calculate correlations
                combined_data = pd.DataFrame(close_data).dropna()
                correlation_matrix = combined_data.corr()
                
                # Create heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                           square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
                plt.title('Stock Correlation Matrix')
                plt.tight_layout()
                plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
                plt.show()
                print("✓ Correlation heatmap saved as 'correlation_heatmap.png'")
            else:
                print("ℹ Need at least 2 stocks for correlation heatmap")
                
        except Exception as e:
            print(f"✗ Correlation heatmap failed: {e}")

    def create_returns_comparison(self):
        """Create cumulative returns comparison"""
        print("Creating returns comparison...")
        
        plt.figure(figsize=(15, 10))
        
        for file_name, df in self.dfs.items():
            if 'Close' in df.columns and len(df) > 1:
                # Calculate cumulative returns
                returns = df['Close'].pct_change().dropna()
                cumulative_returns = (1 + returns).cumprod() - 1
                
                # Align dates
                valid_dates = df['Date'].iloc[cumulative_returns.index]
                
                plt.plot(valid_dates, cumulative_returns * 100, 
                        label=file_name.split(".")[0], linewidth=2)
        
        plt.title('Cumulative Returns Comparison')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('returns_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Returns comparison saved as 'returns_comparison.png'")

    def run_complete_analysis(self):
        """Run all professional-grade analyses with full error handling"""
        print("Starting COMPLETE PROFESSIONAL-GRADE stock analysis...")
        print("="*70)
        print("All plots will be saved in current directory:")
        print(f"Current working directory: {os.getcwd()}")
        print("="*70)
        
        # Run all analyses with error handling
        analyses = [
            ('Fourier Analysis', self.create_fourier_analysis),
            ('Market Regime Analysis', self.create_market_regime_analysis),
            ('Monte Carlo Simulations', self.create_monte_carlo_simulation),
            ('Volatility Analysis', self.create_fixed_volatility_analysis),
            ('Risk Metrics Dashboard', self.create_risk_metrics_dashboard),
            ('Correlation Heatmap', self.create_correlation_heatmap),
            ('Returns Comparison', self.create_returns_comparison),
        ]
        
        for analysis_name, analysis_func in analyses:
            try:
                print(f"\n{'='*50}")
                print(f"Running: {analysis_name}")
                print('='*50)
                analysis_func()
            except Exception as e:
                print(f"✗ {analysis_name} failed: {e}")
                continue
        
        print("\n" + "="*70)
        print("COMPLETE ANALYSIS FINISHED!")
        print("="*70)
        print("All generated files are in your current directory:")
        print(f"Location: {os.getcwd()}")
        print("\nTo view all generated files, run:")
        print("ls -la *.png")
        print("\nGenerated plots include:")
        expected_files = [
            "fourier_analysis.png",
            "market_regime_*.png", 
            "monte_carlo_*.png",
            "volatility_analysis_*.png",
            "risk_metrics_dashboard.png",
            "correlation_heatmap.png",
            "returns_comparison.png"
        ]
        for file in expected_files:
            print(f"- {file}")

# Main execution
if __name__ == "__main__":
    file_paths = {
        'cleaned_all_stocks_combined.csv': '/home/adnan/Music/1B-Fall-2025-Courses/COMP 333-Data-Analytics/1Z-Project-step03/1E-step08/1D-step09-par03/cleaned_financial_data/stock_data/cleaned_all_stocks_combined.csv',
        'cleaned_COST_full_year.csv': '/home/adnan/Music/1B-Fall-2025-Courses/COMP 333-Data-Analytics/1Z-Project-step03/1E-step08/1D-step09-par03/cleaned_financial_data/stock_data/cleaned_COST_full_year.csv',
        'cleaned_V_full_year.csv': '/home/adnan/Music/1B-Fall-2025-Courses/COMP 333-Data-Analytics/1Z-Project-step03/1E-step08/1D-step09-par03/cleaned_financial_data/stock_data/cleaned_V_full_year.csv',
    }
    
    print("Current working directory:", os.getcwd())
    print("\nPlease install required packages first:")
    print("pip install scikit-learn")
    print("\nStarting complete analysis...")
    
    professional_visualizer = ProfessionalStockVisualizer(file_paths)
    professional_visualizer.run_complete_analysis()
