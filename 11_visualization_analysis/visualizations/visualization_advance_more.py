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
                    if period < len(prices)/2:  # Reasonable periods
                        ax.axvline(period, color='red', linestyle='--', alpha=0.7)
                        ax.text(period, np.max(power_spectrum[positive_freq])/2, 
                               f'{period:.1f}d', rotation=90, ha='right')
        
        plt.tight_layout()
        plt.savefig('fourier_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

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
                        plt.savefig(f'market_regime_{file_name.split(".")[0]}.png', 
                                  dpi=300, bbox_inches='tight')
                        plt.show()
                        
        except Exception as e:
            print(f"Market regime analysis failed: {e}")

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
                plt.savefig(f'monte_carlo_{file_name.split(".")[0]}.png', dpi=300, bbox_inches='tight')
                plt.show()

    def create_volatility_surface(self):
        """Create 3D volatility surface (if we had options data)"""
        print("Creating volatility analysis...")
        
        for file_name, df in self.dfs.items():
            if 'Close' in df.columns and len(df) > 100:
                # Calculate rolling volatility at different time horizons
                horizons = [5, 10, 20, 50, 100]
                returns = df['Close'].pct_change().dropna()
                
                # Create volatility surface data
                vol_data = []
                for horizon in horizons:
                    rolling_vol = returns.rolling(window=horizon).std() * np.sqrt(252)
                    vol_data.append(rolling_vol.dropna().values)
                
                # Plot 3D surface
                fig = plt.figure(figsize=(12, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                # Create meshgrid
                X, Y = np.meshgrid(range(len(vol_data[0])), horizons)
                Z = np.array(vol_data)
                
                # Plot surface
                surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
                
                ax.set_xlabel('Time')
                ax.set_ylabel('Horizon (days)')
                ax.set_zlabel('Annualized Volatility')
                ax.set_title(f'{file_name.split(".")[0]} - Volatility Surface')
                
                plt.colorbar(surf)
                plt.tight_layout()
                plt.savefig(f'volatility_surface_{file_name.split(".")[0]}.png', dpi=300, bbox_inches='tight')
                plt.show()

    def create_risk_metrics_dashboard(self):
        """Create comprehensive risk metrics dashboard"""
        print("Creating risk metrics dashboard...")
        
        risk_metrics = {}
        
        for file_name, df in self.dfs.items():
            if 'Close' in df.columns and len(df) > 100:
                returns = df['Close'].pct_change().dropna()
                
                # Calculate various risk metrics
                metrics = {
                    'Volatility (Annual)': returns.std() * np.sqrt(252),
                    'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                    'Max Drawdown': (df['Close'] / df['Close'].expanding().max() - 1).min(),
                    'VaR (95%)': np.percentile(returns, 5),
                    'CVaR (95%)': returns[returns <= np.percentile(returns, 5)].mean(),
                    'Skewness': returns.skew(),
                    'Kurtosis': returns.kurtosis(),
                    'Hurst Exponent': self.calculate_hurst_exponent(returns),
                    'Auto-correlation (lag1)': returns.autocorr(lag=1)
                }
                
                risk_metrics[file_name.split(".")[0]] = metrics
        
        # Create visualization
        metrics_df = pd.DataFrame(risk_metrics).T
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Risk Metrics Dashboard', fontsize=16)
        
        metrics_to_plot = ['Volatility (Annual)', 'Sharpe Ratio', 'Max Drawdown', 
                          'VaR (95%)', 'CVaR (95%)', 'Skewness', 
                          'Kurtosis', 'Hurst Exponent', 'Auto-correlation (lag1)']
        
        for i, metric in enumerate(metrics_to_plot):
            row, col = i // 3, i % 3
            if metric in metrics_df.columns:
                axes[row, col].bar(metrics_df.index, metrics_df[metric])
                axes[row, col].set_title(metric)
                axes[row, col].tick_params(axis='x', rotation=45)
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('risk_metrics_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print table
        print("\nRisk Metrics Summary:")
        print("="*80)
        print(metrics_df.round(4))

    def calculate_hurst_exponent(self, returns, max_lag=50):
        """Calculate Hurst exponent for trend persistence analysis"""
        try:
            lags = range(2, max_lag)
            tau = [np.std(np.subtract(returns[lag:], returns[:-lag])) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        except:
            return np.nan

    def create_quantile_regression(self):
        """Create quantile regression for conditional distributions"""
        print("Creating quantile regression analysis...")
        
        try:
            from sklearn.linear_model import QuantileRegressor
            
            for file_name, df in self.dfs.items():
                if 'Close' in df.columns and len(df) > 100:
                    # Create lagged returns
                    returns = df['Close'].pct_change().dropna()
                    X = returns.shift(1).dropna().values.reshape(-1, 1)
                    y = returns.iloc[1:].values
                    
                    # Fit quantile regressions
                    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
                    predictions = {}
                    
                    plt.figure(figsize=(12, 8))
                    
                    for q in quantiles:
                        qr = QuantileRegressor(quantile=q, alpha=0)
                        qr.fit(X, y)
                        pred = qr.predict(X)
                        predictions[q] = pred
                        
                        plt.plot(X, pred, label=f'{int(q*100)}th percentile', linewidth=2)
                    
                    # Scatter plot of actual data
                    plt.scatter(X, y, alpha=0.3, s=10, color='gray', label='Actual Returns')
                    
                    plt.xlabel('Lag Return (t-1)')
                    plt.ylabel('Return (t)')
                    plt.title(f'{file_name.split(".")[0]} - Quantile Regression\n(Conditional Distribution)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f'quantile_regression_{file_name.split(".")[0]}.png', 
                              dpi=300, bbox_inches='tight')
                    plt.show()
                    
        except Exception as e:
            print(f"Quantile regression failed: {e}")

    def create_structural_break_detection(self):
        """Detect structural breaks in time series"""
        print("Creating structural break detection...")
        
        try:
            from ruptures import Binseg
            
            for file_name, df in self.dfs.items():
                if 'Close' in df.columns and len(df) > 200:
                    prices = df['Close'].values
                    
                    # Detect change points
                    algo = Binseg(model="rbf").fit(prices.reshape(-1, 1))
                    breakpoints = algo.predict(n_bkps=5)  # Find 5 breakpoints
                    
                    # Plot
                    plt.figure(figsize=(15, 8))
                    plt.plot(df['Date'], prices, 'b-', alpha=0.7, label='Price')
                    
                    # Mark breakpoints
                    for bp in breakpoints[:-1]:  # Last one is end of series
                        plt.axvline(df['Date'].iloc[bp], color='red', linestyle='--', 
                                  alpha=0.8, label='Structural Break' if bp == breakpoints[0] else "")
                    
                    plt.title(f'{file_name.split(".")[0]} - Structural Break Detection')
                    plt.xlabel('Date')
                    plt.ylabel('Price')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(f'structural_breaks_{file_name.split(".")[0]}.png', 
                              dpi=300, bbox_inches='tight')
                    plt.show()
                    
        except Exception as e:
            print(f"Structural break detection failed: {e}")

    def create_machine_learning_feature_importance(self):
        """Create feature importance analysis using ML"""
        print("Creating ML feature importance analysis...")
        
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.inspection import permutation_importance
            
            for file_name, df in self.dfs.items():
                if 'Close' in df.columns and len(df) > 100:
                    # Create features
                    df_ml = df.copy()
                    df_ml['returns'] = df_ml['Close'].pct_change()
                    
                    # Technical indicators as features
                    df_ml['SMA_10'] = df_ml['Close'].rolling(10).mean()
                    df_ml['SMA_30'] = df_ml['Close'].rolling(30).mean()
                    df_ml['volatility'] = df_ml['returns'].rolling(20).std()
                    df_ml['momentum'] = df_ml['Close'] / df_ml['Close'].shift(10) - 1
                    
                    # Target: future returns
                    df_ml['target'] = df_ml['Close'].shift(-5) / df_ml['Close'] - 1
                    
                    df_ml = df_ml.dropna()
                    
                    if len(df_ml) > 50:
                        features = ['SMA_10', 'SMA_30', 'volatility', 'momentum']
                        X = df_ml[features]
                        y = df_ml['target']
                        
                        # Train model
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(X, y)
                        
                        # Feature importance
                        importance = model.feature_importances_
                        
                        # Plot
                        plt.figure(figsize=(10, 6))
                        plt.barh(features, importance)
                        plt.xlabel('Feature Importance')
                        plt.title(f'{file_name.split(".")[0]} - ML Feature Importance\n(Predicting 5-day Returns)')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        plt.savefig(f'feature_importance_{file_name.split(".")[0]}.png', 
                                  dpi=300, bbox_inches='tight')
                        plt.show()
                        
        except Exception as e:
            print(f"ML feature importance failed: {e}")

    def run_professional_analyses(self):
        """Run all professional-grade analyses"""
        print("Starting PROFESSIONAL-GRADE stock analysis...")
        print("="*70)
        
        self.create_fourier_analysis()
        self.create_market_regime_analysis()
        self.create_monte_carlo_simulation()
        self.create_volatility_surface()
        self.create_risk_metrics_dashboard()
        self.create_quantile_regression()
        self.create_structural_break_detection()
        self.create_machine_learning_feature_importance()
        
        print("\n" + "="*70)
        print("PROFESSIONAL ANALYSIS COMPLETE!")
        print("="*70)
        print("Generated professional plots:")
        professional_plots = [
            "fourier_analysis.png", "market_regime_*.png", "monte_carlo_*.png",
            "volatility_surface_*.png", "risk_metrics_dashboard.png",
            "quantile_regression_*.png", "structural_breaks_*.png",
            "feature_importance_*.png"
        ]
        for plot in professional_plots:
            print(f"- {plot}")

# Main execution
if __name__ == "__main__":
    file_paths = {
        'cleaned_all_stocks_combined.csv': '/home/adnan/Music/1B-Fall-2025-Courses/COMP 333-Data-Analytics/1Z-Project-step03/1E-step08/1D-step09-par03/cleaned_financial_data/stock_data/cleaned_all_stocks_combined.csv',
        'cleaned_COST_full_year.csv': '/home/adnan/Music/1B-Fall-2025-Courses/COMP 333-Data-Analytics/1Z-Project-step03/1E-step08/1D-step09-par03/cleaned_financial_data/stock_data/cleaned_COST_full_year.csv',
        'cleaned_V_full_year.csv': '/home/adnan/Music/1B-Fall-2025-Courses/COMP 333-Data-Analytics/1Z-Project-step03/1E-step08/1D-step09-par03/cleaned_financial_data/stock_data/cleaned_V_full_year.csv',
    }
    
    # Install required packages first:
    print("Please install required packages:")
    print("pip install scikit-learn ruptures")
    
    professional_visualizer = ProfessionalStockVisualizer(file_paths)
    professional_visualizer.run_professional_analyses()
