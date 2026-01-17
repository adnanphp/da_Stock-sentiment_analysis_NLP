import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdvancedStockVisualizer:
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

    def create_candlestick_charts(self):
        """Create candlestick charts for stock price movements"""
        print("Creating candlestick charts...")
        
        for file_name, df in self.dfs.items():
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Date']):
                # Sample last 30 days for clarity
                plot_df = df.tail(30).copy()
                
                fig = go.Figure(data=[go.Candlestick(
                    x=plot_df['Date'],
                    open=plot_df['Open'],
                    high=plot_df['High'],
                    low=plot_df['Low'],
                    close=plot_df['Close'],
                    name='Price'
                )])
                
                fig.update_layout(
                    title=f'{file_name} - Candlestick Chart (Last 30 Days)',
                    xaxis_title='Date',
                    yaxis_title='Price',
                    template='plotly_white'
                )
                
                fig.write_html(f"candlestick_{file_name.split('.')[0]}.html")
                fig.show()

    def create_technical_indicators(self):
        """Create plots with technical indicators"""
        print("Creating technical indicators plots...")
        
        for file_name, df in self.dfs.items():
            if 'Close' in df.columns and 'Date' in df.columns:
                # Calculate technical indicators
                df_tech = df.copy()
                
                # Moving averages
                df_tech['SMA_20'] = df_tech['Close'].rolling(window=20).mean()
                df_tech['SMA_50'] = df_tech['Close'].rolling(window=50).mean()
                
                # RSI
                delta = df_tech['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df_tech['RSI'] = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                df_tech['BB_Middle'] = df_tech['Close'].rolling(window=20).mean()
                bb_std = df_tech['Close'].rolling(window=20).std()
                df_tech['BB_Upper'] = df_tech['BB_Middle'] + (bb_std * 2)
                df_tech['BB_Lower'] = df_tech['BB_Middle'] - (bb_std * 2)
                
                # Create subplots
                fig = make_subplots(rows=2, cols=1, shared_x=True,
                                  subplot_titles=('Price with Technical Indicators', 'RSI'),
                                  vertical_spacing=0.1,
                                  row_heights=[0.7, 0.3])
                
                # Price and indicators
                fig.add_trace(go.Scatter(x=df_tech['Date'], y=df_tech['Close'], 
                                       name='Close Price', line=dict(color='blue')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_tech['Date'], y=df_tech['SMA_20'], 
                                       name='SMA 20', line=dict(color='orange')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_tech['Date'], y=df_tech['SMA_50'], 
                                       name='SMA 50', line=dict(color='red')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_tech['Date'], y=df_tech['BB_Upper'], 
                                       name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
                fig.add_trace(go.Scatter(x=df_tech['Date'], y=df_tech['BB_Lower'], 
                                       name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
                
                # RSI
                fig.add_trace(go.Scatter(x=df_tech['Date'], y=df_tech['RSI'], 
                                       name='RSI', line=dict(color='purple')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                fig.update_layout(height=800, title_text=f"{file_name} - Technical Analysis")
                fig.write_html(f"technical_indicators_{file_name.split('.')[0]}.html")
                fig.show()

    def create_volatility_analysis(self):
        """Create volatility analysis plots"""
        print("Creating volatility analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Volatility Analysis - Daily Returns and Rolling Volatility', fontsize=16)
        
        for i, (file_name, df) in enumerate(self.dfs.items()):
            if i >= 4:
                break
                
            row, col = i // 2, i % 2
            ax1 = axes[row, col]
            
            if 'Close' in df.columns:
                # Calculate daily returns
                returns = df['Close'].pct_change().dropna()
                
                # Plot daily returns
                ax1.plot(returns.index, returns, alpha=0.7, linewidth=0.8)
                ax1.axhline(y=0, color='red', linestyle='-', alpha=0.3)
                ax1.set_title(f'{file_name}\nDaily Returns')
                ax1.set_ylabel('Daily Return')
                ax1.grid(True, alpha=0.3)
                
                # Calculate and plot rolling volatility
                ax2 = ax1.twinx()
                rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)  # Annualized
                ax2.plot(rolling_vol.index, rolling_vol, color='red', alpha=0.7, linewidth=2, label='20-day Volatility')
                ax2.set_ylabel('Annualized Volatility', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                ax2.legend()
                
        plt.tight_layout()
        plt.savefig('volatility_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_correlation_network(self):
        """Create correlation network between different stocks"""
        print("Creating correlation network...")
        
        try:
            # Extract close prices from all datasets
            close_prices = {}
            for file_name, df in self.dfs.items():
                if 'Close' in df.columns:
                    close_prices[file_name] = df['Close']
            
            # Create correlation matrix
            correlation_df = pd.DataFrame(close_prices).corr()
            
            # Create network plot
            fig = plt.figure(figsize=(12, 10))
            
            # Create a graph from correlation matrix
            import networkx as nx
            G = nx.Graph()
            
            # Add nodes
            for stock in correlation_df.columns:
                G.add_node(stock)
            
            # Add edges with weights based on correlation
            for i, stock1 in enumerate(correlation_df.columns):
                for j, stock2 in enumerate(correlation_df.columns):
                    if i < j:  # Avoid duplicates and self-loops
                        corr = correlation_df.iloc[i, j]
                        if abs(corr) > 0.5:  # Only show strong correlations
                            G.add_edge(stock1, stock2, weight=abs(corr))
            
            # Draw the network
            pos = nx.spring_layout(G, k=1, iterations=50)
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                 node_size=2000, alpha=0.7)
            nx.draw_networkx_edges(G, pos, edge_color='gray', 
                                 width=[G[u][v]['weight']*3 for u,v in G.edges()],
                                 alpha=0.6)
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
            
            plt.title('Stock Correlation Network\n(Strong Correlations > 0.5)')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig('correlation_network.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"Correlation network failed: {e}")

    def create_risk_return_scatter(self):
        """Create risk-return scatter plot (Modern Portfolio Theory)"""
        print("Creating risk-return analysis...")
        
        returns_data = []
        volatility_data = []
        sharpe_data = []
        names = []
        
        for file_name, df in self.dfs.items():
            if 'Close' in df.columns:
                # Calculate daily returns
                daily_returns = df['Close'].pct_change().dropna()
                
                # Annualized return and volatility
                annual_return = daily_returns.mean() * 252
                annual_volatility = daily_returns.std() * np.sqrt(252)
                sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
                
                returns_data.append(annual_return)
                volatility_data.append(annual_volatility)
                sharpe_data.append(sharpe_ratio)
                names.append(file_name)
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(volatility_data, returns_data, c=sharpe_data, 
                            cmap='viridis', s=100, alpha=0.7)
        
        # Add labels
        for i, name in enumerate(names):
            plt.annotate(name.split('.')[0], (volatility_data[i], returns_data[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        plt.colorbar(scatter, label='Sharpe Ratio')
        plt.xlabel('Annualized Volatility (Risk)')
        plt.ylabel('Annualized Return')
        plt.title('Risk-Return Profile of Stocks\n(Modern Portfolio Theory)')
        plt.grid(True, alpha=0.3)
        
        # Add efficient frontier concept
        x = np.linspace(min(volatility_data), max(volatility_data), 100)
        plt.fill_between(x, 0, max(returns_data) * 1.1, alpha=0.1, color='green', 
                        label='Efficient Frontier Region')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig('risk_return_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_heatmap_calendar(self):
        """Create calendar heatmap of returns"""
        print("Creating calendar heatmap...")
        
        for file_name, df in self.dfs.items():
            if 'Date' in df.columns and 'Close' in df.columns:
                # Calculate daily returns
                df_plot = df.copy()
                df_plot['Return'] = df_plot['Close'].pct_change()
                df_plot = df_plot.dropna()
                
                # Extract year, month, day
                df_plot['Year'] = df_plot['Date'].dt.year
                df_plot['Month'] = df_plot['Date'].dt.month
                df_plot['Day'] = df_plot['Date'].dt.day
                
                # Create pivot table for heatmap
                heatmap_data = df_plot.pivot_table(values='Return', 
                                                 index='Month', 
                                                 columns='Year', 
                                                 aggfunc='mean')
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                           center=0, cbar_kws={'label': 'Average Daily Return'})
                plt.title(f'{file_name} - Monthly Returns Heatmap by Year')
                plt.tight_layout()
                plt.savefig(f'calendar_heatmap_{file_name.split(".")[0]}.png', 
                           dpi=300, bbox_inches='tight')
                plt.show()

    def create_drawdown_analysis(self):
        """Create drawdown analysis plots"""
        print("Creating drawdown analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Drawdown Analysis - Maximum Peak-to-Trough Declines', fontsize=16)
        
        for i, (file_name, df) in enumerate(self.dfs.items()):
            if i >= 4:
                break
                
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if 'Close' in df.columns and 'Date' in df.columns:
                # Calculate running maximum
                running_max = df['Close'].expanding().max()
                
                # Calculate drawdown
                drawdown = (df['Close'] - running_max) / running_max * 100
                
                # Plot
                ax.fill_between(df['Date'], drawdown, 0, alpha=0.3, color='red')
                ax.plot(df['Date'], drawdown, color='red', linewidth=1)
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.axhline(y=drawdown.min(), color='darkred', linestyle='--', 
                          label=f'Max Drawdown: {drawdown.min():.1f}%')
                
                ax.set_title(f'{file_name}\nDrawdown Analysis')
                ax.set_ylabel('Drawdown (%)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                
        plt.tight_layout()
        plt.savefig('drawdown_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    def create_volume_profile(self):
        """Create volume profile analysis"""
        print("Creating volume profile...")
        
        for file_name, df in self.dfs.items():
            if all(col in df.columns for col in ['Close', 'Volume', 'High', 'Low']):
                # Create price bins
                price_range = np.linspace(df['Low'].min(), df['High'].max(), 50)
                
                # Calculate volume at each price level
                volume_profile = []
                for i in range(len(price_range)-1):
                    mask = (df['Close'] >= price_range[i]) & (df['Close'] < price_range[i+1])
                    volume_at_level = df.loc[mask, 'Volume'].sum()
                    volume_profile.append(volume_at_level)
                
                # Plot
                plt.figure(figsize=(10, 8))
                plt.barh(price_range[:-1], volume_profile, height=(price_range[1]-price_range[0]), 
                        alpha=0.7, color='skyblue', edgecolor='black')
                plt.axvline(x=np.mean(volume_profile), color='red', linestyle='--', 
                           label=f'Mean Volume: {np.mean(volume_profile):,.0f}')
                plt.xlabel('Volume')
                plt.ylabel('Price Level')
                plt.title(f'{file_name} - Volume Profile Analysis')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'volume_profile_{file_name.split(".")[0]}.png', 
                           dpi=300, bbox_inches='tight')
                plt.show()

    def create_returns_distribution(self):
        """Create detailed returns distribution with statistical analysis"""
        print("Creating returns distribution analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Returns Distribution Analysis with Statistical Tests', fontsize=16)
        
        for i, (file_name, df) in enumerate(self.dfs.items()):
            if i >= 4:
                break
                
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if 'Close' in df.columns:
                returns = df['Close'].pct_change().dropna()
                
                # Plot histogram with normal distribution overlay
                n, bins, patches = ax.hist(returns, bins=50, density=True, alpha=0.7, 
                                         color='lightblue', edgecolor='black')
                
                # Fit normal distribution
                mu, std = stats.norm.fit(returns)
                x = np.linspace(returns.min(), returns.max(), 100)
                p = stats.norm.pdf(x, mu, std)
                ax.plot(x, p, 'r-', linewidth=2, label=f'Normal fit\nμ={mu:.4f}, σ={std:.4f}')
                
                # Add statistical information
                skewness = stats.skew(returns)
                kurtosis = stats.kurtosis(returns)
                ax.text(0.05, 0.95, f'Skew: {skewness:.3f}\nKurtosis: {kurtosis:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax.set_title(f'{file_name}\nReturns Distribution')
                ax.set_xlabel('Daily Returns')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
        plt.tight_layout()
        plt.savefig('returns_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

    def run_all_advanced_analyses(self):
        """Run all advanced visualization analyses"""
        print("Starting advanced stock data analysis...")
        print("="*60)
        
        self.create_candlestick_charts()
        self.create_technical_indicators()
        self.create_volatility_analysis()
        self.create_correlation_network()
        self.create_risk_return_scatter()
        self.create_heatmap_calendar()
        self.create_drawdown_analysis()
        self.create_volume_profile()
        self.create_returns_distribution()
        
        print("\n" + "="*60)
        print("ADVANCED ANALYSIS COMPLETE!")
        print("="*60)
        print("Generated advanced plots:")
        advanced_plots = [
            "candlestick_*.html", "technical_indicators_*.html", 
            "volatility_analysis.png", "correlation_network.png",
            "risk_return_analysis.png", "calendar_heatmap_*.png",
            "drawdown_analysis.png", "volume_profile_*.png",
            "returns_distribution.png"
        ]
        for plot in advanced_plots:
            print(f"- {plot}")

# Main execution with your Linux file paths
if __name__ == "__main__":
    file_paths = {
        'cleaned_all_stocks_combined.csv': '/home/adnan/Music/1B-Fall-2025-Courses/COMP 333-Data-Analytics/1Z-Project-step03/1E-step08/1D-step09-par03/cleaned_financial_data/stock_data/cleaned_all_stocks_combined.csv',
        'cleaned_COST_full_year.csv': '/home/adnan/Music/1B-Fall-2025-Courses/COMP 333-Data-Analytics/1Z-Project-step03/1E-step08/1D-step09-par03/cleaned_financial_data/stock_data/cleaned_COST_full_year.csv',
        'cleaned_V_full_year.csv': '/home/adnan/Music/1B-Fall-2025-Courses/COMP 333-Data-Analytics/1Z-Project-step03/1E-step08/1D-step09-par03/cleaned_financial_data/stock_data/cleaned_V_full_year.csv',
        'feature_engineered_all_stocks_combined.csv': '/home/adnan/Music/1B-Fall-2025-Courses/COMP 333-Data-Analytics/1Z-Project-step03/1E-step08/1D-step09-par03/feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv'
    }
    
    # Initialize and run advanced visualizer
    advanced_visualizer = AdvancedStockVisualizer(file_paths)
    advanced_visualizer.run_all_advanced_analyses()
