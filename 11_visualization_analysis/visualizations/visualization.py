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
plt.rcParams['figure.figsize'] = (12, 8)

class StockDataVisualizer:
    def __init__(self, file_paths):
        """Initialize with actual file paths"""
        self.file_paths = file_paths
        self.dfs = {}
        self.load_data()
        
    def load_data(self):
        """Load all CSV files from file paths"""
        for file_name, file_path in self.file_paths.items():
            try:
                print(f"Loading {file_name} from {file_path}...")
                self.dfs[file_name] = pd.read_csv(file_path)
                
                # Convert Date column to datetime
                if 'Date' in self.dfs[file_name].columns:
                    self.dfs[file_name]['Date'] = pd.to_datetime(self.dfs[file_name]['Date'])
                    
                print(f"Loaded {len(self.dfs[file_name])} rows from {file_name}")
                
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                
    def create_scatter_plots(self):
        """Create various scatter plots"""
        print("Creating scatter plots...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Scatter Plots - Stock Price Relationships', fontsize=16, fontweight='bold')
        
        for i, (file_name, df) in enumerate(self.dfs.items()):
            if i >= 4:  # Limit to first 4 datasets
                break
                
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if 'Close' in df.columns and 'Volume' in df.columns:
                # Sample data if too large for performance
                if len(df) > 10000:
                    plot_df = df.sample(10000, random_state=42)
                else:
                    plot_df = df
                    
                scatter = ax.scatter(plot_df['Close'], plot_df['Volume'], alpha=0.6, 
                                   c=range(len(plot_df)), cmap='viridis', s=30)
                ax.set_xlabel('Close Price')
                ax.set_ylabel('Volume')
                ax.set_title(f'{file_name} - Price vs Volume')
                plt.colorbar(scatter, ax=ax, label='Data Point Index')
                ax.grid(True, alpha=0.3)
                
        plt.tight_layout()
        plt.savefig('scatter_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_density_plots(self):
        """Create density and distribution plots"""
        print("Creating density plots...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Density and Distribution Plots', fontsize=16, fontweight='bold')
        
        price_columns = ['Close', 'Open', 'High', 'Low']
        
        for i, file_name in enumerate(list(self.dfs.keys())[:4]):
            df = self.dfs[file_name]
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Plot density for available price columns
            for col_name in price_columns:
                if col_name in df.columns:
                    data = df[col_name].dropna()
                    if len(data) > 0:
                        sns.kdeplot(data, ax=ax, label=col_name, alpha=0.7)
                    
            ax.set_xlabel('Price')
            ax.set_ylabel('Density')
            ax.set_title(f'{file_name} - Price Distributions')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig('density_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_box_plots(self):
        """Create box plots for price distributions"""
        print("Creating box plots...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Box Plots - Price Distributions by Stock', fontsize=16, fontweight='bold')
        
        for i, (file_name, df) in enumerate(self.dfs.items()):
            if i >= 4:
                break
                
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Prepare data for box plot
            price_data = []
            labels = []
            for col_name in ['Open', 'High', 'Low', 'Close']:
                if col_name in df.columns:
                    data = df[col_name].dropna()
                    if len(data) > 0:
                        price_data.append(data)
                        labels.append(col_name)
                    
            if price_data:
                box = ax.boxplot(price_data, labels=labels, patch_artist=True)
                # Color the boxes
                colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
                for patch, color in zip(box['boxes'], colors[:len(price_data)]):
                    patch.set_facecolor(color)
                    
                ax.set_ylabel('Price')
                ax.set_title(f'{file_name}')
                ax.grid(True, alpha=0.3)
                
        plt.tight_layout()
        plt.savefig('box_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_parallel_plots(self):
        """Create parallel coordinate plots"""
        print("Creating parallel coordinate plots...")
        try:
            # Prepare data for parallel coordinates
            parallel_data = []
            
            for file_name, df in self.dfs.items():
                if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume']):
                    # Sample data to avoid overcrowding
                    sample_size = min(1000, len(df))
                    sample_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().sample(sample_size, random_state=42)
                    
                    sample_df['Stock'] = file_name
                    parallel_data.append(sample_df)
            
            if parallel_data:
                combined_df = pd.concat(parallel_data, ignore_index=True)
                
                # Normalize data for better parallel coordinates
                from sklearn.preprocessing import StandardScaler
                features = ['Open', 'High', 'Low', 'Close', 'Volume']
                scaler = StandardScaler()
                normalized_data = scaler.fit_transform(combined_df[features])
                normalized_df = pd.DataFrame(normalized_data, columns=features)
                normalized_df['Stock'] = combined_df['Stock']
                
                # Create parallel coordinates plot
                fig = px.parallel_coordinates(normalized_df, 
                                            dimensions=features,
                                            color='Stock',
                                            title="Parallel Coordinates - Stock Price Patterns",
                                            color_continuous_scale=px.colors.sequential.Viridis)
                fig.write_html("parallel_coordinates.html")
                fig.show()
                
        except Exception as e:
            print(f"Parallel coordinates plot failed: {e}")
            # Fallback to matplotlib version
            self.create_parallel_plots_matplotlib()
            
    def create_parallel_plots_matplotlib(self):
        """Fallback parallel coordinates using matplotlib"""
        from pandas.plotting import parallel_coordinates
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Parallel Coordinates - Stock Metrics', fontsize=16, fontweight='bold')
        
        for i, (file_name, df) in enumerate(self.dfs.items()):
            if i >= 4:
                break
                
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
                # Sample data for clarity
                sample_size = min(100, len(df))
                sample_df = df[['Open', 'High', 'Low', 'Close']].dropna().sample(sample_size, random_state=42)
                
                # Add an ID column for parallel coordinates
                sample_df = sample_df.reset_index(drop=True)
                
                try:
                    parallel_coordinates(sample_df, alpha=0.5, ax=ax)
                    ax.set_title(f'{file_name}')
                    ax.grid(True, alpha=0.3)
                except Exception as e:
                    print(f"Parallel coordinates failed for {file_name}: {e}")
                    
        plt.tight_layout()
        plt.savefig('parallel_coordinates.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_histogram_plots(self):
        """Create histogram plots for price distributions"""
        print("Creating histogram plots...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Histogram Plots - Price Distributions', fontsize=16, fontweight='bold')
        
        for i, (file_name, df) in enumerate(self.dfs.items()):
            if i >= 4:
                break
                
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if 'Close' in df.columns:
                close_prices = df['Close'].dropna()
                if len(close_prices) > 0:
                    # Plot histogram with KDE
                    sns.histplot(close_prices, kde=True, ax=ax, bins=50, 
                                alpha=0.7, color='skyblue', edgecolor='black')
                    
                    # Add statistical information
                    mean_price = close_prices.mean()
                    std_price = close_prices.std()
                    ax.axvline(mean_price, color='red', linestyle='--', 
                              label=f'Mean: ${mean_price:.2f}')
                    ax.axvline(mean_price + std_price, color='orange', linestyle=':', 
                              alpha=0.7, label=f'+1 STD')
                    ax.axvline(mean_price - std_price, color='orange', linestyle=':', 
                              alpha=0.7, label=f'-1 STD')
                    
                    ax.set_xlabel('Close Price')
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'{file_name} - Close Price Distribution')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
        plt.tight_layout()
        plt.savefig('histogram_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_time_series_plots(self):
        """Create time series plots for stock prices"""
        print("Creating time series plots...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Time Series - Stock Price Movement', fontsize=16, fontweight='bold')
        
        for i, (file_name, df) in enumerate(self.dfs.items()):
            if i >= 4:
                break
                
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if 'Date' in df.columns and 'Close' in df.columns:
                df_sorted = df.sort_values('Date')
                # Sample if too many points for performance
                if len(df_sorted) > 1000:
                    plot_df = df_sorted.iloc[::len(df_sorted)//1000]
                else:
                    plot_df = df_sorted
                    
                ax.plot(plot_df['Date'], plot_df['Close'], 
                       linewidth=2, label='Close Price', color='blue', alpha=0.8)
                
                ax.set_xlabel('Date')
                ax.set_ylabel('Close Price')
                ax.set_title(f'{file_name} - Price Over Time')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Format x-axis dates
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
                
        plt.tight_layout()
        plt.savefig('time_series_plots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_correlation_heatmaps(self):
        """Create correlation heatmaps for price metrics"""
        print("Creating correlation heatmaps...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Correlation Heatmaps - Price Metrics', fontsize=16, fontweight='bold')
        
        for i, (file_name, df) in enumerate(self.dfs.items()):
            if i >= 4:
                break
                
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            price_columns = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] 
                           if col in df.columns]
            
            if len(price_columns) >= 2:
                correlation_matrix = df[price_columns].corr()
                
                im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1, 
                              aspect='auto')
                
                # Add correlation values as text
                for i in range(len(price_columns)):
                    for j in range(len(price_columns)):
                        text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                      ha="center", va="center", color="black", fontweight='bold')
                
                ax.set_xticks(range(len(price_columns)))
                ax.set_yticks(range(len(price_columns)))
                ax.set_xticklabels(price_columns, rotation=45)
                ax.set_yticklabels(price_columns)
                ax.set_title(f'{file_name}')
                
                # Add colorbar
                plt.colorbar(im, ax=ax)
                
        plt.tight_layout()
        plt.savefig('correlation_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_volume_analysis(self):
        """Create volume analysis plots"""
        print("Creating volume analysis plots...")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Volume Analysis', fontsize=16, fontweight='bold')
        
        for i, (file_name, df) in enumerate(self.dfs.items()):
            if i >= 4:
                break
                
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if 'Volume' in df.columns and 'Close' in df.columns:
                # Sample data if too large
                if len(df) > 5000:
                    plot_df = df.sample(5000, random_state=42)
                else:
                    plot_df = df
                    
                # Scatter plot: Price vs Volume
                scatter = ax.scatter(plot_df['Close'], plot_df['Volume'], 
                                   alpha=0.6, c=range(len(plot_df)),
                                   cmap='plasma', s=20)
                ax.set_xlabel('Close Price')
                ax.set_ylabel('Volume')
                ax.set_title(f'{file_name} - Price vs Volume')
                ax.grid(True, alpha=0.3)
                
                # Add colorbar for time
                plt.colorbar(scatter, ax=ax, label='Data Point Index')
                
        plt.tight_layout()
        plt.savefig('volume_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def create_statistical_summary(self):
        """Create statistical summary tables and plots"""
        print("Creating statistical summary...")
        print("=" * 80)
        print("STATISTICAL SUMMARY")
        print("=" * 80)
        
        for file_name, df in self.dfs.items():
            print(f"\n{file_name}:")
            print("-" * 40)
            
            if 'Close' in df.columns:
                close_prices = df['Close'].dropna()
                if len(close_prices) > 0:
                    print(f"Records: {len(close_prices):,}")
                    print(f"Mean: ${close_prices.mean():.2f}")
                    print(f"Median: ${close_prices.median():.2f}")
                    print(f"Std Dev: ${close_prices.std():.2f}")
                    print(f"Min: ${close_prices.min():.2f}")
                    print(f"Max: ${close_prices.max():.2f}")
                    print(f"Range: ${close_prices.max() - close_prices.min():.2f}")
                    print(f"Coefficient of Variation: {close_prices.std()/close_prices.mean():.3f}")
                    print(f"Skewness: {close_prices.skew():.3f}")
                    print(f"Kurtosis: {close_prices.kurtosis():.3f}")
                    
    def create_interactive_plots(self):
        """Create interactive plots using Plotly"""
        print("Creating interactive plots...")
        try:
            # Interactive time series
            fig = go.Figure()
            
            for file_name, df in self.dfs.items():
                if 'Date' in df.columns and 'Close' in df.columns:
                    df_sorted = df.sort_values('Date')
                    # Sample for performance
                    if len(df_sorted) > 1000:
                        plot_df = df_sorted.iloc[::len(df_sorted)//1000]
                    else:
                        plot_df = df_sorted
                        
                    fig.add_trace(go.Scatter(
                        x=plot_df['Date'],
                        y=plot_df['Close'],
                        mode='lines',
                        name=file_name,
                        hovertemplate='<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>'
                    ))
            
            fig.update_layout(
                title='Interactive Stock Price Comparison',
                xaxis_title='Date',
                yaxis_title='Close Price',
                hovermode='x unified',
                template='plotly_white'
            )
            
            fig.write_html("interactive_stock_prices.html")
            fig.show()
            
        except Exception as e:
            print(f"Interactive plots failed: {e}")
            
    def run_all_analyses(self):
        """Run all visualization analyses"""
        print("Starting comprehensive stock data analysis...")
        print(f"Datasets to analyze: {list(self.file_paths.keys())}")
        
        # Create all plots
        self.create_scatter_plots()
        self.create_density_plots()
        self.create_box_plots()
        self.create_parallel_plots()
        self.create_histogram_plots()
        self.create_time_series_plots()
        self.create_correlation_heatmaps()
        self.create_volume_analysis()
        self.create_statistical_summary()
        self.create_interactive_plots()
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE!")
        print("="*50)
        print("Generated files:")
        generated_files = [
            "scatter_plots.png", "density_plots.png", "box_plots.png",
            "parallel_coordinates.html", "parallel_coordinates.png",
            "histogram_plots.png", "time_series_plots.png", 
            "correlation_heatmaps.png", "volume_analysis.png",
            "interactive_stock_prices.html"
        ]
        for file in generated_files:
            print(f"- {file}")

# Main execution - UPDATE THESE PATHS TO YOUR ACTUAL CSV FILES
if __name__ == "__main__":
    # UPDATE THESE PATHS TO YOUR ACTUAL CSV FILE LOCATIONS
    file_paths = {
        'cleaned_all_stocks_combined.csv': r'C:\your\path\to\cleaned_all_stocks_combined.csv',
        'cleaned_COST_full_year.csv': r'C:\your\path\to\cleaned_COST_full_year.csv', 
        'cleaned_V_full_year.csv': r'C:\your\path\to\cleaned_V_full_year.csv',
        'feature_engineered_outlier_treated_all_stocks_combined.csv': r'C:\your\path\to\feature_engineered_outlier_treated_all_stocks_combined.csv'
    }
    
    # Initialize and run visualizer
    visualizer = StockDataVisualizer(file_paths)
    visualizer.run_all_analyses()
