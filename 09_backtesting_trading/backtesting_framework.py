# backtesting_framework.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import warnings
from typing import Dict, List, Optional, Tuple
import logging
import sys
import os

# Add the current directory to path to import from advanced_analytics_integration
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_analytics_integration import AdvancedProductionTradingEngine, EnhancedProductionPredictor
except ImportError:
    print("⚠️  Could not import from advanced_analytics_integration")
    # Create simplified versions for testing
    class EnhancedProductionPredictor:
        def generate_enhanced_signals(self, current_price, symbol):
            return {
                'action': np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.4, 0.3, 0.3]),
                'confidence': np.random.uniform(0.1, 0.9),
                'predicted_return': np.random.uniform(-0.05, 0.05),
                'predicted_volatility': np.random.uniform(0.05, 0.2),
                'risk_level': np.random.choice(['LOW', 'MEDIUM', 'HIGH']),
                'enhancement_factors': {}
            }
    
    class AdvancedProductionTradingEngine:
        def __init__(self, initial_capital=10000):
            self.initial_capital = initial_capital
            self.current_capital = initial_capital
            self.predictor = EnhancedProductionPredictor()
            
        def calculate_enhanced_position(self, capital, confidence, risk_level, enhancement_factors):
            risk_tiers = {'HIGH': 0.01, 'MEDIUM': 0.02, 'LOW': 0.03}
            base_risk = risk_tiers.get(risk_level, 0.02)
            position_size = capital * base_risk * min(confidence, 1.0)
            return min(position_size, capital * 0.05)

warnings.filterwarnings('ignore')

class BacktestingEngine:
    """Comprehensive backtesting framework for trading strategies."""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.results = {}
        self.performance_metrics = {}
        self.comparison_data = {}
        
    def generate_historical_data(self, days: int = 252, symbol: str = "TEST") -> pd.DataFrame:
        """Generate realistic historical price data for backtesting."""
        np.random.seed(42)  # For reproducible results
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        # Generate realistic price data with trends, volatility clusters, and mean reversion
        returns = np.random.normal(0.0005, 0.02, days)  # Daily returns
        
        # Add some market structure
        returns[0:63] += 0.001  # Bullish quarter
        returns[63:126] -= 0.0005  # Correction
        returns[189:252] += 0.002  # Strong finish
        
        # Add volatility clusters
        volatility_regimes = np.random.choice([0.015, 0.025, 0.035], size=days, p=[0.6, 0.3, 0.1])
        returns *= volatility_regimes / 0.02  # Adjust volatility
        
        # Generate prices
        prices = [100]  # Start at $100
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        # Generate volume data (correlated with volatility)
        volumes = [np.random.randint(1000000, 5000000) * (1 + abs(ret)*10) for ret in returns]
        
        historical_data = pd.DataFrame({
            'date': dates,
            'symbol': symbol,
            'open': prices,
            'high': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
            'low': [p * (1 - np.random.uniform(0, 0.015)) for p in prices],
            'close': prices,
            'volume': volumes,
            'returns': returns
        })
        
        return historical_data
    
    def run_backtest(self, strategy_name: str, historical_data: pd.DataFrame, 
                    trading_engine_class=AdvancedProductionTradingEngine,
                    **engine_kwargs) -> Dict:
        """Run comprehensive backtest for a trading strategy."""
        print(f"🔬 Running backtest for: {strategy_name}")
        
        # Filter out unexpected kwargs
        valid_kwargs = {k: v for k, v in engine_kwargs.items() 
                       if k in ['initial_capital']}
        
        # Initialize trading engine
        engine = trading_engine_class(initial_capital=self.initial_capital)
        
        # Backtest results storage
        portfolio_values = []
        trades = []
        daily_returns = []
        dates = []
        
        current_cash = self.initial_capital
        current_positions = {}
        
        for i, row in historical_data.iterrows():
            current_date = row['date']
            current_price = row['close']
            
            # Run trading cycle (simplified for backtesting)
            try:
                # Use enhanced signals but with historical data
                signals = engine.predictor.generate_enhanced_signals(current_price, row['symbol'])
                
                # Simplified trade execution for backtesting
                trade_executed = False
                position_size = engine.calculate_enhanced_position(
                    engine.current_capital,
                    signals['confidence'],
                    signals['risk_level'],
                    signals.get('enhancement_factors', {})
                )
                
                # Execute trade based on signals
                if signals['action'] in ['BUY', 'SELL'] and signals['confidence'] > 0.3:
                    quantity = int(position_size / current_price)
                    if quantity > 0:
                        # Simulate trade execution
                        if signals['action'] == 'BUY' and position_size <= engine.current_capital:
                            cost = quantity * current_price
                            commission = max(cost * 0.001, 1)
                            engine.current_capital -= (cost + commission)
                            
                            if row['symbol'] in current_positions:
                                current_positions[row['symbol']] += quantity
                            else:
                                current_positions[row['symbol']] = quantity
                                
                            trade_executed = True
                            
                        elif signals['action'] == 'SELL' and row['symbol'] in current_positions:
                            if quantity > current_positions[row['symbol']]:
                                quantity = current_positions[row['symbol']]
                            
                            proceeds = quantity * current_price
                            commission = max(proceeds * 0.001, 1)
                            engine.current_capital += (proceeds - commission)
                            
                            current_positions[row['symbol']] -= quantity
                            if current_positions[row['symbol']] == 0:
                                del current_positions[row['symbol']]
                                
                            trade_executed = True
                
                # Calculate portfolio value
                portfolio_value = engine.current_capital
                for symbol, qty in current_positions.items():
                    portfolio_value += qty * current_price
                
                # Store results
                portfolio_values.append(portfolio_value)
                dates.append(current_date)
                
                if i > 0:
                    daily_return = (portfolio_value / portfolio_values[-2]) - 1
                    daily_returns.append(daily_return)
                
                # Record trade if executed
                if trade_executed:
                    trades.append({
                        'date': current_date,
                        'action': signals['action'],
                        'price': current_price,
                        'quantity': quantity,
                        'confidence': signals['confidence'],
                        'portfolio_value': portfolio_value
                    })
                    
            except Exception as e:
                print(f"⚠️ Backtest error on {current_date}: {e}")
                continue
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics(
            portfolio_values, daily_returns, trades, dates
        )
        
        self.results[strategy_name] = {
            'portfolio_values': portfolio_values,
            'daily_returns': daily_returns,
            'trades': trades,
            'dates': dates,
            'performance': performance
        }
        
        print(f"✅ Backtest completed: {strategy_name}")
        print(f"   Final Portfolio: ${portfolio_values[-1]:,.2f}")
        print(f"   Total Return: {performance['total_return']:.2%}")
        print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        
        return self.results[strategy_name]
    
    def calculate_performance_metrics(self, portfolio_values: List[float], 
                                    daily_returns: List[float], 
                                    trades: List[Dict],
                                    dates: List[datetime]) -> Dict:
        """Calculate comprehensive performance metrics."""
        if len(portfolio_values) < 2:
            return {}
        
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        cumulative_returns = [ (pv / portfolio_values[0]) - 1 for pv in portfolio_values ]
        
        # Risk metrics
        volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0
        sharpe_ratio = (np.mean(daily_returns) * 252) / volatility if volatility > 0 else 0
        
        # Drawdown analysis
        peak = portfolio_values[0]
        max_drawdown = 0
        drawdown_duration = 0
        current_drawdown_duration = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
                current_drawdown_duration = 0
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
                current_drawdown_duration += 1
                drawdown_duration = max(drawdown_duration, current_drawdown_duration)
        
        # Trade analysis
        winning_trades = 0
        trade_returns = []
        
        for i in range(1, len(trades)):
            if trades[i]['action'] == 'SELL' and i > 0:
                # Simplified trade P&L calculation
                prev_trade = trades[i-1]
                if prev_trade['action'] == 'BUY':
                    trade_return = (trades[i]['price'] - prev_trade['price']) / prev_trade['price']
                    trade_returns.append(trade_return)
                    if trade_return > 0:
                        winning_trades += 1
        
        win_rate = winning_trades / len(trade_returns) if trade_returns else 0
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        
        return {
            'total_return': total_return,
            'annualized_return': total_return * (252 / len(daily_returns)) if daily_returns else 0,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'drawdown_duration': drawdown_duration,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'profit_factor': abs(sum([r for r in trade_returns if r > 0])) / abs(sum([r for r in trade_returns if r < 0])) if trade_returns and any(r < 0 for r in trade_returns) else float('inf'),
            'calmar_ratio': (total_return * (252 / len(daily_returns))) / max_drawdown if max_drawdown > 0 else float('inf')
        }
    
    def run_comparison_backtest(self, strategies: Dict, historical_data: pd.DataFrame) -> Dict:
        """Run backtest comparison between multiple strategies."""
        print("🔄 Running comparative backtesting...")
        
        for strategy_name, engine_config in strategies.items():
            # Remove description and other non-engine parameters
            engine_kwargs = {k: v for k, v in engine_config.items() 
                           if k not in ['description', 'trading_engine_class']}
            trading_engine_class = engine_config.get('trading_engine_class', AdvancedProductionTradingEngine)
            
            self.run_backtest(strategy_name, historical_data, trading_engine_class, **engine_kwargs)
        
        # Generate comparison report
        self.generate_comparison_report()
        
        return self.results
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report."""
        print("\n" + "="*80)
        print("📊 BACKTESTING COMPARISON REPORT")
        print("="*80)
        
        comparison_data = []
        
        for strategy, results in self.results.items():
            perf = results['performance']
            comparison_data.append({
                'Strategy': strategy,
                'Total Return': f"{perf['total_return']:.2%}",
                'Annual Return': f"{perf['annualized_return']:.2%}",
                'Volatility': f"{perf['volatility']:.2%}",
                'Sharpe Ratio': f"{perf['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{perf['max_drawdown']:.2%}",
                'Win Rate': f"{perf['win_rate']:.1%}",
                'Total Trades': perf['total_trades']
            })
        
        # Display comparison table
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))
        
        # Identify best performing strategy
        if self.results:
            best_sharpe = max(self.results.keys(), 
                             key=lambda x: self.results[x]['performance']['sharpe_ratio'])
            best_return = max(self.results.keys(), 
                             key=lambda x: self.results[x]['performance']['total_return'])
            
            print(f"\n🎯 BEST PERFORMING STRATEGIES:")
            print(f"   Highest Sharpe: {best_sharpe} ({self.results[best_sharpe]['performance']['sharpe_ratio']:.2f})")
            print(f"   Highest Return: {best_return} ({self.results[best_return]['performance']['total_return']:.2%})")
    
    def plot_backtest_results(self, strategies: List[str] = None):
        """Plot comprehensive backtest results."""
        if not strategies:
            strategies = list(self.results.keys())
        
        if not self.results:
            print("❌ No backtest results to plot")
            return
            
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Backtesting Results Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Portfolio Value Over Time
        ax1 = axes[0, 0]
        for strategy in strategies:
            if strategy in self.results:
                data = self.results[strategy]
                ax1.plot(data['dates'], data['portfolio_values'], label=strategy, linewidth=2)
        
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Cumulative Returns
        ax2 = axes[0, 1]
        for strategy in strategies:
            if strategy in self.results:
                data = self.results[strategy]
                initial_value = data['portfolio_values'][0]
                cumulative_returns = [(pv / initial_value) - 1 for pv in data['portfolio_values']]
                ax2.plot(data['dates'], cumulative_returns, label=strategy, linewidth=2)
        
        ax2.set_title('Cumulative Returns')
        ax2.set_ylabel('Cumulative Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown Analysis
        ax3 = axes[1, 0]
        for strategy in strategies:
            if strategy in self.results:
                data = self.results[strategy]
                portfolio_values = data['portfolio_values']
                peak = portfolio_values[0]
                drawdowns = []
                
                for value in portfolio_values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    drawdowns.append(drawdown)
                
                ax3.plot(data['dates'], drawdowns, label=strategy, linewidth=2)
        
        ax3.set_title('Drawdown Analysis')
        ax3.set_ylabel('Drawdown')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Performance Metrics Heatmap
        ax4 = axes[1, 1]
        metrics_data = []
        metric_names = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
        
        for strategy in strategies:
            if strategy in self.results:
                perf = self.results[strategy]['performance']
                metrics_data.append([
                    perf['total_return'],
                    perf['sharpe_ratio'],
                    perf['max_drawdown'],
                    perf['win_rate']
                ])
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data, index=strategies, columns=metric_names)
            sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                       center=0, ax=ax4, cbar_kws={'label': 'Metric Value'})
            ax4.set_title('Performance Metrics Comparison')
        
        plt.tight_layout()
        plt.savefig('backtesting_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self, strategy_name: str):
        """Generate detailed performance report for a specific strategy."""
        if strategy_name not in self.results:
            print(f"❌ Strategy '{strategy_name}' not found in results")
            return
        
        results = self.results[strategy_name]
        performance = results['performance']
        trades = results['trades']
        
        print(f"\n{'='*60}")
        print(f"📈 DETAILED PERFORMANCE REPORT: {strategy_name}")
        print(f"{'='*60}")
        
        print(f"\n📊 RETURN METRICS:")
        print(f"   Total Return: {performance['total_return']:.2%}")
        print(f"   Annualized Return: {performance['annualized_return']:.2%}")
        print(f"   Cumulative Return: {(results['portfolio_values'][-1] / results['portfolio_values'][0] - 1):.2%}")
        
        print(f"\n⚡ RISK METRICS:")
        print(f"   Volatility (Annualized): {performance['volatility']:.2%}")
        print(f"   Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {performance['max_drawdown']:.2%}")
        print(f"   Worst Drawdown Duration: {performance['drawdown_duration']} days")
        if performance['calmar_ratio'] != float('inf'):
            print(f"   Calmar Ratio: {performance['calmar_ratio']:.2f}")
        
        print(f"\n🎯 TRADING METRICS:")
        print(f"   Total Trades: {performance['total_trades']}")
        print(f"   Win Rate: {performance['win_rate']:.1%}")
        print(f"   Average Trade Return: {performance['avg_trade_return']:.2%}")
        if performance['profit_factor'] != float('inf'):
            print(f"   Profit Factor: {performance['profit_factor']:.2f}")
        
        print(f"\n💰 PORTFOLIO METRICS:")
        print(f"   Initial Capital: ${self.initial_capital:,.2f}")
        print(f"   Final Portfolio Value: ${results['portfolio_values'][-1]:,.2f}")
        print(f"   Net Profit: ${results['portfolio_values'][-1] - self.initial_capital:,.2f}")
        
        # Trade analysis
        if trades:
            print(f"\n🔍 RECENT TRADES (Last 10):")
            recent_trades = trades[-10:] if len(trades) > 10 else trades
            for trade in recent_trades:
                print(f"   {trade['date'].strftime('%Y-%m-%d')}: {trade['action']} "
                      f"{trade['quantity']} shares @ ${trade['price']:.2f} "
                      f"(Confidence: {trade['confidence']:.1%})")

class StrategyComparator:
    """Compare different trading strategies."""
    
    def __init__(self):
        self.strategies = {
            'enhanced_analytics': {
                'trading_engine_class': AdvancedProductionTradingEngine,
                'description': 'Enhanced with Technical + Sentiment + Regime Analysis'
            },
            'buy_hold': {
                'trading_engine_class': AdvancedProductionTradingEngine,
                'description': 'Simple Buy & Hold Strategy'
            }
        }
    
    def run_comprehensive_comparison(self, days: int = 252, initial_capital: float = 10000):
        """Run comprehensive strategy comparison."""
        print("🚀 COMPREHENSIVE STRATEGY BACKTESTING COMPARISON")
        print("="*70)
        
        # Generate historical data
        backtester = BacktestingEngine(initial_capital=initial_capital)
        historical_data = backtester.generate_historical_data(days=days)
        
        print(f"📈 Generated {len(historical_data)} days of historical data")
        print(f"💰 Initial Capital: ${initial_capital:,.2f}")
        
        # Run backtests for all strategies
        backtester.run_comparison_backtest(self.strategies, historical_data)
        
        # Generate plots
        backtester.plot_backtest_results()
        
        # Generate detailed reports
        for strategy in self.strategies.keys():
            backtester.generate_detailed_report(strategy)
        
        return backtester.results

def run_backtesting_demo():
    """Run a demonstration of the backtesting framework."""
    print("🎯 BACKTESTING FRAMEWORK DEMONSTRATION")
    print("="*60)
    
    # Initialize comparator
    comparator = StrategyComparator()
    
    # Run comprehensive comparison
    results = comparator.run_comprehensive_comparison(
        days=90,  # 3 months of data for quick testing
        initial_capital=50000
    )
    
    print("\n✅ Backtesting demonstration completed!")
    print("📊 Results saved to: backtesting_results.png")
    print("📋 Detailed reports generated above")
    
    return results

if __name__ == "__main__":
    run_backtesting_demo()
