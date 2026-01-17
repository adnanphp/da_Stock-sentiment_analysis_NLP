# quick_backtest.py
import pandas as pd
import numpy as np
from backtesting_framework import BacktestingEngine
import warnings
warnings.filterwarnings('ignore')

# Import the trading engine
try:
    from advanced_analytics_integration import AdvancedProductionTradingEngine
except ImportError:
    print("⚠️  Could not import from advanced_analytics_integration")
    # Create a simple version for testing
    class AdvancedProductionTradingEngine:
        def __init__(self, initial_capital=10000):
            self.initial_capital = initial_capital
            self.current_capital = initial_capital
            
            class SimplePredictor:
                def generate_enhanced_signals(self, current_price, symbol):
                    return {
                        'action': np.random.choice(['BUY', 'SELL', 'HOLD'], p=[0.4, 0.3, 0.3]),
                        'confidence': np.random.uniform(0.1, 0.9),
                        'predicted_return': np.random.uniform(-0.05, 0.05),
                        'predicted_volatility': np.random.uniform(0.05, 0.2),
                        'risk_level': np.random.choice(['LOW', 'MEDIUM', 'HIGH']),
                        'enhancement_factors': {}
                    }
            
            self.predictor = SimplePredictor()
            
        def calculate_enhanced_position(self, capital, confidence, risk_level, enhancement_factors):
            risk_tiers = {'HIGH': 0.01, 'MEDIUM': 0.02, 'LOW': 0.03}
            base_risk = risk_tiers.get(risk_level, 0.02)
            position_size = capital * base_risk * min(confidence, 1.0)
            return min(position_size, capital * 0.05)

def quick_strategy_test():
    """Quick test of individual strategies."""
    print("⚡ QUICK STRATEGY BACKTEST")
    print("="*50)
    
    # Initialize backtester
    backtester = BacktestingEngine(initial_capital=10000)
    
    # Generate shorter historical data for quick testing
    historical_data = backtester.generate_historical_data(days=30)  # 1 month
    
    print("Testing Enhanced Analytics Strategy...")
    
    # Test enhanced strategy
    results = backtester.run_backtest(
        "Enhanced_Analytics_Quick",
        historical_data,
        trading_engine_class=AdvancedProductionTradingEngine
    )
    
    # Display quick results
    perf = results['performance']
    print(f"\n📊 QUICK RESULTS:")
    print(f"   Period: {len(historical_data)} days")
    print(f"   Final Value: ${results['portfolio_values'][-1]:,.2f}")
    print(f"   Total Return: {perf['total_return']:.2%}")
    print(f"   Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {perf['max_drawdown']:.2%}")
    print(f"   Total Trades: {perf['total_trades']}")
    
    return results

if __name__ == "__main__":
    quick_strategy_test()
