# integrated_interpretation.py
import pandas as pd
import numpy as np
from backtesting_framework import BacktestingEngine, StrategyComparator
from model_interpretation import ModelInterpretationDashboard
import warnings
warnings.filterwarnings('ignore')

def interpret_backtest_results():
    """Run backtest and then interpret the results."""
    print("🔍 INTERPRETING BACKTEST RESULTS")
    print("="*60)
    
    # Run a quick backtest first
    backtester = BacktestingEngine(initial_capital=50000)
    historical_data = backtester.generate_historical_data(days=60)
    
    print("Running quick backtest...")
    results = backtester.run_backtest(
        "Interpretation_Test",
        historical_data
    )
    
    # Initialize interpretation dashboard
    dashboard = ModelInterpretationDashboard()
    
    # Create mock trading engine for interpretation
    class MockTradingEngine:
        def __init__(self):
            self.predictor = None
    
    trading_engine = MockTradingEngine()
    
    # Run interpretation analysis
    print("\n" + "="*60)
    print("🤖 RUNNING MODEL INTERPRETATION")
    print("="*60)
    
    analysis_results = dashboard.run_comprehensive_analysis(
        trading_engine,
        historical_data,
        results['trades']
    )
    
    # Generate visualization
    dashboard.plot_interpretation_results(analysis_results)
    
    print("\n✅ Interpretation Complete!")
    print("📊 Check the generated reports and visualizations")
    
    return analysis_results, results

if __name__ == "__main__":
    interpret_backtest_results()
