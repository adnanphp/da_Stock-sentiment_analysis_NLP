# run_fixed_system.py
import sys
import os
from enhanced_dashboard import EnhancedModelDashboard
from fixed_enhanced_predictor import FixedEnhancedStockPredictor, MockDataFetcher
import pandas as pd

def main():
    """Main execution function for the fixed production system."""
    print("🚀 FIXED STOCK PREDICTION PRODUCTION SYSTEM")
    print("=" * 60)
    
    # Step 1: Create enhanced dashboard
    print("\n1. 📊 CREATING ENHANCED DASHBOARD")
    dashboard = EnhancedModelDashboard()
    perf_df, risk_metrics = dashboard.create_enhanced_dashboard()
    
    # Step 2: Initialize fixed predictor
    print("\n2. 🎯 INITIALIZING FIXED ENHANCED PREDICTOR")
    predictor = FixedEnhancedStockPredictor()
    
    # Show detailed model info
    model_info = predictor.get_model_info()
    print("\n📋 MODEL DETAILS:")
    for target, info in model_info.items():
        print(f"   • {target}: {info['type']} model")
        if 'r2' in info and info['r2'] != 'N/A':
            print(f"     R²: {info['r2']:.4f}")
        if info.get('expected_features'):
            print(f"     Features: {len(info['expected_features'])} expected")
    
    # Step 3: Run backtest
    print("\n3. 📈 RUNNING BACKTEST")
    try:
        historical_data = pd.read_csv("feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv")
        backtest_results = predictor.backtest_strategy(historical_data)
        
        if backtest_results:
            print("   ✅ Backtest completed successfully")
        else:
            print("   ⚠️  Backtest had issues - check model compatibility")
            
    except Exception as e:
        print(f"   ❌ Backtest failed: {e}")
    
    # Step 4: Demo predictions
    print("\n4. 🎯 DEMO PREDICTIONS")
    try:
        data_fetcher = MockDataFetcher()
        new_data = data_fetcher.fetch_latest_data()
        predictions = predictor.predict(new_data)
        
        current_price = new_data['Close'].iloc[-1] if 'Close' in new_data.columns else 100
        signals = predictor.enhanced_trading_signals(predictions, current_price)
        
        position_size = predictor.calculate_position_size(
            predictor.current_capital,
            signals['confidence'],
            signals['predicted_volatility'],
            signals['risk_level']
        )
        
        print(f"   Latest Signal: {signals['action']}")
        print(f"   Confidence: {signals['confidence']:.2%}")
        print(f"   Risk Level: {signals['risk_level']}")
        print(f"   Recommended Position: ${position_size:,.2f}")
        print(f"   Predicted Return: {signals['predicted_return']:.4f}")
        
    except Exception as e:
        print(f"   ❌ Demo predictions failed: {e}")
    
    print("\n" + "=" * 60)
    print("✅ FIXED PRODUCTION SYSTEM READY")
    print("\n🎯 Next steps:")
    print("1. Review enhanced_dashboard.png")
    print("2. Check model feature compatibility above")
    print("3. Run: python fixed_enhanced_predictor.py for detailed testing")
    print("4. Use predictor.real_time_prediction_loop() for live trading")

if __name__ == "__main__":
    main()
