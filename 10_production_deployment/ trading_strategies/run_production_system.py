# run_production_system.py
import sys
import os
from enhanced_dashboard import EnhancedModelDashboard
from enhanced_predictor import EnhancedStockPredictor, MockDataFetcher
from enhanced_training import EnhancedRobustTrainer
import pandas as pd

def main():
    """Main execution function for the production system."""
    print("🚀 STOCK PREDICTION PRODUCTION SYSTEM")
    print("=" * 60)
    
    # Step 1: Create enhanced dashboard
    print("\n1. 📊 CREATING ENHANCED DASHBOARD")
    dashboard = EnhancedModelDashboard()
    perf_df, risk_metrics = dashboard.create_enhanced_dashboard()
    
    # Step 2: Check if models exist, train if not
    print("\n2. 🤖 CHECKING/TRAINING MODELS")
    if not os.path.exists("robust_models"):
        print("   No existing models found. Starting enhanced training...")
        try:
            df = pd.read_csv("feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv")
            trainer = EnhancedRobustTrainer()
            
            targets = ['daily_return', 'log_return', 'volatility_5d']
            for target in targets:
                if target in df.columns:
                    print(f"   Training {target}...")
                    model, mse, r2, features = trainer.robust_time_series_training(df, target)
                    trainer.save_enhanced_model(model, target, mse, r2, features)
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
    else:
        print("   ✅ Existing models found")
    
    # Step 3: Initialize enhanced predictor
    print("\n3. 🎯 INITIALIZING ENHANCED PREDICTOR")
    predictor = EnhancedStockPredictor()
    
    # Step 4: Run backtest
    print("\n4. 📈 RUNNING BACKTEST")
    try:
        historical_data = pd.read_csv("feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv")
        backtest_results = predictor.backtest_strategy(historical_data.tail(200))
    except Exception as e:
        print(f"   ❌ Backtest failed: {e}")
    
    # Step 5: Start monitoring (optional - for demo)
    print("\n5. 🔄 STARTING MONITORING SYSTEM")
    print("   Monitoring system ready - use real_time_prediction_loop() for live trading")
    
    # Demo predictions
    print("\n6. 🎯 DEMO PREDICTIONS")
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
    print("✅ PRODUCTION SYSTEM READY")
    print("\nNext steps:")
    print("1. Review enhanced_dashboard.png for model performance")
    print("2. Run backtest with more data if needed")
    print("3. Use predictor.real_time_prediction_loop() for live trading")
    print("4. Monitor model performance regularly")

if __name__ == "__main__":
    main()
