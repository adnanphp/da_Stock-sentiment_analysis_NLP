# quick_test.py
from fixed_live_trading_engine import LiveTradingManager

def quick_test():
    print("🧪 QUICK TEST - Live Trading System")
    print("=" * 40)
    
    # Test with small capital and quick cycles
    manager = LiveTradingManager(initial_capital=1000, trading_interval=10)
    
    try:
        # Run just 3 cycles for quick test
        manager.start_live_trading(symbol="TEST", max_cycles=3)
        
        # Show results
        manager.get_performance_report()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    quick_test()
