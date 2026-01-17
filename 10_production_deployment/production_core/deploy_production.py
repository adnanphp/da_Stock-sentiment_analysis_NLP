# deploy_production.py
import sys
import time
import schedule
from datetime import datetime
from live_trading_engine import LiveTradingManager
from performance_monitor import RealTimeMonitor
import threading

class ProductionDeployment:
    """
    Production deployment and scheduling system.
    """
    
    def __init__(self, initial_capital=5000):
        self.initial_capital = initial_capital
        self.trading_manager = None
        self.monitor = None
        self.is_running = False
        
    def start_production_trading(self):
        """Start full production trading system."""
        print("🏢 STARTING PRODUCTION TRADING SYSTEM")
        print("=" * 60)
        
        # Initialize systems
        self.trading_manager = LiveTradingManager(
            initial_capital=self.initial_capital,
            trading_interval=300  # 5-minute intervals
        )
        
        self.monitor = RealTimeMonitor(
            trading_manager=self.trading_manager,
            check_interval=60  # 1-minute monitoring
        )
        
        self.is_running = True
        
        # Start systems in separate threads
        trading_thread = threading.Thread(target=self._start_trading)
        monitoring_thread = threading.Thread(target=self._start_monitoring)
        
        trading_thread.daemon = True
        monitoring_thread.daemon = True
        
        trading_thread.start()
        monitoring_thread.start()
        
        print("✅ Production system started")
        print("   • Trading: 5-minute intervals")
        print("   • Monitoring: 1-minute checks")
        print("   • Capital: ${:,.2f}".format(self.initial_capital))
        print("\nPress Ctrl+C to stop the system")
        
        # Keep main thread alive
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_production_trading()
    
    def _start_trading(self):
        """Start trading in a separate thread."""
        try:
            self.trading_manager.start_live_trading(
                symbol="PRODUCTION",
                max_cycles=None  # Run indefinitely
            )
        except Exception as e:
            print(f"❌ Trading error: {e}")
    
    def _start_monitoring(self):
        """Start monitoring in a separate thread."""
        try:
            self.monitor.start_monitoring()
        except Exception as e:
            print(f"❌ Monitoring error: {e}")
    
    def stop_production_trading(self):
        """Stop the production trading system."""
        print("\n🛑 Stopping production trading system...")
        self.is_running = False
        
        if self.trading_manager:
            self.trading_manager.stop_live_trading()
        
        if self.monitor:
            self.monitor.stop_monitoring()
        
        print("✅ Production system stopped")
    
    def schedule_market_hours(self):
        """Schedule trading during market hours only."""
        print("⏰ Scheduling market hours trading (9:30 AM - 4:00 PM)")
        
        # Schedule start at market open
        schedule.every().monday.at("09:25").do(self.start_production_trading)
        schedule.every().tuesday.at("09:25").do(self.start_production_trading)
        schedule.every().wednesday.at("09:25").do(self.start_production_trading)
        schedule.every().thursday.at("09:25").do(self.start_production_trading)
        schedule.every().friday.at("09:25").do(self.start_production_trading)
        
        # Schedule stop at market close
        schedule.every().monday.at("16:05").do(self.stop_production_trading)
        schedule.every().tuesday.at("16:05").do(self.stop_production_trading)
        schedule.every().wednesday.at("16:05").do(self.stop_production_trading)
        schedule.every().thursday.at("16:05").do(self.stop_production_trading)
        schedule.every().friday.at("16:05").do(self.stop_production_trading)
        
        print("✅ Market hours scheduling configured")
        
        # Keep scheduler running
        while True:
            schedule.run_pending()
            time.sleep(1)

def main():
    """Main deployment function."""
    print("🚀 STOCK PRADICTION PRODUCTION DEPLOYMENT")
    print("=" * 50)
    
    print("Select deployment mode:")
    print("1. Continuous Trading (24/7)")
    print("2. Market Hours Only (9:30 AM - 4:00 PM)")
    print("3. Demo Mode (5 cycles)")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        deployment = ProductionDeployment(initial_capital=5000)
        
        if choice == "1":
            print("\n🏢 STARTING CONTINUOUS TRADING")
            deployment.start_production_trading()
            
        elif choice == "2":
            print("\n⏰ STARTING MARKET HOURS TRADING")
            deployment.schedule_market_hours()
            
        elif choice == "3":
            print("\n🧪 STARTING DEMO MODE")
            # Run quick demo
            manager = LiveTradingManager(initial_capital=1000, trading_interval=30)
            manager.start_live_trading(symbol="DEMO", max_cycles=5)
            manager.get_performance_report()
            
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n🛑 Deployment stopped by user")
    except Exception as e:
        print(f"❌ Deployment error: {e}")

if __name__ == "__main__":
    main()
