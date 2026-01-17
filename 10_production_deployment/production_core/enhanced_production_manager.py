# enhanced_production_manager.py
import time
from datetime import datetime
from typing import Dict, List, Optional
from advanced_analytics_integration import AdvancedProductionTradingEngine

class EnhancedProductionTradingManager:
    """Enhanced production trading manager with advanced analytics."""
    
    def __init__(self, initial_capital: float = 5000, trading_interval: int = 300):
        self.engine = AdvancedProductionTradingEngine(initial_capital)
        self.trading_interval = trading_interval
        self.is_running = False
        self.market_hours_only = True
    
    def start_enhanced_trading(self, symbol: str = "ENHANCED_PROD", max_cycles: Optional[int] = None):
        """Start enhanced production trading."""
        print(f"🚀 STARTING ENHANCED PRODUCTION TRADING")
        print(f"   Symbol: {symbol}")
        print(f"   Interval: {self.trading_interval} seconds")
        print(f"   Advanced Features: ✅ Technical + Sentiment + Regime")
        print(f"   Max Cycles: {max_cycles or 'Continuous'}")
        print("=" * 70)
        
        self.is_running = True
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_trades = 0
        start_time = datetime.now()
        
        try:
            cycle_count = 0
            while self.is_running and (max_cycles is None or cycle_count < max_cycles):
                cycle_count += 1
                
                if self.market_hours_only and not self._is_market_hours():
                    print(f"⏸️  Market closed - waiting for next check...")
                    time.sleep(60)
                    continue
                
                print(f"\n🔬 Enhanced Production Cycle {cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                result = self.engine.run_enhanced_production_cycle(symbol)
                
                if result['success']:
                    decision = result['decision']
                    enhancement = decision['enhancement_factors']
                    
                    # Display enhanced results
                    print(f"   📊 Decision: {decision['action']}")
                    print(f"   🎯 Confidence: {decision['confidence']:.1%}")
                    print(f"   📈 Predicted Return: {decision['predicted_return']:.2%}")
                    print(f"   🌡️  Market Regime: {enhancement.get('market_regime', 'N/A')}")
                    print(f"   📰 Sentiment Score: {enhancement.get('news_sentiment', {}).get('sentiment_score', 0):.3f}")
                    print(f"   💰 Position Size: ${decision['position_size']:,.2f}")
                    print(f"   📊 Portfolio Value: ${result['portfolio_value']:,.2f}")
                    print(f"   ⚠️  VaR 95%: {result['risk_metrics']['var_95']:.2%}")
                    
                    if result['trade_executed']:
                        print(f"   ✅ ENHANCED TRADE EXECUTED: {decision['action']}")
                        total_trades += 1
                    else:
                        print(f"   ❌ No trade executed")
                    
                    signal_counts[decision['action']] += 1
                
                if cycle_count % 10 == 0:
                    self.engine.save_production_state()
                
                if cycle_count % 20 == 0:
                    self._print_enhanced_performance_report(cycle_count, total_trades, start_time)
                
                if cycle_count < (max_cycles or float('inf')):
                    print(f"   ⏰ Next enhanced cycle in {self.trading_interval} seconds...")
                    time.sleep(self.trading_interval)
                    
        except KeyboardInterrupt:
            print("\n🛑 Enhanced production trading stopped by user")
        except Exception as e:
            print(f"❌ Enhanced production trading error: {e}")
        finally:
            self.stop_enhanced_trading(signal_counts, total_trades, cycle_count, start_time)
    
    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours."""
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        market_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_end = now.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_start <= now <= market_end
    
    def _print_enhanced_performance_report(self, cycle_count: int, total_trades: int, start_time: datetime):
        """Print enhanced performance report."""
        # Implementation similar to basic version but with enhanced metrics
        runtime = datetime.now() - start_time
        print(f"\n📊 ENHANCED PERFORMANCE REPORT")
        print(f"   Runtime: {runtime}")
        print(f"   Total Cycles: {cycle_count}")
        print(f"   Total Trades: {total_trades}")
        print(f"   Advanced Analytics: ✅ Active")
    
    def stop_enhanced_trading(self, signal_counts: Dict, total_trades: int, total_cycles: int, start_time: datetime):
        """Stop enhanced production trading."""
        self.is_running = False
        runtime = datetime.now() - start_time
        print(f"\n📈 ENHANCED TRADING SUMMARY")
        print(f"   Total Runtime: {runtime}")
        print(f"   Total Cycles: {total_cycles}")
        print(f"   Total Trades: {total_trades}")
        print(f"   Advanced Features Used: ✅ Technical + Sentiment + Regime")
        print("🛑 Enhanced production trading stopped")
