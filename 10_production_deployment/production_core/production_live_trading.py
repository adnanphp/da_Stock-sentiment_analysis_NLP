# production_live_trading.py
import pandas as pd
import numpy as np
import time
import json
import logging
from datetime import datetime, timedelta
from production_predictor import ProductionStockPredictor, ProductionDataFetcher
import warnings
warnings.filterwarnings('ignore')

class ProductionTradingEngine:
    """
    Production live trading engine with all enhancements.
    """
    
    def __init__(self, initial_capital=10000, model_dir="robust_models"):
        self.predictor = ProductionStockPredictor(model_dir)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.performance_log = []
        self.cycle_count = 0
        
        # Production trading parameters
        self.max_position_size = 0.07
        self.daily_loss_limit = 0.025
        self.max_drawdown_limit = 0.08
        
        # Setup production logging
        self.setup_logging()
        
        print("🚀 PRODUCTION TRADING ENGINE INITIALIZED")
        print(f"   Initial Capital: ${initial_capital:,.2f}")
        print(f"   Max Position Size: {self.max_position_size:.1%}")
        print(f"   Daily Loss Limit: {self.daily_loss_limit:.1%}")
        print(f"   Max Drawdown Limit: {self.max_drawdown_limit:.1%}")
    
    def setup_logging(self):
        """Setup production logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('production_trading.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_trading_cycle(self, data_fetcher, symbol="STOCK"):
        """Run one production trading cycle."""
        try:
            self.cycle_count += 1
            self.logger.info(f"🔄 Production cycle {self.cycle_count}...")
            
            # 1. Fetch production data
            new_data = data_fetcher.fetch_latest_data()
            self.logger.info(f"📊 Fetched {len(new_data)} data points")
            
            # 2. Make production predictions
            predictions = self.predictor.predict(new_data)
            self.logger.info("🎯 Production predictions generated")
            
            # 3. Generate trading decision
            current_price = new_data['Close'].iloc[-1] if 'Close' in new_data.columns else 100
            decision = self.generate_trading_decision(predictions, new_data, symbol)
            
            # 4. Execute trade with production threshold (15% confidence)
            if decision['action'] != 'HOLD' and decision['confidence'] > 0.15:
                executed = self.execute_trade(
                    symbol=symbol,
                    action=decision['action'],
                    quantity=decision['quantity'],
                    price=decision['price'],
                    confidence=decision['confidence']
                )
                
                if executed:
                    self.logger.info(f"✅ Trade executed: {decision['action']} {decision['quantity']} shares")
                else:
                    self.logger.info("❌ Trade execution skipped")
            else:
                self.logger.info("⏸️  HOLD signal - no trade executed")
            
            # 5. Log performance
            portfolio_value = self.current_capital + sum(
                pos['quantity'] * decision['price'] for pos in self.positions.values()
            )
            
            performance_entry = {
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_value,
                'cash': self.current_capital,
                'positions': len(self.positions),
                'decision': decision
            }
            
            self.performance_log.append(performance_entry)
            
            # 6. Check risk limits
            risk_breaches, risk_metrics = self.check_risk_limits()
            if risk_breaches:
                self.logger.error(f"🚨 RISK LIMIT BREACH: {risk_breaches}")
            
            return {
                'success': True,
                'decision': decision,
                'risk_metrics': risk_metrics,
                'risk_breaches': risk_breaches,
                'portfolio_value': portfolio_value
            }
            
        except Exception as e:
            self.logger.error(f"❌ Production cycle error: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def generate_trading_decision(self, predictions, current_data, symbol="STOCK"):
        """Generate production trading decision."""
        current_price = current_data['Close'].iloc[-1] if 'Close' in current_data.columns else 100
        
        signals = self.predictor.enhanced_trading_signals(predictions, current_price)
        
        position_size = self.predictor.calculate_position_size(
            self.current_capital,
            signals['confidence'],
            signals['predicted_volatility'],
            signals['risk_level']
        )
        
        quantity = int(position_size / current_price)
        
        if quantity < 1:
            quantity = 1
            position_size = quantity * current_price
        
        decision = {
            'action': signals['action'],
            'quantity': quantity,
            'price': current_price,
            'confidence': signals['confidence'],
            'risk_level': signals['risk_level'],
            'predicted_return': signals['predicted_return'],
            'predicted_volatility': signals['predicted_volatility'],  # Fixed: include volatility
            'position_size': position_size,
            'timestamp': datetime.now()
        }
        
        return decision
    
    def execute_trade(self, symbol, action, quantity, price, confidence):
        """Execute production trade."""
        trade_cost = quantity * price
        commission = max(trade_cost * 0.001, 1)
        
        if action.upper() == 'BUY':
            if trade_cost + commission > self.current_capital:
                self.logger.warning("Insufficient capital for BUY order")
                return False
            
            self.current_capital -= (trade_cost + commission)
            if symbol in self.positions:
                self.positions[symbol]['quantity'] += quantity
            else:
                self.positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'entry_time': datetime.now()
                }
            
        elif action.upper() == 'SELL':
            if symbol not in self.positions or self.positions[symbol]['quantity'] < quantity:
                self.logger.warning("Insufficient position for SELL order")
                return False
            
            entry_value = self.positions[symbol]['avg_price'] * quantity
            exit_value = price * quantity
            pnl = exit_value - entry_value - commission
            
            self.current_capital += (exit_value - commission)
            
            if self.positions[symbol]['quantity'] == quantity:
                del self.positions[symbol]
            else:
                self.positions[symbol]['quantity'] -= quantity
        else:
            return False
        
        # Record trade
        trade_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'action': action.upper(),
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'confidence': confidence,
            'portfolio_value': self.current_capital,
            'pnl': pnl if action.upper() == 'SELL' else 0
        }
        
        self.trade_history.append(trade_record)
        return True
    
    def check_risk_limits(self):
        """Check production risk limits."""
        portfolio_value = self.current_capital + sum(
            pos['quantity'] * 100 for pos in self.positions.values()
        )
        
        if self.performance_log:
            peak_capital = max([self.initial_capital] + 
                              [entry['portfolio_value'] for entry in self.performance_log])
            drawdown = (peak_capital - portfolio_value) / peak_capital
        else:
            drawdown = 0
        
        risk_breaches = []
        
        if drawdown > self.max_drawdown_limit:
            risk_breaches.append(f"Max drawdown limit breached: {drawdown:.2%}")
        
        return risk_breaches, {
            'portfolio_value': portfolio_value,
            'drawdown': drawdown
        }
    
    def get_portfolio_summary(self):
        """Get production portfolio summary."""
        return self.predictor.get_portfolio_summary()
    
    def save_portfolio_state(self):
        """Save production portfolio state."""
        self.predictor.save_portfolio_state()

# Production Trading Manager
class ProductionTradingManager:
    def __init__(self, initial_capital=5000, trading_interval=300):
        self.engine = ProductionTradingEngine(initial_capital)
        self.data_fetcher = ProductionDataFetcher()
        self.trading_interval = trading_interval
        self.is_running = False
        
    def start_live_trading(self, symbol="PRODUCTION", max_cycles=None):
        """Start production live trading."""
        print(f"🚀 STARTING PRODUCTION LIVE TRADING")
        print(f"   Symbol: {symbol}")
        print(f"   Interval: {self.trading_interval} seconds")
        print(f"   Max Cycles: {max_cycles or 'Unlimited'}")
        print("=" * 50)
        
        self.is_running = True
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_confidence = 0
        
        try:
            cycle_count = 0
            while self.is_running and (max_cycles is None or cycle_count < max_cycles):
                cycle_count += 1
                print(f"\n🔄 Production Cycle {cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                result = self.engine.run_trading_cycle(self.data_fetcher, symbol)
                
                if result['success']:
                    decision = result['decision']
                    risk_metrics = result['risk_metrics']
                    
                    print(f"   📊 Decision: {decision['action']}")
                    print(f"   🎯 Confidence: {decision['confidence']:.1%}")
                    print(f"   📈 Predicted Return: {decision['predicted_return']:.4f}")
                    print(f"   📊 Volatility: {decision['predicted_volatility']:.4f}")
                    print(f"   ⚠️  Risk Level: {decision['risk_level']}")
                    print(f"   💰 Position Size: ${decision['position_size']:,.2f}")
                    print(f"   📊 Portfolio Value: ${result['portfolio_value']:,.2f}")
                    print(f"   📉 Drawdown: {risk_metrics['drawdown']:.2%}")
                    
                    signal_counts[decision['action']] += 1
                    total_confidence += decision['confidence']
                    
                    if result['risk_breaches']:
                        print(f"   🚨 RISK ALERT: {result['risk_breaches']}")
                
                if cycle_count < (max_cycles or float('inf')):
                    print(f"   ⏰ Next cycle in {self.trading_interval} seconds...")
                    time.sleep(self.trading_interval)
                    
        except KeyboardInterrupt:
            print("\n🛑 Production trading stopped by user")
        finally:
            self.stop_production_trading(signal_counts, total_confidence, cycle_count)
    
    def stop_production_trading(self, signal_counts, total_confidence, total_cycles):
        """Stop production trading with detailed summary."""
        self.is_running = False
        self.engine.save_portfolio_state()
        
        if total_cycles > 0:
            avg_confidence = total_confidence / total_cycles
            
            print(f"\n📈 PRODUCTION TRADING SUMMARY:")
            print(f"   Total Cycles: {total_cycles}")
            for action, count in signal_counts.items():
                percentage = (count / total_cycles) * 100
                print(f"   {action}: {count} cycles ({percentage:.1f}%)")
            print(f"   Average Confidence: {avg_confidence:.1%}")
        
        print("🛑 Production trading stopped")
    
    def get_performance_report(self):
        """Generate production performance report."""
        summary = self.engine.get_portfolio_summary()
        
        print("\n" + "="*60)
        print("📊 PRODUCTION PERFORMANCE REPORT")
        print("="*60)
        
        print(f"💰 Portfolio Value: ${summary['current_value']:,.2f}")
        print(f"📈 Total Return: {summary['total_return']:.2%}")
        print(f"💵 Cash Balance: ${summary['cash']:,.2f}")
        print(f"🔢 Total Trades: {summary['total_trades']}")
        print(f"📦 Active Positions: {len(summary['active_positions'])}")
        print(f"📊 Daily P&L: ${summary['daily_pnl']:,.2f}")
        print(f"🏆 Daily Win Rate: {summary['daily_win_rate']:.1%}")
        
        return summary

# Production Demo
def production_demo():
    """Production demo with enhanced signals."""
    print("🧪 PRODUCTION DEMO - Enhanced Signals")
    print("=" * 50)
    
    # Quick demo with fast cycles
    manager = ProductionTradingManager(initial_capital=1000, trading_interval=10)
    
    try:
        # Run 8 cycles to see signal variation
        manager.start_live_trading(symbol="PROD_DEMO", max_cycles=8)
        
        # Show production report
        manager.get_performance_report()
        
    except Exception as e:
        print(f"❌ Production demo failed: {e}")

# Production Deployment
def production_deployment():
    """Full production deployment."""
    print("🚀 PRODUCTION DEPLOYMENT")
    print("=" * 40)
    
    print("Select deployment mode:")
    print("1. Quick Demo (8 cycles, 10-second intervals)")
    print("2. Extended Test (20 cycles, 30-second intervals)") 
    print("3. Production Mode (continuous, 5-minute intervals)")
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            production_demo()
        elif choice == "2":
            manager = ProductionTradingManager(initial_capital=1000, trading_interval=30)
            manager.start_live_trading(symbol="EXTENDED_TEST", max_cycles=20)
            manager.get_performance_report()
        elif choice == "3":
            manager = ProductionTradingManager(initial_capital=5000, trading_interval=300)
            print("🚀 STARTING PRODUCTION MODE")
            print("Press Ctrl+C to stop production trading")
            manager.start_live_trading(symbol="PRODUCTION")
        else:
            print("❌ Invalid choice, running demo...")
            production_demo()
            
    except KeyboardInterrupt:
        print("\n🛑 Production deployment stopped by user")
    except Exception as e:
        print(f"❌ Production deployment error: {e}")

if __name__ == "__main__":
    production_deployment()
