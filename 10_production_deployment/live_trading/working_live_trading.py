# working_live_trading.py
import pandas as pd
import numpy as np
import time
import json
import logging
from datetime import datetime, timedelta
from working_enhanced_predictor import WorkingEnhancedStockPredictor, WorkingDataFetcher
import warnings
warnings.filterwarnings('ignore')

class WorkingLiveTradingEngine:
    """
    Working live trading engine with all fixes applied.
    """
    
    def __init__(self, initial_capital=10000, model_dir="robust_models"):
        self.predictor = WorkingEnhancedStockPredictor(model_dir)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.performance_log = []  # Initialize performance_log
        self.cycle_count = 0
        
        # Enhanced trading parameters
        self.max_position_size = 0.08
        self.daily_loss_limit = 0.03
        self.max_drawdown_limit = 0.10
        
        # Setup logging
        self.setup_logging()
        
        print("🚀 WORKING LIVE TRADING ENGINE INITIALIZED")
        print(f"   Initial Capital: ${initial_capital:,.2f}")
        print(f"   Max Position Size: {self.max_position_size:.1%}")
        print(f"   Daily Loss Limit: {self.daily_loss_limit:.1%}")
        print(f"   Max Drawdown Limit: {self.max_drawdown_limit:.1%}")
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('working_trading_log.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_trading_cycle(self, data_fetcher, symbol="STOCK"):
        """Run one complete trading cycle with enhanced variation."""
        try:
            self.cycle_count += 1
            self.logger.info(f"🔄 Starting trading cycle {self.cycle_count}...")
            
            # 1. Fetch latest data with enhanced variation
            new_data = data_fetcher.fetch_latest_data()
            self.logger.info(f"📊 Fetched {len(new_data)} data points")
            
            # 2. Make predictions with enhanced variation
            predictions = self.predictor.predict(new_data)
            self.logger.info("🎯 Enhanced predictions generated")
            
            # 3. Generate trading decision
            current_price = new_data['Close'].iloc[-1] if 'Close' in new_data.columns else 100
            decision = self.generate_trading_decision(predictions, new_data, symbol)
            
            # 4. Execute trade if confidence > 20% (more aggressive)
            if decision['action'] != 'HOLD' and decision['confidence'] > 0.20:
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
                    self.logger.info("❌ Trade execution failed or skipped")
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
            self.logger.error(f"❌ Trading cycle error: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    # fixed_working_live_trading.py
    # Just replace the generate_trading_decision method in WorkingLiveTradingEngine:

    def generate_trading_decision(self, predictions, current_data, symbol="STOCK"):
        """Generate complete trading decision with fixed volatility display."""
        current_price = current_data['Close'].iloc[-1] if 'Close' in current_data.columns else 100

        # Get enhanced signals from predictor
        signals = self.predictor.enhanced_trading_signals(predictions, current_price)

        # Enhanced position sizing
        position_size = self.predictor.calculate_position_size(
            self.current_capital,
            signals['confidence'],
            signals['predicted_volatility'],
            signals['risk_level']
        )

        # Calculate quantity
        quantity = int(position_size / current_price)

        # Minimum quantity check
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
            'predicted_volatility': signals['predicted_volatility'],  # ADD THIS LINE
            'position_size': position_size,
            'timestamp': datetime.now()
        }

        return decision
    def execute_trade(self, symbol, action, quantity, price, confidence):
        """Execute a trade with enhanced logic."""
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
            
            # Calculate P&L
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
        """Check if any risk limits are breached."""
        portfolio_value = self.current_capital + sum(
            pos['quantity'] * 100 for pos in self.positions.values()
        )
        
        # Calculate drawdown
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
        """Get current portfolio summary."""
        return self.predictor.get_portfolio_summary()
    
    def save_portfolio_state(self):
        """Save current portfolio state to file."""
        self.predictor.save_portfolio_state()

# Working Live Trading Manager
class WorkingLiveTradingManager:
    def __init__(self, initial_capital=5000, trading_interval=300):
        self.engine = WorkingLiveTradingEngine(initial_capital)
        self.data_fetcher = WorkingDataFetcher()
        self.trading_interval = trading_interval
        self.is_running = False
        
    def start_live_trading(self, symbol="STOCK", max_cycles=None):
        """Start live trading loop."""
        print(f"🚀 STARTING WORKING LIVE TRADING")
        print(f"   Symbol: {symbol}")
        print(f"   Interval: {self.trading_interval} seconds")
        print(f"   Max Cycles: {max_cycles or 'Unlimited'}")
        print("=" * 50)
        
        self.is_running = True
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        try:
            cycle_count = 0
            while self.is_running and (max_cycles is None or cycle_count < max_cycles):
                cycle_count += 1
                print(f"\n🔄 Trading Cycle {cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Run trading cycle
                result = self.engine.run_trading_cycle(self.data_fetcher, symbol)
                
                # Display results
                if result['success']:
                    decision = result['decision']
                    risk_metrics = result['risk_metrics']
                    
                    print(f"   📊 Decision: {decision['action']}")
                    print(f"   🎯 Confidence: {decision['confidence']:.1%}")
                    print(f"   📈 Predicted Return: {decision['predicted_return']:.4f}")
                    print(f"   📊 Volatility: {decision.get('predicted_volatility', 0):.4f}")
                    print(f"   ⚠️  Risk Level: {decision['risk_level']}")
                    print(f"   💰 Position Size: ${decision['position_size']:,.2f}")
                    print(f"   📊 Portfolio Value: ${result['portfolio_value']:,.2f}")
                    print(f"   📉 Drawdown: {risk_metrics['drawdown']:.2%}")
                    
                    # Count signals
                    signal_counts[decision['action']] += 1
                    
                    if result['risk_breaches']:
                        print(f"   🚨 RISK ALERT: {result['risk_breaches']}")
                
                # Wait for next cycle
                if cycle_count < (max_cycles or float('inf')):
                    print(f"   ⏰ Next cycle in {self.trading_interval} seconds...")
                    time.sleep(self.trading_interval)
                    
        except KeyboardInterrupt:
            print("\n🛑 Live trading stopped by user")
        finally:
            self.stop_live_trading(signal_counts)
    
    def stop_live_trading(self, signal_counts=None):
        """Stop live trading with summary."""
        self.is_running = False
        self.engine.save_portfolio_state()
        
        if signal_counts:
            print(f"\n📈 TRADING SUMMARY:")
            total_cycles = sum(signal_counts.values())
            for action, count in signal_counts.items():
                percentage = (count / total_cycles) * 100 if total_cycles > 0 else 0
                print(f"   {action}: {count} cycles ({percentage:.1f}%)")
        
        print("🛑 Live trading stopped")
    
    def get_performance_report(self):
        """Generate performance report."""
        summary = self.engine.get_portfolio_summary()
        
        print("\n" + "="*60)
        print("📊 WORKING LIVE TRADING PERFORMANCE REPORT")
        print("="*60)
        
        print(f"💰 Portfolio Value: ${summary['current_value']:,.2f}")
        print(f"📈 Total Return: {summary['total_return']:.2%}")
        print(f"💵 Cash Balance: ${summary['cash']:,.2f}")
        print(f"🔢 Total Trades: {summary['total_trades']}")
        print(f"📦 Active Positions: {len(summary['active_positions'])}")
        
        return summary

# Final Working Test
def working_final_test():
    """Final test of the working system."""
    print("🧪 WORKING FINAL TEST - Enhanced Variation")
    print("=" * 50)
    
    # Test with quick cycles to see variation
    manager = WorkingLiveTradingManager(initial_capital=1000, trading_interval=10)
    
    try:
        # Run 5 cycles to see signal variation
        manager.start_live_trading(symbol="WORKING_TEST", max_cycles=5)
        
        # Show results
        manager.get_performance_report()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    working_final_test()
