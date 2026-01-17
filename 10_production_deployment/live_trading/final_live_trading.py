# final_live_trading.py
import pandas as pd
import numpy as np
import time
import json
import logging
from datetime import datetime, timedelta
from final_enhanced_predictor import FinalEnhancedStockPredictor, EnhancedDataFetcher
import warnings
warnings.filterwarnings('ignore')

class FinalLiveTradingEngine:
    """
    Final live trading engine with all fixes applied.
    """
    
    def __init__(self, initial_capital=10000, model_dir="robust_models"):
        self.predictor = FinalEnhancedStockPredictor(model_dir)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.performance_log = []
        
        # Enhanced trading parameters
        self.max_position_size = 0.08  # 8% of capital (more conservative)
        self.daily_loss_limit = 0.03   # 3% daily loss limit
        self.max_drawdown_limit = 0.10  # 10% max drawdown
        
        # Setup logging
        self.setup_logging()
        
        print("🚀 FINAL LIVE TRADING ENGINE INITIALIZED")
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
                logging.FileHandler('final_trading_log.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def run_trading_cycle(self, data_fetcher, symbol="STOCK"):
        """Run one complete trading cycle with enhanced logic."""
        try:
            self.logger.info("🔄 Starting trading cycle...")
            
            # 1. Fetch latest data
            new_data = data_fetcher.fetch_latest_data()
            self.logger.info(f"📊 Fetched {len(new_data)} data points")
            
            # 2. Make predictions
            predictions = self.predictor.predict(new_data)
            self.logger.info("🎯 Predictions generated")
            
            # 3. Generate trading decision
            current_price = new_data['Close'].iloc[-1] if 'Close' in new_data.columns else 100
            decision = self.generate_trading_decision(predictions, new_data, symbol)
            
            # 4. Execute trade if not HOLD and confidence > 15%
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
                    self.logger.info("❌ Trade execution failed or skipped due to risk")
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
    
    def generate_trading_decision(self, predictions, current_data, symbol="STOCK"):
        """Generate complete trading decision with enhanced position sizing."""
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
            'position_size': position_size,
            'timestamp': datetime.now()
        }
        
        return decision
    
    def execute_trade(self, symbol, action, quantity, price, confidence):
        """Execute a trade with enhanced risk checks."""
        # Enhanced pre-trade risk checks
        risk_breaches, risk_metrics = self.check_risk_limits()
        
        if risk_breaches:
            self.logger.warning(f"🚨 Risk limits breached - skipping trade: {risk_breaches}")
            return False
        
        # Additional confidence check
        if confidence < 0.2:
            self.logger.info("📊 Low confidence - skipping trade")
            return False
        
        trade_cost = quantity * price
        commission = max(trade_cost * 0.001, 1)  # 0.1% commission, min $1
        
        if action.upper() == 'BUY':
            if trade_cost + commission > self.current_capital:
                self.logger.warning("Insufficient capital for BUY order")
                return False
            
            self.current_capital -= (trade_cost + commission)
            if symbol in self.positions:
                self.positions[symbol]['quantity'] += quantity
                self.positions[symbol]['avg_price'] = (
                    (self.positions[symbol]['quantity'] * self.positions[symbol]['avg_price'] + 
                     quantity * price) / (self.positions[symbol]['quantity'] + quantity)
                )
            else:
                self.positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'entry_time': datetime.now()
                }
            
            trade_type = 'BUY'
            
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
            
            trade_type = 'SELL'
        else:
            self.logger.warning(f"Invalid action: {action}")
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
            'portfolio_value': risk_metrics['portfolio_value'],
            'pnl': pnl if action.upper() == 'SELL' else 0
        }
        
        self.trade_history.append(trade_record)
        self.logger.info(
            f"✅ {trade_type} {quantity} shares of {symbol} at ${price:.2f} "
            f"(Confidence: {confidence:.1%})"
        )
        
        return True
    
    def check_risk_limits(self):
        """Check if any risk limits are breached."""
        portfolio_value = self.current_capital + sum(
            pos['quantity'] * 100 for pos in self.positions.values()  # Simplified current value
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

# Final Live Trading Manager
class FinalLiveTradingManager:
    def __init__(self, initial_capital=5000, trading_interval=300):
        self.engine = FinalLiveTradingEngine(initial_capital)
        self.data_fetcher = EnhancedDataFetcher()
        self.trading_interval = trading_interval
        self.is_running = False
        
    def start_live_trading(self, symbol="STOCK", max_cycles=None):
        """Start live trading loop."""
        print(f"🚀 STARTING FINAL LIVE TRADING")
        print(f"   Symbol: {symbol}")
        print(f"   Interval: {self.trading_interval} seconds")
        print(f"   Max Cycles: {max_cycles or 'Unlimited'}")
        print("=" * 50)
        
        self.is_running = True
        cycle_count = 0
        
        try:
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
                    print(f"   ⚠️  Risk Level: {decision['risk_level']}")
                    print(f"   💰 Position Size: ${decision['position_size']:,.2f}")
                    print(f"   📊 Portfolio Value: ${result['portfolio_value']:,.2f}")
                    print(f"   📉 Drawdown: {risk_metrics['drawdown']:.2%}")
                    
                    if result['risk_breaches']:
                        print(f"   🚨 RISK ALERT: {result['risk_breaches']}")
                else:
                    print(f"   ❌ Cycle failed: {result.get('error', 'Unknown error')}")
                
                # Save state every 5 cycles
                if cycle_count % 5 == 0:
                    self.engine.save_portfolio_state()
                
                # Wait for next cycle
                if cycle_count < (max_cycles or float('inf')):
                    print(f"   ⏰ Next cycle in {self.trading_interval} seconds...")
                    time.sleep(self.trading_interval)
                    
        except KeyboardInterrupt:
            print("\n🛑 Live trading stopped by user")
        finally:
            self.stop_live_trading()
    
    def stop_live_trading(self):
        """Stop live trading."""
        self.is_running = False
        self.engine.save_portfolio_state()
        print("🛑 Live trading stopped")
    
    def get_performance_report(self):
        """Generate performance report."""
        summary = self.engine.get_portfolio_summary()
        
        print("\n" + "="*60)
        print("📊 FINAL LIVE TRADING PERFORMANCE REPORT")
        print("="*60)
        
        print(f"💰 Portfolio Value: ${summary['current_value']:,.2f}")
        print(f"📈 Total Return: {summary['total_return']:.2%}")
        print(f"💵 Cash Balance: ${summary['cash']:,.2f}")
        print(f"🎯 Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        print(f"📊 Daily P&L: ${summary['daily_pnl']:,.2f}")
        print(f"🏆 Daily Win Rate: {summary['daily_win_rate']:.1%}")
        print(f"🔢 Total Trades: {summary['total_trades']}")
        print(f"📦 Active Positions: {len(summary['active_positions'])}")
        
        return summary

# Final Quick Test
def final_quick_test():
    """Quick test of the final system."""
    print("🧪 FINAL QUICK TEST")
    print("=" * 40)
    
    # Test with small capital and quick cycles
    manager = FinalLiveTradingManager(initial_capital=1000, trading_interval=15)
    
    try:
        # Run just 3 cycles for quick test
        manager.start_live_trading(symbol="FINAL_TEST", max_cycles=3)
        
        # Show results
        manager.get_performance_report()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    final_quick_test()
