# live_trading_engine.py
import pandas as pd
import numpy as np
import time
import json
import logging
from datetime import datetime, timedelta
from fixed_enhanced_predictor import FixedEnhancedStockPredictor
import warnings
warnings.filterwarnings('ignore')

class LiveTradingEngine:
    """
    Live trading engine for production deployment.
    """
    
    def __init__(self, initial_capital=10000, model_dir="robust_models"):
        self.predictor = FixedEnhancedStockPredictor(model_dir)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.performance_log = []
        
        # Trading parameters
        self.max_position_size = 0.1  # 10% of capital
        self.daily_loss_limit = 0.05  # 5% daily loss limit
        self.max_drawdown_limit = 0.15  # 15% max drawdown
        
        # Setup logging
        self.setup_logging()
        
        print("🚀 LIVE TRADING ENGINE INITIALIZED")
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
                logging.FileHandler('trading_log.txt'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def calculate_daily_performance(self):
        """Calculate today's performance metrics."""
        today = datetime.now().date()
        today_trades = [t for t in self.trade_history if t['timestamp'].date() == today]
        
        if not today_trades:
            return {'daily_pnl': 0, 'daily_trades': 0, 'win_rate': 0}
        
        daily_pnl = sum(trade['pnl'] for trade in today_trades)
        winning_trades = len([t for t in today_trades if t['pnl'] > 0])
        win_rate = winning_trades / len(today_trades)
        
        return {
            'daily_pnl': daily_pnl,
            'daily_trades': len(today_trades),
            'win_rate': win_rate
        }
    
    def check_risk_limits(self):
        """Check if any risk limits are breached."""
        # Calculate current portfolio value
        portfolio_value = self.current_capital + sum(
            position['current_value'] for position in self.positions.values()
        )
        
        # Calculate drawdown
        peak_capital = max([self.initial_capital] + 
                          [entry['portfolio_value'] for entry in self.performance_log])
        drawdown = (peak_capital - portfolio_value) / peak_capital
        
        # Check daily loss limit
        daily_perf = self.calculate_daily_performance()
        daily_loss = -daily_perf['daily_pnl'] / self.current_capital if self.current_capital > 0 else 0
        
        risk_breaches = []
        
        if drawdown > self.max_drawdown_limit:
            risk_breaches.append(f"Max drawdown limit breached: {drawdown:.2%}")
        
        if daily_loss > self.daily_loss_limit:
            risk_breaches.append(f"Daily loss limit breached: {daily_loss:.2%}")
        
        return risk_breaches, {
            'portfolio_value': portfolio_value,
            'drawdown': drawdown,
            'daily_loss': daily_loss
        }
    
    def execute_trade(self, symbol, action, quantity, price, confidence):
        """Execute a trade with risk checks."""
        # Pre-trade risk checks
        risk_breaches, risk_metrics = self.check_risk_limits()
        
        if risk_breaches:
            self.logger.warning(f"🚨 Risk limits breached - skipping trade: {risk_breaches}")
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
    
    def generate_trading_decision(self, predictions, current_data, symbol="STOCK"):
        """Generate complete trading decision with position sizing."""
        current_price = current_data['Close'].iloc[-1] if 'Close' in current_data.columns else 100
        
        # Get signals from predictor
        signals = self.predictor.enhanced_trading_signals(predictions, current_price)
        
        # Enhanced position sizing with volatility adjustment
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
    
    def run_trading_cycle(self, data_fetcher, symbol="STOCK"):
        """Run one complete trading cycle."""
        try:
            self.logger.info("🔄 Starting trading cycle...")
            
            # 1. Fetch latest data
            new_data = data_fetcher.fetch_latest_data()
            self.logger.info(f"📊 Fetched {len(new_data)} data points")
            
            # 2. Make predictions
            predictions = self.predictor.predict(new_data)
            self.logger.info("🎯 Predictions generated")
            
            # 3. Check for model drift
            drift_detected, drift_message = self.predictor.monitor_model_drift(new_data)
            if drift_detected:
                self.logger.warning(f"🚨 Model drift detected: {drift_message}")
            
            # 4. Generate trading decision
            decision = self.generate_trading_decision(predictions, new_data, symbol)
            
            # 5. Execute trade if not HOLD
            if decision['action'] != 'HOLD':
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
                    self.logger.warning("❌ Trade execution failed")
            else:
                self.logger.info("⏸️  HOLD signal - no trade executed")
            
            # 6. Log performance
            portfolio_value = self.current_capital + sum(
                pos['quantity'] * decision['price'] for pos in self.positions.values()
            )
            
            performance_entry = {
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_value,
                'cash': self.current_capital,
                'positions': len(self.positions),
                'decision': decision,
                'drift_detected': drift_detected
            }
            
            self.performance_log.append(performance_entry)
            
            # 7. Check risk limits
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
    
    def get_portfolio_summary(self):
        """Get current portfolio summary."""
        if not self.performance_log:
            current_value = self.initial_capital
        else:
            current_value = self.performance_log[-1]['portfolio_value']
        
        total_return = (current_value - self.initial_capital) / self.initial_capital
        daily_perf = self.calculate_daily_performance()
        
        # Calculate Sharpe ratio (simplified)
        returns = []
        for i in range(1, len(self.performance_log)):
            ret = (self.performance_log[i]['portfolio_value'] - 
                   self.performance_log[i-1]['portfolio_value']) / self.performance_log[i-1]['portfolio_value']
            returns.append(ret)
        
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns and np.std(returns) > 0 else 0
        
        summary = {
            'initial_capital': self.initial_capital,
            'current_value': current_value,
            'total_return': total_return,
            'cash': self.current_capital,
            'positions_count': len(self.positions),
            'total_trades': len(self.trade_history),
            'daily_pnl': daily_perf['daily_pnl'],
            'daily_win_rate': daily_perf['win_rate'],
            'sharpe_ratio': sharpe,
            'active_positions': list(self.positions.keys())
        }
        
        return summary
    
    def save_portfolio_state(self):
        """Save current portfolio state to file."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'current_capital': self.current_capital,
            'positions': self.positions,
            'trade_history': [{
                **trade,
                'timestamp': trade['timestamp'].isoformat()
            } for trade in self.trade_history[-100:]],  # Last 100 trades
            'performance_log': [{
                **log,
                'timestamp': log['timestamp'].isoformat()
            } for log in self.performance_log[-50:]]  # Last 50 performance entries
        }
        
        with open('portfolio_state.json', 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info("💾 Portfolio state saved")

# Enhanced Data Fetcher for Live Trading
class LiveDataFetcher:
    def __init__(self, data_file=None):
        self.data_file = data_file or "feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv"
    
    def fetch_latest_data(self, lookback=50):
        """Fetch latest data for prediction."""
        try:
            data = pd.read_csv(self.data_file)
            return data.tail(lookback)
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            # Return mock data as fallback
            return self._generate_mock_data(lookback)
    
    def _generate_mock_data(self, n_samples):
        """Generate mock data when real data is unavailable."""
        dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='D')
        
        data = {
            'Date': dates,
            'Open': np.random.normal(100, 10, n_samples),
            'High': np.random.normal(105, 10, n_samples),
            'Low': np.random.normal(95, 10, n_samples),
            'Close': np.random.normal(102, 10, n_samples),
            'Volume': np.random.normal(1000000, 200000, n_samples),
            'daily_return': np.random.normal(0.001, 0.02, n_samples),
            'log_return': np.random.normal(0.0005, 0.015, n_samples),
            'volatility_5d': np.random.normal(0.15, 0.05, n_samples)
        }
        
        return pd.DataFrame(data)

# Live Trading Manager
class LiveTradingManager:
    def __init__(self, initial_capital=5000, trading_interval=300):
        self.engine = LiveTradingEngine(initial_capital)
        self.data_fetcher = LiveDataFetcher()
        self.trading_interval = trading_interval  # seconds
        self.is_running = False
        
    def start_live_trading(self, symbol="STOCK", max_cycles=None):
        """Start live trading loop."""
        print(f"🚀 STARTING LIVE TRADING")
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
                
                # Save state every 10 cycles
                if cycle_count % 10 == 0:
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
        print("📊 LIVE TRADING PERFORMANCE REPORT")
        print("="*60)
        
        print(f"💰 Portfolio Value: ${summary['current_value']:,.2f}")
        print(f"📈 Total Return: {summary['total_return']:.2%}")
        print(f"💵 Cash Balance: ${summary['cash']:,.2f}")
        print(f"🎯 Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        print(f"📊 Daily P&L: ${summary['daily_pnl']:,.2f}")
        print(f"🏆 Daily Win Rate: {summary['daily_win_rate']:.1%}")
        print(f"🔢 Total Trades: {summary['total_trades']}")
        print(f"📦 Active Positions: {len(summary['active_positions'])}")
        
        if summary['active_positions']:
            print(f"   📋 Positions: {', '.join(summary['active_positions'])}")
        
        return summary

# Demo and Testing
if __name__ == "__main__":
    # Initialize trading manager
    manager = LiveTradingManager(initial_capital=5000, trading_interval=60)  # 1-minute cycles for demo
    
    try:
        # Run demo trading (5 cycles)
        print("🧪 RUNNING DEMO TRADING (5 cycles)")
        manager.start_live_trading(symbol="DEMO", max_cycles=5)
        
        # Show performance report
        manager.get_performance_report()
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
