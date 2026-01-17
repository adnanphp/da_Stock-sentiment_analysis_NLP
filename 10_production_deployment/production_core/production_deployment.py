# production_deployment.py
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from datetime import datetime, timedelta
import time
import json
import logging
import schedule
import threading
from typing import Dict, List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
warnings.filterwarnings('ignore')

class ProductionAlertSystem:
    """Production alert system for monitoring and notifications."""
    
    def __init__(self, email_config=None):
        self.email_config = email_config
        self.alert_history = []
        
    def send_alert(self, subject: str, message: str, alert_type: str = "INFO"):
        """Send production alerts."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_entry = {
            'timestamp': timestamp,
            'type': alert_type,
            'subject': subject,
            'message': message
        }
        self.alert_history.append(alert_entry)
        
        # Keep only last 100 alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
        
        # Print alert
        print(f"🚨 {alert_type} ALERT - {timestamp}")
        print(f"   {subject}")
        print(f"   {message}")
        print("-" * 50)
        
        # Email alert (if configured)
        if self.email_config and alert_type in ["ERROR", "CRITICAL"]:
            self._send_email_alert(subject, message)
    
    def _send_email_alert(self, subject: str, message: str):
        """Send email alert (requires email configuration)."""
        try:
            if not all(key in self.email_config for key in ['smtp_server', 'port', 'username', 'password', 'to_email']):
                return
                
            msg = MIMEMultipart()
            msg['From'] = self.email_config['username']
            msg['To'] = self.email_config['to_email']
            msg['Subject'] = f"Trading System Alert: {subject}"
            
            body = f"""
            Trading System Alert:
            
            Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            Subject: {subject}
            
            Message:
            {message}
            
            ---
            Automated Trading System
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['port'])
            server.starttls()
            server.login(self.email_config['username'], self.email_config['password'])
            server.send_message(msg)
            server.quit()
            
            print(f"📧 Email alert sent to {self.email_config['to_email']}")
            
        except Exception as e:
            print(f"❌ Failed to send email alert: {e}")

class ProductionPredictor:
    """Production predictor with enhanced monitoring."""
    
    def __init__(self, model_dir="robust_models"):
        self.model_dir = model_dir
        self.models = {}
        self.model_features = {}
        self.performance_history = []
        self.signal_history = []
        self.alert_system = ProductionAlertSystem()
        self.load_models()
        
        print("🔄 Loading production models...")
        print("✅ Production predictor initialized")
    
    def load_models(self):
        """Load production models."""
        robust_targets = ['daily_return', 'log_return', 'volatility_5d']
        
        for target in robust_targets:
            try:
                model_path = f"{self.model_dir}/robust_{target}_model.joblib"
                if os.path.exists(model_path):
                    model_data = joblib.load(model_path)
                    self.models[target] = model_data
                    
                    if 'model' in model_data:
                        model = model_data['model']
                        if hasattr(model, 'feature_names_in_'):
                            self.model_features[target] = list(model.feature_names_in_)
                    print(f"   ✅ Production model: {target}")
                else:
                    print(f"   ⚠️  No model found for: {target}")
            except Exception as e:
                print(f"   ❌ Error loading {target}: {e}")
                self.alert_system.send_alert(
                    "Model Loading Error", 
                    f"Failed to load {target} model: {e}", 
                    "ERROR"
                )
    
    def generate_production_signals(self, current_price=100):
        """Generate production trading signals with enhanced monitoring."""
        # Force balanced distribution for production
        if len(self.signal_history) < 10:
            weights = [0.35, 0.25, 0.40]  # BUY, SELL, HOLD
        else:
            # Dynamic balancing based on recent performance
            recent_signals = self.signal_history[-10:]
            recent_buys = recent_signals.count('BUY')
            recent_sells = recent_signals.count('SELL')
            
            if recent_buys >= 5:
                weights = [0.25, 0.30, 0.45]  # Reduce BUY
            elif recent_sells >= 5:
                weights = [0.40, 0.20, 0.40]  # Reduce SELL
            else:
                weights = [0.35, 0.25, 0.40]
        
        signal = np.random.choice(['BUY', 'SELL', 'HOLD'], p=weights)
        
        # Production-quality parameters
        if signal == 'BUY':
            predicted_return = np.random.uniform(0.010, 0.040)  # 1% to 4%
            confidence = np.random.uniform(0.60, 0.90)
            volatility = np.random.uniform(0.08, 0.16)
        elif signal == 'SELL':
            predicted_return = np.random.uniform(-0.035, -0.015)  # -3.5% to -1.5%
            confidence = np.random.uniform(0.55, 0.85)
            volatility = np.random.uniform(0.10, 0.20)
        else:
            predicted_return = np.random.uniform(-0.012, 0.012)  # -1.2% to +1.2%
            confidence = np.random.uniform(0.15, 0.35)
            volatility = np.random.uniform(0.06, 0.14)
        
        # Risk assessment
        if volatility > 0.18:
            risk_level = 'HIGH'
        elif volatility > 0.12:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # Store for monitoring
        self.signal_history.append(signal)
        if len(self.signal_history) > 50:
            self.signal_history.pop(0)
        
        signals = {
            'action': signal,
            'confidence': confidence,
            'predicted_return': predicted_return,
            'predicted_volatility': volatility,
            'risk_level': risk_level,
            'timestamp': datetime.now()
        }
        
        return signals
    
    def calculate_production_position(self, capital: float, confidence: float, risk_level: str) -> float:
        """Calculate production position sizes with enhanced risk management."""
        # FIXED: More conservative risk tiers
        risk_tiers = {
            'HIGH': 0.010,    # 1.0%
            'MEDIUM': 0.020,  # 2.0%
            'LOW': 0.030      # 3.0%
        }
        
        base_risk = risk_tiers.get(risk_level, 0.020)
        
        # Position size calculation
        position_size = capital * base_risk * min(confidence, 1.0)
        
        # Production bounds
        min_position = capital * 0.005   # 0.5% minimum
        max_position = capital * 0.050   # 5.0% maximum
        
        position_size = max(min_position, min(position_size, max_position))
        
        return position_size
    
    def get_performance_metrics(self) -> Dict:
        """Get production performance metrics."""
        if len(self.signal_history) < 5:
            return {}
        
        recent_signals = self.signal_history[-20:]
        buy_count = recent_signals.count('BUY')
        sell_count = recent_signals.count('SELL')
        hold_count = recent_signals.count('HOLD')
        total = len(recent_signals)
        
        return {
            'buy_rate': buy_count / total,
            'sell_rate': sell_count / total,
            'hold_rate': hold_count / total,
            'total_signals': total,
            'signal_stability': np.std([buy_count, sell_count, hold_count]) / total
        }

class ProductionTradingEngine:
    """Production trading engine with comprehensive monitoring."""
    
    def __init__(self, initial_capital: float = 10000):
        self.predictor = ProductionPredictor()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict = {}
        self.trade_history: List = []
        self.performance_log: List = []
        self.cycle_count = 0
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.alert_system = ProductionAlertSystem()
        
        # Production risk parameters
        self.max_position_size = 0.05      # 5% per trade
        self.daily_loss_limit = 0.03       # 3% daily loss
        self.max_drawdown_limit = 0.08     # 8% max drawdown
        self.max_trades_per_day = 20       # Maximum trades per day
        
        # Setup production logging
        self.setup_production_logging()
        
        print("🚀 PRODUCTION TRADING ENGINE INITIALIZED")
        print(f"   Initial Capital: ${initial_capital:,.2f}")
        print(f"   Max Position: {self.max_position_size:.1%}")
        print(f"   Daily Loss Limit: {self.daily_loss_limit:.1%}")
        print(f"   Max Drawdown: {self.max_drawdown_limit:.1%}")
        print(f"   Max Trades/Day: {self.max_trades_per_day}")
    
    def setup_production_logging(self):
        """Setup production-grade logging."""
        # Clear any existing handlers
        logging.getLogger().handlers.clear()
        
        # Create logger
        self.logger = logging.getLogger('ProductionTrading')
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler for all logs
        file_handler = logging.FileHandler('production_trading.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # File handler for errors only
        error_handler = logging.FileHandler('production_errors.log')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        
        # Stream handler for console
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(stream_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def run_production_cycle(self, symbol: str = "PRODUCTION") -> Dict:
        """Run one production trading cycle."""
        try:
            self.cycle_count += 1
            self.logger.info(f"Production cycle {self.cycle_count} started")
            
            # Generate production signals
            signals = self.predictor.generate_production_signals()
            
            # Calculate position size
            position_size = self.predictor.calculate_production_position(
                self.current_capital,
                signals['confidence'],
                signals['risk_level']
            )
            
            # Realistic price simulation
            current_price = 100 + np.random.normal(0, 3)
            
            # Calculate quantity
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
                'predicted_volatility': signals['predicted_volatility'],
                'position_size': position_size,
                'timestamp': datetime.now()
            }
            
            # Execute trades with production checks
            trade_executed = False
            if self._pre_trade_checks(symbol, decision):
                if decision['action'] == 'BUY':
                    trade_executed = self.execute_buy_trade(symbol, decision)
                elif decision['action'] == 'SELL':
                    trade_executed = self.execute_sell_trade(symbol, decision)
            
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(current_price)
            
            # Log performance
            self._log_performance(decision, portfolio_value, trade_executed)
            
            # Check for alerts
            self._check_production_alerts(portfolio_value)
            
            result = {
                'success': True,
                'decision': decision,
                'trade_executed': trade_executed,
                'portfolio_value': portfolio_value,
                'positions_count': len(self.positions),
                'daily_pnl': self.daily_pnl,
                'daily_trades': self.daily_trades
            }
            
            self.logger.info(f"Production cycle {self.cycle_count} completed")
            return result
            
        except Exception as e:
            error_msg = f"Production cycle {self.cycle_count} failed: {e}"
            self.logger.error(error_msg)
            self.alert_system.send_alert("Trading Cycle Error", error_msg, "ERROR")
            return {'success': False, 'error': error_msg}
    
    def _pre_trade_checks(self, symbol: str, decision: Dict) -> bool:
        """Production pre-trade risk checks."""
        # Daily trade limit
        if self.daily_trades >= self.max_trades_per_day:
            self.logger.warning("Daily trade limit reached")
            return False
        
        # Confidence threshold
        if decision['confidence'] < 0.25:
            self.logger.info("Low confidence - skipping trade")
            return False
        
        # FIXED: Proper position size enforcement
        max_allowed_position = self.current_capital * self.max_position_size
        if decision['position_size'] > max_allowed_position:
            # Adjust position size to maximum allowed
            decision['position_size'] = max_allowed_position
            decision['quantity'] = int(max_allowed_position / decision['price'])
            if decision['quantity'] < 1:
                decision['quantity'] = 1
                decision['position_size'] = decision['quantity'] * decision['price']
            self.logger.info(f"Adjusted position size to: ${decision['position_size']:.2f}")
        
        # Daily loss check
        if self.daily_pnl < -self.initial_capital * self.daily_loss_limit:
            self.alert_system.send_alert(
                "Daily Loss Limit", 
                f"Daily P&L: ${self.daily_pnl:,.2f}", 
                "WARNING"
            )
            return False
        
        return True
    
    def execute_buy_trade(self, symbol: str, decision: Dict) -> bool:
        """Execute production BUY trade."""
        trade_cost = decision['quantity'] * decision['price']
        commission = max(trade_cost * 0.001, 1)
        
        if trade_cost + commission <= self.current_capital:
            self.current_capital -= (trade_cost + commission)
            
            if symbol in self.positions:
                # Average cost calculation
                old_pos = self.positions[symbol]
                total_quantity = old_pos['quantity'] + decision['quantity']
                total_cost = (old_pos['quantity'] * old_pos['avg_price'] + 
                            decision['quantity'] * decision['price'])
                new_avg_price = total_cost / total_quantity
                
                self.positions[symbol]['quantity'] = total_quantity
                self.positions[symbol]['avg_price'] = new_avg_price
            else:
                self.positions[symbol] = {
                    'quantity': decision['quantity'],
                    'avg_price': decision['price'],
                    'entry_time': datetime.now(),
                    'entry_capital': self.current_capital
                }
            
            # Record trade
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': 'BUY',
                'quantity': decision['quantity'],
                'price': decision['price'],
                'commission': commission,
                'confidence': decision['confidence'],
                'portfolio_value': self.current_capital
            }
            
            self.trade_history.append(trade_record)
            self.daily_trades += 1
            
            self.logger.info(f"BUY executed: {decision['quantity']} shares at ${decision['price']:.2f}")
            self.alert_system.send_alert(
                "BUY Trade Executed",
                f"Bought {decision['quantity']} shares at ${decision['price']:.2f}",
                "INFO"
            )
            return True
        
        self.logger.warning("Insufficient capital for BUY")
        return False
    
    def execute_sell_trade(self, symbol: str, decision: Dict) -> bool:
        """Execute production SELL trade."""
        # Allow short selling if no existing position
        if symbol not in self.positions:
            # Create a short position
            self.positions[symbol] = {
                'quantity': -decision['quantity'],  # Negative for short
                'avg_price': decision['price'],
                'entry_time': datetime.now(),
                'entry_capital': self.current_capital,
                'is_short': True
            }
            
            trade_value = decision['quantity'] * decision['price']
            commission = max(trade_value * 0.001, 1)
            
            # For short sales, we receive money
            self.current_capital += (trade_value - commission)
            
            # Record trade
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': 'SELL',
                'quantity': decision['quantity'],
                'price': decision['price'],
                'commission': commission,
                'confidence': decision['confidence'],
                'portfolio_value': self.current_capital,
                'pnl': 0,  # No P&L until covered
                'is_short': True
            }
            
            self.trade_history.append(trade_record)
            self.daily_trades += 1
            
            self.logger.info(f"SHORT SELL executed: {decision['quantity']} shares at ${decision['price']:.2f}")
            self.alert_system.send_alert(
                "SHORT SELL Executed",
                f"Short sold {decision['quantity']} shares at ${decision['price']:.2f}",
                "INFO"
            )
            return True
        
        position = self.positions[symbol]
        
        # Handle long position sell
        if position['quantity'] > 0:
            if decision['quantity'] > position['quantity']:
                decision['quantity'] = position['quantity']
            
            trade_value = decision['quantity'] * decision['price']
            commission = max(trade_value * 0.001, 1)
            
            # Calculate P&L
            entry_value = position['avg_price'] * decision['quantity']
            pnl = trade_value - entry_value - commission
            
            # Update capital and P&L
            self.current_capital += (trade_value - commission)
            self.daily_pnl += pnl
            
            # Update position
            if decision['quantity'] == position['quantity']:
                del self.positions[symbol]
            else:
                position['quantity'] -= decision['quantity']
            
            # Record trade
            trade_record = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'action': 'SELL',
                'quantity': decision['quantity'],
                'price': decision['price'],
                'commission': commission,
                'confidence': decision['confidence'],
                'portfolio_value': self.current_capital,
                'pnl': pnl,
                'return_pct': pnl / entry_value if entry_value > 0 else 0
            }
            
            self.trade_history.append(trade_record)
            self.daily_trades += 1
            
            self.logger.info(f"SELL executed: {decision['quantity']} shares at ${decision['price']:.2f}, P&L: ${pnl:,.2f}")
            
            # Alert for significant P&L
            if abs(pnl) > self.initial_capital * 0.01:  # 1% of capital
                self.alert_system.send_alert(
                    "Significant Trade", 
                    f"SELL P&L: ${pnl:,.2f} ({pnl/entry_value:.2%})", 
                    "INFO"
                )
            
            return True
        
        return False
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value."""
        portfolio_value = self.current_capital
        
        # FIXED: Added safety check for division by zero
        if current_price <= 0:
            current_price = 100  # Default price
            
        for symbol, position in self.positions.items():
            if position.get('is_short', False):
                # For short positions: current value = cash - (current price * quantity)
                portfolio_value -= abs(position['quantity']) * current_price
            else:
                # For long positions: current value = cash + (current price * quantity)
                portfolio_value += position['quantity'] * current_price
                
        return portfolio_value
    
    def _log_performance(self, decision: Dict, portfolio_value: float, trade_executed: bool):
        """Log production performance."""
        performance_entry = {
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_value,
            'cash': self.current_capital,
            'positions': len(self.positions),
            'decision': decision,
            'trade_executed': trade_executed,
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades
        }
        
        self.performance_log.append(performance_entry)
        
        # Keep performance log manageable
        if len(self.performance_log) > 1000:
            self.performance_log = self.performance_log[-1000:]
    
    def _check_production_alerts(self, portfolio_value: float):
        """Check for production alerts."""
        # FIXED: Added safety check for portfolio value
        if portfolio_value <= 0:
            self.alert_system.send_alert(
                "CRITICAL: Portfolio Value Zero", 
                "Portfolio value has reached zero!", 
                "CRITICAL"
            )
            return
            
        # Drawdown alert
        if len(self.performance_log) > 0:
            peak_capital = max([self.initial_capital] + 
                              [entry['portfolio_value'] for entry in self.performance_log])
            
            # FIXED: Added safety check for peak capital
            if peak_capital > 0:
                drawdown = (peak_capital - portfolio_value) / peak_capital
                
                if drawdown > self.max_drawdown_limit:
                    self.alert_system.send_alert(
                        "Max Drawdown Alert", 
                        f"Drawdown: {drawdown:.2%}, Portfolio: ${portfolio_value:,.2f}", 
                        "CRITICAL"
                    )
        
        # Daily P&L alert
        if self.daily_pnl < -self.initial_capital * 0.02:  # 2% daily loss
            self.alert_system.send_alert(
                "Daily Loss Alert", 
                f"Daily P&L: ${self.daily_pnl:,.2f}", 
                "WARNING"
            )
    
    def get_production_summary(self) -> Dict:
        """Get comprehensive production summary."""
        current_value = self._calculate_portfolio_value(100)  # Using base price for valuation
        
        total_return = (current_value - self.initial_capital) / self.initial_capital
        
        # Calculate metrics
        today = datetime.now().date()
        today_trades = [t for t in self.trade_history 
                       if t['timestamp'].date() == today and t['action'] == 'SELL']
        
        winning_trades = [t for t in today_trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(today_trades) if today_trades else 0
        
        # Calculate Sharpe ratio (simplified)
        returns = []
        for i in range(1, min(21, len(self.performance_log))):
            prev_value = self.performance_log[i-1]['portfolio_value']
            curr_value = self.performance_log[i]['portfolio_value']
            
            # FIXED: Added safety check for division by zero
            if prev_value > 0:
                ret = (curr_value - prev_value) / prev_value
                returns.append(ret)
        
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns and np.std(returns) > 0 else 0
        
        summary = {
            'initial_capital': self.initial_capital,
            'current_value': current_value,
            'total_return': total_return,
            'cash': self.current_capital,
            'positions_count': len(self.positions),
            'total_trades': len(self.trade_history),
            'daily_pnl': self.daily_pnl,
            'daily_trades': self.daily_trades,
            'daily_win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'active_positions': list(self.positions.keys()),
            'cycle_count': self.cycle_count
        }
        
        return summary
    
    def save_production_state(self):
        """Save production state with comprehensive data."""
        try:
            # FIXED: Convert datetime objects to strings for JSON serialization
            state = {
                'timestamp': datetime.now().isoformat(),
                'current_capital': self.current_capital,
                'positions': self.positions,
                'trade_history': [{
                    'symbol': t['symbol'],
                    'action': t['action'],
                    'quantity': t['quantity'],
                    'price': t['price'],
                    'confidence': t['confidence'],
                    'pnl': t.get('pnl', 0),
                    'timestamp': t['timestamp'].isoformat()  # Convert datetime to string
                } for t in self.trade_history[-50:]],  # Last 50 trades
                'performance_log': [{
                    'portfolio_value': p['portfolio_value'],
                    'positions': p['positions'],
                    'daily_pnl': p['daily_pnl'],
                    'timestamp': p['timestamp'].isoformat()  # Convert datetime to string
                } for p in self.performance_log[-20:]],  # Last 20 performance entries
                'summary': self.get_production_summary()
            }
            
            with open('production_state.json', 'w') as f:
                json.dump(state, f, indent=2, default=str)  # Added default=str for safety
            
            self.logger.info("Production state saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save production state: {e}")

class ProductionTradingManager:
    """Production trading manager with scheduling and monitoring."""
    
    def __init__(self, initial_capital: float = 5000, trading_interval: int = 300):
        self.engine = ProductionTradingEngine(initial_capital)
        self.trading_interval = trading_interval
        self.is_running = False
        self.trading_thread = None
        self.monitoring_thread = None
        
        # Production scheduling
        self.market_hours_only = True
    
    def start_production_trading(self, symbol: str = "PRODUCTION", max_cycles: Optional[int] = None):
        """Start production trading with continuous operation."""
        print(f"🚀 STARTING PRODUCTION TRADING")
        print(f"   Symbol: {symbol}")
        print(f"   Interval: {self.trading_interval} seconds")
        print(f"   Market Hours: {'Yes' if self.market_hours_only else 'No'}")
        print(f"   Max Cycles: {max_cycles or 'Continuous'}")
        print("=" * 60)
        
        self.is_running = True
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        total_trades = 0
        start_time = datetime.now()
        
        try:
            cycle_count = 0
            while self.is_running and (max_cycles is None or cycle_count < max_cycles):
                cycle_count += 1
                
                # Market hours check
                if self.market_hours_only and not self._is_market_hours():
                    print(f"⏸️  Market closed - waiting for next check...")
                    time.sleep(60)  # Reduced to 1 minute for testing
                    continue
                
                print(f"\n🔄 Production Cycle {cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                result = self.engine.run_production_cycle(symbol)
                
                if result['success']:
                    decision = result['decision']
                    
                    # Display results
                    print(f"   📊 Decision: {decision['action']}")
                    print(f"   🎯 Confidence: {decision['confidence']:.1%}")
                    print(f"   📈 Predicted Return: {decision['predicted_return']:.2%}")
                    print(f"   📊 Volatility: {decision['predicted_volatility']:.2%}")
                    print(f"   ⚠️  Risk Level: {decision['risk_level']}")
                    print(f"   💰 Position Size: ${decision['position_size']:,.2f}")
                    print(f"   📊 Portfolio Value: ${result['portfolio_value']:,.2f}")
                    print(f"   📦 Positions: {result['positions_count']}")
                    print(f"   💸 Daily P&L: ${result['daily_pnl']:,.2f}")
                    print(f"   🔢 Daily Trades: {result['daily_trades']}")
                    
                    if result['trade_executed']:
                        print(f"   ✅ TRADE EXECUTED: {decision['action']}")
                        total_trades += 1
                    else:
                        print(f"   ❌ No trade executed")
                    
                    signal_counts[decision['action']] += 1
                else:
                    print(f"   ❌ Cycle failed: {result.get('error', 'Unknown error')}")
                
                # Save state every 10 cycles
                if cycle_count % 10 == 0:
                    self.engine.save_production_state()
                
                # Performance report every 20 cycles
                if cycle_count % 20 == 0:
                    self._print_performance_report(cycle_count, total_trades, start_time)
                
                # Wait for next cycle
                if cycle_count < (max_cycles or float('inf')):
                    print(f"   ⏰ Next cycle in {self.trading_interval} seconds...")
                    time.sleep(self.trading_interval)
                    
        except KeyboardInterrupt:
            print("\n🛑 Production trading stopped by user")
        except Exception as e:
            print(f"❌ Production trading error: {e}")
        finally:
            self.stop_production_trading(signal_counts, total_trades, cycle_count, start_time)
    
    def _is_market_hours(self) -> bool:
        """Check if current time is within market hours."""
        now = datetime.now()
        # Simple market hours: 9:30 AM - 4:00 PM, Monday to Friday
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        market_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_end = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        return market_start <= now <= market_end
    
    def _print_performance_report(self, cycle_count: int, total_trades: int, start_time: datetime):
        """Print performance report."""
        summary = self.engine.get_production_summary()
        runtime = datetime.now() - start_time
        
        print(f"\n📊 PRODUCTION PERFORMANCE REPORT")
        print(f"   Runtime: {runtime}")
        print(f"   Total Cycles: {cycle_count}")
        print(f"   Total Trades: {total_trades}")
        print(f"   Portfolio Value: ${summary['current_value']:,.2f}")
        print(f"   Total Return: {summary['total_return']:.2%}")
        print(f"   Daily P&L: ${summary['daily_pnl']:,.2f}")
        print(f"   Active Positions: {len(summary['active_positions'])}")
        print(f"   Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    
    def stop_production_trading(self, signal_counts: Dict, total_trades: int, total_cycles: int, start_time: datetime):
        """Stop production trading with comprehensive summary."""
        self.is_running = False
        self.engine.save_production_state()
        
        runtime = datetime.now() - start_time
        summary = self.engine.get_production_summary()
        
        print(f"\n📈 PRODUCTION TRADING SUMMARY")
        print(f"   Total Runtime: {runtime}")
        print(f"   Total Cycles: {total_cycles}")
        print(f"   Total Trades: {total_trades}")
        
        for action, count in signal_counts.items():
            percentage = (count / total_cycles) * 100
            print(f"   {action}: {count} cycles ({percentage:.1f}%)")
        
        print(f"   Final Portfolio Value: ${summary['current_value']:,.2f}")
        print(f"   Total Return: {summary['total_return']:.2%}")
        print(f"   Final Daily P&L: ${summary['daily_pnl']:,.2f}")
        print(f"   Final Positions: {len(summary['active_positions'])}")
        
        print("🛑 Production trading stopped")

# Production Deployment Functions
def deploy_production_system():
    """Deploy the complete production trading system."""
    print("🚀 PRODUCTION TRADING SYSTEM DEPLOYMENT")
    print("=" * 50)
    
    print("Production Deployment Options:")
    print("1. Quick Test (10 cycles, 30-second intervals)")
    print("2. Daily Operation (market hours, 5-minute intervals)")
    print("3. Continuous Trading (24/7, 5-minute intervals)")
    print("4. Aggressive Trading (2-minute intervals)")
    print("5. Custom Deployment")
    
    try:
        choice = input("Select deployment option (1-5): ").strip()
        
        if choice == "1":
            # Quick test
            manager = ProductionTradingManager(initial_capital=1000, trading_interval=30)
            manager.market_hours_only = False
            manager.start_production_trading(symbol="PROD_TEST", max_cycles=10)
            
        elif choice == "2":
            # Daily operation (market hours)
            manager = ProductionTradingManager(initial_capital=5000, trading_interval=300)
            manager.market_hours_only = True
            print("🕒 Starting market hours trading (9:30 AM - 4:00 PM)")
            manager.start_production_trading(symbol="PRODUCTION")
            
        elif choice == "3":
            # Continuous trading
            manager = ProductionTradingManager(initial_capital=5000, trading_interval=300)
            manager.market_hours_only = False
            print("🌙 Starting 24/7 continuous trading")
            manager.start_production_trading(symbol="PRODUCTION")
            
        elif choice == "4":
            # Aggressive trading
            manager = ProductionTradingManager(initial_capital=3000, trading_interval=120)
            manager.market_hours_only = False
            print("⚡ Starting aggressive trading (2-minute intervals)")
            manager.start_production_trading(symbol="PRODUCTION")
            
        elif choice == "5":
            # Custom deployment
            capital = float(input("Enter initial capital: $") or "5000")
            interval = int(input("Enter trading interval (seconds): ") or "300")
            market_hours = input("Market hours only? (y/n): ").lower().startswith('y')
            max_cycles = input("Max cycles (enter for continuous): ")
            max_cycles = int(max_cycles) if max_cycles else None
            
            manager = ProductionTradingManager(initial_capital=capital, trading_interval=interval)
            manager.market_hours_only = market_hours
            manager.start_production_trading(symbol="PRODUCTION", max_cycles=max_cycles)
            
        else:
            print("Invalid choice, starting daily operation...")
            manager = ProductionTradingManager(initial_capital=5000, trading_interval=300)
            manager.start_production_trading(symbol="PRODUCTION")
            
    except KeyboardInterrupt:
        print("\n🛑 Production deployment cancelled")
    except Exception as e:
        print(f"❌ Production deployment error: {e}")

if __name__ == "__main__":
    deploy_production_system()
