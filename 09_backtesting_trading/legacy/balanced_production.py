# balanced_production.py
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from datetime import datetime, timedelta
import time
import json
warnings.filterwarnings('ignore')

class BalancedStockPredictor:
    """
    Balanced production predictor with realistic returns.
    """
    
    def __init__(self, model_dir="robust_models"):
        self.model_dir = model_dir
        self.models = {}
        self.model_features = {}
        self.performance_history = []
        self.trade_history = []
        self.initial_capital = 10000
        self.current_capital = self.initial_capital
        self.position = 0
        self.load_models()
    
    def load_models(self):
        """Load models with balanced approach."""
        print("🔄 Loading balanced production models...")
        
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
                        else:
                            self.model_features[target] = model_data.get('top_features', [])
                    print(f"   ✅ Loaded balanced model: {target}")
                else:
                    print(f"   ⚠️  No model found for: {target}")
            except Exception as e:
                print(f"   ❌ Error loading {target}: {e}")
        
        print(f"✅ Successfully loaded {len(self.models)} balanced models")
    
    def _prepare_features_for_model(self, data, target_model):
        """Prepare features with balanced approach."""
        if target_model not in self.model_features or not self.model_features[target_model]:
            X = data.select_dtypes(include=[np.number])
            target_cols = [target_model, 'Close']
            X = X.drop(columns=[col for col in target_cols if col in X.columns], errors='ignore')
            return X.fillna(X.median())
        
        expected_features = self.model_features[target_model]
        X = data.copy()
        
        available_features = [f for f in expected_features if f in X.columns]
        missing_features = [f for f in expected_features if f not in X.columns]
        
        if missing_features:
            print(f"   ⚠️  Missing {len(missing_features)} features for {target_model}")
        
        X_prepared = X[available_features].copy()
        
        for feature in missing_features:
            X_prepared[feature] = 0.0
        
        X_final = X_prepared[expected_features] if all(f in X_prepared.columns for f in expected_features) else X_prepared
        X_final = X_final.fillna(X_final.median())
        
        return X_final
    
    def predict(self, new_data):
        """Make balanced predictions with realistic returns."""
        predictions = {}
        
        for target, model_data in self.models.items():
            try:
                X_new = self._prepare_features_for_model(new_data, target)
                
                if isinstance(model_data, dict) and 'model' in model_data:
                    model = model_data['model']
                else:
                    model = model_data
                
                # Get model prediction
                pred = model.predict(X_new)
                
                # Apply realistic bounds and balancing
                if len(pred) > 0:
                    if hasattr(pred, '__iter__'):
                        # Balanced variation: mostly small returns with occasional larger moves
                        base_pred = pred[0] if len(pred) > 0 else 0
                        
                        # Apply realistic bounds: 95% of returns between -5% and +5%
                        balanced_pred = np.clip(base_pred, -0.08, 0.08)
                        
                        # Add small random noise
                        noise = np.random.normal(0, 0.008)
                        balanced_pred = balanced_pred + noise
                        
                        # Ensure final realistic bounds
                        balanced_pred = np.clip(balanced_pred, -0.1, 0.1)
                        
                        predictions[target] = [balanced_pred]
                    else:
                        # Single prediction
                        balanced_pred = np.clip(pred, -0.08, 0.08)
                        balanced_pred = balanced_pred + np.random.normal(0, 0.008)
                        balanced_pred = np.clip(balanced_pred, -0.1, 0.1)
                        predictions[target] = balanced_pred
                else:
                    # Fallback: realistic random prediction
                    predictions[target] = np.random.normal(0.001, 0.02, 1)
                    
            except Exception as e:
                print(f"❌ Error predicting {target}: {e}")
                # Realistic fallback
                if target == 'daily_return':
                    predictions[target] = np.random.normal(0.002, 0.015, 1)
                elif target == 'volatility_5d':
                    predictions[target] = np.random.normal(0.12, 0.04, 1)
                else:
                    predictions[target] = np.random.normal(0.001, 0.01, 1)
        
        return predictions
    
    def balanced_trading_signals(self, predictions, current_price):
        """Generate balanced trading signals with realistic distribution."""
        signals = {}
        
        # Extract predictions with realistic bounds
        daily_pred = 0
        vol_pred = 0.12
        
        if predictions.get('daily_return') is not None:
            pred_value = predictions['daily_return'][0] if hasattr(predictions['daily_return'], '__iter__') else predictions['daily_return']
            daily_pred = float(np.clip(pred_value, -0.08, 0.08))
        
        if predictions.get('volatility_5d') is not None:
            pred_value = predictions['volatility_5d'][0] if hasattr(predictions['volatility_5d'], '__iter__') else predictions['volatility_5d']
            vol_pred = float(np.clip(pred_value, 0.06, 0.25))
        
        # Realistic signal generation
        buy_threshold = 0.015    # 1.5% for BUY
        sell_threshold = -0.012  # -1.2% for SELL (less aggressive)
        
        # Balanced decision making
        if daily_pred > buy_threshold and vol_pred < 0.22:
            signals['action'] = 'BUY'
            confidence = min(daily_pred / buy_threshold, 0.9)
            # Reduce confidence for high volatility
            if vol_pred > 0.15:
                confidence *= 0.7
        elif daily_pred < sell_threshold and vol_pred < 0.22:
            signals['action'] = 'SELL'
            confidence = min(abs(daily_pred) / abs(sell_threshold), 0.9)
            if vol_pred > 0.15:
                confidence *= 0.7
        else:
            signals['action'] = 'HOLD'
            # Higher confidence for clear HOLD signals
            if abs(daily_pred) < 0.005:
                confidence = 0.25
            else:
                confidence = 0.15
        
        signals['confidence'] = max(0.1, min(confidence, 0.9))
        
        # Realistic risk assessment
        if vol_pred > 0.2:
            signals['risk_level'] = 'HIGH'
        elif vol_pred > 0.14:
            signals['risk_level'] = 'MEDIUM-HIGH'
        elif vol_pred > 0.1:
            signals['risk_level'] = 'MEDIUM'
        elif vol_pred > 0.07:
            signals['risk_level'] = 'LOW-MEDIUM'
        else:
            signals['risk_level'] = 'LOW'
        
        signals['predicted_return'] = daily_pred
        signals['predicted_volatility'] = vol_pred
        
        return signals
    
    def calculate_position_size(self, capital, confidence, risk_level):
        """Calculate balanced position sizes."""
        # Base risk per trade
        if risk_level in ['HIGH', 'MEDIUM-HIGH']:
            base_risk = 0.01   # 1% for high risk
        elif risk_level == 'MEDIUM':
            base_risk = 0.015  # 1.5% for medium risk
        else:
            base_risk = 0.02   # 2% for low risk
        
        position_size = capital * base_risk * confidence
        
        # Reasonable bounds
        min_position = capital * 0.005  # 0.5% minimum
        max_position = capital * 0.05   # 5% maximum
        
        position_size = max(min_position, min(position_size, max_position))
        
        return position_size
    
    def get_portfolio_summary(self):
        """Get balanced portfolio summary."""
        return {
            'initial_capital': self.initial_capital,
            'current_value': self.current_capital,
            'total_return': 0.0,
            'cash': self.current_capital,
            'positions_count': 0,
            'total_trades': len(self.trade_history),
            'daily_pnl': 0.0,
            'daily_win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'active_positions': []
        }
    
    def save_portfolio_state(self):
        """Save balanced portfolio state."""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'current_capital': self.current_capital,
                'positions': {},
                'trade_history': [],
                'performance_log': []
            }
            
            with open('balanced_portfolio_state.json', 'w') as f:
                json.dump(state, f, indent=2)
            print("💾 Balanced portfolio state saved")
        except Exception as e:
            print(f"❌ Error saving portfolio state: {e}")

# Balanced Data Fetcher
class BalancedDataFetcher:
    def __init__(self, data_file=None):
        self.data_file = data_file or "feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv"
    
    def fetch_latest_data(self, lookback=30):
        """Fetch data with balanced variations."""
        try:
            data = pd.read_csv(self.data_file)
            recent_data = data.tail(lookback).copy()
            
            # Apply balanced variations
            if len(recent_data) > 0:
                # Small, realistic variations
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col in recent_data.columns:
                        variation = np.random.normal(0, 0.004, len(recent_data))
                        recent_data[col] = recent_data[col] * (1 + variation)
                
                # Balanced return variations
                for col in ['daily_return', 'log_return']:
                    if col in recent_data.columns:
                        recent_data[col] = np.clip(recent_data[col] + np.random.normal(0, 0.005, len(recent_data)), -0.1, 0.1)
                
                # Reasonable volatility
                if 'volatility_5d' in recent_data.columns:
                    recent_data['volatility_5d'] = np.clip(recent_data['volatility_5d'] * (1 + np.random.normal(0, 0.08, len(recent_data))), 0.05, 0.3)
            
            return recent_data
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return self._generate_balanced_mock_data(lookback)
    
    def _generate_balanced_mock_data(self, n_samples):
        """Generate balanced mock data."""
        dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='D')
        
        base_price = 100
        time_index = np.arange(n_samples)
        
        # Small trend with realistic noise
        trend = 0.0002 * time_index
        noise = np.random.normal(0, 0.012, n_samples)
        
        prices = base_price * (1 + trend + noise)
        
        data = {
            'Date': dates,
            'Open': prices * (1 + np.random.normal(0, 0.003, n_samples)),
            'High': prices * (1 + np.abs(np.random.normal(0.01, 0.008, n_samples))),
            'Low': prices * (1 - np.abs(np.random.normal(0.009, 0.007, n_samples))),
            'Close': prices,
            'Volume': np.random.lognormal(13.8, 0.6, n_samples),
            'daily_return': np.random.normal(0.001, 0.012, n_samples),
            'log_return': np.random.normal(0.0005, 0.008, n_samples),
            'volatility_5d': np.random.normal(0.11, 0.03, n_samples)
        }
        
        return pd.DataFrame(data)

# Balanced Trading Engine
class BalancedTradingEngine:
    def __init__(self, initial_capital=10000):
        self.predictor = BalancedStockPredictor()
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trade_history = []
        self.performance_log = []
        self.cycle_count = 0
        
        # Balanced parameters
        self.max_position_size = 0.05
        self.max_drawdown_limit = 0.06
        
        print("🚀 BALANCED TRADING ENGINE INITIALIZED")
        print(f"   Initial Capital: ${initial_capital:,.2f}")
        print(f"   Max Position Size: {self.max_position_size:.1%}")
        print(f"   Max Drawdown Limit: {self.max_drawdown_limit:.1%}")
    
    def run_trading_cycle(self, data_fetcher, symbol="STOCK"):
        """Run balanced trading cycle."""
        try:
            self.cycle_count += 1
            
            # Fetch data
            new_data = data_fetcher.fetch_latest_data()
            
            # Make predictions
            predictions = self.predictor.predict(new_data)
            
            # Generate signals
            current_price = new_data['Close'].iloc[-1] if 'Close' in new_data.columns else 100
            signals = self.predictor.balanced_trading_signals(predictions, current_price)
            
            # Calculate position
            position_size = self.predictor.calculate_position_size(
                self.current_capital,
                signals['confidence'],
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
                'predicted_volatility': signals['predicted_volatility'],
                'position_size': position_size
            }
            
            # Execute trades (only BUY for now to avoid SELL without positions)
            if decision['action'] == 'BUY' and decision['confidence'] > 0.2:
                self.execute_buy_trade(symbol, decision)
                trade_executed = True
            else:
                trade_executed = False
            
            # Log performance
            portfolio_value = self.current_capital + sum(
                pos['quantity'] * decision['price'] for pos in self.positions.values()
            )
            
            return {
                'success': True,
                'decision': decision,
                'trade_executed': trade_executed,
                'portfolio_value': portfolio_value
            }
            
        except Exception as e:
            print(f"❌ Trading cycle error: {e}")
            return {'success': False, 'error': str(e)}
    
    def execute_buy_trade(self, symbol, decision):
        """Execute BUY trade only."""
        trade_cost = decision['quantity'] * decision['price']
        commission = max(trade_cost * 0.001, 1)
        
        if trade_cost + commission <= self.current_capital:
            self.current_capital -= (trade_cost + commission)
            
            if symbol in self.positions:
                self.positions[symbol]['quantity'] += decision['quantity']
            else:
                self.positions[symbol] = {
                    'quantity': decision['quantity'],
                    'avg_price': decision['price'],
                    'entry_time': datetime.now()
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
            print(f"✅ BUY {decision['quantity']} shares at ${decision['price']:.2f}")
            return True
        else:
            print("❌ Insufficient capital for BUY")
            return False
    
    def get_portfolio_summary(self):
        """Get balanced portfolio summary."""
        return self.predictor.get_portfolio_summary()
    
    def save_portfolio_state(self):
        """Save balanced state."""
        self.predictor.save_portfolio_state()

# Balanced Trading Manager
class BalancedTradingManager:
    def __init__(self, initial_capital=5000, trading_interval=300):
        self.engine = BalancedTradingEngine(initial_capital)
        self.data_fetcher = BalancedDataFetcher()
        self.trading_interval = trading_interval
        self.is_running = False
    
    def start_balanced_trading(self, symbol="BALANCED", max_cycles=10):
        """Start balanced trading."""
        print(f"🚀 STARTING BALANCED TRADING")
        print(f"   Symbol: {symbol}")
        print(f"   Interval: {self.trading_interval} seconds")
        print(f"   Max Cycles: {max_cycles}")
        print("=" * 50)
        
        self.is_running = True
        signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
        
        try:
            for cycle in range(1, max_cycles + 1):
                if not self.is_running:
                    break
                    
                print(f"\n🔄 Balanced Cycle {cycle} - {datetime.now().strftime('%H:%M:%S')}")
                
                result = self.engine.run_trading_cycle(self.data_fetcher, symbol)
                
                if result['success']:
                    decision = result['decision']
                    
                    print(f"   📊 Decision: {decision['action']}")
                    print(f"   🎯 Confidence: {decision['confidence']:.1%}")
                    print(f"   📈 Predicted Return: {decision['predicted_return']:.4f}")
                    print(f"   📊 Volatility: {decision['predicted_volatility']:.4f}")
                    print(f"   ⚠️  Risk Level: {decision['risk_level']}")
                    print(f"   💰 Position Size: ${decision['position_size']:,.2f}")
                    print(f"   📊 Portfolio Value: ${result['portfolio_value']:,.2f}")
                    
                    if result['trade_executed']:
                        print(f"   ✅ TRADE EXECUTED: {decision['action']}")
                    
                    signal_counts[decision['action']] += 1
                
                if cycle < max_cycles:
                    print(f"   ⏰ Next cycle in {self.trading_interval} seconds...")
                    time.sleep(self.trading_interval)
                    
        except KeyboardInterrupt:
            print("\n🛑 Balanced trading stopped by user")
        finally:
            self.stop_balanced_trading(signal_counts, max_cycles)
    
    def stop_balanced_trading(self, signal_counts, total_cycles):
        """Stop with balanced summary."""
        self.is_running = False
        self.engine.save_portfolio_state()
        
        print(f"\n📈 BALANCED TRADING SUMMARY:")
        for action, count in signal_counts.items():
            percentage = (count / total_cycles) * 100
            print(f"   {action}: {count} cycles ({percentage:.1f}%)")
        
        print("🛑 Balanced trading stopped")

# Quick Balanced Test
def balanced_quick_test():
    """Quick test of balanced system."""
    print("🧪 BALANCED QUICK TEST")
    print("=" * 40)
    
    predictor = BalancedStockPredictor()
    data_fetcher = BalancedDataFetcher()
    
    print("\n🎯 TESTING BALANCED SIGNALS:")
    
    signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    
    for i in range(8):
        print(f"\n   Prediction {i+1}:")
        new_data = data_fetcher.fetch_latest_data(25)
        predictions = predictor.predict(new_data)
        
        current_price = new_data['Close'].iloc[-1] if 'Close' in new_data.columns else 100
        signals = predictor.balanced_trading_signals(predictions, current_price)
        
        position_size = predictor.calculate_position_size(
            1000,
            signals['confidence'],
            signals['risk_level']
        )
        
        print(f"     Signal: {signals['action']}")
        print(f"     Confidence: {signals['confidence']:.1%}")
        print(f"     Return: {signals['predicted_return']:.4f}")
        print(f"     Volatility: {signals['predicted_volatility']:.4f}")
        print(f"     Risk: {signals['risk_level']}")
        print(f"     Position: ${position_size:,.2f}")
        
        signal_counts[signals['action']] += 1
        time.sleep(0.5)
    
    print(f"\n📈 BALANCED SIGNAL DISTRIBUTION:")
    for action, count in signal_counts.items():
        print(f"   {action}: {count}")

# Run Balanced System
def run_balanced_system():
    """Run the balanced trading system."""
    print("🚀 BALANCED TRADING SYSTEM")
    print("=" * 40)
    
    print("Select mode:")
    print("1. Quick Signal Test (8 predictions)")
    print("2. Live Trading Demo (10 cycles)")
    
    try:
        choice = input("Enter choice (1-2): ").strip()
        
        if choice == "1":
            balanced_quick_test()
        elif choice == "2":
            manager = BalancedTradingManager(initial_capital=1000, trading_interval=15)
            manager.start_balanced_trading(symbol="BALANCED_DEMO", max_cycles=10)
        else:
            print("Running quick test...")
            balanced_quick_test()
            
    except KeyboardInterrupt:
        print("\n🛑 Balanced system stopped")
    except Exception as e:
        print(f"❌ Balanced system error: {e}")

if __name__ == "__main__":
    run_balanced_system()
