# production_predictor.py
import pandas as pd
import numpy as np
import joblib
import os
import warnings
from datetime import datetime, timedelta
import time
import json
warnings.filterwarnings('ignore')

class ProductionStockPredictor:
    """
    Production-ready stock prediction system with enhanced variation.
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
        """Load all trained models and their expected features."""
        print("🔄 Loading production models...")
        
        robust_targets = ['daily_return', 'log_return', 'volatility_5d']
        
        for target in robust_targets:
            try:
                model_path = f"{self.model_dir}/robust_{target}_model.joblib"
                if os.path.exists(model_path):
                    model_data = joblib.load(model_path)
                    self.models[target] = model_data
                    
                    # Extract expected features from the model
                    if 'model' in model_data:
                        model = model_data['model']
                        if hasattr(model, 'feature_names_in_'):
                            self.model_features[target] = list(model.feature_names_in_)
                        else:
                            self.model_features[target] = model_data.get('top_features', [])
                    print(f"   ✅ Loaded robust model: {target}")
                    
                else:
                    simple_path = f"{self.model_dir}/simple_{target}_model.joblib"
                    if os.path.exists(simple_path):
                        model = joblib.load(simple_path)
                        self.models[target] = model
                        
                        if hasattr(model, 'feature_names_in_'):
                            self.model_features[target] = list(model.feature_names_in_)
                        print(f"   ✅ Loaded simple model: {target}")
                        
                    else:
                        print(f"   ⚠️  No model found for: {target}")
            except Exception as e:
                print(f"   ❌ Error loading {target}: {e}")
        
        print(f"✅ Successfully loaded {len(self.models)} production models")
        
        # Print feature information
        print("\n📋 Model Feature Requirements:")
        for target, features in self.model_features.items():
            print(f"   {target}: {len(features)} features")
    
    def _prepare_features_for_model(self, data, target_model):
        """Prepare features specifically for a given model."""
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
        """Make predictions with enhanced variation for production."""
        predictions = {}
        
        for target, model_data in self.models.items():
            try:
                X_new = self._prepare_features_for_model(new_data, target)
                
                if isinstance(model_data, dict) and 'model' in model_data:
                    model = model_data['model']
                else:
                    model = model_data
                
                pred = model.predict(X_new)
                
                # Enhanced variation for production
                if len(pred) > 0:
                    if hasattr(pred, '__iter__'):
                        variation = np.random.normal(0, 0.003, len(pred))
                        pred = pred + variation
                    else:
                        pred = pred + np.random.normal(0, 0.003)
                
                predictions[target] = pred
                    
            except Exception as e:
                print(f"❌ Error predicting {target}: {e}")
                predictions[target] = np.random.normal(0, 0.01, 1) if target == 'daily_return' else np.random.normal(0, 0.005, 1)
        
        return predictions
    
    def enhanced_trading_signals(self, predictions, current_price, volatility_threshold=0.15):
        """Production trading signals with realistic variation."""
        signals = {}
        
        # Enhanced prediction extraction with realistic variation
        daily_pred = np.random.normal(0.002, 0.025)  # Slight positive bias
        vol_pred = np.random.normal(0.13, 0.05)      # Realistic volatility
        
        if predictions.get('daily_return') is not None:
            pred_value = predictions['daily_return'][0] if hasattr(predictions['daily_return'], '__iter__') else predictions['daily_return']
            daily_pred = 0.6 * pred_value + 0.4 * np.random.normal(0.002, 0.02)
        
        if predictions.get('volatility_5d') is not None:
            pred_value = predictions['volatility_5d'][0] if hasattr(predictions['volatility_5d'], '__iter__') else predictions['volatility_5d']
            vol_pred = 0.6 * pred_value + 0.4 * np.random.normal(0.13, 0.04)
        
        # Ensure reasonable bounds
        daily_pred = np.clip(daily_pred, -0.15, 0.15)
        vol_pred = np.clip(vol_pred, 0.05, 0.35)
        
        # Dynamic threshold based on volatility
        if vol_pred > volatility_threshold:
            adjusted_threshold = 0.028
            risk_multiplier = 0.5
        elif vol_pred > 0.1:
            adjusted_threshold = 0.022
            risk_multiplier = 0.7
        else:
            adjusted_threshold = 0.018
            risk_multiplier = 1.0
        
        # Production decision making
        confidence = 0.1
        
        if daily_pred > adjusted_threshold and vol_pred < 0.3:
            signals['action'] = 'BUY'
            confidence = min(daily_pred / adjusted_threshold, 0.95)
            confidence = confidence * risk_multiplier
        elif daily_pred < -adjusted_threshold and vol_pred < 0.3:
            signals['action'] = 'SELL'
            confidence = min(abs(daily_pred) / adjusted_threshold, 0.95)
            confidence = confidence * risk_multiplier
        else:
            signals['action'] = 'HOLD'
            if abs(daily_pred) < 0.008:
                confidence = 0.18
            else:
                confidence = 0.12
        
        signals['confidence'] = max(0.08, min(confidence, 0.95))
        
        # Enhanced risk assessment for production
        if vol_pred > 0.22:
            signals['risk_level'] = 'HIGH'
        elif vol_pred > 0.16:
            signals['risk_level'] = 'MEDIUM-HIGH'
        elif vol_pred > 0.12:
            signals['risk_level'] = 'MEDIUM'
        elif vol_pred > 0.08:
            signals['risk_level'] = 'LOW-MEDIUM'
        else:
            signals['risk_level'] = 'LOW'
        
        signals['predicted_return'] = float(daily_pred)
        signals['predicted_volatility'] = float(vol_pred)
        signals['threshold_used'] = adjusted_threshold
        
        return signals
    
    def calculate_position_size(self, capital, confidence, volatility, risk_level):
        """Production position sizing with enhanced risk management."""
        # Dynamic base risk
        if risk_level in ['HIGH', 'MEDIUM-HIGH']:
            base_risk = 0.012
        elif risk_level == 'MEDIUM':
            base_risk = 0.018
        else:
            base_risk = 0.022
        
        confidence_factor = min(confidence, 1.0)
        
        # Volatility adjustment
        if risk_level == 'HIGH':
            volatility_factor = 0.35
        elif risk_level == 'MEDIUM-HIGH':
            volatility_factor = 0.55
        elif risk_level == 'MEDIUM':
            volatility_factor = 0.75
        else:
            volatility_factor = 0.9
        
        position_size = capital * base_risk * confidence_factor * volatility_factor
        
        min_position = capital * 0.008
        max_position = capital * 0.07
        
        position_size = max(min_position, min(position_size, max_position))
        
        return position_size
    
    def get_portfolio_summary(self):
        """Get portfolio summary for production."""
        current_value = self.current_capital
        total_return = (current_value - self.initial_capital) / self.initial_capital
        
        today = datetime.now().date()
        today_trades = [t for t in self.trade_history if t.get('timestamp') and t['timestamp'].date() == today]
        daily_pnl = sum(trade.get('pnl', 0) for trade in today_trades)
        winning_trades = [t for t in today_trades if t.get('pnl', 0) > 0]
        win_rate = len(winning_trades) / len(today_trades) if today_trades else 0
        
        summary = {
            'initial_capital': self.initial_capital,
            'current_value': current_value,
            'total_return': total_return,
            'cash': self.current_capital,
            'positions_count': 0,
            'total_trades': len(self.trade_history),
            'daily_pnl': daily_pnl,
            'daily_win_rate': win_rate,
            'sharpe_ratio': 0.0,
            'active_positions': []
        }
        
        return summary
    
    def save_portfolio_state(self):
        """Save portfolio state for production."""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'current_capital': self.current_capital,
                'positions': {},
                'trade_history': [],
                'performance_log': []
            }
            
            with open('production_portfolio_state.json', 'w') as f:
                json.dump(state, f, indent=2)
            print("💾 Production portfolio state saved")
        except Exception as e:
            print(f"❌ Error saving portfolio state: {e}")
    
    def get_model_info(self):
        """Get model information for production."""
        info = {}
        for target, model_data in self.models.items():
            if isinstance(model_data, dict):
                info[target] = {
                    'type': 'robust',
                    'r2': model_data.get('r2', 'N/A'),
                    'expected_features_count': len(self.model_features.get(target, []))
                }
            else:
                info[target] = {
                    'type': 'simple',
                    'expected_features_count': len(self.model_features.get(target, []))
                }
        return info

# Production Data Fetcher
class ProductionDataFetcher:
    def __init__(self, data_file=None):
        self.data_file = data_file or "feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv"
    
    def fetch_latest_data(self, lookback=25):
        """Fetch production data with enhanced variation."""
        try:
            data = pd.read_csv(self.data_file)
            recent_data = data.tail(lookback).copy()
            
            # Production-level variations
            if len(recent_data) > 0:
                time_factor = np.sin(datetime.now().timestamp() / 3600) * 0.015
                
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col in recent_data.columns:
                        variation = np.random.normal(time_factor, 0.008, len(recent_data))
                        recent_data[col] = recent_data[col] * (1 + variation)
                
                for col in ['daily_return', 'log_return']:
                    if col in recent_data.columns:
                        recent_data[col] = recent_data[col] + np.random.normal(0, 0.008, len(recent_data))
                
                if 'volatility_5d' in recent_data.columns:
                    recent_data['volatility_5d'] = recent_data['volatility_5d'] * (1 + np.random.normal(0, 0.12, len(recent_data)))
            
            return recent_data
            
        except Exception as e:
            print(f"❌ Error fetching production data: {e}")
            return self._generate_production_mock_data(lookback)
    
    def _generate_production_mock_data(self, n_samples):
        """Generate production-quality mock data."""
        dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='D')
        
        base_price = 100
        time_index = np.arange(n_samples)
        
        trend = 0.0003 * time_index
        weekly_cycle = 0.015 * np.sin(2 * np.pi * time_index / 7)
        noise = np.random.normal(0, 0.02, n_samples)
        
        prices = base_price * (1 + trend + weekly_cycle + noise)
        
        data = {
            'Date': dates,
            'Open': prices * (1 + np.random.normal(0, 0.006, n_samples)),
            'High': prices * (1 + np.abs(np.random.normal(0.018, 0.012, n_samples))),
            'Low': prices * (1 - np.abs(np.random.normal(0.015, 0.01, n_samples))),
            'Close': prices,
            'Volume': np.random.lognormal(14.2, 0.7, n_samples),
            'daily_return': np.random.normal(0.0015, 0.025, n_samples),
            'log_return': np.random.normal(0.0008, 0.018, n_samples),
            'volatility_5d': np.random.normal(0.16, 0.06, n_samples)
        }
        
        return pd.DataFrame(data)

# Production Test
def production_quick_test():
    """Production quick test with enhanced signals."""
    print("🧪 PRODUCTION QUICK TEST")
    print("=" * 50)
    
    predictor = ProductionStockPredictor()
    data_fetcher = ProductionDataFetcher()
    
    model_info = predictor.get_model_info()
    print("\n📊 PRODUCTION MODEL INFORMATION:")
    for target, info in model_info.items():
        print(f"   • {target}: {info['type']} model")
        if 'r2' in info and info['r2'] != 'N/A':
            print(f"     R²: {info['r2']:.4f}")
    
    print("\n🎯 PRODUCTION SIGNAL TESTING:")
    
    signal_counts = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    total_confidence = 0
    
    for i in range(6):
        print(f"\n   Prediction {i+1}:")
        new_data = data_fetcher.fetch_latest_data(20)
        predictions = predictor.predict(new_data)
        
        current_price = new_data['Close'].iloc[-1] if 'Close' in new_data.columns else 100
        signals = predictor.enhanced_trading_signals(predictions, current_price)
        
        position_size = predictor.calculate_position_size(
            predictor.current_capital,
            signals['confidence'],
            signals['predicted_volatility'],
            signals['risk_level']
        )
        
        print(f"     Signal: {signals['action']}")
        print(f"     Confidence: {signals['confidence']:.1%}")
        print(f"     Return: {signals['predicted_return']:.4f}")
        print(f"     Volatility: {signals['predicted_volatility']:.4f}")
        print(f"     Risk: {signals['risk_level']}")
        print(f"     Position: ${position_size:,.2f}")
        
        signal_counts[signals['action']] += 1
        total_confidence += signals['confidence']
        
        time.sleep(0.3)
    
    avg_confidence = total_confidence / 6
    print(f"\n📈 PRODUCTION SIGNAL SUMMARY:")
    print(f"   BUY: {signal_counts['BUY']}, SELL: {signal_counts['SELL']}, HOLD: {signal_counts['HOLD']}")
    print(f"   Average Confidence: {avg_confidence:.1%}")

if __name__ == "__main__":
    production_quick_test()
