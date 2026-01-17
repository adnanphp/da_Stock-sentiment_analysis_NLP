# fixed_production_predictor.py
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class StockPredictor:
    """
    Production-ready stock prediction system.
    """
    
    def __init__(self, model_dir="robust_models"):
        self.model_dir = model_dir
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all trained models with fallback options."""
        print("🔄 Loading models...")
        
        # Try to load robust models first
        robust_targets = ['daily_return', 'log_return', 'volatility_5d']
        
        for target in robust_targets:
            try:
                # Try robust model first
                model_path = f"{self.model_dir}/robust_{target}_model.joblib"
                if os.path.exists(model_path):
                    self.models[target] = joblib.load(model_path)
                    print(f"   ✅ Loaded robust model: {target}")
                else:
                    # Fallback to simple model
                    simple_path = f"{self.model_dir}/simple_{target}_model.joblib"
                    if os.path.exists(simple_path):
                        self.models[target] = joblib.load(simple_path)
                        print(f"   ✅ Loaded simple model: {target}")
                    else:
                        # Fallback to fixed trained models
                        fixed_path = f"fixed_trained_models/feature_engineered_outlier_treated_all_stocks_combined_{target}_model.joblib"
                        if os.path.exists(fixed_path):
                            self.models[target] = joblib.load(fixed_path)
                            print(f"   ✅ Loaded fixed model: {target}")
                        else:
                            print(f"   ⚠️  No model found for: {target}")
            except Exception as e:
                print(f"   ❌ Error loading {target}: {e}")
        
        print(f"✅ Successfully loaded {len(self.models)} models")
    
    def predict(self, new_data):
        """
        Make predictions on new stock data.
        """
        predictions = {}
        
        for target, model_data in self.models.items():
            try:
                # Handle different model formats
                if isinstance(model_data, dict) and 'model' in model_data:
                    # Robust model format
                    model = model_data['model']
                    X_new = self._prepare_features(new_data)
                    pred = model.predict(X_new)
                    predictions[target] = pred
                else:
                    # Simple model or fixed model format
                    model = model_data
                    X_new = self._prepare_features(new_data)
                    pred = model.predict(X_new)
                    predictions[target] = pred
                    
            except Exception as e:
                print(f"❌ Error predicting {target}: {e}")
                predictions[target] = None
        
        return predictions
    
    def _prepare_features(self, data):
        """Prepare features for prediction."""
        X = data.select_dtypes(include=[np.number])
        
        # Remove target columns if present
        target_cols = ['daily_return', 'log_return', 'volatility_5d', 'Close']
        X = X.drop(columns=[col for col in target_cols if col in X.columns], errors='ignore')
        
        # Handle missing values
        X = X.fillna(X.median())
        
        return X
    
    def generate_trading_signals(self, predictions, threshold=0.02):
        """
        Generate simple trading signals based on predictions.
        """
        signals = {}
        
        daily_return_pred = predictions.get('daily_return')
        
        if daily_return_pred is not None and len(daily_return_pred) > 0:
            # Get the latest prediction
            if hasattr(daily_return_pred, '__iter__'):
                latest_return = daily_return_pred[-1] if len(daily_return_pred) > 0 else daily_return_pred[0]
            else:
                latest_return = daily_return_pred
            
            # Generate signal
            if latest_return > threshold:
                signals['action'] = 'BUY'
                signals['confidence'] = min(abs(latest_return) / threshold, 1.0)
            elif latest_return < -threshold:
                signals['action'] = 'SELL' 
                signals['confidence'] = min(abs(latest_return) / threshold, 1.0)
            else:
                signals['action'] = 'HOLD'
                signals['confidence'] = 0.0
                
            signals['predicted_return'] = float(latest_return)
            signals['threshold_used'] = threshold
        
        return signals

    def get_model_info(self):
        """Get information about loaded models."""
        info = {}
        for target, model_data in self.models.items():
            if isinstance(model_data, dict):
                info[target] = {
                    'type': 'robust',
                    'mse': model_data.get('mse', 'N/A'),
                    'r2': model_data.get('r2', 'N/A'),
                    'top_features': model_data.get('top_features', [])[:3]
                }
            else:
                info[target] = {
                    'type': 'simple',
                    'features_used': 'All available'
                }
        return info

# Example usage with better error handling
if __name__ == "__main__":
    print("🚀 INITIALIZING STOCK PREDICTION SYSTEM")
    print("=" * 50)
    
    # Initialize predictor
    predictor = StockPredictor()
    
    # Show model info
    model_info = predictor.get_model_info()
    print("\n📊 LOADED MODELS:")
    for target, info in model_info.items():
        print(f"   • {target}: {info['type']} model")
        if 'r2' in info and info['r2'] != 'N/A':
            print(f"     R²: {info['r2']:.4f}, MSE: {info['mse']:.4f}")
        if 'top_features' in info:
            print(f"     Top features: {', '.join(info['top_features'])}")
    
    # Example: Load new data for prediction
    try:
        print(f"\n🎯 MAKING PREDICTIONS")
        print("=" * 40)
        
        # Use the last 5 rows of your data as "new" data
        new_data = pd.read_csv("feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv").tail(5)
        
        print(f"   Using {len(new_data)} samples for prediction")
        
        # Make predictions
        predictions = predictor.predict(new_data)
        
        print(f"\n📈 PREDICTION RESULTS:")
        print("-" * 30)
        for target, pred in predictions.items():
            if pred is not None:
                if hasattr(pred, '__iter__'):
                    # Show predictions for all samples
                    for i, p in enumerate(pred):
                        print(f"   {target} [Sample {i+1}]: {p:.6f}")
                else:
                    print(f"   {target}: {pred:.6f}")
            else:
                print(f"   {target}: No prediction available")
        
        # Generate trading signals (use first prediction)
        if predictions.get('daily_return') is not None:
            first_pred = predictions['daily_return'][0] if hasattr(predictions['daily_return'], '__iter__') else predictions['daily_return']
            signals = predictor.generate_trading_signals({'daily_return': [first_pred]})
            
            print(f"\n💰 TRADING SIGNALS:")
            print("-" * 30)
            print(f"   Action: {signals.get('action', 'UNKNOWN')}")
            print(f"   Confidence: {signals.get('confidence', 0):.2%}")
            print(f"   Predicted Return: {signals.get('predicted_return', 0):.6f}")
            print(f"   Threshold: {signals.get('threshold_used', 0):.4f}")
        
    except Exception as e:
        print(f"❌ Error in prediction example: {e}")
        print("💡 Make sure you've run the model training scripts first!")
