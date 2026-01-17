# check_data_leakage.py
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def check_temporal_leakage(df, target_col='Close'):
    """Check for temporal data leakage in time series data."""
    print("🔍 CHECKING FOR TEMPORAL DATA LEAKAGE")
    print("=" * 50)
    
    # Sort by date if available
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if date_cols:
        df = df.sort_values(date_cols[0])
        print(f"   Sorted by: {date_cols[0]}")
    
    # Prepare features and target
    X = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    y = df[target_col]
    
    # Time Series Cross Validation
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Simple model
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = mean_squared_error(y_test, y_pred)
        scores.append(score)
    
    print(f"   Time Series CV MSE: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    
    # Check for features that might cause leakage
    suspicious_features = []
    for col in X.columns:
        if any(term in col.lower() for term in ['future', 'lead', 'next', 'tomorrow', 'ahead']):
            suspicious_features.append(col)
        # Check for features that use future information
        if 'lag' in col.lower():
            lag_value = int(col.split('_')[-1]) if col.split('_')[-1].isdigit() else 1
            if lag_value < 0:  # Negative lag means future values
                suspicious_features.append(col)
    
    if suspicious_features:
        print(f"   ⚠️  Suspicious features found: {suspicious_features}")
    else:
        print("   ✅ No obvious temporal leakage features found")
    
    return np.mean(scores)

# Run leakage check
if __name__ == "__main__":
    df = pd.read_csv("feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv")
    check_temporal_leakage(df, 'Close')
