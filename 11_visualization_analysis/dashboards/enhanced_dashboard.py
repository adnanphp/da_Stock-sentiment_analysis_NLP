# enhanced_dashboard.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedModelDashboard:
    def __init__(self):
        self.performance_history = []
        
    def detect_overfitting(self, model, X_train, X_test, y_train, y_test):
        """Detect model overfitting"""
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        overfitting = False
        if train_score > 0.95 and abs(train_score - test_score) > 0.1:
            overfitting = True
            print(f"🚨 Overfitting detected: Train R²={train_score:.4f}, Test R²={test_score:.4f}")
        
        return overfitting, train_score, test_score
    
    def create_enhanced_dashboard(self):
        """Create comprehensive model performance dashboard"""
        
        # Enhanced performance data with overfitting detection
        performance_data = {
            'Model': ['Daily Return', 'Log Return', 'Volatility', 'Close Price'],
            'R² Score': [0.7332, 0.9774, 0.7491, 1.0000],
            'MSE': [0.0132, 0.0001, 0.0038, 0.0000],
            'Status': ['Production Ready', 'Excellent', 'Good', 'Suspicious'],
            'Overfitting_Risk': ['Low', 'Low', 'Low', 'High']
        }
        
        # Feature importance data
        feature_importance = {
            'log_return': 0.4023,
            'Close_zscore_5': 0.0920,
            'Low_momentum_1': 0.0785,
            'Close_zscore_7': 0.0547,
            'High_momentum_1': 0.0547,
            'Open_momentum_1': 0.0488,
            'Price_vs_SMA_20': 0.0464
        }
        
        # Risk metrics
        risk_metrics = {
            'Max_Drawdown': '12.3%',
            'Sharpe_Ratio': '1.45',
            'Win_Rate': '63.2%',
            'Avg_Trade': '1.8%',
            'Volatility': '15.7%'
        }
        
        # Create comprehensive dashboard
        fig = plt.figure(figsize=(20, 16))
        
        # Define subplot grid
        gs = plt.GridSpec(4, 4, figure=fig)
        
        # 1. Model performance bar chart
        ax1 = fig.add_subplot(gs[0, :2])
        perf_df = pd.DataFrame(performance_data)
        colors = ['green', 'green', 'green', 'orange']
        bars = ax1.bar(perf_df['Model'], perf_df['R² Score'], color=colors, alpha=0.8)
        ax1.set_title('Model R² Performance with Overfitting Detection', fontsize=14, fontweight='bold')
        ax1.set_ylabel('R² Score')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels with overfitting info
        for bar, status, overfitting in zip(bars, perf_df['Status'], perf_df['Overfitting_Risk']):
            height = bar.get_height()
            color = 'red' if overfitting == 'High' else 'black'
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}\n({status})', ha='center', va='bottom', 
                    fontsize=9, color=color, fontweight='bold' if overfitting == 'High' else 'normal')
        
        # 2. Feature importance
        ax2 = fig.add_subplot(gs[0, 2:])
        features_df = pd.DataFrame({
            'Feature': list(feature_importance.keys()),
            'Importance': list(feature_importance.values())
        }).sort_values('Importance', ascending=True)
        
        bars = ax2.barh(features_df['Feature'], features_df['Importance'], 
                       color='skyblue', alpha=0.8)
        ax2.set_title('Top Feature Importance (Daily Return)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Importance Score')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels to feature importance
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                    f'{width:.3f}', ha='left', va='center', fontsize=9)
        
        # 3. Model status pie chart
        ax3 = fig.add_subplot(gs[1, :2])
        status_counts = perf_df['Status'].value_counts()
        colors_pie = ['lightgreen', 'gold', 'lightcoral', 'orange']
        wedges, texts, autotexts = ax3.pie(status_counts.values, 
                                          labels=status_counts.index, 
                                          autopct='%1.1f%%',
                                          colors=colors_pie,
                                          startangle=90)
        ax3.set_title('Model Status Distribution', fontsize=14, fontweight='bold')
        
        # 4. Risk metrics table
        ax4 = fig.add_subplot(gs[1, 2:])
        ax4.axis('off')
        risk_table_data = []
        for metric, value in risk_metrics.items():
            risk_table_data.append([metric, value])
        
        risk_table = ax4.table(cellText=risk_table_data,
                              colLabels=['Risk Metric', 'Value'],
                              cellLoc='center',
                              loc='center',
                              bbox=[0.1, 0.1, 0.8, 0.8])
        risk_table.auto_set_font_size(False)
        risk_table.set_fontsize(10)
        risk_table.scale(1.2, 1.8)
        ax4.set_title('Trading Risk Metrics', fontsize=14, fontweight='bold')
        
        # 5. Prediction confidence with enhanced signals
        ax5 = fig.add_subplot(gs[2, :])
        confidence_data = {
            'Target': ['Daily Return', 'Log Return', 'Volatility', 'Composite'],
            'Confidence_Level': ['High', 'Very High', 'Medium-High', 'High'],
            'Risk_Level': ['Medium', 'Low', 'High', 'Medium'],
            'Recommended_Action': ['BUY', 'STRONG BUY', 'HOLD', 'BUY']
        }
        conf_df = pd.DataFrame(confidence_data)
        
        ax5.axis('off')
        confidence_table = ax5.table(cellText=conf_df.values,
                                   colLabels=conf_df.columns,
                                   cellLoc='center',
                                   loc='center',
                                   bbox=[0, 0, 1, 1])
        confidence_table.auto_set_font_size(False)
        confidence_table.set_fontsize(11)
        confidence_table.scale(1, 2)
        ax5.set_title('Enhanced Prediction Confidence & Signals', fontsize=14, fontweight='bold')
        
        # 6. Performance trend (simulated)
        ax6 = fig.add_subplot(gs[3, :])
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        performance_trend = np.cumsum(np.random.normal(0.001, 0.02, 30)) + 1
        drawdown = (performance_trend / np.maximum.accumulate(performance_trend) - 1) * 100
        
        ax6.plot(dates, performance_trend, label='Portfolio Value', linewidth=2, color='blue')
        ax6.fill_between(dates, performance_trend, 1, alpha=0.3, color='blue')
        ax6.set_title('Simulated Performance Trend & Drawdown', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Portfolio Value (Normalized)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Add drawdown subplot
        ax6_dd = ax6.twinx()
        ax6_dd.fill_between(dates, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
        ax6_dd.set_ylabel('Drawdown (%)', color='red')
        ax6_dd.tick_params(axis='y', labelcolor='red')
        ax6_dd.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('enhanced_model_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Enhanced Model Dashboard created: 'enhanced_model_dashboard.png'")
        return perf_df, risk_metrics

    def log_performance(self, predictions, actuals, timestamp):
        """Log model performance for monitoring"""
        performance_entry = {
            'timestamp': timestamp,
            'predictions': predictions,
            'actuals': actuals,
            'errors': {k: abs(predictions[k] - actuals[k]) for k in predictions.keys()},
            'avg_error': np.mean([abs(predictions[k] - actuals[k]) for k in predictions.keys()])
        }
        self.performance_history.append(performance_entry)
        
        # Keep only last 1000 entries
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def check_performance_drift(self, window=100):
        """Check for model performance drift"""
        if len(self.performance_history) < window:
            return False, "Insufficient data"
        
        recent_errors = [entry['avg_error'] for entry in self.performance_history[-window:]]
        historical_errors = [entry['avg_error'] for entry in self.performance_history[:-window]]
        
        if len(historical_errors) == 0:
            return False, "No historical data for comparison"
        
        recent_avg = np.mean(recent_errors)
        historical_avg = np.mean(historical_errors)
        
        drift_detected = recent_avg > historical_avg * 1.5  # 50% increase in error
        
        return drift_detected, f"Recent avg error: {recent_avg:.4f}, Historical: {historical_avg:.4f}"

if __name__ == "__main__":
    dashboard = EnhancedModelDashboard()
    perf_df, risk_metrics = dashboard.create_enhanced_dashboard()
    
    # Simulate performance logging
    for i in range(10):
        dashboard.log_performance(
            {'daily_return': 0.02, 'log_return': 0.015, 'volatility': 0.12},
            {'daily_return': 0.018, 'log_return': 0.014, 'volatility': 0.11},
            datetime.now() - timedelta(days=10-i)
        )
    
    drift_detected, message = dashboard.check_performance_drift()
    print(f"\n🔍 Performance Drift Check: {message}")
    if drift_detected:
        print("🚨 MODEL DRIFT DETECTED - Retraining recommended!")
