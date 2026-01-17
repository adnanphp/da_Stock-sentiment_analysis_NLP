# model_dashboard.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def create_model_dashboard():
    """Create a dashboard showing model performance and feature importance."""
    
    # Model performance data
    performance_data = {
        'Model': ['Daily Return', 'Log Return', 'Volatility', 'Close Price'],
        'R² Score': [0.7332, 0.9774, 0.7491, 1.0000],
        'MSE': [0.0132, 0.0001, 0.0038, 0.0000],
        'Status': ['Production Ready', 'Excellent', 'Good', 'Suspicious']
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
    
    # Create plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Model performance bar chart
    perf_df = pd.DataFrame(performance_data)
    colors = ['green', 'green', 'green', 'orange']
    bars = ax1.bar(perf_df['Model'], perf_df['R² Score'], color=colors)
    ax1.set_title('Model R² Performance', fontsize=14, fontweight='bold')
    ax1.set_ylabel('R² Score')
    ax1.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bar, status in zip(bars, perf_df['Status']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}\n({status})', ha='center', va='bottom', fontsize=9)
    
    # Feature importance
    features_df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    }).sort_values('Importance', ascending=True)
    
    ax2.barh(features_df['Feature'], features_df['Importance'], color='skyblue')
    ax2.set_title('Top Feature Importance (Daily Return)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Importance Score')
    
    # Model status pie chart
    status_counts = perf_df['Status'].value_counts()
    ax3.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%', 
            colors=['lightgreen', 'gold', 'lightcoral'])
    ax3.set_title('Model Status Distribution', fontsize=14, fontweight='bold')
    
    # Prediction confidence
    confidence_data = {
        'Target': ['Daily Return', 'Log Return', 'Volatility'],
        'Confidence_Level': ['High', 'Very High', 'Medium-High']
    }
    conf_df = pd.DataFrame(confidence_data)
    ax4.axis('off')
    ax4.table(cellText=conf_df.values,
              colLabels=conf_df.columns,
              cellLoc='center',
              loc='center',
              bbox=[0, 0, 1, 1])
    ax4.set_title('Prediction Confidence Levels', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Model Performance Dashboard created: 'model_performance_dashboard.png'")

if __name__ == "__main__":
    create_model_dashboard()
