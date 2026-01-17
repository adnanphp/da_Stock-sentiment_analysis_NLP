"""
classification_stock_news_experiment_fixed.py - Fixed version handling non-numeric data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime
import json
import pickle

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_clean_data():
    """Load and clean the integrated dataset"""
    print("="*80)
    print("LOADING AND CLEANING INTEGRATED DATASET")
    print("="*80)
    
    # Load the FULL integrated dataset
    data_file = "./integrated_stock_news/stock_news_integrated_20251202_231018.csv"
    df = pd.read_csv(data_file)
    
    print(f"📂 Initial dataset shape: {df.shape}")
    
    # First, let's identify and handle non-numeric columns
    print(f"\n🔍 IDENTIFYING COLUMN TYPES:")
    
    numeric_cols = []
    non_numeric_cols = []
    problem_cols = []
    
    for col in df.columns:
        if col in ['date', 'ticker', 'price_change', 'target']:
            continue
            
        try:
            # Try to convert to numeric
            pd.to_numeric(df[col], errors='raise')
            numeric_cols.append(col)
        except:
            non_numeric_cols.append(col)
            
            # Check sample values
            sample_values = df[col].dropna().unique()[:5]
            if len(sample_values) > 0:
                print(f"   • {col}: Non-numeric values like {sample_values[:3]}")
                problem_cols.append(col)
    
    print(f"\n📊 COLUMN STATISTICS:")
    print(f"   • Numeric columns: {len(numeric_cols)}")
    print(f"   • Non-numeric columns: {len(non_numeric_cols)}")
    print(f"   • Problem columns identified: {len(problem_cols)}")
    
    # Try to fix common issues
    print(f"\n🛠️  ATTEMPTING TO FIX DATA ISSUES...")
    
    df_clean = df.copy()
    fixed_cols = []
    
    for col in problem_cols:
        try:
            # Try to convert to numeric, coercing errors to NaN
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Check if we successfully converted any values
            if df_clean[col].notna().any():
                fixed_cols.append(col)
                print(f"   ✅ Fixed {col}: Converted to numeric")
            else:
                print(f"   ❌ Could not fix {col}: All values became NaN")
        except:
            print(f"   ❌ Failed to fix {col}")
    
    # Update numeric columns list
    numeric_cols = []
    for col in df_clean.columns:
        if col in ['date', 'ticker', 'price_change', 'target']:
            continue
        if pd.api.types.is_numeric_dtype(df_clean[col]):
            numeric_cols.append(col)
    
    print(f"\n📊 AFTER CLEANING:")
    print(f"   • Numeric columns available: {len(numeric_cols)}")
    print(f"   • Problem columns fixed: {len(fixed_cols)}")
    
    # Separate stock and news features
    stock_features = [col for col in numeric_cols if not col.startswith('news_')]
    news_features = [col for col in numeric_cols if col.startswith('news_')]
    
    print(f"\n📊 FEATURE BREAKDOWN:")
    print(f"   • Clean stock features: {len(stock_features)}")
    print(f"   • Clean news features: {len(news_features)}")
    print(f"   • Total clean features: {len(stock_features) + len(news_features)}")
    
    # Check target distribution
    print(f"\n🎯 TARGET DISTRIBUTION:")
    print(f"   • UP (target=1): {(df_clean['target'] == 1).sum()} ({df_clean['target'].mean():.2%})")
    print(f"   • DOWN (target=0): {(df_clean['target'] == 0).sum()} ({1 - df_clean['target'].mean():.2%})")
    
    # Check news data presence
    print(f"\n📰 NEWS DATA COVERAGE:")
    tickers = df_clean['ticker'].unique()
    for ticker in tickers:
        ticker_data = df_clean[df_clean['ticker'] == ticker]
        count_col = f'news_{ticker}_count'
        
        if count_col in df_clean.columns:
            news_records = ticker_data[count_col].notna().sum()
            print(f"   • {ticker}: {news_records}/{len(ticker_data)} records have news ({news_records/len(ticker_data):.1%})")
    
    return df_clean, stock_features, news_features

def prepare_ticker_data_simple(df, ticker, stock_features, news_features):
    """Simple data preparation for a ticker"""
    print(f"\n📊 PREPARING DATA FOR {ticker}:")
    
    # Filter data for this ticker
    ticker_data = df[df['ticker'] == ticker].copy()
    
    # Select only numeric features that exist for this ticker
    available_stock_features = [col for col in stock_features if col in ticker_data.columns]
    available_news_features = [col for col in news_features if col in ticker_data.columns and ticker in col]
    
    print(f"   • Total records: {len(ticker_data)}")
    print(f"   • Available stock features: {len(available_stock_features)}")
    print(f"   • Available news features: {len(available_news_features)}")
    print(f"   • Total features: {len(available_stock_features) + len(available_news_features)}")
    
    # Create feature sets
    X_stock = ticker_data[available_stock_features].fillna(0).values
    X_news = ticker_data[available_news_features].fillna(0).values
    X_all = ticker_data[available_stock_features + available_news_features].fillna(0).values
    y = ticker_data['target'].values
    
    # Check if we have any real news data (not all zeros)
    has_news_data = False
    if len(available_news_features) > 0:
        # Check if any news column has non-zero values beyond the zeros we filled
        original_news_data = ticker_data[available_news_features]
        has_news_data = (original_news_data.notna().sum().sum() > 0)
    
    print(f"   • Has actual news data: {'✅ YES' if has_news_data else '❌ NO'}")
    
    if has_news_data and len(available_news_features) > 0:
        for col in available_news_features:
            if 'count' in col:
                news_count = ticker_data[col].notna().sum()
                if news_count > 0:
                    print(f"   • {col}: {news_count} non-NaN values")
            elif 'sentiment' in col:
                avg_sentiment = ticker_data[col].mean()
                print(f"   • {col}: average {avg_sentiment:.3f}")
    
    return {
        'stock': {'X': X_stock, 'features': available_stock_features, 'name': 'Stock-only'},
        'news': {'X': X_news, 'features': available_news_features, 'name': 'News-only'},
        'all': {'X': X_all, 'features': available_stock_features + available_news_features, 'name': 'All-features'},
        'y': y,
        'ticker_data': ticker_data,
        'has_news_data': has_news_data
    }

def run_simple_classification(ticker, data_dict):
    """Run simple classification for a ticker"""
    print(f"\n{'='*60}")
    print(f"CLASSIFICATION FOR: {ticker}")
    print(f"{'='*60}")
    
    results = []
    predictions = {}
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, class_weight='balanced', 
                                               random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42)
    }
    
    # Get data
    X_all = data_dict['all']['X']
    y = data_dict['y']
    
    if len(X_all) < 50:
        print(f"   ⚠️ Not enough data for {ticker} (only {len(X_all)} samples)")
        return [], {}
    
    # Time-based split (70/30)
    split_idx = int(len(X_all) * 0.7)
    X_train_all, X_test_all = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Also get stock-only split
    X_stock = data_dict['stock']['X']
    X_stock_train, X_stock_test = X_stock[:split_idx], X_stock[split_idx:]
    
    print(f"\n   📈 DATA SPLIT:")
    print(f"     • Training samples: {len(X_train_all)} ({len(X_train_all)/len(X_all):.1%})")
    print(f"     • Testing samples: {len(X_test_all)} ({len(X_test_all)/len(X_all):.1%})")
    print(f"     • Features in all-features set: {X_all.shape[1]}")
    print(f"     • Features in stock-only set: {X_stock.shape[1]}")
    
    for model_name, model in models.items():
        print(f"\n   🔍 {model_name}:")
        
        # Test both feature sets
        for feature_set, X_train, X_test in [
            ('Stock-only', X_stock_train, X_stock_test),
            ('All-features', X_train_all, X_test_all)
        ]:
            if X_train.shape[1] == 0:
                print(f"     {feature_set:<12} Skipped (no features)")
                continue
            
            try:
                # Scale the data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Clone and train model
                model_clone = model.__class__(**model.get_params())
                model_clone.fit(X_train_scaled, y_train)
                
                # Predict
                y_pred = model_clone.predict(X_test_scaled)
                y_pred_proba = model_clone.predict_proba(X_test_scaled)[:, 1] if hasattr(model_clone, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                auc_score = 0.5
                if y_pred_proba is not None and len(np.unique(y_test)) > 1:
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                
                cm = confusion_matrix(y_test, y_pred)
                
                print(f"     {feature_set:<12} Acc={accuracy:.3f}, F1={f1:.3f}, AUC={auc_score:.3f}")
                
                # Store results
                results.append({
                    'ticker': ticker,
                    'model': model_name,
                    'feature_set': feature_set,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc_score,
                    'test_samples': len(y_test),
                    'training_samples': len(y_train),
                    'num_features': X_test.shape[1],
                    'has_news_data': data_dict['has_news_data']
                })
                
                # Store predictions for the all-features model
                if feature_set == 'All-features':
                    predictions[model_name] = {
                        'y_true': y_test,
                        'y_pred': y_pred,
                        'y_proba': y_pred_proba,
                        'feature_set': feature_set,
                        'metrics': {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1': f1,
                            'auc': auc_score
                        },
                        'confusion_matrix': cm
                    }
                    
            except Exception as e:
                print(f"     {feature_set:<12} Failed: {str(e)[:50]}...")
    
    return results, predictions

def create_visualizations(results_df, all_predictions):
    """Create visualizations"""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Create output directory
    output_dir = "./classification_results_fixed"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"   📁 Results will be saved to: {results_dir}")
    
    # 1. Model Performance Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # 1.1 Accuracy by Model and Ticker
    ax = axes[0]
    if not results_df.empty:
        pivot_acc = results_df.pivot_table(
            index='model', 
            columns='ticker', 
            values='accuracy',
            aggfunc='mean'
        )
        pivot_acc.plot(kind='bar', ax=ax, alpha=0.7)
        ax.set_title('Accuracy by Model and Ticker', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.legend(title='Ticker')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
    
    # 1.2 F1-Score Comparison
    ax = axes[1]
    if not results_df.empty:
        pivot_f1 = results_df.pivot_table(
            index='model', 
            columns='ticker', 
            values='f1',
            aggfunc='mean'
        )
        pivot_f1.plot(kind='bar', ax=ax, alpha=0.7)
        ax.set_title('F1-Score by Model and Ticker', fontsize=12, fontweight='bold')
        ax.set_xlabel('Model')
        ax.set_ylabel('F1-Score')
        ax.legend(title='Ticker')
        ax.grid(True, alpha=0.3, axis='y')
    
    # 1.3 Feature Set Comparison
    ax = axes[2]
    if not results_df.empty:
        feature_set_stats = results_df.groupby('feature_set')['accuracy'].agg(['mean', 'std', 'count'])
        feature_set_stats['mean'].plot(kind='bar', yerr=feature_set_stats['std'], 
                                      capsize=4, ax=ax, alpha=0.7)
        ax.set_title('Accuracy by Feature Set', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Set')
        ax.set_ylabel('Accuracy')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add count labels
        for i, (idx, row) in enumerate(feature_set_stats.iterrows()):
            ax.text(i, row['mean'], f"n={int(row['count'])}", 
                   ha='center', va='bottom' if row['mean'] < 0.5 else 'top', fontsize=9)
    
    # 1.4 News Impact Analysis
    ax = axes[3]
    if not results_df.empty:
        # Compare Stock-only vs All-features
        comparison_data = []
        tickers = results_df['ticker'].unique()
        
        for ticker in tickers:
            ticker_results = results_df[results_df['ticker'] == ticker]
            
            for model in ticker_results['model'].unique():
                model_results = ticker_results[ticker_results['model'] == model]
                
                stock_only = model_results[model_results['feature_set'] == 'Stock-only']
                all_features = model_results[model_results['feature_set'] == 'All-features']
                
                if len(stock_only) > 0 and len(all_features) > 0:
                    improvement = (all_features['accuracy'].iloc[0] - stock_only['accuracy'].iloc[0]) * 100
                    comparison_data.append({
                        'ticker': ticker,
                        'model': model,
                        'improvement': improvement
                    })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            
            # Plot by ticker
            ticker_improvement = comparison_df.groupby('ticker')['improvement'].mean()
            
            colors = ['green' if imp > 0 else 'red' for imp in ticker_improvement.values]
            ticker_improvement.plot(kind='bar', color=colors, ax=ax, alpha=0.7)
            ax.set_title('News Data Impact by Ticker', fontsize=12, fontweight='bold')
            ax.set_xlabel('Ticker')
            ax.set_ylabel('Accuracy Improvement (%)')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for i, imp in enumerate(ticker_improvement.values):
                ax.text(i, imp, f'{imp:+.1f}%', 
                       ha='center', va='bottom' if imp > 0 else 'top', fontsize=9)
    
    plt.suptitle('Stock+News Classification Results', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    dashboard_file = os.path.join(results_dir, "performance_dashboard.png")
    plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   📊 Performance dashboard saved: {dashboard_file}")
    
    # 2. Confusion Matrices
    for ticker, ticker_predictions in all_predictions.items():
        if not ticker_predictions:
            continue
        
        num_models = len(ticker_predictions)
        if num_models == 0:
            continue
        
        fig, axes = plt.subplots(1, num_models, figsize=(5*num_models, 4))
        if num_models == 1:
            axes = [axes]
        
        for idx, (model_name, pred_data) in enumerate(ticker_predictions.items()):
            ax = axes[idx] if idx < len(axes) else None
            if ax is None:
                break
            
            cm = pred_data['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
            
            metrics_text = (f"Acc: {pred_data['metrics']['accuracy']:.3f}\n"
                          f"Prec: {pred_data['metrics']['precision']:.3f}\n"
                          f"Rec: {pred_data['metrics']['recall']:.3f}")
            
            ax.set_title(f'{ticker}\n{model_name}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            
            ax.text(1.2, 0.5, metrics_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle(f'Confusion Matrices - {ticker}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        cm_file = os.path.join(results_dir, f"confusion_matrices_{ticker}.png")
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📈 Confusion matrices for {ticker} saved: {cm_file}")
    
    # 3. Save results
    results_file = os.path.join(results_dir, "detailed_results.csv")
    results_df.to_csv(results_file, index=False)
    
    # Create summary
    summary_file = os.path.join(results_dir, "experiment_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STOCK+NEWS CLASSIFICATION EXPERIMENT SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        if not results_df.empty:
            f.write(f"Total models tested: {len(results_df)}\n")
            f.write(f"Average accuracy: {results_df['accuracy'].mean():.3f}\n")
            f.write(f"Average F1-score: {results_df['f1'].mean():.3f}\n\n")
            
            f.write("Best performing models by ticker:\n")
            f.write("-"*40 + "\n")
            
            for ticker in results_df['ticker'].unique():
                ticker_results = results_df[results_df['ticker'] == ticker]
                if len(ticker_results) > 0:
                    best = ticker_results.loc[ticker_results['accuracy'].idxmax()]
                    f.write(f"{ticker}: {best['model']} ({best['feature_set']})\n")
                    f.write(f"  Accuracy: {best['accuracy']:.3f}, F1: {best['f1']:.3f}\n\n")
            
            f.write("\nFeature set comparison:\n")
            f.write("-"*40 + "\n")
            
            for feature_set in results_df['feature_set'].unique():
                fs_results = results_df[results_df['feature_set'] == feature_set]
                if len(fs_results) > 0:
                    f.write(f"{feature_set}: {fs_results['accuracy'].mean():.3f} avg accuracy\n")
    
    print(f"   📄 Detailed results saved: {results_file}")
    print(f"   📋 Summary saved: {summary_file}")
    
    return results_dir, results_df

def print_final_report(results_df, results_dir):
    """Print final report"""
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE - FINAL REPORT")
    print("="*80)
    
    if results_df is None or results_df.empty:
        print("❌ No results to report")
        return
    
    print(f"\n📊 OVERALL RESULTS:")
    print(f"   • Models tested: {len(results_df)}")
    print(f"   • Average accuracy: {results_df['accuracy'].mean():.3f}")
    print(f"   • Average F1-score: {results_df['f1'].mean():.3f}")
    print(f"   • Best accuracy: {results_df['accuracy'].max():.3f}")
    print(f"   • Worst accuracy: {results_df['accuracy'].min():.3f}")
    
    print(f"\n🏆 BEST MODELS:")
    
    # Best overall
    best_overall = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"   • Overall: {best_overall['model']} on {best_overall['ticker']}")
    print(f"     Accuracy: {best_overall['accuracy']:.3f}")
    print(f"     Feature set: {best_overall['feature_set']}")
    
    # Best by ticker
    print(f"\n   By ticker:")
    tickers = results_df['ticker'].unique()
    for ticker in tickers:
        ticker_results = results_df[results_df['ticker'] == ticker]
        if len(ticker_results) > 0:
            best_ticker = ticker_results.loc[ticker_results['accuracy'].idxmax()]
            print(f"     {ticker}: {best_ticker['model']} ({best_ticker['accuracy']:.3f})")
    
    print(f"\n📈 FEATURE SET ANALYSIS:")
    
    for feature_set in results_df['feature_set'].unique():
        fs_results = results_df[results_df['feature_set'] == feature_set]
        if len(fs_results) > 0:
            avg_acc = fs_results['accuracy'].mean()
            count = len(fs_results)
            print(f"   • {feature_set}: {avg_acc:.3f} avg ({count} models)")
    
    # News impact analysis
    print(f"\n🔍 NEWS DATA IMPACT:")
    
    stock_results = results_df[results_df['feature_set'] == 'Stock-only']
    all_results = results_df[results_df['feature_set'] == 'All-features']
    
    if len(stock_results) > 0 and len(all_results) > 0:
        stock_avg = stock_results['accuracy'].mean()
        all_avg = all_results['accuracy'].mean()
        improvement = (all_avg - stock_avg) * 100
        
        print(f"   • Stock-only average: {stock_avg:.3f}")
        print(f"   • With news average: {all_avg:.3f}")
        print(f"   • Improvement: {improvement:+.2f}%")
        
        if improvement > 1:
            print(f"   ✅ News data provides meaningful improvement")
        elif improvement > 0:
            print(f"   ⚠️ News data provides marginal improvement")
        else:
            print(f"   ❌ News data does not improve predictions")
    
    print(f"\n📁 ALL FILES SAVED IN:")
    print(f"   {results_dir}")
    
    print(f"\n🎯 KEY QUESTIONS ANSWERED:")
    print(f"   1. Can we predict stock direction? {'Yes' if results_df['accuracy'].mean() > 0.5 else 'No'}")
    print(f"   2. Does news help? {'Yes' if 'improvement' in locals() and improvement > 0 else 'No'}")
    print(f"   3. Best model? {best_overall['model']}")
    print(f"   4. Best ticker for prediction? {best_overall['ticker']}")

def main():
    """Main function"""
    print("="*80)
    print("FIXED STOCK+NEWS CLASSIFICATION EXPERIMENT")
    print("="*80)
    
    # Step 1: Load and clean data
    df, stock_features, news_features = load_and_clean_data()
    
    # Step 2: Run classification for each ticker
    all_results = []
    all_predictions = {}
    
    tickers = ['AAPL', 'GOOGL', 'TSLA']
    
    for ticker in tickers:
        # Prepare data
        data_dict = prepare_ticker_data_simple(df, ticker, stock_features, news_features)
        
        # Run classification
        ticker_results, ticker_predictions = run_simple_classification(ticker, data_dict)
        
        all_results.extend(ticker_results)
        all_predictions[ticker] = ticker_predictions
    
    # Step 3: Create visualizations and save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_dir, results_df = create_visualizations(results_df, all_predictions)
        
        # Step 4: Print final report
        print_final_report(results_df, results_dir)
        
        return results_df, results_dir
    else:
        print("\n❌ No results generated. Please check your data.")
        return None, None

if __name__ == "__main__":
    results_df, results_dir = main()
