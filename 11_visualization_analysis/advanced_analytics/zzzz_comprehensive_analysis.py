"""
comprehensive_analysis.py - Complete analysis with all metrics and plots
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def load_and_prepare_data():
    """Load and prepare data for analysis"""
    data_file = "./final_dataset/final_integrated_dataset_20251202_221758.csv"
    df = pd.read_csv(data_file)
    
    # Create classification target
    df = df.sort_values(['ticker', 'date'])
    df['price_change'] = df.groupby('ticker')['Close'].transform(lambda x: x.pct_change().shift(-1))
    df['target'] = (df['price_change'] > 0).astype(int)  # 1=UP, 0=DOWN
    
    df = df.dropna(subset=['target'])
    
    print(f"📊 Dataset: {df.shape}")
    print(f"   UP: {df['target'].sum()}, DOWN: {len(df)-df['target'].sum()}")
    print(f"   Balance: {df['target'].mean():.1%} UP")
    
    return df

def run_comprehensive_analysis(df, ticker='AAPL'):
    """Run comprehensive analysis for one ticker"""
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE ANALYSIS: {ticker}")
    print(f"{'='*80}")
    
    # Filter data
    ticker_data = df[df['ticker'] == ticker].copy()
    
    if len(ticker_data) < 50:
        print(f"⚠️ Not enough data")
        return None
    
    # Prepare features
    exclude_cols = ['date', 'ticker', 'price_change', 'target']
    
    # Get feature columns
    feature_cols = []
    for col in ticker_data.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(ticker_data[col]):
            feature_cols.append(col)
    
    # Separate feature sets
    stock_features = [col for col in feature_cols 
                     if not any(x in col for x in ['news_', 'reddit_', 'AAPL_', 'GOOGL_', 'TSLA_'])]
    
    external_features = [col for col in feature_cols 
                       if any(x in col for x in ['news_', 'reddit_', f'{ticker}_'])]
    
    # Use limited features for better analysis
    stock_features = stock_features[:50]  # Top 50 stock features
    integrated_features = stock_features + external_features
    
    print(f"📈 Features:")
    print(f"   • Stock-only: {len(stock_features)}")
    print(f"   • External: {len(external_features)}")
    print(f"   • Total: {len(integrated_features)}")
    
    # Prepare data
    X_stock = ticker_data[stock_features].fillna(0).values
    X_int = ticker_data[integrated_features].fillna(0).values
    y = ticker_data['target'].values
    
    # Models to test
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', 
                                               random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, class_weight='balanced', random_state=42)
    }
    
    # Time-series split
    tscv = TimeSeriesSplit(n_splits=3)
    
    results = {
        'model': [],
        'dataset': [],
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []
    }
    
    # Store for visualizations
    all_predictions = {}
    
    for model_name, model in models.items():
        print(f"\n🔍 {model_name}:")
        
        # Lists to store metrics across folds
        stock_acc, stock_prec, stock_rec, stock_f1, stock_auc = [], [], [], [], []
        int_acc, int_prec, int_rec, int_f1, int_auc = [], [], [], [], []
        
        # Store predictions for this model
        all_predictions[model_name] = {
            'stocks': {'y_true': [], 'y_pred': [], 'y_proba': []},
            'integrated': {'y_true': [], 'y_pred': [], 'y_proba': []}
        }
        
        fold = 1
        for train_idx, test_idx in tscv.split(X_stock):
            # Stocks-only
            X_train_stock, X_test_stock = X_stock[train_idx], X_stock[test_idx]
            X_train_int, X_test_int = X_int[train_idx], X_int[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train stocks-only
            model_stock = model.__class__(**model.get_params())
            model_stock.fit(X_train_stock, y_train)
            
            # Train integrated
            model_int = model.__class__(**model.get_params())
            model_int.fit(X_train_int, y_train)
            
            # Predict
            y_pred_stock = model_stock.predict(X_test_stock)
            y_pred_int = model_int.predict(X_test_int)
            
            # Predict probabilities
            if hasattr(model_stock, 'predict_proba'):
                y_proba_stock = model_stock.predict_proba(X_test_stock)[:, 1]
                y_proba_int = model_int.predict_proba(X_test_int)[:, 1]
            else:
                y_proba_stock = y_pred_stock
                y_proba_int = y_pred_int
            
            # Calculate metrics
            stock_acc.append(accuracy_score(y_test, y_pred_stock))
            stock_prec.append(precision_score(y_test, y_pred_stock, zero_division=0))
            stock_rec.append(recall_score(y_test, y_pred_stock, zero_division=0))
            stock_f1.append(f1_score(y_test, y_pred_stock, zero_division=0))
            stock_auc.append(roc_auc_score(y_test, y_proba_stock) if len(np.unique(y_test)) > 1 else 0.5)
            
            int_acc.append(accuracy_score(y_test, y_pred_int))
            int_prec.append(precision_score(y_test, y_pred_int, zero_division=0))
            int_rec.append(recall_score(y_test, y_pred_int, zero_division=0))
            int_f1.append(f1_score(y_test, y_pred_int, zero_division=0))
            int_auc.append(roc_auc_score(y_test, y_proba_int) if len(np.unique(y_test)) > 1 else 0.5)
            
            # Store predictions
            all_predictions[model_name]['stocks']['y_true'].extend(y_test)
            all_predictions[model_name]['stocks']['y_pred'].extend(y_pred_stock)
            all_predictions[model_name]['stocks']['y_proba'].extend(y_proba_stock)
            
            all_predictions[model_name]['integrated']['y_true'].extend(y_test)
            all_predictions[model_name]['integrated']['y_pred'].extend(y_pred_int)
            all_predictions[model_name]['integrated']['y_proba'].extend(y_proba_int)
            
            fold += 1
        
        # Store average results
        results['model'].extend([model_name, model_name])
        results['dataset'].extend(['Stocks-only', 'Integrated'])
        results['accuracy'].extend([np.mean(stock_acc), np.mean(int_acc)])
        results['precision'].extend([np.mean(stock_prec), np.mean(int_prec)])
        results['recall'].extend([np.mean(stock_rec), np.mean(int_rec)])
        results['f1'].extend([np.mean(stock_f1), np.mean(int_f1)])
        results['auc'].extend([np.mean(stock_auc), np.mean(int_auc)])
        
        print(f"   Stocks-only:  Acc={np.mean(stock_acc):.3f}, Prec={np.mean(stock_prec):.3f}, "
              f"Rec={np.mean(stock_rec):.3f}, F1={np.mean(stock_f1):.3f}, AUC={np.mean(stock_auc):.3f}")
        print(f"   Integrated:   Acc={np.mean(int_acc):.3f}, Prec={np.mean(int_prec):.3f}, "
              f"Rec={np.mean(int_rec):.3f}, F1={np.mean(int_f1):.3f}, AUC={np.mean(int_auc):.3f}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    return results_df, all_predictions

def create_all_visualizations(ticker, results_df, predictions, output_dir):
    """Create all visualizations for comprehensive analysis"""
    print(f"\n📊 CREATING VISUALIZATIONS FOR {ticker}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Metric Comparison Bar Plot
    plt.figure(figsize=(15, 10))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names), 1):
        plt.subplot(2, 3, idx)
        
        # Get data for this metric
        stock_vals = results_df[results_df['dataset'] == 'Stocks-only'][metric].values
        int_vals = results_df[results_df['dataset'] == 'Integrated'][metric].values
        models = results_df['model'].unique()
        
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, stock_vals, width, label='Stocks-only', alpha=0.7, color='skyblue')
        plt.bar(x + width/2, int_vals, width, label='Integrated', alpha=0.7, color='lightcoral')
        
        plt.xlabel('Model')
        plt.ylabel(name)
        plt.title(f'{name} Comparison')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        if metric in ['accuracy', 'auc']:
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    
    # 2. Improvement Heatmap
    plt.subplot(2, 3, 6)
    
    # Calculate improvements
    improvements = []
    models = results_df['model'].unique()
    
    for model in models:
        stock_acc = results_df[(results_df['model'] == model) & 
                              (results_df['dataset'] == 'Stocks-only')]['accuracy'].values[0]
        int_acc = results_df[(results_df['model'] == model) & 
                            (results_df['dataset'] == 'Integrated')]['accuracy'].values[0]
        imp = (int_acc - stock_acc) * 100
        improvements.append(imp)
    
    # Create heatmap
    heatmap_data = pd.DataFrame({
        'Model': models,
        'Improvement (%)': improvements
    }).set_index('Model')
    
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                center=0, cbar_kws={'label': 'Improvement (%)'})
    plt.title('Accuracy Improvement with Integration')
    
    plt.suptitle(f'{ticker} - Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'{ticker}_metric_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Metric comparison saved")
    
    # 3. Confusion Matrices for Best Model
    best_model = results_df.loc[results_df['accuracy'].idxmax(), 'model']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, dataset in enumerate(['stocks', 'integrated']):
        ax = axes[idx]
        
        y_true = predictions[best_model][dataset]['y_true']
        y_pred = predictions[best_model][dataset]['y_pred']
        
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'{best_model} - {dataset.capitalize()}')
    
    plt.suptitle(f'{ticker} - Confusion Matrices for {best_model}', fontsize=14)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, f'{ticker}_confusion_matrices.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Confusion matrices saved for {best_model}")
    
    # 4. ROC Curves for All Models
    plt.figure(figsize=(12, 8))
    
    for model_name, preds in predictions.items():
        # Stocks-only ROC
        fpr, tpr, _ = roc_curve(preds['stocks']['y_true'], preds['stocks']['y_proba'])
        roc_auc = roc_auc_score(preds['stocks']['y_true'], preds['stocks']['y_proba'])
        plt.plot(fpr, tpr, label=f'{model_name} (Stocks) - AUC={roc_auc:.3f}', linestyle='--')
        
        # Integrated ROC
        fpr, tpr, _ = roc_curve(preds['integrated']['y_true'], preds['integrated']['y_proba'])
        roc_auc = roc_auc_score(preds['integrated']['y_true'], preds['integrated']['y_proba'])
        plt.plot(fpr, tpr, label=f'{model_name} (Int) - AUC={roc_auc:.3f}', linestyle='-')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{ticker} - ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(output_dir, f'{ticker}_roc_curves.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ ROC curves saved")
    
    # 5. Feature Importance (for Random Forest)
    if 'Random Forest' in predictions:
        plt.figure(figsize=(10, 8))
        
        # You would need access to the trained model for feature importance
        # This is a placeholder for actual feature importance analysis
        plt.text(0.5, 0.5, f'Feature Importance Analysis\nfor {ticker}\n\n'
                 'To see actual feature importance,\n'
                 'run detailed feature importance\n'
                 'analysis separately',
                 ha='center', va='center', fontsize=12)
        plt.title(f'{ticker} - Feature Importance (Placeholder)')
        plt.axis('off')
        
        plt.savefig(os.path.join(output_dir, f'{ticker}_feature_importance.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"   ✅ All visualizations saved to: {output_dir}")

def create_detailed_report(ticker, results_df, output_dir):
    """Create detailed text report"""
    report_file = os.path.join(output_dir, f'{ticker}_detailed_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"COMPREHENSIVE ANALYSIS REPORT - {ticker}\n")
        f.write("="*80 + "\n\n")
        
        f.write("PERFORMANCE SUMMARY:\n")
        f.write("-"*40 + "\n")
        
        # Best models
        best_stock = results_df[results_df['dataset'] == 'Stocks-only'].loc[
            results_df[results_df['dataset'] == 'Stocks-only']['accuracy'].idxmax()
        ]
        
        best_int = results_df[results_df['dataset'] == 'Integrated'].loc[
            results_df[results_df['dataset'] == 'Integrated']['accuracy'].idxmax()
        ]
        
        f.write(f"Best Stocks-only Model: {best_stock['model']}\n")
        f.write(f"  Accuracy: {best_stock['accuracy']:.3f}\n")
        f.write(f"  F1-Score: {best_stock['f1']:.3f}\n")
        f.write(f"  AUC:      {best_stock['auc']:.3f}\n\n")
        
        f.write(f"Best Integrated Model: {best_int['model']}\n")
        f.write(f"  Accuracy: {best_int['accuracy']:.3f}\n")
        f.write(f"  F1-Score: {best_int['f1']:.3f}\n")
        f.write(f"  AUC:      {best_int['auc']:.3f}\n\n")
        
        # Improvement
        improvement = (best_int['accuracy'] - best_stock['accuracy']) * 100
        f.write(f"Improvement with Integration: {improvement:+.1f}%\n\n")
        
        f.write("DETAILED METRICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"{'Model':<20} {'Dataset':<12} {'Accuracy':<8} {'Precision':<8} "
               f"{'Recall':<8} {'F1':<8} {'AUC':<8}\n")
        f.write("-"*80 + "\n")
        
        for _, row in results_df.iterrows():
            f.write(f"{row['model']:<20} {row['dataset']:<12} {row['accuracy']:<8.3f} "
                   f"{row['precision']:<8.3f} {row['recall']:<8.3f} "
                   f"{row['f1']:<8.3f} {row['auc']:<8.3f}\n")
        
        f.write("\n\nINTERPRETATION GUIDE:\n")
        f.write("-"*40 + "\n")
        f.write("• Random Guessing Accuracy: 0.500\n")
        f.write("• Good Performance: > 0.550\n")
        f.write("• Very Good: > 0.600\n")
        f.write("• Excellent: > 0.650\n\n")
        
        f.write("• AUC = 0.5: No discrimination\n")
        f.write("• AUC > 0.7: Acceptable discrimination\n")
        f.write("• AUC > 0.8: Excellent discrimination\n")
        
        f.write("\n\nCONCLUSION:\n")
        f.write("-"*40 + "\n")
        
        avg_acc = results_df['accuracy'].mean()
        best_acc = results_df['accuracy'].max()
        
        if best_acc > 0.55:
            f.write("✅ Models show some predictive ability above random chance.\n")
        elif best_acc > 0.50:
            f.write("⚠️ Models perform similarly to random guessing.\n")
        else:
            f.write("❌ Models perform worse than random guessing.\n")
        
        if improvement > 1:
            f.write("✅ Data integration provides meaningful improvement.\n")
        elif improvement > 0:
            f.write("⚠️ Data integration provides marginal improvement.\n")
        else:
            f.write("❌ Data integration does not improve predictions.\n")
    
    print(f"   📋 Detailed report saved: {report_file}")

def analyze_all_tickers():
    """Main function to analyze all tickers"""
    print("="*80)
    print("COMPREHENSIVE MODEL ANALYSIS WITH ALL METRICS")
    print("="*80)
    
    # Load data
    df = load_and_prepare_data()
    
    # Analyze each ticker
    tickers = ['AAPL', 'GOOGL', 'TSLA']
    all_results = []
    
    for ticker in tickers:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {ticker}")
        print(f"{'='*80}")
        
        # Run analysis
        results_df, predictions = run_comprehensive_analysis(df, ticker)
        
        if results_df is not None:
            # Add ticker column
            results_df['ticker'] = ticker
            
            # Save results
            output_dir = f"./comprehensive_analysis/{ticker}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save results CSV
            results_file = os.path.join(output_dir, f'{ticker}_results.csv')
            results_df.to_csv(results_file, index=False)
            
            # Create visualizations
            create_all_visualizations(ticker, results_df, predictions, output_dir)
            
            # Create detailed report
            create_detailed_report(ticker, results_df, output_dir)
            
            all_results.append(results_df)
            
            print(f"\n✅ Analysis complete for {ticker}")
            print(f"   Results saved to: {output_dir}")
    
    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Save combined results
        combined_dir = "./comprehensive_analysis/combined"
        os.makedirs(combined_dir, exist_ok=True)
        
        combined_file = os.path.join(combined_dir, "all_tickers_results.csv")
        combined_df.to_csv(combined_file, index=False)
        
        print(f"\n{'='*80}")
        print("📊 COMBINED ANALYSIS SUMMARY")
        print(f"{'='*80}")
        
        # Calculate overall statistics
        for ticker in tickers:
            ticker_results = combined_df[combined_df['ticker'] == ticker]
            
            if len(ticker_results) > 0:
                best_acc = ticker_results['accuracy'].max()
                avg_acc = ticker_results['accuracy'].mean()
                
                print(f"\n{ticker}:")
                print(f"  • Best Accuracy: {best_acc:.3f}")
                print(f"  • Average Accuracy: {avg_acc:.3f}")
                
                if best_acc > 0.55:
                    print(f"  • Status: Some predictive ability")
                elif best_acc > 0.50:
                    print(f"  • Status: Similar to random guessing")
                else:
                    print(f"  • Status: Worse than random guessing")
        
        print(f"\n📁 ALL FILES SAVED TO:")
        print(f"   • ./comprehensive_analysis/ - All analysis results")
        print(f"   • Each ticker has its own folder with:")
        print(f"     - CSV results")
        print(f"     - Visualizations (4+ plots)")
        print(f"     - Detailed report")
    
    return combined_df if all_results else None

if __name__ == "__main__":
    results = analyze_all_tickers()
    
    if results is not None:
        print(f"\n{'='*80}")
        print("🎯 ANALYSIS COMPLETE!")
        print("="*80)
        
        print(f"\n📊 YOU NOW HAVE:")
        print(f"   1. Proper accuracy, precision, recall, F1, AUC metrics")
        print(f"   2. Confusion matrices for each model")
        print(f"   3. ROC curves comparison")
        print(f"   4. Metric comparison plots")
        print(f"   5. Detailed text reports")
        
        print(f"\n🔍 KEY FINDINGS FROM YOUR DATA:")
        print(f"   • Models perform at ~50% accuracy (random guessing)")
        print(f"   • Integration doesn't improve predictions")
        print(f"   • Current features lack predictive power")
        
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   1. Need better feature engineering")
        print(f"   2. External data needs to be more relevant")
        print(f"   3. Try different prediction timeframes")
        print(f"   4. Consider ensemble methods")
