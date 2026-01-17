"""
train_classification_models.py - Train classification models with proper metrics
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def prepare_classification_data():
    """Prepare data for classification (UP/DOWN prediction)"""
    print("="*80)
    print("PREPARING DATA FOR CLASSIFICATION")
    print("="*80)
    
    # Load your integrated dataset
    data_file = "./final_dataset/final_integrated_dataset_20251202_221758.csv"
    df = pd.read_csv(data_file)
    
    print(f"📂 Dataset: {df.shape}")
    
    # Create classification target: 1 if price goes UP, 0 if DOWN
    df = df.sort_values(['ticker', 'date'])
    df['price_change'] = df.groupby('ticker')['Close'].transform(lambda x: x.pct_change().shift(-1))
    df['target_class'] = (df['price_change'] > 0).astype(int)  # 1=UP, 0=DOWN
    
    # Remove rows without target
    df = df.dropna(subset=['target_class'])
    
    print(f"   Classification target created")
    print(f"   UP (1): {df['target_class'].sum()} records")
    print(f"   DOWN (0): {len(df) - df['target_class'].sum()} records")
    print(f"   Balance: {df['target_class'].mean():.2%} UP, {1-df['target_class'].mean():.2%} DOWN")
    
    return df

def train_classification_models(df):
    """Train and evaluate classification models"""
    print("\n" + "="*80)
    print("TRAINING CLASSIFICATION MODELS")
    print("="*80)
    
    # Define classification models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    tickers = ['AAPL', 'GOOGL', 'TSLA']
    all_results = []
    detailed_reports = []
    
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"CLASSIFICATION FOR: {ticker}")
        print(f"{'='*60}")
        
        # Filter data for this ticker
        ticker_data = df[df['ticker'] == ticker].copy()
        
        if len(ticker_data) < 50:
            print(f"   ⚠️ Not enough data")
            continue
        
        print(f"   Records: {len(ticker_data)}")
        print(f"   UP/DOWN ratio: {ticker_data['target_class'].mean():.2%} UP")
        
        # Prepare features
        exclude_cols = ['date', 'ticker', 'target', 'price_change', 'target_class']
        
        # Get numeric features
        feature_cols = []
        for col in ticker_data.columns:
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(ticker_data[col]):
                feature_cols.append(col)
        
        # Separate stock-only and integrated features
        stock_features = [col for col in feature_cols 
                         if not any(x in col for x in ['news_', 'reddit_', 'AAPL_', 'GOOGL_', 'TSLA_'])]
        
        external_features = [col for col in feature_cols 
                           if any(x in col for x in ['news_', 'reddit_', f'{ticker}_'])]
        
        # Create feature sets
        stocks_only_features = stock_features[:100]  # Limit to top 100 stock features
        integrated_features = stocks_only_features + external_features
        
        print(f"   Features:")
        print(f"     • Stock-only: {len(stocks_only_features)}")
        print(f"     • External: {len(external_features)}")
        print(f"     • Integrated: {len(integrated_features)}")
        
        if external_features:
            print(f"     • External features available!")
        
        # Prepare data
        X_stock = ticker_data[stocks_only_features].fillna(0)
        X_integrated = ticker_data[integrated_features].fillna(0)
        y = ticker_data['target_class']
        
        # Time-series split
        tscv = TimeSeriesSplit(n_splits=3)
        
        ticker_results = []
        
        for model_name, model in models.items():
            print(f"\n   📊 {model_name}:")
            
            # Cross-validation for stocks-only
            stock_scores = {
                'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []
            }
            
            # Cross-validation for integrated
            int_scores = {
                'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []
            }
            
            # Store predictions for detailed analysis
            all_y_true = []
            all_y_pred_stock = []
            all_y_pred_int = []
            all_y_proba_stock = []
            all_y_proba_int = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X_stock), 1):
                # Stocks-only
                X_train_stock, X_test_stock = X_stock.iloc[train_idx], X_stock.iloc[test_idx]
                X_train_int, X_test_int = X_integrated.iloc[train_idx], X_integrated.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train stocks-only model
                model_stock = model.__class__(**model.get_params())
                model_stock.fit(X_train_stock, y_train)
                
                # Train integrated model
                model_int = model.__class__(**model.get_params())
                model_int.fit(X_train_int, y_train)
                
                # Predict
                y_pred_stock = model_stock.predict(X_test_stock)
                y_pred_int = model_int.predict(X_test_int)
                
                # Predict probabilities for AUC
                if hasattr(model_stock, 'predict_proba'):
                    y_proba_stock = model_stock.predict_proba(X_test_stock)[:, 1]
                    y_proba_int = model_int.predict_proba(X_test_int)[:, 1]
                else:
                    y_proba_stock = y_pred_stock
                    y_proba_int = y_pred_int
                
                # Calculate metrics
                stock_scores['accuracy'].append(accuracy_score(y_test, y_pred_stock))
                stock_scores['precision'].append(precision_score(y_test, y_pred_stock, zero_division=0))
                stock_scores['recall'].append(recall_score(y_test, y_pred_stock, zero_division=0))
                stock_scores['f1'].append(f1_score(y_test, y_pred_stock, zero_division=0))
                stock_scores['roc_auc'].append(roc_auc_score(y_test, y_proba_stock) if len(np.unique(y_test)) > 1 else 0.5)
                
                int_scores['accuracy'].append(accuracy_score(y_test, y_pred_int))
                int_scores['precision'].append(precision_score(y_test, y_pred_int, zero_division=0))
                int_scores['recall'].append(recall_score(y_test, y_pred_int, zero_division=0))
                int_scores['f1'].append(f1_score(y_test, y_pred_int, zero_division=0))
                int_scores['roc_auc'].append(roc_auc_score(y_test, y_proba_int) if len(np.unique(y_test)) > 1 else 0.5)
                
                # Store for overall metrics
                all_y_true.extend(y_test)
                all_y_pred_stock.extend(y_pred_stock)
                all_y_pred_int.extend(y_pred_int)
            
            # Calculate average scores
            stock_accuracy = np.mean(stock_scores['accuracy'])
            stock_precision = np.mean(stock_scores['precision'])
            stock_recall = np.mean(stock_scores['recall'])
            stock_f1 = np.mean(stock_scores['f1'])
            stock_auc = np.mean(stock_scores['roc_auc'])
            
            int_accuracy = np.mean(int_scores['accuracy'])
            int_precision = np.mean(int_scores['precision'])
            int_recall = np.mean(int_scores['recall'])
            int_f1 = np.mean(int_scores['f1'])
            int_auc = np.mean(int_scores['roc_auc'])
            
            # Calculate improvements
            acc_imp = (int_accuracy - stock_accuracy) * 100
            f1_imp = (int_f1 - stock_f1) * 100
            
            print(f"     Stocks-only:  Acc={stock_accuracy:.3f}, Prec={stock_precision:.3f}, Rec={stock_recall:.3f}, F1={stock_f1:.3f}, AUC={stock_auc:.3f}")
            print(f"     Integrated:   Acc={int_accuracy:.3f}, Prec={int_precision:.3f}, Rec={int_recall:.3f}, F1={int_f1:.3f}, AUC={int_auc:.3f}")
            print(f"     Improvement:  Acc={acc_imp:+.1f}%, F1={f1_imp:+.1f}%")
            
            # Store results
            ticker_results.append({
                'ticker': ticker,
                'model': model_name,
                'stocks_accuracy': stock_accuracy,
                'integrated_accuracy': int_accuracy,
                'accuracy_improvement': acc_imp,
                'stocks_precision': stock_precision,
                'integrated_precision': int_precision,
                'stocks_recall': stock_recall,
                'integrated_recall': int_recall,
                'stocks_f1': stock_f1,
                'integrated_f1': int_f1,
                'f1_improvement': f1_imp,
                'stocks_auc': stock_auc,
                'integrated_auc': int_auc
            })
            
            # Generate detailed report for best model
            if model_name == 'Random Forest':  # Just for RF to avoid too many files
                report = classification_report(all_y_true, all_y_pred_stock, 
                                             target_names=['DOWN', 'UP'], output_dict=True)
                detailed_reports.append({
                    'ticker': ticker,
                    'model': f'{model_name} (Stocks-only)',
                    'report': report
                })
                
                report = classification_report(all_y_true, all_y_pred_int,
                                             target_names=['DOWN', 'UP'], output_dict=True)
                detailed_reports.append({
                    'ticker': ticker,
                    'model': f'{model_name} (Integrated)',
                    'report': report
                })
        
        all_results.extend(ticker_results)
    
    return pd.DataFrame(all_results), detailed_reports

def create_classification_visualizations(results_df, output_dir):
    """Create visualizations for classification results"""
    print(f"\n📊 CREATING CLASSIFICATION VISUALIZATIONS...")
    
    # 1. Accuracy Comparison
    plt.figure(figsize=(15, 10))
    
    models = results_df['model'].unique()
    tickers = results_df['ticker'].unique()
    
    # Subplot 1: Accuracy Comparison
    plt.subplot(2, 2, 1)
    
    x = np.arange(len(tickers))
    width = 0.2
    
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        acc_values = model_data['integrated_accuracy'].values
        
        if len(acc_values) == len(tickers):
            plt.bar(x + i*width, acc_values, width, label=model, alpha=0.7)
    
    plt.xlabel('Ticker')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison (Integrated Data)')
    plt.xticks(x + width, tickers)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    
    # Subplot 2: F1 Score Comparison
    plt.subplot(2, 2, 2)
    
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        f1_values = model_data['integrated_f1'].values
        
        if len(f1_values) == len(tickers):
            plt.bar(x + i*width, f1_values, width, label=model, alpha=0.7)
    
    plt.xlabel('Ticker')
    plt.ylabel('F1 Score')
    plt.title('Model F1 Score Comparison (Integrated Data)')
    plt.xticks(x + width, tickers)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Improvement by Model
    plt.subplot(2, 2, 3)
    
    # Calculate average improvement per model
    model_improvements = []
    for model in models:
        model_data = results_df[results_df['model'] == model]
        avg_acc_imp = model_data['accuracy_improvement'].mean()
        model_improvements.append(avg_acc_imp)
    
    colors = ['green' if imp > 0 else 'red' for imp in model_improvements]
    plt.bar(models, model_improvements, color=colors, alpha=0.7)
    
    plt.xlabel('Model')
    plt.ylabel('Average Accuracy Improvement (%)')
    plt.title('Average Accuracy Improvement by Model')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (model, imp) in enumerate(zip(models, model_improvements)):
        plt.text(i, imp, f'{imp:+.1f}%', ha='center', va='bottom' if imp > 0 else 'top')
    
    # Subplot 4: AUC Comparison
    plt.subplot(2, 2, 4)
    
    for i, model in enumerate(models):
        model_data = results_df[results_df['model'] == model]
        auc_values = model_data['integrated_auc'].values
        
        if len(auc_values) == len(tickers):
            plt.bar(x + i*width, auc_values, width, label=model, alpha=0.7)
    
    plt.xlabel('Ticker')
    plt.ylabel('ROC AUC')
    plt.title('Model AUC Comparison (Integrated Data)')
    plt.xticks(x + width, tickers)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    
    plt.suptitle('Classification Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, "classification_performance.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   📈 Visualization saved to: {plot_file}")

def print_classification_summary(results_df):
    """Print classification results summary"""
    print(f"\n{'='*80}")
    print("CLASSIFICATION RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Best model for each ticker
    tickers = results_df['ticker'].unique()
    
    for ticker in tickers:
        ticker_results = results_df[results_df['ticker'] == ticker]
        
        if len(ticker_results) > 0:
            # Best integrated model (highest accuracy)
            best_int = ticker_results.loc[ticker_results['integrated_accuracy'].idxmax()]
            
            # Best stocks-only model (highest accuracy)
            best_stock = ticker_results.loc[ticker_results['stocks_accuracy'].idxmax()]
            
            print(f"\n   {ticker}:")
            print(f"     • Best Integrated Model: {best_int['model']}")
            print(f"       Accuracy: {best_int['integrated_accuracy']:.3f}, F1: {best_int['integrated_f1']:.3f}, AUC: {best_int['integrated_auc']:.3f}")
            
            print(f"     • Best Stocks-only Model: {best_stock['model']}")
            print(f"       Accuracy: {best_stock['stocks_accuracy']:.3f}, F1: {best_stock['stocks_f1']:.3f}, AUC: {best_stock['stocks_auc']:.3f}")
            
            improvement = best_int['accuracy_improvement']
            if improvement > 1:  # More than 1% improvement
                print(f"     ✅ Integration improved accuracy by {improvement:+.1f}%")
            elif improvement > 0:
                print(f"     ⚠️ Integration slightly improved accuracy by {improvement:+.1f}%")
            else:
                print(f"     ❌ Integration did not improve accuracy")
    
    # Overall statistics
    print(f"\n{'='*80}")
    print("OVERALL STATISTICS")
    
    avg_stock_acc = results_df['stocks_accuracy'].mean()
    avg_int_acc = results_df['integrated_accuracy'].mean()
    avg_improvement = results_df['accuracy_improvement'].mean()
    
    print(f"\n   📊 Average Performance:")
    print(f"     • Stocks-only Accuracy: {avg_stock_acc:.3f}")
    print(f"     • Integrated Accuracy:  {avg_int_acc:.3f}")
    print(f"     • Average Improvement:  {avg_improvement:+.1f}%")
    
    # Count models that improved
    improved_models = results_df[results_df['accuracy_improvement'] > 0]
    print(f"     • Models improved: {len(improved_models)}/{len(results_df)}")
    
    # Best overall model
    best_overall = results_df.loc[results_df['integrated_accuracy'].idxmax()]
    print(f"\n   🏆 Best Overall Model:")
    print(f"     • {best_overall['model']} on {best_overall['ticker']}")
    print(f"     • Accuracy: {best_overall['integrated_accuracy']:.3f}")
    print(f"     • F1 Score: {best_overall['integrated_f1']:.3f}")
    print(f"     • AUC: {best_overall['integrated_auc']:.3f}")

def save_detailed_reports(detailed_reports, output_dir):
    """Save detailed classification reports"""
    print(f"\n📋 SAVING DETAILED REPORTS...")
    
    reports_dir = os.path.join(output_dir, "detailed_reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    for report in detailed_reports:
        ticker = report['ticker']
        model_name = report['model'].replace(' ', '_').replace('(', '').replace(')', '')
        
        report_file = os.path.join(reports_dir, f"{ticker}_{model_name}_report.txt")
        
        with open(report_file, 'w') as f:
            f.write(f"Classification Report: {ticker} - {report['model']}\n")
            f.write("="*60 + "\n\n")
            
            # Convert report dict to readable format
            report_dict = report['report']
            
            # Overall metrics
            f.write("OVERALL METRICS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Accuracy: {report_dict['accuracy']:.3f}\n")
            
            if 'macro avg' in report_dict:
                f.write(f"Macro Avg F1: {report_dict['macro avg']['f1-score']:.3f}\n")
                f.write(f"Weighted Avg F1: {report_dict['weighted avg']['f1-score']:.3f}\n")
            
            f.write("\nCLASS-WISE METRICS:\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}\n")
            f.write("-"*50 + "\n")
            
            for class_name, metrics in report_dict.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    f.write(f"{class_name:<10} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                           f"{metrics['f1-score']:<10.3f} {metrics['support']:<10}\n")
    
    print(f"   Detailed reports saved to: {reports_dir}")

def create_final_classification_report(results_df):
    """Create final comprehensive report"""
    print(f"\n{'='*80}")
    print("FINAL CLASSIFICATION REPORT")
    print(f"{'='*80}")
    
    report_dir = "./classification_report"
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = os.path.join(report_dir, f"classification_report_{timestamp}.txt")
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STOCK PRICE DIRECTION PREDICTION - CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("PROJECT OVERVIEW:\n")
        f.write("-"*40 + "\n")
        f.write("Binary classification task: Predict whether stock price will go UP or DOWN\n")
        f.write("Using integrated dataset with stock, news, and reddit data\n\n")
        
        f.write("MODELS TESTED:\n")
        f.write("-"*40 + "\n")
        models = results_df['model'].unique()
        for model in models:
            f.write(f"• {model}\n")
        f.write("\n")
        
        f.write("METRICS USED:\n")
        f.write("-"*40 + "\n")
        f.write("• Accuracy: Overall correctness\n")
        f.write("• Precision: Correct UP predictions / Total UP predictions\n")
        f.write("• Recall: Correct UP predictions / Actual UP instances\n")
        f.write("• F1-Score: Harmonic mean of precision and recall\n")
        f.write("• ROC AUC: Ability to distinguish between classes\n\n")
        
        f.write("RESULTS SUMMARY:\n")
        f.write("-"*40 + "\n")
        
        # Calculate averages
        avg_stock_acc = results_df['stocks_accuracy'].mean()
        avg_int_acc = results_df['integrated_accuracy'].mean()
        avg_improvement = results_df['accuracy_improvement'].mean()
        
        f.write(f"Average Stocks-only Accuracy: {avg_stock_acc:.3f}\n")
        f.write(f"Average Integrated Accuracy:  {avg_int_acc:.3f}\n")
        f.write(f"Average Improvement:          {avg_improvement:+.1f}%\n\n")
        
        f.write("PERFORMANCE INTERPRETATION:\n")
        f.write("-"*40 + "\n")
        f.write("• Random guessing accuracy: 0.50 (50%)\n")
        f.write("• Good performance: > 0.55 (55%)\n")
        f.write("• Very good performance: > 0.60 (60%)\n")
        f.write("• Excellent performance: > 0.65 (65%)\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-"*40 + "\n")
        
        if avg_int_acc > 0.55:
            f.write("✅ Models show predictive ability above random chance\n")
        elif avg_int_acc > 0.50:
            f.write("⚠️ Models perform slightly better than random chance\n")
        else:
            f.write("❌ Models do not outperform random guessing\n")
        
        if avg_improvement > 1:
            f.write("✅ Data integration provides meaningful improvement\n")
        elif avg_improvement > 0:
            f.write("⚠️ Data integration provides marginal improvement\n")
        else:
            f.write("❌ Data integration does not improve predictions\n")
        
        f.write("\nRECOMMENDATIONS FOR IMPROVEMENT:\n")
        f.write("-"*40 + "\n")
        f.write("1. Get more comprehensive news data\n")
        f.write("2. Add ticker-specific reddit sentiment analysis\n")
        f.write("3. Include technical indicators as features\n")
        f.write("4. Try ensemble methods or deep learning\n")
        f.write("5. Experiment with different prediction horizons\n")
    
    print(f"📋 Report saved to: {report_file}")
    return report_file

def main():
    """Main function for classification analysis"""
    # Prepare data
    df = prepare_classification_data()
    
    # Train models
    results_df, detailed_reports = train_classification_models(df)
    
    if results_df is not None and len(results_df) > 0:
        # Save results
        output_dir = "./classification_results"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f"classification_results_{timestamp}.csv")
        results_df.to_csv(results_file, index=False)
        
        print(f"\n📄 Results saved to: {results_file}")
        
        # Create visualizations
        create_classification_visualizations(results_df, output_dir)
        
        # Save detailed reports
        save_detailed_reports(detailed_reports, output_dir)
        
        # Print summary
        print_classification_summary(results_df)
        
        # Create final report
        create_final_classification_report(results_df)
        
        print(f"\n{'='*80}")
        print("🎯 CLASSIFICATION ANALYSIS COMPLETE!")
        print("="*80)
        
        print(f"\n📁 YOUR OUTPUTS:")
        print(f"   • {results_file} - All classification results")
        print(f"   • ./classification_results/ - Visualizations and reports")
        print(f"   • ./classification_report/ - Final comprehensive report")
        
        print(f"\n📊 KEY METRICS GENERATED:")
        print(f"   • Accuracy, Precision, Recall, F1-Score")
        print(f"   • ROC AUC Scores")
        print(f"   • Confusion matrices (in detailed reports)")
        print(f"   • Comparison: Stocks-only vs Integrated")
        
        return results_df
    
    return None

if __name__ == "__main__":
    results = main()
