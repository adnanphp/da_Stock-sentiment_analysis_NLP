# final_perfect_metrics_with_plots.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class PerfectMetricsVisualizer:
    """
    Final perfect version with all bugs fixed.
    """
    
    def __init__(self):
        self.results = {}
        self.plots_dir = "perfect_metrics_plots"
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set style for better plots
        plt.style.use('default')
        sns.set_palette("husl")
    
    def prepare_features(self, model_path, test_data_path, target_column):
        """Prepare features for prediction."""
        try:
            # Load model
            model_data = joblib.load(model_path)
            model = model_data['model']
            scaler = model_data.get('scaler')
            feature_selector = model_data.get('feature_selector')
            
            # Load test data
            df = pd.read_csv(test_data_path, low_memory=False)
            
            # Prepare features
            X = df.drop(columns=[target_column], errors='ignore')
            y_true_reg = df[target_column].values
            
            # Remove non-numeric columns
            X = X.select_dtypes(include=[np.number])
            
            # Handle missing values
            X = X.fillna(X.median())
            y_true_reg = np.nan_to_num(y_true_reg, nan=np.nanmedian(y_true_reg))
            
            # Apply feature selection if available
            if feature_selector is not None:
                try:
                    X = feature_selector.transform(X)
                except:
                    # If feature selection fails, use all features
                    pass
            
            # Apply scaling if available
            if scaler is not None:
                try:
                    X = scaler.transform(X)
                except:
                    # If scaling fails, use unscaled features
                    pass
            
            return X, y_true_reg, model
            
        except Exception as e:
            print(f"   ❌ Error preparing features: {e}")
            return None, None, None
    
    def convert_to_movement_classes(self, y_true, y_pred):
        """Convert continuous values to movement classes (Up/Down)."""
        if len(y_true) < 2:
            return None, None, None
            
        # Convert to price movement direction (1=up, 0=down)
        movement_true = np.where(np.diff(y_true) > 0, 1, 0)
        movement_pred = np.where(np.diff(y_pred) > 0, 1, 0)
        
        class_names = ['Down', 'Up']
        
        return movement_true, movement_pred, class_names
    
    def calculate_all_metrics(self, y_true, y_pred, class_names):
        """Calculate all requested metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        if len(np.unique(y_true)) > 1:  # Only calculate if multiple classes exist
            metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None, zero_division=0)
            metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None, zero_division=0)
            metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None, zero_division=0)
        else:
            metrics['precision_per_class'] = [metrics['precision']]
            metrics['recall_per_class'] = [metrics['recall']]
            metrics['f1_per_class'] = [metrics['f1_score']]
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report as string
        metrics['classification_report_str'] = classification_report(y_true, y_pred, 
                                                                   target_names=class_names, 
                                                                   zero_division=0)
        
        return metrics
    
    def create_comprehensive_plots(self, metrics, model_name, target_column, class_names):
        """Create comprehensive plots for all metrics."""
        base_name = model_name.replace('.joblib', '').replace(' ', '_')
        
        # 1. Confusion Matrix Heatmap
        self.plot_confusion_matrix(metrics['confusion_matrix'], class_names, base_name)
        
        # 2. Metrics Comparison Bar Chart
        self.plot_metrics_comparison(metrics, base_name, target_column)
        
        # 3. Per-Class Metrics Plot (only if multiple classes)
        if len(class_names) > 1:
            self.plot_per_class_metrics(metrics, class_names, base_name, target_column)
        
        # 4. Performance Summary Dashboard
        self.create_performance_dashboard(metrics, model_name, target_column, class_names, base_name)
        
        # 5. Metrics Distribution Plot
        self.plot_metrics_distribution(metrics, base_name, target_column)
    
    def plot_confusion_matrix(self, cm, class_names, base_name):
        """Plot enhanced confusion matrix."""
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names,
                        cbar_kws={'label': 'Count'}, annot_kws={"size": 12})
        
        plt.title(f'Confusion Matrix: {base_name}', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/{base_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {base_name}_confusion_matrix.png")
    
    def plot_metrics_comparison(self, metrics, base_name, target_column):
        """Plot metrics comparison bar chart."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Main metrics
        main_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [metrics[metric] for metric in main_metrics]
        
        # Create bar plot
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        bars = ax.bar(metric_names, metric_values, color=colors)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        
        # Customize plot
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Model Performance Metrics: {base_name}\nTarget: {target_column}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        # Add horizontal line at 0.5 for reference
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Baseline (0.5)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/{base_name}_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {base_name}_metrics_comparison.png")
    
    def plot_per_class_metrics(self, metrics, class_names, base_name, target_column):
        """Plot per-class metrics."""
        if len(class_names) > 1 and len(metrics['precision_per_class']) == len(class_names):
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = np.arange(len(class_names))
            width = 0.25
            
            # Plot bars for each metric
            bars1 = ax.bar(x - width, metrics['precision_per_class'], width, label='Precision', alpha=0.8, color='#2E86AB')
            bars2 = ax.bar(x, metrics['recall_per_class'], width, label='Recall', alpha=0.8, color='#A23B72')
            bars3 = ax.bar(x + width, metrics['f1_per_class'], width, label='F1-Score', alpha=0.8, color='#F18F01')
            
            # Add value labels
            for bars in [bars1, bars2, bars3]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Customize plot
            ax.set_xlabel('Classes', fontsize=12, fontweight='bold')
            ax.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax.set_title(f'Per-Class Metrics: {base_name}\nTarget: {target_column}', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(class_names)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim(0, 1.1)
            
            plt.tight_layout()
            plt.savefig(f'{self.plots_dir}/{base_name}_per_class_metrics.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   💾 Saved: {base_name}_per_class_metrics.png")
    
    def plot_metrics_distribution(self, metrics, base_name, target_column):
        """Plot metrics distribution as a horizontal bar chart."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics_data = {
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score']
        }
        
        # Sort by value
        sorted_metrics = dict(sorted(metrics_data.items(), key=lambda x: x[1]))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_metrics)))
        bars = ax.barh(list(sorted_metrics.keys()), list(sorted_metrics.values()), color=colors)
        
        # Add value labels
        for bar, (metric, value) in zip(bars, sorted_metrics.items()):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.4f}', va='center', fontweight='bold')
        
        ax.set_xlim(0, 1.1)
        ax.set_xlabel('Score', fontweight='bold')
        ax.set_title(f'Metrics Distribution: {base_name}\nTarget: {target_column}', 
                    fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/{base_name}_metrics_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {base_name}_metrics_distribution.png")
    
    def create_performance_dashboard(self, metrics, model_name, target_column, class_names, base_name):
        """Create a comprehensive performance dashboard."""
        fig = plt.figure(figsize=(20, 12))
        
        # Create subplots grid
        gs = fig.add_gridspec(2, 2)
        
        # 1. Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title('Confusion Matrix', fontweight='bold', fontsize=12)
        ax1.set_ylabel('True Label', fontweight='bold')
        ax1.set_xlabel('Predicted Label', fontweight='bold')
        
        # 2. Main Metrics Bar Chart
        ax2 = fig.add_subplot(gs[0, 1])
        main_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [metrics[metric] for metric in main_metrics]
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        bars = ax2.bar(metric_names, metric_values, color=colors)
        
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_ylim(0, 1.1)
        ax2.set_title('Performance Metrics', fontweight='bold', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        
        # 3. Metrics Summary Table
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('tight')
        ax3.axis('off')
        
        # Create summary table
        summary_data = [
            ['Accuracy', f"{metrics['accuracy']:.4f}"],
            ['Precision', f"{metrics['precision']:.4f}"],
            ['Recall', f"{metrics['recall']:.4f}"],
            ['F1-Score', f"{metrics['f1_score']:.4f}"],
            ['Model', model_name[:50]],
            ['Target', target_column],
            ['Classes', ', '.join(class_names)],
            ['Samples', f"{metrics['confusion_matrix'].sum()}"]
        ]
        
        table = ax3.table(cellText=summary_data,
                         colLabels=['Metric', 'Value'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0.1, 0.1, 0.8, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Main title
        plt.suptitle(f'COMPREHENSIVE MODEL PERFORMANCE DASHBOARD\n{model_name}', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/{base_name}_performance_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   💾 Saved: {base_name}_performance_dashboard.png")
    
    def evaluate_model(self, model_path, test_data_path, target_column):
        """Evaluate a single model and create comprehensive plots."""
        print(f"\n🔍 Evaluating: {os.path.basename(model_path)}")
        print("=" * 50)
        
        try:
            # Prepare features and make predictions
            X, y_true_reg, model = self.prepare_features(model_path, test_data_path, target_column)
            
            if X is None or model is None:
                return None
            
            # Make predictions
            y_pred_reg = model.predict(X)
            
            # Convert to classification
            y_true_class, y_pred_class, class_names = self.convert_to_movement_classes(y_true_reg, y_pred_reg)
            
            if y_true_class is None:
                return None
            
            # Calculate all metrics
            metrics = self.calculate_all_metrics(y_true_class, y_pred_class, class_names)
            
            # Create comprehensive plots
            self.create_comprehensive_plots(metrics, os.path.basename(model_path), target_column, class_names)
            
            # Store results
            model_name = os.path.basename(model_path)
            self.results[model_name] = {
                'metrics': metrics,
                'target_column': target_column,
                'class_names': class_names
            }
            
            # Print results
            self.print_results(metrics, model_name, target_column, class_names)
            
            return metrics
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return None
    
    def print_results(self, metrics, model_name, target_column, class_names):
        """Print all results in a clean format."""
        print(f"\n📊 **RESULTS FOR: {model_name}**")
        print(f"🎯 Target: {target_column}")
        print("=" * 60)
        
        print(f"✅ **Accuracy**:  {metrics['accuracy']:.4f}")
        print(f"🎯 **Precision**: {metrics['precision']:.4f}")
        print(f"🔁 **Recall**:    {metrics['recall']:.4f}")
        print(f"⚖️  **F1-Score**:  {metrics['f1_score']:.4f}")
        
        print(f"\n📋 **Confusion Matrix**:")
        print(metrics['confusion_matrix'])
        print(f"   (Rows: True, Columns: Predicted)")
        print(f"   Classes: {class_names}")
        
        print(f"\n📝 **Classification Report**:")
        print(metrics['classification_report_str'])
        
        print(f"💾 **All plots saved to**: {self.plots_dir}/")
        print("=" * 60)
    
    def evaluate_all_models(self):
        """Evaluate all models and create comprehensive visualizations."""
        print("🚀 CREATING COMPREHENSIVE METRICS VISUALIZATIONS")
        print("=" * 60)
        
        # Model to test data mapping
        model_test_data_mapping = {
            'all_stocks_combined': "feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv",
            'COST_full_year': "feature_engineered_data/stock_data/feature_engineered_outlier_treated_COST_full_year.csv", 
            'V_full_year': "feature_engineered_data/stock_data/feature_engineered_outlier_treated_V_full_year.csv"
        }
        
        models_dir = "trained_models"
        evaluated_count = 0
        
        for model_file in os.listdir(models_dir):
            if model_file.endswith('.joblib'):
                model_path = os.path.join(models_dir, model_file)
                
                # Determine test data and target
                test_data_path = None
                target_column = None
                
                if 'all_stocks_combined' in model_file:
                    test_data_path = model_test_data_mapping['all_stocks_combined']
                elif 'COST_full_year' in model_file:
                    test_data_path = model_test_data_mapping['COST_full_year']
                elif 'V_full_year' in model_file:
                    test_data_path = model_test_data_mapping['V_full_year']
                
                if 'Close' in model_file:
                    target_column = 'Close'
                elif 'daily_return' in model_file:
                    target_column = 'daily_return'
                elif 'volatility_5d' in model_file:
                    target_column = 'volatility_5d'
                
                if test_data_path and target_column and os.path.exists(test_data_path):
                    print(f"🎯 Processing: {model_file}")
                    metrics = self.evaluate_model(model_path, test_data_path, target_column)
                    if metrics:
                        evaluated_count += 1
                        print(f"✅ Successfully evaluated: {model_file}")
                else:
                    print(f"❌ Skipping {model_file} - test data not found")
        
        return evaluated_count
    
    def create_comparative_analysis(self):
        """Create comparative analysis across all models."""
        if not self.results:
            print("No results for comparative analysis.")
            return
        
        print("\n📈 CREATING COMPARATIVE ANALYSIS ACROSS ALL MODELS")
        print("=" * 60)
        
        # Prepare data for comparison
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'Target': result['target_column'],
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create comparative plots
        self.plot_model_comparison(df_comparison)
        self.plot_performance_heatmap(df_comparison)
        
        # Save comparative data
        df_comparison.to_csv(f'{self.plots_dir}/model_comparison.csv', index=False)
        print(f"💾 Comparative analysis saved to: {self.plots_dir}/model_comparison.csv")
        
        return len(comparison_data)
    
    def plot_model_comparison(self, df):
        """Create comparative bar plot across models."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        axes = axes.flatten()
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, metric in enumerate(metrics_to_plot):
            # Sort by metric value
            df_sorted = df.sort_values(metric, ascending=True)
            
            bars = axes[i].barh(range(len(df_sorted)), df_sorted[metric], 
                               color=plt.cm.viridis(np.linspace(0, 1, len(df_sorted))))
            
            # Add value labels
            for j, (bar, value) in enumerate(zip(bars, df_sorted[metric])):
                axes[i].text(value + 0.01, bar.get_y() + bar.get_height()/2,
                           f'{value:.4f}', va='center', fontsize=9, fontweight='bold')
            
            axes[i].set_yticks(range(len(df_sorted)))
            axes[i].set_yticklabels([f"{row['Model'][:30]}...\n({row['Target']})" 
                                   for _, row in df_sorted.iterrows()], fontsize=8)
            axes[i].set_xlabel(metric, fontweight='bold')
            axes[i].set_xlim(0, 1.1)
            axes[i].grid(axis='x', alpha=0.3)
            axes[i].set_title(f'{metric} Comparison', fontweight='bold')
        
        plt.tight_layout()
        plt.suptitle('MODEL PERFORMANCE COMPARISON ACROSS ALL METRICS', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.savefig(f'{self.plots_dir}/model_comparison_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: model_comparison_chart.png")
    
    def plot_performance_heatmap(self, df):
        """Create performance heatmap across models and metrics."""
        # Pivot data for heatmap
        heatmap_data = df.set_index(['Model', 'Target'])[['Accuracy', 'Precision', 'Recall', 'F1-Score']]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Score'}, linewidths=0.5)
        plt.title('MODEL PERFORMANCE HEATMAP\n(Higher is Better)', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.plots_dir}/performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"💾 Saved: performance_heatmap.png")

def main():
    visualizer = PerfectMetricsVisualizer()
    
    # Evaluate all models and create comprehensive plots
    evaluated_count = visualizer.evaluate_all_models()
    
    # Create comparative analysis
    if evaluated_count > 0:
        comparative_count = visualizer.create_comparative_analysis()
    
    print(f"\n🎉 **COMPREHENSIVE VISUALIZATION COMPLETED!**")
    print(f"📊 Models evaluated: {evaluated_count}")
    print(f"📁 All plots saved to: {visualizer.plots_dir}/")
    print(f"\n📋 **GENERATED PLOTS FOR EACH MODEL**:")
    print("   • Confusion Matrix Heatmap")
    print("   • Metrics Comparison Bar Chart") 
    print("   • Per-Class Metrics Plot")
    print("   • Metrics Distribution Chart")
    print("   • Comprehensive Dashboard")
    print(f"\n📊 **COMPARATIVE ANALYSIS**:")
    print("   • Model Comparison Charts")
    print("   • Performance Heatmap")
    print("   • Comparative CSV Report")

if __name__ == "__main__":
    main()
