# regression_to_classification_metrics.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class RegressionToClassificationEvaluator:
    """
    Evaluate regression models using classification metrics by converting
    continuous predictions to categorical movements.
    """
    
    def __init__(self):
        self.results = {}
        self.plots_dir = "classification_metrics_plots"
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def convert_to_movement_classes(self, y_true, y_pred, method='direction', bins=3):
        """
        Convert continuous values to movement classes.
        
        Parameters:
        - method: 'direction' (up/down) or 'magnitude' (low/medium/high)
        - bins: number of bins for magnitude method
        """
        if method == 'direction':
            # Convert to price movement direction (1=up, 0=down)
            movement_true = np.where(np.diff(y_true) > 0, 1, 0)
            movement_pred = np.where(np.diff(y_pred) > 0, 1, 0)
            
            # Since diff reduces length by 1, align arrays
            y_true_aligned = y_true[1:]
            y_pred_aligned = y_pred[1:]
            
            class_names = ['Down', 'Up']
            
        elif method == 'magnitude':
            # Convert to magnitude-based classes using quantiles
            true_quantiles = pd.qcut(y_true, q=bins, labels=False, duplicates='drop')
            pred_quantiles = pd.qcut(y_pred, q=bins, labels=False, duplicates='drop')
            
            movement_true = true_quantiles
            movement_pred = pred_quantiles
            
            class_names = [f'Class_{i}' for i in range(bins)]
            
        else:
            raise ValueError("Method must be 'direction' or 'magnitude'")
        
        return movement_true, movement_pred, class_names
    
    def calculate_classification_metrics(self, y_true, y_pred, class_names):
        """Calculate comprehensive classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report
        metrics['classification_report'] = classification_report(y_true, y_pred, 
                                                               target_names=class_names, 
                                                               output_dict=True,
                                                               zero_division=0)
        
        return metrics
    
    def plot_confusion_matrix(self, cm, class_names, title):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix: {title}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontweight='bold')
        plt.xlabel('Predicted Label', fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        filename = f"{title.replace(' ', '_').replace(':', '')}_confusion_matrix.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def plot_prediction_comparison(self, y_true_reg, y_pred_reg, y_true_class, y_pred_class, title):
        """Plot regression predictions vs actual with classification overlay."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Regression comparison
        ax1.plot(y_true_reg, label='Actual', alpha=0.7, linewidth=2)
        ax1.plot(y_pred_reg, label='Predicted', alpha=0.7, linewidth=2)
        ax1.set_title(f'Regression Predictions: {title}', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Classification comparison
        time_points = range(len(y_true_class))
        ax2.plot(time_points, y_true_class, 'o-', label='True Movement', alpha=0.7, linewidth=2)
        ax2.plot(time_points, y_pred_class, 'x-', label='Predicted Movement', alpha=0.7, linewidth=2)
        ax2.set_title(f'Classification Movement: {title}', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Time Points')
        ax2.set_ylabel('Movement Class')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{title.replace(' ', '_').replace(':', '')}_comparison.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def evaluate_regression_model(self, model_path, test_data_path, target_column, method='direction'):
        """
        Evaluate a regression model using classification metrics.
        
        Parameters:
        - model_path: Path to the saved .joblib model
        - test_data_path: Path to the test data CSV
        - target_column: Name of the target column
        - method: 'direction' or 'magnitude'
        """
        print(f"\n🔍 Evaluating: {os.path.basename(model_path)}")
        print("=" * 60)
        
        try:
            # Load model
            model_data = joblib.load(model_path)
            model = model_data['model']
            scaler = model_data.get('scaler')
            feature_selector = model_data.get('feature_selector')
            
            # Load test data
            df = pd.read_csv(test_data_path, low_memory=False)
            
            # Prepare features and target
            X = df.drop(columns=[target_column], errors='ignore')
            y_true_reg = df[target_column].values
            
            # Remove non-numeric columns
            X = X.select_dtypes(include=[np.number])
            
            # Handle missing values
            X = X.fillna(X.median())
            y_true_reg = np.nan_to_num(y_true_reg, nan=np.nanmedian(y_true_reg))
            
            # Apply feature selection if available
            if feature_selector is not None:
                X = feature_selector.transform(X)
            
            # Scale features if scaler is available
            if scaler is not None:
                X = scaler.transform(X)
            
            # Make predictions
            y_pred_reg = model.predict(X)
            
            # Convert to classification
            y_true_class, y_pred_class, class_names = self.convert_to_movement_classes(
                y_true_reg, y_pred_reg, method=method
            )
            
            # Calculate metrics
            metrics = self.calculate_classification_metrics(y_true_class, y_pred_class, class_names)
            
            # Generate plots
            cm_plot_path = self.plot_confusion_matrix(
                metrics['confusion_matrix'], class_names, 
                f"{os.path.basename(model_path)} - {target_column}"
            )
            
            comparison_plot_path = self.plot_prediction_comparison(
                y_true_reg, y_pred_reg, y_true_class, y_pred_class,
                f"{os.path.basename(model_path)} - {target_column}"
            )
            
            # Store results
            model_name = os.path.basename(model_path)
            self.results[model_name] = {
                'metrics': metrics,
                'regression_data': {
                    'y_true': y_true_reg,
                    'y_pred': y_pred_reg
                },
                'classification_data': {
                    'y_true': y_true_class,
                    'y_pred': y_pred_class,
                    'class_names': class_names
                },
                'plot_paths': {
                    'confusion_matrix': cm_plot_path,
                    'comparison': comparison_plot_path
                }
            }
            
            # Print results
            self.print_detailed_metrics(metrics, model_name, target_column, class_names)
            
            return metrics
            
        except Exception as e:
            print(f"❌ Error evaluating {model_path}: {e}")
            return None
    
    def print_detailed_metrics(self, metrics, model_name, target_column, class_names):
        """Print comprehensive metrics in a readable format."""
        print(f"\n📊 MODEL: {model_name}")
        print(f"🎯 TARGET: {target_column}")
        print(f"🏷️  CLASSES: {class_names}")
        print("-" * 50)
        
        print(f"✅ Accuracy:  {metrics['accuracy']:.4f}")
        print(f"🎯 Precision: {metrics['precision']:.4f}")
        print(f"🔁 Recall:    {metrics['recall']:.4f}")
        print(f"⚖️  F1-Score:  {metrics['f1_score']:.4f}")
        
        print(f"\n📋 Confusion Matrix:")
        print(metrics['confusion_matrix'])
        
        print(f"\n📝 Detailed Classification Report:")
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        print(report_df.round(4))
        
        print(f"\n💾 Plots saved to: {self.plots_dir}")
        print("=" * 60)
    
    def evaluate_all_models(self, models_dir, test_data_path, target_columns):
        """Evaluate all models in a directory."""
        print("🚀 STARTING REGRESSION TO CLASSIFICATION EVALUATION")
        print("=" * 60)
        
        for model_file in os.listdir(models_dir):
            if model_file.endswith('.joblib'):
                model_path = os.path.join(models_dir, model_file)
                
                # Extract target column from filename
                for target_col in target_columns:
                    if target_col in model_file:
                        self.evaluate_regression_model(model_path, test_data_path, target_col)
                        break
    
    def generate_summary_report(self):
        """Generate a summary report of all evaluations."""
        if not self.results:
            print("No results to summarize.")
            return
        
        print("\n📈 SUMMARY REPORT")
        print("=" * 60)
        
        summary_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            summary_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score']
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.round(4))
        
        # Save summary to CSV
        summary_path = os.path.join(self.plots_dir, "evaluation_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n💾 Summary saved to: {summary_path}")
        
        return summary_df

# Usage example
def main():
    # Initialize evaluator
    evaluator = RegressionToClassificationEvaluator()
    
    # Define paths and targets
    models_directory = "trained_models"  # Your models directory
    test_data_file = "feature_engineered_data/stock_data/feature_engineered_outlier_treated_all_stocks_combined.csv"
    
    target_columns = ['Close', 'daily_return', 'volatility_5d']
    
    # Evaluate all models
    evaluator.evaluate_all_models(models_directory, test_data_file, target_columns)
    
    # Generate summary report
    evaluator.generate_summary_report()
    
    print(f"\n🎉 EVALUATION COMPLETED!")
    print(f"📊 Models evaluated: {len(evaluator.results)}")
    print(f"📁 Results saved to: {evaluator.plots_dir}")

# Quick evaluation function for single model
def quick_evaluate_single_model(model_path, test_data_path, target_column):
    """Quick evaluation for a single model."""
    evaluator = RegressionToClassificationEvaluator()
    metrics = evaluator.evaluate_regression_model(model_path, test_data_path, target_column)
    return metrics

if __name__ == "__main__":
    # Install required packages if needed
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
    except ImportError:
        print("📦 Installing required packages...")
        os.system("pip install seaborn matplotlib")
    
    # Run main evaluation
    main()
