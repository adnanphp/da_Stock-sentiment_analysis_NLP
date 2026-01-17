# fixed_regression_to_classification_metrics.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class FixedRegressionToClassificationEvaluator:
    """
    Fixed evaluator that handles feature dimension mismatches.
    """
    
    def __init__(self):
        self.results = {}
        self.plots_dir = "classification_metrics_plots"
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def prepare_features_with_original_data(self, model_path, test_data_path, target_column):
        """
        Prepare features using the original training data structure to avoid dimension mismatches.
        """
        try:
            # Load model and get original training data info
            model_data = joblib.load(model_path)
            model = model_data['model']
            scaler = model_data.get('scaler')
            feature_selector = model_data.get('feature_selector')
            
            # Load test data
            df = pd.read_csv(test_data_path, low_memory=False)
            
            # Prepare features exactly as done during training
            X = df.drop(columns=[target_column], errors='ignore')
            y_true_reg = df[target_column].values
            
            # Remove non-numeric columns (same as training)
            X = X.select_dtypes(include=[np.number])
            
            # Handle missing values (same as training)
            X = X.fillna(X.median())
            y_true_reg = np.nan_to_num(y_true_reg, nan=np.nanmedian(y_true_reg))
            
            # Get the original feature names that were used during training
            original_features = X.columns.tolist()
            
            # If we have a feature selector, we need to handle it differently
            if feature_selector is not None:
                try:
                    # Try to apply the feature selector
                    X_transformed = feature_selector.transform(X)
                    print(f"   ✅ Applied feature selection: {X.shape[1]} -> {X_transformed.shape[1]} features")
                    X = X_transformed
                except ValueError as e:
                    print(f"   ⚠️  Feature selection failed: {e}")
                    print(f"   🔧 Using original {X.shape[1]} features without selection")
            
            # If we have a scaler, apply it
            if scaler is not None:
                try:
                    X = scaler.transform(X)
                    print(f"   ✅ Applied feature scaling")
                except ValueError as e:
                    print(f"   ⚠️  Feature scaling failed: {e}")
                    print(f"   🔧 Using original features without scaling")
            
            return X, y_true_reg, model
            
        except Exception as e:
            print(f"   ❌ Error preparing features: {e}")
            return None, None, None
    
    def convert_to_movement_classes(self, y_true, y_pred, method='direction'):
        """
        Convert continuous values to movement classes.
        """
        if len(y_true) < 2:
            raise ValueError("Not enough data points for movement analysis")
            
        if method == 'direction':
            # Convert to price movement direction (1=up, 0=down)
            movement_true = np.where(np.diff(y_true) > 0, 1, 0)
            movement_pred = np.where(np.diff(y_pred) > 0, 1, 0)
            
            class_names = ['Down', 'Up']
            
        elif method == 'magnitude':
            # Use percent change for more robust movement detection
            true_pct_change = np.diff(y_true) / y_true[:-1]
            pred_pct_change = np.diff(y_pred) / y_pred[:-1]
            
            # Define thresholds for movement classification
            threshold = 0.005  # 0.5% threshold
            
            movement_true = np.where(abs(true_pct_change) > threshold, 
                                   np.where(true_pct_change > 0, 2, 0),  # 2=Up, 0=Down
                                   1)  # 1=Stable
            movement_pred = np.where(abs(pred_pct_change) > threshold,
                                   np.where(pred_pct_change > 0, 2, 0),
                                   1)
            
            class_names = ['Down', 'Stable', 'Up']
        
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
        filename = f"{title.replace(' ', '_').replace(':', '').replace('.joblib', '')}_confusion_matrix.png"
        filepath = os.path.join(self.plots_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def evaluate_single_model(self, model_path, test_data_path, target_column, method='direction'):
        """
        Evaluate a single regression model using classification metrics.
        """
        print(f"\n🔍 Evaluating: {os.path.basename(model_path)}")
        print("=" * 60)
        
        try:
            # Prepare features with proper dimension handling
            X, y_true_reg, model = self.prepare_features_with_original_data(model_path, test_data_path, target_column)
            
            if X is None or model is None:
                print(f"   ❌ Failed to prepare data for {os.path.basename(model_path)}")
                return None
            
            # Make predictions
            y_pred_reg = model.predict(X)
            
            print(f"   📊 Predictions made: {len(y_pred_reg)} samples")
            print(f"   📈 True values range: {y_true_reg.min():.4f} to {y_true_reg.max():.4f}")
            print(f"   📉 Pred values range: {y_pred_reg.min():.4f} to {y_pred_reg.max():.4f}")
            
            # Convert to classification
            y_true_class, y_pred_class, class_names = self.convert_to_movement_classes(
                y_true_reg, y_pred_reg, method=method
            )
            
            # Calculate metrics
            metrics = self.calculate_classification_metrics(y_true_class, y_pred_class, class_names)
            
            # Generate plots
            cm_plot_path = self.plot_confusion_matrix(
                metrics['confusion_matrix'], class_names, 
                f"{os.path.basename(model_path)}"
            )
            
            # Store results
            model_name = os.path.basename(model_path)
            self.results[model_name] = {
                'metrics': metrics,
                'target_column': target_column,
                'class_names': class_names,
                'plot_path': cm_plot_path
            }
            
            # Print results
            self.print_detailed_metrics(metrics, model_name, target_column, class_names)
            
            return metrics
            
        except Exception as e:
            print(f"   ❌ Error evaluating {model_path}: {e}")
            import traceback
            traceback.print_exc()
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
        report_str = classification_report(
            metrics['classification_report']['support'].keys(),
            [1] * len(metrics['classification_report']['support']),  # dummy values for display
            target_names=class_names,
            zero_division=0
        )
        print(report_str)
        
        # Print per-class metrics from the stored report
        for class_name in class_names:
            if class_name in metrics['classification_report']:
                class_metrics = metrics['classification_report'][class_name]
                print(f"   {class_name}: Precision={class_metrics['precision']:.3f}, "
                      f"Recall={class_metrics['recall']:.3f}, F1={class_metrics['f1-score']:.3f}")
        
        print(f"\n💾 Plot saved to: {self.results[model_name]['plot_path']}")
        print("=" * 60)
    
    def evaluate_all_models(self):
        """Evaluate all models with proper test data mapping."""
        print("🚀 STARTING FIXED REGRESSION TO CLASSIFICATION EVALUATION")
        print("=" * 60)
        
        # Define model to test data mapping
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
                
                # Determine which test data to use based on model filename
                test_data_path = None
                target_column = None
                
                if 'all_stocks_combined' in model_file:
                    test_data_path = model_test_data_mapping['all_stocks_combined']
                elif 'COST_full_year' in model_file:
                    test_data_path = model_test_data_mapping['COST_full_year']
                elif 'V_full_year' in model_file:
                    test_data_path = model_test_data_mapping['V_full_year']
                
                # Determine target column
                if 'Close' in model_file:
                    target_column = 'Close'
                elif 'daily_return' in model_file:
                    target_column = 'daily_return'
                elif 'volatility_5d' in model_file:
                    target_column = 'volatility_5d'
                
                if test_data_path and target_column and os.path.exists(test_data_path):
                    print(f"\n🎯 Processing: {model_file}")
                    print(f"📁 Test data: {test_data_path}")
                    print(f"🎯 Target: {target_column}")
                    
                    metrics = self.evaluate_single_model(model_path, test_data_path, target_column)
                    if metrics:
                        evaluated_count += 1
                else:
                    print(f"❌ Could not find test data for {model_file}")
        
        print(f"\n📊 Successfully evaluated: {evaluated_count} models")
        return evaluated_count
    
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
                'Target': result['target_column'],
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save summary to CSV
        summary_path = os.path.join(self.plots_dir, "evaluation_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n💾 Summary saved to: {summary_path}")
        
        return summary_df

def main():
    # Initialize evaluator
    evaluator = FixedRegressionToClassificationEvaluator()
    
    # Evaluate all models
    evaluated_count = evaluator.evaluate_all_models()
    
    # Generate summary report
    if evaluated_count > 0:
        evaluator.generate_summary_report()
    
    print(f"\n🎉 EVALUATION COMPLETED!")
    print(f"📊 Models evaluated: {evaluated_count}")
    print(f"📁 Results saved to: {evaluator.plots_dir}")

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
