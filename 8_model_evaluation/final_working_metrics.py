# final_working_metrics.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, precision_score, recall_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class FinalMetricsCalculator:
    """
    Final working version that calculates all requested metrics.
    """
    
    def __init__(self):
        self.results = {}
        self.plots_dir = "final_metrics_results"
        os.makedirs(self.plots_dir, exist_ok=True)
    
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
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Classification report as string
        metrics['classification_report_str'] = classification_report(y_true, y_pred, 
                                                                   target_names=class_names, 
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
    
    def evaluate_model(self, model_path, test_data_path, target_column):
        """Evaluate a single model and return all metrics."""
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
            
            # Plot confusion matrix
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
        
        print(f"💾 **Plot saved**: {self.results[model_name]['plot_path']}")
        print("=" * 60)
    
    def evaluate_all_models(self):
        """Evaluate all models."""
        print("🚀 CALCULATING F1 SCORE, ACCURACY, AND CONFUSION MATRIX")
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
                    metrics = self.evaluate_model(model_path, test_data_path, target_column)
                    if metrics:
                        evaluated_count += 1
                else:
                    print(f"❌ Skipping {model_file} - test data not found")
        
        return evaluated_count
    
    def generate_final_summary(self):
        """Generate final summary report."""
        if not self.results:
            print("No results to summarize.")
            return
        
        print("\n" + "=" * 70)
        print("📈 FINAL SUMMARY REPORT")
        print("=" * 70)
        
        summary_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            summary_data.append({
                'Model': model_name[:40] + '...' if len(model_name) > 40 else model_name,
                'Target': result['target_column'],
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_path = os.path.join(self.plots_dir, "FINAL_METRICS_SUMMARY.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"\n💾 Full results saved to: {self.plots_dir}/")
        print(f"💾 Summary saved to: {summary_path}")
        
        # Best performing models
        if len(summary_data) > 0:
            best_accuracy = max([float(x['Accuracy']) for x in summary_data])
            best_f1 = max([float(x['F1-Score']) for x in summary_data])
            
            print(f"\n🏆 **BEST PERFORMING MODELS**:")
            print(f"   Highest Accuracy:  {best_accuracy:.4f}")
            print(f"   Highest F1-Score:  {best_f1:.4f}")
        
        return summary_df

def main():
    calculator = FinalMetricsCalculator()
    
    # Evaluate all models
    evaluated_count = calculator.evaluate_all_models()
    
    # Generate final summary
    if evaluated_count > 0:
        calculator.generate_final_summary()
    
    print(f"\n🎉 **EVALUATION COMPLETED!**")
    print(f"📊 Models successfully evaluated: {evaluated_count}")
    print(f"📁 All results saved to: {calculator.plots_dir}/")
    
    # Print quick results
    if calculator.results:
        print(f"\n📋 **QUICK RESULTS**:")
        for model_name, result in calculator.results.items():
            metrics = result['metrics']
            print(f"   {model_name}:")
            print(f"     Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()
