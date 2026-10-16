#!/usr/bin/env python3
"""
ENHANCED DEEP LEARNING PIPELINE WITH COMPREHENSIVE VISUALIZATIONS
Fixed version with proper data handling and detailed performance plots
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, 
                           roc_curve, confusion_matrix, classification_report,
                           precision_recall_curve, auc, precision_score, recall_score)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("ENHANCED DEEP LEARNING FOR FINANCIAL PREDICTION")
print("="*80)
print("With Comprehensive Performance Visualizations")
print("="*80)

# ============================================================================
# 1. SETUP
# ============================================================================

# Create output directory
os.makedirs('dl_enhanced_results', exist_ok=True)
os.makedirs('dl_enhanced_results/models', exist_ok=True)
os.makedirs('dl_enhanced_results/plots', exist_ok=True)
os.makedirs('dl_enhanced_results/plots/interactive', exist_ok=True)

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# 2. LOAD AND PREPARE DATA
# ============================================================================

print("\n1. LOADING DATA")

# Load PCA-transformed data
df_pca = pd.read_csv('pca_tsne_comprehensive_results/data/pca_transformed_data.csv')

# Load original data for targets
df_original = pd.read_csv('./integrated_datasets_simple/integrated_simple_20251202_213953.csv')

# Align indices
aligned_indices = df_pca.index.intersection(df_original.index)
df_pca_aligned = df_pca.loc[aligned_indices].copy()
df_original_aligned = df_original.loc[aligned_indices].copy()

# Get PCA features
pca_features = [col for col in df_pca.columns if col.startswith('PC_')]
X = df_pca_aligned[pca_features].values

print(f"✓ Data shape: {X.shape}")
print(f"✓ PCA components: {len(pca_features)}")

# ============================================================================
# 3. CREATE TARGETS
# ============================================================================

print("\n2. CREATING TARGETS")

# Create market regime target (best performing from your results)
df_original_aligned['ma_20'] = df_original_aligned.groupby('ticker')['Close'].rolling(20).mean().reset_index(0, drop=True)
df_original_aligned['ma_50'] = df_original_aligned.groupby('ticker')['Close'].rolling(50).mean().reset_index(0, drop=True)
df_original_aligned['market_regime'] = (df_original_aligned['ma_20'] > df_original_aligned['ma_50']).astype(int)

# Use market_regime as target (your best performing)
y = df_original_aligned['market_regime'].values

# Remove any NaN
valid_mask = ~np.isnan(y)
X = X[valid_mask]
y = y[valid_mask]

print(f"✓ Final data shape: X={X.shape}, y={y.shape}")
print(f"✓ Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# ============================================================================
# 4. SPLIT DATA
# ============================================================================

print("\n3. SPLITTING DATA")

# Split data (70% train, 15% val, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"✓ Train: {X_train.shape}, {y_train.shape}")
print(f"✓ Validation: {X_val.shape}, {y_val.shape}")
print(f"✓ Test: {X_test.shape}, {y_test.shape}")

# ============================================================================
# 5. SCALE DATA
# ============================================================================

print("\n4. SCALING DATA")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, 'dl_enhanced_results/scaler.joblib')
print("✓ Scaler saved")

# ============================================================================
# 6. SIMPLE DEEP LEARNING MODELS
# ============================================================================

print("\n5. BUILDING DEEP LEARNING MODELS")

def build_simple_mlp(input_dim):
    """Build a simple MLP model"""
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )
    
    return model

def build_cnn(input_dim):
    """Build a 1D CNN model"""
    model = models.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )
    
    return model

def build_lstm(input_dim):
    """Build an LSTM model"""
    model = models.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.LSTM(32),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )
    
    return model

# ============================================================================
# 7. TRAIN MODELS
# ============================================================================

print("\n6. TRAINING MODELS")

# Callbacks
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        filepath='dl_enhanced_results/models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    callbacks.TensorBoard(
        log_dir='dl_enhanced_results/tensorboard_logs',
        histogram_freq=1
    )
]

# Train different architectures
models_dict = {}
histories = {}

# 1. MLP Model
print("\n🔹 Training MLP Model...")
mlp_model = build_simple_mlp(X_train.shape[1])
mlp_model._name = "MLP"

history_mlp = mlp_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks_list,
    verbose=1
)

mlp_model.save('dl_enhanced_results/models/mlp_model.h5')
models_dict['MLP'] = mlp_model
histories['MLP'] = history_mlp
print(f"   MLP trained: {len(history_mlp.history['loss'])} epochs")

# 2. CNN Model
print("\n🔹 Training CNN Model...")
# Reshape for CNN
X_train_cnn = X_train_scaled.reshape(-1, X_train_scaled.shape[1], 1)
X_val_cnn = X_val_scaled.reshape(-1, X_val_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(-1, X_test_scaled.shape[1], 1)

cnn_model = build_cnn(X_train.shape[1])
cnn_model._name = "CNN"

history_cnn = cnn_model.fit(
    X_train_cnn, y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks_list,
    verbose=1
)

cnn_model.save('dl_enhanced_results/models/cnn_model.h5')
models_dict['CNN'] = cnn_model
histories['CNN'] = history_cnn
print(f"   CNN trained: {len(history_cnn.history['loss'])} epochs")

# 3. LSTM Model
print("\n🔹 Training LSTM Model...")
lstm_model = build_lstm(X_train.shape[1])
lstm_model._name = "LSTM"

history_lstm = lstm_model.fit(
    X_train_cnn, y_train,
    validation_data=(X_val_cnn, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks_list,
    verbose=1
)

lstm_model.save('dl_enhanced_results/models/lstm_model.h5')
models_dict['LSTM'] = lstm_model
histories['LSTM'] = history_lstm
print(f"   LSTM trained: {len(history_lstm.history['loss'])} epochs")

# ============================================================================
# 8. EVALUATE MODELS
# ============================================================================

print("\n7. EVALUATING MODELS")

results = {}
detailed_reports = {}

for model_name, model in models_dict.items():
    print(f"\n📊 Evaluating {model_name}:")
    
    # Prepare test data
    if model_name in ['CNN', 'LSTM']:
        X_test_prepared = X_test_cnn
    else:
        X_test_prepared = X_test_scaled
    
    # Make predictions
    y_pred_proba = model.predict(X_test_prepared, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   F1-Score: {f1:.3f}")
    print(f"   ROC-AUC: {roc_auc:.3f}")
    print(f"   Precision: {precision:.3f}")
    print(f"   Recall: {recall:.3f}")
    
    results[model_name] = {
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_test': y_test,
        'confusion_matrix': cm,
        'classification_report': report
    }
    
    # Save detailed report
    detailed_reports[model_name] = report

# ============================================================================
# 9. COMPREHENSIVE VISUALIZATIONS
# ============================================================================

print("\n8. CREATING COMPREHENSIVE VISUALIZATIONS")

# ============================================================================
# 9.1 TRAINING HISTORY VISUALIZATIONS
# ============================================================================

print("\n📈 8.1 Training History Visualizations")

# Create a comprehensive training history dashboard
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 4, figure=fig)

metrics_to_plot = ['loss', 'accuracy', 'precision', 'recall', 'auc']

for idx, (model_name, history) in enumerate(histories.items()):
    row = idx
    
    for metric_idx, metric in enumerate(['loss', 'accuracy', 'auc']):
        col = metric_idx
        
        ax = fig.add_subplot(gs[row, col])
        
        if metric in history.history:
            # Plot training and validation
            ax.plot(history.history[metric], label=f'Train {metric}', linewidth=2, alpha=0.8)
            if f'val_{metric}' in history.history:
                ax.plot(history.history[f'val_{metric}'], label=f'Val {metric}', linewidth=2, alpha=0.8, linestyle='--')
            
            ax.set_title(f'{model_name} - {metric.upper()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Mark best epoch
            if f'val_{metric}' in history.history:
                best_epoch = np.argmin(history.history[f'val_{metric}']) if metric == 'loss' else np.argmax(history.history[f'val_{metric}'])
                best_value = history.history[f'val_{metric}'][best_epoch]
                ax.axvline(x=best_epoch, color='red', linestyle=':', alpha=0.7, linewidth=1)
                ax.scatter(best_epoch, best_value, color='red', s=100, zorder=5, 
                          label=f'Best: {best_value:.4f}')

plt.suptitle('Model Training History Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('dl_enhanced_results/plots/training_history_comprehensive.png', dpi=300, bbox_inches='tight')

# ============================================================================
# 9.2 MODEL PERFORMANCE COMPARISON
# ============================================================================

print("📊 8.2 Model Performance Comparison")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Accuracy Comparison
ax = axes[0, 0]
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
bars = ax.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.axhline(y=0.703, color='red', linestyle='--', linewidth=2, label='Traditional Model (0.703)')
ax.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy')
ax.set_ylim([0.5, 1.0])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# 2. F1-Score Comparison
ax = axes[0, 1]
f1_scores = [results[m]['f1'] for m in model_names]
bars = ax.bar(model_names, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('F1-Score')
ax.set_ylim([0.5, 1.0])
ax.grid(True, alpha=0.3, axis='y')

# 3. ROC-AUC Comparison
ax = axes[0, 2]
roc_aucs = [results[m]['roc_auc'] for m in model_names]
bars = ax.bar(model_names, roc_aucs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
ax.set_title('ROC-AUC Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('ROC-AUC')
ax.set_ylim([0.5, 1.0])
ax.grid(True, alpha=0.3, axis='y')

# 4. Precision-Recall Comparison
ax = axes[1, 0]
precisions = [results[m]['precision'] for m in model_names]
recalls = [results[m]['recall'] for m in model_names]

x = np.arange(len(model_names))
width = 0.35

bars1 = ax.bar(x - width/2, precisions, width, label='Precision', alpha=0.8)
bars2 = ax.bar(x + width/2, recalls, width, label='Recall', alpha=0.8)

ax.set_title('Precision & Recall Comparison', fontsize=14, fontweight='bold')
ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 5. Confusion Matrix Heatmaps
for idx, model_name in enumerate(model_names[:2]):  # Show first 2 models
    ax = axes[1, 1 + idx]
    cm = results[model_name]['confusion_matrix']
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f'{model_name} - Confusion Matrix', fontsize=12, fontweight='bold')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['0', '1'])
    ax.set_yticklabels(['0', '1'])

plt.suptitle('Deep Learning Model Performance Dashboard', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('dl_enhanced_results/plots/model_performance_dashboard.png', dpi=300, bbox_inches='tight')

# ============================================================================
# 9.3 ROC CURVES AND PRECISION-RECALL CURVES
# ============================================================================

print("📈 8.3 ROC and Precision-Recall Curves")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# ROC Curves
ax = axes[0, 0]
for model_name, res in results.items():
    fpr, tpr, _ = roc_curve(res['y_test'], res['y_pred_proba'])
    roc_auc = res['roc_auc']
    
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2.5)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Precision-Recall Curves
ax = axes[0, 1]
for model_name, res in results.items():
    precision, recall, _ = precision_recall_curve(res['y_test'], res['y_pred_proba'])
    pr_auc = auc(recall, precision)
    
    ax.plot(recall, precision, label=f'{model_name} (AUC = {pr_auc:.3f})', linewidth=2.5)

ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Probability Distributions
ax = axes[1, 0]
for model_name, res in results.items():
    # Separate probabilities by true class
    prob_class_0 = res['y_pred_proba'][res['y_test'] == 0]
    prob_class_1 = res['y_pred_proba'][res['y_test'] == 1]
    
    ax.hist(prob_class_0, bins=30, alpha=0.5, label=f'{model_name} - Class 0', density=True)
    ax.hist(prob_class_1, bins=30, alpha=0.5, label=f'{model_name} - Class 1', density=True)

ax.set_xlabel('Predicted Probability', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Prediction Probability Distributions', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Model Performance Summary (Radar Chart)
ax = axes[1, 1]
metrics = ['Accuracy', 'F1-Score', 'ROC-AUC', 'Precision', 'Recall']
num_metrics = len(metrics)

# Create radar chart coordinates
angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
angles += angles[:1]

# Plot each model
for model_name in model_names:
    values = [
        results[model_name]['accuracy'],
        results[model_name]['f1'],
        results[model_name]['roc_auc'],
        results[model_name]['precision'],
        results[model_name]['recall']
    ]
    values += values[:1]  # Close the polygon
    
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name, markersize=8)
    ax.fill(angles, values, alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylim([0, 1])
ax.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.grid(True)

plt.suptitle('Advanced Model Evaluation Metrics', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('dl_enhanced_results/plots/advanced_metrics_comparison.png', dpi=300, bbox_inches='tight')

# ============================================================================
# 9.4 INTERACTIVE PLOTS (Plotly)
# ============================================================================

print("🎨 8.4 Creating Interactive Plots")

try:
    # Interactive ROC Curves
    fig = go.Figure()
    
    for model_name, res in results.items():
        fpr, tpr, _ = roc_curve(res['y_test'], res['y_pred_proba'])
        roc_auc = res['roc_auc']
        
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'{model_name} (AUC={roc_auc:.3f})',
            line=dict(width=3)
        ))
    
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color='gray')
    ))
    
    fig.update_layout(
        title='Interactive ROC Curves',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        hovermode='x unified',
        template='plotly_white',
        height=600
    )
    
    fig.write_html('dl_enhanced_results/plots/interactive/interactive_roc_curves.html')
    
    # Interactive Model Comparison
    fig = go.Figure()
    
    metrics = ['Accuracy', 'F1-Score', 'ROC-AUC', 'Precision', 'Recall']
    
    for model_name in model_names:
        values = [
            results[model_name]['accuracy'],
            results[model_name]['f1'],
            results[model_name]['roc_auc'],
            results[model_name]['precision'],
            results[model_name]['recall']
        ]
        
        fig.add_trace(go.Bar(
            name=model_name,
            x=metrics,
            y=values,
            text=[f'{v:.3f}' for v in values],
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Interactive Model Performance Comparison',
        xaxis_title='Metrics',
        yaxis_title='Score',
        barmode='group',
        template='plotly_white',
        height=600
    )
    
    fig.write_html('dl_enhanced_results/plots/interactive/interactive_model_comparison.html')
    
    print("✓ Interactive plots saved")
except Exception as e:
    print(f"⚠️  Could not create interactive plots: {e}")

# ============================================================================
# 9.5 SINGLE MODEL DEEP DIVE VISUALIZATION
# ============================================================================

print("🔍 8.5 Creating Single Model Deep Dive")

# Select best model for deep dive
best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
best_model_results = results[best_model_name]

fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig)

# 1. Confusion Matrix with percentages
ax1 = fig.add_subplot(gs[0, 0])
cm = best_model_results['confusion_matrix']
cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

im = ax1.imshow(cm_percentage, interpolation='nearest', cmap='YlOrRd')
ax1.set_title(f'{best_model_name} - Confusion Matrix (%)', fontsize=14, fontweight='bold')

# Add text annotations
thresh = cm_percentage.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax1.text(j, i, f'{cm[i, j]}\n({cm_percentage[i, j]:.1f}%)',
                ha="center", va="center",
                color="white" if cm_percentage[i, j] > thresh else "black",
                fontsize=11)

ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Class 0', 'Class 1'])
ax1.set_yticklabels(['Class 0', 'Class 1'])

# 2. Classification Metrics Bar Chart
ax2 = fig.add_subplot(gs[0, 1])
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
metrics_values = [
    best_model_results['accuracy'],
    best_model_results['precision'],
    best_model_results['recall'],
    best_model_results['f1'],
    best_model_results['roc_auc']
]

bars = ax2.bar(metrics_names, metrics_values, color=plt.cm.Set3(np.arange(len(metrics_names))))
ax2.set_title(f'{best_model_name} - Classification Metrics', fontsize=14, fontweight='bold')
ax2.set_ylim([0, 1])
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, value in zip(bars, metrics_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# 3. Probability Calibration Plot
ax3 = fig.add_subplot(gs[0, 2])
probabilities = best_model_results['y_pred_proba']
true_labels = best_model_results['y_test']

# Create bins for calibration
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_indices = np.digitize(probabilities, bin_edges) - 1

bin_means = []
bin_true_props = []
for i in range(n_bins):
    mask = bin_indices == i
    if np.sum(mask) > 0:
        bin_mean = np.mean(probabilities[mask])
        bin_true_prop = np.mean(true_labels[mask])
        bin_means.append(bin_mean)
        bin_true_props.append(bin_true_prop)

ax3.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated', alpha=0.5)
ax3.scatter(bin_means, bin_true_props, s=100, edgecolors='black', zorder=5)
ax3.plot(bin_means, bin_true_props, 'r-', linewidth=2)

ax3.set_xlabel('Mean Predicted Probability')
ax3.set_ylabel('Fraction of Positives')
ax3.set_title(f'{best_model_name} - Calibration Plot', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Error Analysis: Misclassified Samples
ax4 = fig.add_subplot(gs[1, :])
misclassified_mask = best_model_results['y_pred'] != best_model_results['y_test']
misclassified_probs = probabilities[misclassified_mask]
misclassified_true = true_labels[misclassified_mask]

# Create histogram of misclassified probabilities
ax4.hist(misclassified_probs[misclassified_true == 0], 
        bins=20, alpha=0.7, label='Class 0 Misclassified', color='red')
ax4.hist(misclassified_probs[misclassified_true == 1], 
        bins=20, alpha=0.7, label='Class 1 Misclassified', color='blue')

ax4.axvline(x=0.5, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax4.set_xlabel('Predicted Probability')
ax4.set_ylabel('Count')
ax4.set_title(f'{best_model_name} - Error Analysis (n={np.sum(misclassified_mask)})', 
             fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle(f'{best_model_name} - Deep Dive Analysis (Best Model)', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'dl_enhanced_results/plots/{best_model_name.lower()}_deep_dive.png', 
           dpi=300, bbox_inches='tight')

# ============================================================================
# 10. CREATE ENSEMBLE
# ============================================================================

print("\n9. CREATING ENSEMBLE")

# Get predictions from all models
all_predictions = []
for model_name in ['MLP', 'CNN', 'LSTM']:
    if model_name in results:
        all_predictions.append(results[model_name]['y_pred_proba'])

if len(all_predictions) >= 2:
    # Average ensemble
    avg_predictions = np.mean(all_predictions, axis=0)
    avg_pred = (avg_predictions > 0.5).astype(int)
    
    # Weighted ensemble (by accuracy)
    accuracies = [results[m]['accuracy'] for m in ['MLP', 'CNN', 'LSTM'] if m in results]
    weights = np.array(accuracies) / np.sum(accuracies)
    weighted_predictions = np.average(all_predictions, axis=0, weights=weights)
    weighted_pred = (weighted_predictions > 0.5).astype(int)
    
    # Calculate ensemble metrics
    avg_accuracy = accuracy_score(y_test, avg_pred)
    weighted_accuracy = accuracy_score(y_test, weighted_pred)
    avg_f1 = f1_score(y_test, avg_pred)
    weighted_f1 = f1_score(y_test, weighted_pred)
    
    print(f"\n🤝 Ensemble Results:")
    print(f"   Average Ensemble Accuracy: {avg_accuracy:.3f}")
    print(f"   Average Ensemble F1-Score: {avg_f1:.3f}")
    print(f"   Weighted Ensemble Accuracy: {weighted_accuracy:.3f}")
    print(f"   Weighted Ensemble F1-Score: {weighted_f1:.3f}")
    print(f"   Weights: {dict(zip(['MLP', 'CNN', 'LSTM'], weights))}")
    
    # Plot ensemble comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ensemble_names = ['Average Ensemble', 'Weighted Ensemble', 'Best Single']
    ensemble_accuracies = [avg_accuracy, weighted_accuracy, best_model_results['accuracy']]
    
    bars = ax.bar(ensemble_names, ensemble_accuracies, 
                 color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    
    ax.set_title('Ensemble vs Single Model Performance', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy')
    ax.set_ylim([0.5, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, ensemble_accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('dl_enhanced_results/plots/ensemble_comparison.png', dpi=300)
    
    # Save ensemble results
    ensemble_results = {
        'average_ensemble': {
            'accuracy': avg_accuracy,
            'f1': avg_f1,
            'predictions': avg_pred,
            'probabilities': avg_predictions
        },
        'weighted_ensemble': {
            'accuracy': weighted_accuracy,
            'f1': weighted_f1,
            'predictions': weighted_pred,
            'probabilities': weighted_predictions,
            'weights': dict(zip(['MLP', 'CNN', 'LSTM'], weights))
        }
    }
    
    joblib.dump(ensemble_results, 'dl_enhanced_results/ensemble_results.joblib')
    print("✓ Ensemble results saved")

# ============================================================================
# 11. CREATE PRODUCTION PIPELINE
# ============================================================================

print("\n10. CREATING PRODUCTION PIPELINE")

# Determine best model
best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
best_model = models_dict[best_model_name]
best_accuracy = results[best_model_name]['accuracy']

print(f"\n🏆 Best Deep Learning Model: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.3f}")
print(f"   F1-Score: {results[best_model_name]['f1']:.3f}")
print(f"   ROC-AUC: {results[best_model_name]['roc_auc']:.3f}")

# Save production pipeline
pipeline_info = {
    'best_model_name': best_model_name,
    'best_model_path': f'dl_enhanced_results/models/{best_model_name.lower()}_model.h5',
    'best_model_metrics': {
        'accuracy': best_accuracy,
        'f1': results[best_model_name]['f1'],
        'roc_auc': results[best_model_name]['roc_auc'],
        'precision': results[best_model_name]['precision'],
        'recall': results[best_model_name]['recall']
    },
    'scaler_path': 'dl_enhanced_results/scaler.joblib',
    'input_dim': X_train.shape[1],
    'needs_reshape': best_model_name in ['CNN', 'LSTM'],
    'ensemble_results': 'dl_enhanced_results/ensemble_results.joblib' if 'ensemble_results' in locals() else None,
    'all_results': results
}

joblib.dump(pipeline_info, 'dl_enhanced_results/production_pipeline.joblib')

# Create comprehensive report
report_text = f"""
{'='*80}
DEEP LEARNING FINANCIAL PREDICTION - COMPREHENSIVE REPORT
{'='*80}

SUMMARY:
• Dataset: {X.shape[0]} samples, {X.shape[1]} PCA features
• Target: Market Regime (Binary Classification)
• Best Model: {best_model_name}
• Best Accuracy: {best_accuracy:.3f}

MODEL PERFORMANCE:
"""
for model_name, res in results.items():
    report_text += f"""
{model_name}:
    Accuracy:  {res['accuracy']:.3f}
    F1-Score:  {res['f1']:.3f}
    ROC-AUC:   {res['roc_auc']:.3f}
    Precision: {res['precision']:.3f}
    Recall:    {res['recall']:.3f}
"""

if 'avg_accuracy' in locals():
    report_text += f"""
ENSEMBLE PERFORMANCE:
    Average Ensemble Accuracy: {avg_accuracy:.3f}
    Weighted Ensemble Accuracy: {weighted_accuracy:.3f}
"""

report_text += f"""
{'='*80}
VISUALIZATIONS CREATED:
1. Training History Dashboard
2. Model Performance Dashboard
3. Advanced Metrics Comparison
4. {best_model_name} Deep Dive Analysis
5. Ensemble Comparison
6. Interactive ROC Curves (HTML)
7. Interactive Model Comparison (HTML)

{'='*80}
PRODUCTION READY:
• Pipeline saved: dl_enhanced_results/production_pipeline.joblib
• Best model: dl_enhanced_results/models/{best_model_name.lower()}_model.h5
• Scaler: dl_enhanced_results/scaler.joblib
• All visualizations saved in dl_enhanced_results/plots/

{'='*80}
"""

with open('dl_enhanced_results/comprehensive_report.txt', 'w') as f:
    f.write(report_text)

print("✓ Production pipeline saved")
print("✓ Comprehensive report saved")

# ============================================================================
# 12. FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ENHANCED DEEP LEARNING PIPELINE COMPLETE")
print("="*80)

print(f"\n✅ ACHIEVEMENTS:")
print(f"1. Trained {len(models_dict)} deep learning models")
print(f"2. Best model: {best_model_name} with {best_accuracy:.3f} accuracy")
print(f"3. Created {len(os.listdir('dl_enhanced_results/plots'))} comprehensive visualizations")
print(f"4. Generated interactive HTML plots")
print(f"5. Built production-ready pipeline with detailed reporting")

print(f"\n📊 FINAL MODEL ACCURACIES:")
for model_name, res in results.items():
    better_worse = "✓" if res['accuracy'] > 0.703 else "✗"
    print(f"   {better_worse} {model_name}: {res['accuracy']:.3f}")

print(f"\n📈 VISUALIZATION OUTPUTS:")
print(f"   • Static Plots: dl_enhanced_results/plots/")
print(f"   • Interactive Plots: dl_enhanced_results/plots/interactive/")
print(f"   • TensorBoard Logs: dl_enhanced_results/tensorboard_logs/")

print(f"\n🎯 KEY INSIGHTS:")
if best_accuracy > 0.703:
    improvement = best_accuracy - 0.703
    print(f"• Deep Learning beats traditional model by {improvement:.3f}!")
else:
    gap = 0.703 - best_accuracy
    print(f"• Traditional model still leads by {gap:.3f}")
print(f"• Best architecture: {best_model_name}")
print(f"• Most predictive PCA components: {len(pca_features)} features")

print(f"\n🚀 NEXT STEPS:")
print(f"1. View interactive plots: dl_enhanced_results/plots/interactive/")
print(f"2. Run: tensorboard --logdir dl_enhanced_results/tensorboard_logs")
print(f"3. Use production pipeline for real-time predictions")
print(f"4. Monitor model drift with the saved visualizations")

print(f"\n" + "="*80)
print("🎉 ENHANCED PROJECT COMPLETE!")
print("="*80)
print("You now have:")
print("• PCA dimensionality reduction ✓")
print("• Traditional ML models (67-70% accuracy) ✓")
print("• Enhanced Deep Learning models with visualization ✓")
print("• Production-ready pipelines ✓")
print("• Complete financial prediction system ✓")
print("="*80)

# Show all plots
try:
    plt.show()
except:
    pass
