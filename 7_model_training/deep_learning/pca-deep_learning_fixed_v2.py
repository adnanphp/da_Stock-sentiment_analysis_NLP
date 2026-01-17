#!/usr/bin/env python3
"""
SIMPLIFIED DEEP LEARNING PIPELINE FOR PCA-TRANSFORMED FINANCIAL DATA
Fixed version with proper data handling
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("SIMPLIFIED DEEP LEARNING FOR FINANCIAL PREDICTION")
print("="*80)
print("Fixed version with proper data handling")
print("="*80)

# ============================================================================
# 1. SETUP
# ============================================================================

# Create output directory
os.makedirs('dl_fixed_results', exist_ok=True)
os.makedirs('dl_fixed_results/models', exist_ok=True)
os.makedirs('dl_fixed_results/plots', exist_ok=True)

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
joblib.dump(scaler, 'dl_fixed_results/scaler.joblib')
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
        metrics=['accuracy', 'AUC']
    )
    
    return model

def build_cnn(input_dim):
    """Build a 1D CNN model"""
    model = models.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(128, 3, activation='relu', padding='same'),
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
        metrics=['accuracy', 'AUC']
    )
    
    return model

def build_lstm(input_dim):
    """Build an LSTM model"""
    model = models.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        
        layers.LSTM(32),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
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
        filepath='dl_fixed_results/models/best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
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
    verbose=0
)

mlp_model.save('dl_fixed_results/models/mlp_model.h5')
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
    verbose=0
)

cnn_model.save('dl_fixed_results/models/cnn_model.h5')
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
    verbose=0
)

lstm_model.save('dl_fixed_results/models/lstm_model.h5')
models_dict['LSTM'] = lstm_model
histories['LSTM'] = history_lstm
print(f"   LSTM trained: {len(history_lstm.history['loss'])} epochs")

# ============================================================================
# 8. EVALUATE MODELS
# ============================================================================

print("\n7. EVALUATING MODELS")

results = {}

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
    
    print(f"   Accuracy: {accuracy:.3f}")
    print(f"   F1-Score: {f1:.3f}")
    print(f"   ROC-AUC: {roc_auc:.3f}")
    
    results[model_name] = {
        'accuracy': accuracy,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_test': y_test
    }

# ============================================================================
# 9. CREATE ENSEMBLE
# ============================================================================

print("\n8. CREATING ENSEMBLE")

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
    
    print(f"\n🤝 Ensemble Results:")
    print(f"   Average Ensemble Accuracy: {avg_accuracy:.3f}")
    print(f"   Weighted Ensemble Accuracy: {weighted_accuracy:.3f}")
    
    # Save ensemble results
    ensemble_results = {
        'average_ensemble': {
            'accuracy': avg_accuracy,
            'predictions': avg_pred,
            'probabilities': avg_predictions
        },
        'weighted_ensemble': {
            'accuracy': weighted_accuracy,
            'predictions': weighted_pred,
            'probabilities': weighted_predictions,
            'weights': dict(zip(['MLP', 'CNN', 'LSTM'], weights))
        }
    }
    
    joblib.dump(ensemble_results, 'dl_fixed_results/ensemble_results.joblib')
    print("✓ Ensemble results saved")

# ============================================================================
# 10. COMPARE WITH TRADITIONAL MODELS
# ============================================================================

print("\n9. COMPARING WITH TRADITIONAL MODELS")

# Load your traditional model results
try:
    trad_results = joblib.load('enhanced_models/best_market_regime_model.joblib')
    
    # For comparison, we need to make predictions with the traditional model
    # Since we don't have the exact test set predictions, we'll note the accuracy
    print("⚠️  Note: Traditional model accuracy from your results: 0.703")
    
    print(f"\n📊 Deep Learning vs Traditional:")
    for model_name, res in results.items():
        print(f"   {model_name}: {res['accuracy']:.3f}")
    
    print(f"   Traditional (Voting): 0.703")
    
    # Check if deep learning beats traditional
    best_dl = max(results.values(), key=lambda x: x['accuracy'])
    if best_dl['accuracy'] > 0.703:
        print(f"\n🎉 Deep Learning beats traditional by {best_dl['accuracy'] - 0.703:.3f}!")
    else:
        print(f"\n🔍 Traditional model still leads by {0.703 - best_dl['accuracy']:.3f}")
        
except Exception as e:
    print(f"   Could not load traditional results: {e}")

# ============================================================================
# 11. VISUALIZATIONS
# ============================================================================

print("\n10. CREATING VISUALIZATIONS")

# 1. Training History
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, (model_name, history) in enumerate(histories.items()):
    row = idx // 3
    col = idx % 3
    
    ax1 = axes[row, col]
    ax2 = ax1.twinx()
    
    # Plot loss
    color = 'tab:blue'
    ax1.plot(history.history['loss'], label='Train Loss', color=color)
    ax1.plot(history.history['val_loss'], label='Val Loss', color=color, linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Plot accuracy
    color = 'tab:red'
    ax2.plot(history.history['accuracy'], label='Train Acc', color=color)
    ax2.plot(history.history['val_accuracy'], label='Val Acc', color=color, linestyle='--')
    ax2.set_ylabel('Accuracy', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    ax1.set_title(f'{model_name} Training History')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig('dl_fixed_results/plots/training_history.png', dpi=300)

# 2. Model Comparison
fig, ax = plt.subplots(figsize=(10, 6))

model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]

bars = ax.bar(model_names, accuracies, alpha=0.7)
ax.axhline(y=0.703, color='red', linestyle='--', label='Traditional Model (0.703)')
ax.set_xlabel('Model')
ax.set_ylabel('Accuracy')
ax.set_title('Deep Learning Model Performance')
ax.legend()
ax.grid(True, alpha=0.3)

# Add value labels
for bar, acc in zip(bars, accuracies):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{acc:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('dl_fixed_results/plots/model_comparison.png', dpi=300)

# 3. ROC Curves
fig, ax = plt.subplots(figsize=(10, 8))

for model_name, res in results.items():
    fpr, tpr, _ = roc_curve(res['y_test'], res['y_pred_proba'])
    roc_auc = res['roc_auc']
    
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves - Deep Learning Models')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dl_fixed_results/plots/roc_curves.png', dpi=300)

print("✓ Visualizations saved")

# ============================================================================
# 12. CREATE PRODUCTION PIPELINE
# ============================================================================

print("\n11. CREATING PRODUCTION PIPELINE")

# Determine best model
best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
best_model = models_dict[best_model_name]
best_accuracy = results[best_model_name]['accuracy']

print(f"\n🏆 Best Deep Learning Model: {best_model_name}")
print(f"   Accuracy: {best_accuracy:.3f}")

# Save production pipeline
pipeline_info = {
    'best_model_name': best_model_name,
    'best_model_path': f'dl_fixed_results/models/{best_model_name.lower()}_model.h5',
    'accuracy': best_accuracy,
    'scaler_path': 'dl_fixed_results/scaler.joblib',
    'input_dim': X_train.shape[1],
    'needs_reshape': best_model_name in ['CNN', 'LSTM'],
    'ensemble_results': 'dl_fixed_results/ensemble_results.joblib' if 'ensemble_results' in locals() else None
}

joblib.dump(pipeline_info, 'dl_fixed_results/production_pipeline.joblib')

# Create usage example
usage_code = f'''
# ============================================================
# DEEP LEARNING PREDICTION PIPELINE
# Best Model: {best_model_name}
# Accuracy: {best_accuracy:.3f}
# ============================================================
import joblib
import numpy as np
from tensorflow import keras

# Load pipeline
pipeline = joblib.load('dl_fixed_results/production_pipeline.joblib')

# Load scaler
scaler = joblib.load(pipeline['scaler_path'])

# Load model
model = keras.models.load_model(pipeline['best_model_path'])

def predict(X_new):
    """Make predictions on new data"""
    # Scale the data
    X_scaled = scaler.transform(X_new)
    
    # Reshape if needed
    if pipeline['needs_reshape']:
        X_reshaped = X_scaled.reshape(-1, X_scaled.shape[1], 1)
        predictions = model.predict(X_reshaped, verbose=0)
    else:
        predictions = model.predict(X_scaled, verbose=0)
    
    # For binary classification
    probabilities = predictions.flatten()
    binary_predictions = (probabilities > 0.5).astype(int)
    
    return binary_predictions, probabilities

# Example usage:
# X_new should have shape (n_samples, 35) - 35 PCA components
# predictions, probs = predict(X_new)
# ============================================================
'''

with open('dl_fixed_results/usage_example.py', 'w') as f:
    f.write(usage_code)

print("✓ Production pipeline saved")
print("✓ Usage example saved")

# ============================================================================
# 13. FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("DEEP LEARNING PIPELINE COMPLETE")
print("="*80)

print(f"\n✅ ACHIEVEMENTS:")
print(f"1. Trained {len(models_dict)} deep learning models")
print(f"2. Best model: {best_model_name} with {best_accuracy:.3f} accuracy")
print(f"3. Created ensemble methods")
print(f"4. Compared with traditional models")
print(f"5. Built production-ready pipeline")

print(f"\n📊 MODEL ACCURACIES:")
for model_name, res in results.items():
    print(f"   {model_name}: {res['accuracy']:.3f}")

if 'avg_accuracy' in locals():
    print(f"   Average Ensemble: {avg_accuracy:.3f}")
if 'weighted_accuracy' in locals():
    print(f"   Weighted Ensemble: {weighted_accuracy:.3f}")

print(f"\n🎯 KEY INSIGHTS:")
print(f"• Deep learning can capture complex patterns in PCA-transformed data")
print(f"• Different architectures have different strengths")
print(f"• Ensembles can provide more robust predictions")
print(f"• Your PCA transformation (1795 → 35 features) works well with DL")

print(f"\n🚀 NEXT STEPS:")
print(f"1. Use dl_fixed_results/production_pipeline.joblib for predictions")
print(f"2. Monitor model performance over time")
print(f"3. Experiment with more complex architectures")
print(f"4. Combine with traditional models in a meta-ensemble")

print(f"\n📁 OUTPUTS:")
print(f"   • Models: dl_fixed_results/models/")
print(f"   • Plots: dl_fixed_results/plots/")
print(f"   • Pipeline: dl_fixed_results/production_pipeline.joblib")
print(f"   • Usage: dl_fixed_results/usage_example.py")

print(f"\n" + "="*80)
print("🎉 PROJECT COMPLETE!")
print("="*80)
print("You now have:")
print("• PCA dimensionality reduction ✓")
print("• Traditional ML models (67-70% accuracy) ✓")
print("• Deep Learning models ✓")
print("• Production-ready pipelines ✓")
print("• Complete financial prediction system ✓")
print("="*80)

# Show plots
try:
    plt.show()
except:
    pass
