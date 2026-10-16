#!/usr/bin/env python3
"""
PCA and t-SNE Dimensionality Reduction for Financial Data
Dataset: integrated_simple_20251202_213953.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the dataset
print("="*70)
print(f"LOADING DATASET: ./integrated_datasets_simple/integrated_simple_20251202_213953.csv")
print("="*70)

df = pd.read_csv("./integrated_datasets_simple/integrated_simple_20251202_213953.csv")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {len(df.columns)}")
print(f"Rows: {len(df)}")
print(f"\nColumn dtypes:")
print(df.dtypes.value_counts())

# Display basic info
print("\n" + "="*70)
print("DATASET INFORMATION")
print("="*70)
print(f"Memory usage: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

if 'ticker' in df.columns:
    print(f"\nTickers in dataset:")
    print(df['ticker'].value_counts())
    
    # Filter for AAPL, GOOGL, TSLA if they exist
    target_tickers = ['AAPL', 'GOOGL', 'TSLA', 'GOOG']
    available_tickers = [t for t in target_tickers if t in df['ticker'].unique()]
    
    if available_tickers:
        print(f"\nFocusing on tickers: {available_tickers}")
        df_filtered = df[df['ticker'].isin(available_tickers)].copy()
    else:
        print("Target tickers not found, using all data")
        df_filtered = df.copy()
else:
    print("No 'ticker' column found, using all data")
    df_filtered = df.copy()

print(f"\nFiltered dataset shape: {df_filtered.shape}")

# Identify numeric columns for PCA
print("\n" + "="*70)
print("IDENTIFYING NUMERIC FEATURES FOR PCA")
print("="*70)

numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
print(f"Found {len(numeric_cols)} numeric columns")

# Remove potential target columns and low-variance columns
potential_targets = ['Close', 'close', 'price', 'target', 'label', 'return', 'next_return']
feature_cols = [col for col in numeric_cols if not any(target in col.lower() for target in [t.lower() for t in potential_targets])]

# Also remove columns with too many missing values
missing_ratios = df_filtered[feature_cols].isnull().mean()
feature_cols = [col for col in feature_cols if missing_ratios[col] < 0.5]

print(f"Selected {len(feature_cols)} features for PCA after filtering")

# Prepare feature matrix
X = df_filtered[feature_cols].fillna(df_filtered[feature_cols].mean()).values
print(f"Feature matrix shape: {X.shape}")

# Scale features
print("\n" + "="*70)
print("SCALING FEATURES")
print("="*70)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Features scaled using StandardScaler")

# ========== PCA ANALYSIS ==========
print("\n" + "="*70)
print("PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("="*70)

# Determine optimal number of components
pca_full = PCA().fit(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Find components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
n_components_90 = np.argmax(cumulative_variance >= 0.90) + 1

print(f"Original dimensions: {X.shape[1]}")
print(f"Components for 90% variance: {n_components_90}")
print(f"Components for 95% variance: {n_components_95}")
print(f"Components for 99% variance: {np.argmax(cumulative_variance >= 0.99) + 1}")

# Apply PCA with optimal components
pca = PCA(n_components=n_components_95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA reduced dimensions to: {X_pca.shape[1]}")
print(f"Variance explained: {np.sum(pca.explained_variance_ratio_):.3f}")

# ========== t-SNE ANALYSIS ==========
print("\n" + "="*70)
print("t-SNE DIMENSIONALITY REDUCTION")
print("="*70)

# Use PCA output for t-SNE (faster and often better for high-dim data)
print("Applying t-SNE on PCA-reduced data...")

# For t-SNE, we can use different perplexity values
perplexities = [5, 30, 50]
tsne_results = {}

for perplexity in perplexities:
    print(f"  Running t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, 
                perplexity=perplexity,
                random_state=42,
                n_iter=1000,
                learning_rate=200)
    
    # Use first 1000 samples for t-SNE if dataset is large
    if X_pca.shape[0] > 1000:
        sample_indices = np.random.choice(X_pca.shape[0], 1000, replace=False)
        X_sample = X_pca[sample_indices]
        tsne_result = tsne.fit_transform(X_sample)
        tsne_results[perplexity] = (tsne_result, sample_indices)
    else:
        tsne_result = tsne.fit_transform(X_pca)
        tsne_results[perplexity] = (tsne_result, np.arange(X_pca.shape[0]))
    
    print(f"    t-SNE completed with perplexity={perplexity}")

# ========== VISUALIZATION ==========
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# Create visualization directory
import os
os.makedirs("pca_tsne_results", exist_ok=True)

# 1. PCA Scree Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
        pca.explained_variance_ratio_, 
        alpha=0.5, align='center')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Scree Plot')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), 
         cumulative_variance, 'ro-', linewidth=2)
plt.axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
plt.axhline(y=0.90, color='b', linestyle='--', label='90% threshold')
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_tsne_results/pca_variance_analysis.png', dpi=300, bbox_inches='tight')

# 2. PCA Loadings Heatmap (Top 20 features)
plt.figure(figsize=(14, 8))
n_top_features = min(20, len(feature_cols))
n_top_components = min(10, pca.n_components_)

# Get feature loadings for top components
loadings = pca.components_[:n_top_components, :n_top_features]

sns.heatmap(loadings, 
           xticklabels=feature_cols[:n_top_features],
           yticklabels=[f'PC{i+1}' for i in range(n_top_components)],
           cmap='RdBu_r', center=0,
           cbar_kws={'label': 'Loading Coefficient'})
plt.title(f'PCA Component Loadings (Top {n_top_features} Features)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('pca_tsne_results/pca_loadings_heatmap.png', dpi=300, bbox_inches='tight')

# 3. t-SNE visualizations
fig, axes = plt.subplots(1, len(perplexities), figsize=(18, 5))

if len(perplexities) == 1:
    axes = [axes]

for idx, (perplexity, (tsne_data, sample_idx)) in enumerate(tsne_results.items()):
    ax = axes[idx]
    
    # Color by ticker if available
    if 'ticker' in df_filtered.columns:
        sample_tickers = df_filtered.iloc[sample_idx]['ticker'] if len(sample_idx) < len(df_filtered) else df_filtered['ticker']
        
        # Encode tickers as colors
        le = LabelEncoder()
        ticker_encoded = le.fit_transform(sample_tickers)
        
        scatter = ax.scatter(tsne_data[:, 0], tsne_data[:, 1], 
                           c=ticker_encoded, cmap='tab10', 
                           alpha=0.6, s=30)
        
        # Create legend
        handles = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=plt.cm.tab10(i/len(le.classes_)), 
                             markersize=10, label=label)
                  for i, label in enumerate(le.classes_)]
        ax.legend(handles=handles, title='Ticker', fontsize=8)
        
    else:
        ax.scatter(tsne_data[:, 0], tsne_data[:, 1], alpha=0.6, s=30)
    
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_title(f't-SNE (perplexity={perplexity})')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_tsne_results/tsne_visualizations.png', dpi=300, bbox_inches='tight')

# 4. 3D PCA plot (if we have enough components)
if X_pca.shape[1] >= 3:
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if 'ticker' in df_filtered.columns:
        le = LabelEncoder()
        ticker_encoded = le.fit_transform(df_filtered['ticker'])
        scatter = ax.scatter(X_pca[:1000, 0], X_pca[:1000, 1], X_pca[:1000, 2], 
                           c=ticker_encoded[:1000], cmap='tab10', alpha=0.6, s=20)
    else:
        ax.scatter(X_pca[:1000, 0], X_pca[:1000, 1], X_pca[:1000, 2], alpha=0.6, s=20)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title('3D PCA Visualization')
    plt.savefig('pca_tsne_results/pca_3d_visualization.png', dpi=300, bbox_inches='tight')

# 5. Feature importance from PCA
plt.figure(figsize=(12, 6))

# Feature importance based on absolute loadings of PC1
pc1_loadings = np.abs(pca.components_[0])
top_n = min(15, len(feature_cols))
top_indices = np.argsort(pc1_loadings)[-top_n:][::-1]

plt.subplot(1, 2, 1)
plt.barh(range(top_n), pc1_loadings[top_indices])
plt.yticks(range(top_n), [feature_cols[i] for i in top_indices])
plt.xlabel('Absolute Loading on PC1')
plt.title('Top Features by PC1 Loading')
plt.gca().invert_yaxis()

# Feature importance based on variance across all PCs
total_importance = np.sum(pca.components_**2, axis=0)
top_indices_var = np.argsort(total_importance)[-top_n:][::-1]

plt.subplot(1, 2, 2)
plt.barh(range(top_n), total_importance[top_indices_var])
plt.yticks(range(top_n), [feature_cols[i] for i in top_indices_var])
plt.xlabel('Total Importance (Sum of squared loadings)')
plt.title('Top Features by Total PCA Importance')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig('pca_tsne_results/feature_importance_pca.png', dpi=300, bbox_inches='tight')

# ========== SAVE RESULTS ==========
print("\n" + "="*70)
print("SAVING RESULTS")
print("="*70)

# Save PCA-transformed data
df_pca = pd.DataFrame(X_pca, columns=[f'PC_{i+1}' for i in range(X_pca.shape[1])])

# Add original identifiers if available
if 'ticker' in df_filtered.columns:
    df_pca['ticker'] = df_filtered['ticker'].values

date_col = next((col for col in df_filtered.columns if 'date' in col.lower()), None)
if date_col:
    df_pca['date'] = df_filtered[date_col].values

df_pca.to_csv('pca_tsne_results/pca_transformed_data.csv', index=False)
print("✓ PCA-transformed data saved to: pca_tsne_results/pca_transformed_data.csv")

# Save t-SNE data
for perplexity, (tsne_data, sample_idx) in tsne_results.items():
    df_tsne = pd.DataFrame(tsne_data, columns=[f'TSNE1_perplexity{perplexity}', f'TSNE2_perplexity{perplexity}'])
    
    if 'ticker' in df_filtered.columns:
        df_tsne['ticker'] = df_filtered.iloc[sample_idx]['ticker'].values if len(sample_idx) < len(df_filtered) else df_filtered['ticker'].values
    
    df_tsne.to_csv(f'pca_tsne_results/tsne_perplexity{perplexity}_data.csv', index=False)
    print(f"✓ t-SNE data (perplexity={perplexity}) saved")

# Save PCA model and scaler
import joblib
joblib.dump({'pca': pca, 'scaler': scaler, 'feature_names': feature_cols}, 
            'pca_tsne_results/pca_model.joblib')
print("✓ PCA model saved to: pca_tsne_results/pca_model.joblib")

# Create summary report
with open('pca_tsne_results/summary_report.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("PCA & t-SNE DIMENSIONALITY REDUCTION REPORT\n")
    f.write("="*70 + "\n\n")
    f.write(f"Dataset: {dataset_path}\n")
    f.write(f"Original dimensions: {X.shape[1]}\n")
    f.write(f"PCA reduced to: {X_pca.shape[1]} components\n")
    f.write(f"Variance explained: {np.sum(pca.explained_variance_ratio_):.3f}\n\n")
    
    f.write("PCA Explained Variance by Component:\n")
    for i, var in enumerate(pca.explained_variance_ratio_, 1):
        f.write(f"  PC{i}: {var:.4f}\n")
    
    f.write("\nTop 10 Most Important Features:\n")
    pc1_loadings = np.abs(pca.components_[0])
    top_indices = np.argsort(pc1_loadings)[-10:][::-1]
    for idx in top_indices:
        f.write(f"  {feature_cols[idx]}: {pc1_loadings[idx]:.4f}\n")

print("✓ Summary report saved to: pca_tsne_results/summary_report.txt")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)
print("\nAll results saved in 'pca_tsne_results/' directory:")
print("  - PCA variance analysis plots")
print("  - t-SNE visualizations")
print("  - PCA-transformed data (CSV)")
print("  - t-SNE transformed data (CSV)")
print("  - PCA model (joblib)")
print("  - Summary report")

# Show some plots
plt.show()
