# PCA and t-SNE Analysis

Dimensionality reduction on the integrated dataset. Reduces 2,704
features to ~35 principal components while retaining ~95% of variance.

## Files

| File | Purpose |
| --- | --- |
| `pca_tsne_analysis.py` | Full PCA + t-SNE analysis with visualizations |
| `find_and_analyze_dataset.py` | Utility to explore candidate integrated datasets and report their shape, columns, and date ranges |

## Legacy

`legacy/` contains the iteration of the analysis:

- **v1** — basic analysis of the integrated dataset
- **v2** — comprehensive analysis expanded to all features
- **v2_fixed** — fixed issues in v2
- **v3_fixed** → renamed to `pca_tsne_analysis.py`

## Input

`14_data_storage/integrated_datasets/integrated_simple_*.csv`
(gitignored — regeneratable from the pipeline).

## Output

- Explained-variance plots: `20_images/1_pca_variance_analysis.png`
- 2D/3D t-SNE plots: `20_images/3_tsne_comprehensive_visualizations.png`
- Feature contribution analysis: `20_images/4_feature_analysis_visualizations.png`

## Results

| Metric | Value |
| --- | --- |
| Original features | 2,704 |
| PCA components kept | 35 |
| Variance retained | 95.1% |
| Dimensionality reduction | 98.0% |
