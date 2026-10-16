
05 — Data Scaling and Normalization
Scales features to comparable ranges so that distance-based methods
(PCA, t-SNE, neural networks) work correctly. Without this stage, a
feature in the thousands (e.g., price) dominates a feature in the
single digits (e.g., a sentiment score).

Files
File	Purpose
data_scaler.py	StandardScaler, MinMaxScaler, RobustScaler
scaler_selector.py	Chooses a scaler per feature based on distribution
scaling_config.py	Configuration
normalization_pipeline.py	End-to-end pipeline
scale_inverter.py	Inverse-transforms predictions back to original scale
Why scale_inverter.py
Predictions are made in scaled space. To interpret them (e.g., "predicted
next-day return of 1.2%"), we need to invert the scaling. This file keeps
that transformation explicit.

Inputs
Outlier-treated features from 04_outlier_management/.

Output
Scaled features, ready for integration and modeling.

Stage boundary
Input: outlier-treated features.
Output: scaled features.
