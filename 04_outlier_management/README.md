
04 — Outlier Management
Detects and treats outliers in the engineered feature space. Financial
data has heavy tails — extreme price moves, sentiment spikes, volume
anomalies — so treating them carefully matters.

Files
File	Purpose
outlier_config.py	Configuration — which method per feature
outlier_detector.py	IQR, Z-score detection
outlier_treatment.py	Winsorization and other treatments
outlier_manager.py	Orchestrates detect-then-treat per column
outlier_pipeline.py	End-to-end pipeline
fixed_outlier_*.py	Corrected versions with additional handling
Methods
Method	When to use
IQR	Robust to skewed distributions
Z-score	Standard for near-normal distributions
Winsorization	Cap extremes rather than delete
Inputs
Feature-engineered data from 03_feature_engineering/.

Output
Feature data with outliers treated, ready for scaling.

Stage boundary
Input: engineered features.
Output: outlier-treated features.
