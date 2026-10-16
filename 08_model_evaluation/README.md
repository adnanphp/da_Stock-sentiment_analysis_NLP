# Model Evaluation

Evaluates the trained classification and regression models, checks for
common methodological issues in financial ML, and reports metrics.

## Files

| File | Purpose |
| --- | --- |
| `check_data_leakage.py` | Checks the pipeline for temporal leakage — overlapping samples, feature leakage, and improper train/test splitting. Uses `TimeSeriesSplit` to compute honest cross-validation. |
| `realistic_verification.py` | Produces conservative, honest metrics for the model evaluation, avoiding the inflated numbers that arise from random splits on time-series data. |

## Legacy

`legacy/` contains six iterations of a "compute classification metrics
from regression predictions" script. The final working version was used
for the metrics reported in `16_reports_results/classification_reports/`.

Files:
- `regression_to_classification_metrics.py` — first attempt
- `fixed_regression_to_classification_metrics.py` — fixes
- `enhanced_metrics_with_plots.py` — added plots
- `fixed_enhanced_metrics_with_plots.py` — fixes
- `final_working_metrics.py` — another attempt
- `final_perfect_metrics_with_plots.py` — last attempt (name is a leftover from debugging)

## Why `check_data_leakage.py` matters

The MLP model reports 92% accuracy, which is unusual for financial
time-series prediction. This file was written to test the pipeline for
the causes of inflated results:

- Overlapping train/test samples
- Features computed with knowledge of the future
- Random splits on time-series data

Any result from the pipeline should be re-verified with the temporal
splits used in this file before being treated as evidence of real
predictive skill.
