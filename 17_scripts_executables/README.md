# 17 — Scripts and Executables

Top-level entry points that orchestrate multi-stage runs.

## Files

| File | Purpose |
| --- | --- |
| `main.py` | Runs the full pipeline |
| `run_data_quality_assessment.py` | Runs data-quality checks in isolation |
| `run_feature_engineering.py` | Runs feature engineering in isolation |
| `run_missing_values_pipeline.py` | Runs missing-value handling in isolation |

## Usage

```bash
python main.py                    # full pipeline
python run_feature_engineering.py # single stage
Note
Individual stages also have their own runnable scripts at the top of
their folders. These entry points are conveniences for running the
whole thing or a specific slice.
