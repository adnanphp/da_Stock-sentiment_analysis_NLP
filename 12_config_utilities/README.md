# 12 — Configuration and Utilities

Shared configuration and helper modules used across the pipeline.

## Files

| File | Purpose |
| --- | --- |
| `config.py` | Main configuration |
| `datetime_config.py` | Date/time conventions shared across stages |
| `feature_config.py` | Feature-generation settings |
| `scaling_config.py` | Scaling settings |
| `fix_specific_columns.py` | One-off column-fixing utility |

## Note

Configuration that differs per stage lives with that stage
(e.g., `03_feature_engineering/feature_config.py`). This folder holds
only what is shared.
