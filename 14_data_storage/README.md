# 14 — Data Storage

Storage for datasets at each stage of the pipeline.

## Contents

| Folder | Contains |
| --- | --- |
| `integrated_datasets/` | The final integrated dataset used for modeling |

## Gitignored

Large data files (`*.csv`, `*.parquet`) are gitignored. They are
regeneratable from the pipeline:
01_data_collection → 02_data_processing → 03_feature_engineering
→ 04_outlier_management → 05_data_scaling → 06_integration

text

## Why the folder still exists in git

`.gitkeep`-style tracking keeps the structure visible. The README
documents what should be here after the pipeline runs.
