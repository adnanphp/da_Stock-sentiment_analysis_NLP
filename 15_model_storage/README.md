# 15 — Model Storage

Trained model artifacts and training logs.

## Contents

| Folder | Contains |
| --- | --- |
| `dl_models/` | Deep learning models (`.h5`), tensorboard logs, checkpoints |

## Gitignored

Model files (`.h5`, `.pkl`, `.joblib`, `.pt`, `.pth`) and tensorboard
logs (`events.out.tfevents.*`) are gitignored. They are regeneratable
by rerunning the training scripts.

## Related

Training code lives in `07_model_training/`. Metrics and evaluation
plots live in `13_experiments_tests/` and `16_reports_results/`.
