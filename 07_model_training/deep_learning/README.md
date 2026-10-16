# Deep Learning

Trains neural network models (MLP, CNN, LSTM) on PCA-transformed
financial features.

## Files

| File | Purpose |
| --- | --- |
| `deep_learning_pipeline.py` | Trains MLP, CNN, and LSTM models with plot saving (uses the `Agg` backend for headless execution) |

## Legacy

`legacy/` contains the iteration that led to the current pipeline:

- **v1** — first attempt, added neural networks on top of the existing pipeline
- **v2** — simplified because v1 was too complex
- **v3** — added comprehensive visualizations
- **v4** → renamed to `deep_learning_pipeline.py` (fixed headless execution)

## Inputs

PCA-transformed features from `07_model_training/pca_analysis/`.

## Outputs

- Model files: `15_model_storage/dl_models/` (gitignored)
- Training logs: `15_model_storage/dl_models/.../tensorboard_logs/` (gitignored)
- Metrics and plots: `13_experiments_tests/` and `20_images/`

## Note

The MLP model reports very high accuracy (92%) on the classification
task, which is unusual for financial time-series prediction. The root
README flags this as suspicious and recommends re-evaluation with proper
temporal splits before treating it as evidence of predictive skill.
