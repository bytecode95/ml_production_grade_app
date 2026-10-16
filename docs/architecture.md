# Architecture

## Runtime flow

React/TypeScript -> FastAPI -> PredictionService -> Predictor -> fitted scikit-learn pipeline.

The API loads one configured trusted artifact. Training is not performed in a prediction request. Health checks are independent from readiness; readiness indicates whether the configured model is available.

## Training flow

CSV source -> validation -> split -> fit candidate/baseline on train -> evaluate on validation -> candidate gate -> final test metrics -> artifact and metadata.

Learned preprocessing is fit only on the training partition. The test partition is not used for model selection.

## Boundaries

- `data`: source loading and data checks.
- `features`: reusable transformations.
- `training`: model fitting, evaluation and artifact creation.
- `inference`: artifact loading and prediction.
- `services`: application use cases.
- `api`: HTTP contracts and error mapping.
- `pipelines`: orchestration adapters only.
- `frontend`: UI and typed HTTP client.

## Known limitations

The starter's generic validation cannot establish domain correctness for every dataset. Add dataset-specific ranges, category rules, target units and leakage review. The current candidate gate compares validation MAE with a DummyRegressor baseline; production thresholds require empirical calibration. Joblib artifacts must only be loaded from trusted storage.
