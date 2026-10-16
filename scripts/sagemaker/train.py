"""SageMaker training entry point; uses the project's fitted sklearn pipeline."""
import argparse
import json
import os
import sys
from pathlib import Path

# SageMaker uploads the repository as /opt/ml/code when source_dir=ROOT.
# Add the src-layout package to Python import resolution.
CODE_ROOT = Path("/opt/ml/code")
sys.path.insert(0, str(CODE_ROOT / "src"))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge

from price_prediction.domain.schemas import FeatureSchema
from price_prediction.features.transformations import build_sklearn_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-column", default="SalePrice")
    parser.add_argument("--model-name", choices=["random_forest", "extra_trees", "ridge"], default="random_forest")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=15)
    args, _ = parser.parse_known_args()
    folder = Path("/opt/ml/input/data/train")
    csvs = list(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No training CSV in {folder}")
    df = pd.read_csv(csvs[0])
    if args.target_column not in df or df[args.target_column].isna().any():
        raise ValueError("Training data must contain a non-null target column.")
    schema = FeatureSchema(target=args.target_column)
    X, y = df.drop(columns=[args.target_column]), df[args.target_column]
    if args.model_name == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            random_state=args.random_state, n_jobs=-1,
        )
    elif args.model_name == "extra_trees":
        estimator = ExtraTreesRegressor(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            random_state=args.random_state, n_jobs=-1,
        )
    else:
        estimator = Ridge()
    pipeline = build_sklearn_pipeline(schema, estimator, reference_year=2026)
    pipeline.fit(X, y)
    model_dir = Path(os.getenv("SM_MODEL_DIR", "/opt/ml/model"))
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "model.joblib")
    (model_dir / "training_metadata.json").write_text(json.dumps({
        "model_name": args.model_name,
        "target_column": args.target_column,
        "numeric_features": schema.numeric_features,
        "categorical_features": schema.categorical_features,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
