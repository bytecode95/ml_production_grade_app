
"""Train the selected model and create a loadable model artifact."""

import argparse
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from price_prediction.data.validation import validate_schema
from price_prediction.domain.model_policy import ModelPolicy
from price_prediction.domain.schemas import FeatureSchema
from price_prediction.training.artifact_metadata import ArtifactMetadata
from price_prediction.inference.artifact import save_artifact
from price_prediction.training.workflow import run_training_workflow

logger = logging.getLogger(__name__)

# File: src/price_prediction/training/run_training.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "AmesHousing.csv"
DEFAULT_ARTIFACT_PATH = (
    PROJECT_ROOT / "artifacts" / "house_price_model.joblib"
)


def get_code_version() -> str:
    """Return the current Git commit when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def train_and_save(
    data_path: Path,
    artifact_path: Path,
    policy: ModelPolicy,
) -> Path:
    """Validate data, train candidates, and save the selected pipeline."""

    logger.info("Loading dataset: %s", data_path)

    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {data_path}")

    df = pd.read_csv(data_path)

    # Validate the raw dataset before training.
    df = validate_schema(df)

    schema = FeatureSchema()

    # Ensure all configured training features exist.
    required_features = set(schema.all_features + [schema.target])
    missing = required_features - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing training columns: {sorted(missing)}"
        )

    # Keep only the configured features and target.
    # This avoids accidentally passing IDs or unrelated columns into training.
    df = df[schema.all_features + [schema.target]].copy()

    logger.info(
        "Training dataset ready: rows=%d, features=%d",
        len(df),
        len(schema.all_features),
    )

    result = run_training_workflow(
        df=df,
        schema=schema,
        policy=policy,
    )

    if not isinstance(result.selected_model, Pipeline):
        raise TypeError(
            "The selected model must be a fitted sklearn Pipeline."
        )

    # Use the actual model metrics produced by the workflow.
    metadata = ArtifactMetadata(
        model_name=result.selected_model_name,
        model_version=datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        ),
        target_column=schema.target,
        numeric_features=tuple(schema.numeric_features),
        categorical_features=tuple(schema.categorical_features),
        metrics={
            "baseline_mae": result.baseline_metrics.mae,
            "baseline_rmse": result.baseline_metrics.rmse,
            "baseline_r2": result.baseline_metrics.r2,
            "validation_mae": result.validation_metrics.mae,
            "validation_rmse": result.validation_metrics.rmse,
            "validation_r2": result.validation_metrics.r2,
            "test_mae": result.test_metrics.mae,
            "test_rmse": result.test_metrics.rmse,
            "test_r2": result.test_metrics.r2,
            "test_mape": result.test_metrics.mape,
        },
        dataset_version=data_path.name,
        code_version=get_code_version(),
    )

    saved_path = save_artifact(
        model=result.selected_model,
        metadata=metadata,
        path=artifact_path,
    )

    # Verify the saved artifact contract and model inference.
    payload = joblib.load(saved_path)

    if payload.get("format_version") != 1:
        raise ValueError("Unexpected artifact format version.")

    loaded_model = payload.get("model")
    loaded_metadata = payload.get("metadata")

    if not isinstance(loaded_model, Pipeline):
        raise TypeError("Saved artifact does not contain a Pipeline.")

    if not isinstance(loaded_metadata, dict):
        raise TypeError("Saved artifact metadata is invalid.")

    sample = df[schema.all_features].head(3)
    predictions = loaded_model.predict(sample)

    if len(predictions) != len(sample):
        raise RuntimeError("Artifact smoke test returned an invalid result.")

    logger.info("Artifact verification passed.")
    logger.info("Artifact path: %s", saved_path)
    logger.info("Selected model: %s", metadata.model_name)
    logger.info("Model version: %s", metadata.model_version)
    logger.info("Test metrics: %s", result.test_metrics.to_dict())
    logger.info("Sample predictions: %s", predictions.tolist())

    return saved_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and save the house-price prediction model."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the Ames Housing CSV.",
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
        help="Output path for the model artifact.",
    )
    parser.add_argument(
        "--max-mae",
        type=float,
        default=40000.0,
        help="Maximum acceptable validation MAE, in dataset currency units.",
    )
    parser.add_argument(
        "--max-rmse",
        type=float,
        default=60000.0,
        help="Maximum acceptable validation RMSE.",
    )
    parser.add_argument(
        "--min-r2",
        type=float,
        default=0.50,
        help="Minimum acceptable validation R-squared.",
    )
    args = parser.parse_args()

    # These are initial example gates, not calibrated production thresholds.
    policy = ModelPolicy(
        max_mae=args.max_mae,
        max_rmse=args.max_rmse,
        min_r2=args.min_r2,
    )

    saved_path = train_and_save(
        data_path=args.data.resolve(),
        artifact_path=args.artifact.resolve(),
        policy=policy,
    )

    print(f"\nTraining completed successfully.")
    print(f"Model artifact: {saved_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    main()
