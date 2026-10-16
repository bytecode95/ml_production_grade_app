"""Train a housing-price model and save its artifact."""

import argparse
import logging
import os
from pathlib import Path

import pandas as pd

from price_prediction.domain.model_policy import ModelPolicy
from price_prediction.domain.schemas import FeatureSchema
from price_prediction.inference.artifact import save_artifact
from price_prediction.training.artifact_metadata import ArtifactMetadata
from price_prediction.training.workflow import run_training_workflow
from price_prediction.observability.logging import configure_logging


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to validated CSV data")
    parser.add_argument(
        "--output",
        default="artifacts/house_price_model.joblib",
    )
    parser.add_argument("--model-version", default="local-001")
    parser.add_argument("--dataset-version", default="local")
    parser.add_argument(
        "--code-version",
        default=os.getenv("GIT_COMMIT", "unknown"),
    )
    args = parser.parse_args()

    configure_logging()
    logger = logging.getLogger(__name__)

    df = pd.read_csv(args.data)
    schema = FeatureSchema()

    policy = ModelPolicy(
        max_mae=25000,
        max_rmse=40000,
        min_r2=0.70,
    )

    # These thresholds are illustrative. Calibrate them using your
    # validation results and business requirements.
    result = run_training_workflow(
        df=df,
        schema=schema,
        policy=policy,
    )

    metrics = result.validation_metrics.to_dict()

    metadata = ArtifactMetadata(
        model_name=result.selected_model_name,
        model_version=args.model_version,
        target_column=schema.target,
        numeric_features=tuple(schema.numeric_features),
        categorical_features=tuple(schema.categorical_features),
        metrics=metrics,
        target_transform="identity",
        dataset_version=args.dataset_version,
        code_version=args.code_version,
    )

    output_path = save_artifact(
        model=result.selected_model,
        metadata=metadata,
        path=Path(args.output),
    )

    logger.info("Selected model: %s", result.selected_model_name)
    logger.info("Validation metrics: %s", metrics)
    logger.info("Held-out test metrics: %s", result.test_metrics.to_dict())
    logger.info("Artifact saved: %s", output_path)


if __name__ == "__main__":
    main()