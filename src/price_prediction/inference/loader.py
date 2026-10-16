"""Load and validate trusted model artifacts."""

import logging
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from price_prediction.training.artifact_metadata import ArtifactMetadata
from price_prediction.inference.artifact import ARTIFACT_FORMAT_VERSION

logger = logging.getLogger(__name__)


def load_artifact(
    path: str | Path,
) -> tuple[Pipeline, ArtifactMetadata]:
    """Load a trusted artifact and validate its contract."""
    artifact_path = Path(path)

    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Model artifact does not exist: {artifact_path}"
        )

    payload: Any = joblib.load(artifact_path)

    if not isinstance(payload, dict):
        raise ValueError("Invalid artifact: expected a dictionary.")

    if payload.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError("Unsupported artifact format version.")

    model = payload.get("model")
    raw_metadata = payload.get("metadata")

    if not isinstance(model, Pipeline):
        raise ValueError("Invalid artifact: model must be a Pipeline.")

    if not isinstance(raw_metadata, dict):
        raise ValueError("Invalid artifact: metadata is missing.")

    try:
        metadata = ArtifactMetadata(
            model_name=raw_metadata["model_name"],
            model_version=raw_metadata["model_version"],
            target_column=raw_metadata["target_column"],
            numeric_features=tuple(raw_metadata["numeric_features"]),
            categorical_features=tuple(raw_metadata["categorical_features"]),
            metrics=raw_metadata["metrics"],
            target_transform=raw_metadata["target_transform"],
            dataset_version=raw_metadata["dataset_version"],
            code_version=raw_metadata["code_version"],
            created_at=raw_metadata["created_at"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid artifact metadata.") from exc

    logger.info(
        "Model artifact loaded: path=%s version=%s",
        artifact_path,
        metadata.model_version,
    )
    return model, metadata