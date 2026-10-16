"""Save and loadable contract for model artifacts."""

import logging
import os
from pathlib import Path
import tempfile
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from price_prediction.training.artifact_metadata import ArtifactMetadata

logger = logging.getLogger(__name__)

ARTIFACT_FORMAT_VERSION = 1


def save_artifact(
    model: Pipeline,
    metadata: ArtifactMetadata,
    path: str | Path,
) -> Path:
    """Persist a fitted pipeline and metadata."""
    if not isinstance(model, Pipeline):
        raise TypeError("model must be a fitted sklearn Pipeline.")

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "model": model,
        "metadata": metadata.to_dict(),
    }

    # Write to a temporary file first so an interrupted write does not
    # leave a partially written artifact at the destination.
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)

        joblib.dump(payload, temp_path)
        os.replace(temp_path, destination)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        logger.exception("Failed to save model artifact: %s", destination)
        raise

    logger.info(
        "Model artifact saved: path=%s version=%s",
        destination,
        metadata.model_version,
    )
    return destination