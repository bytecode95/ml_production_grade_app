"""Prediction service for saved housing-price models."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from price_prediction.inference.loader import load_artifact
from price_prediction.inference.target_transform import (
    inverse_transform_target,
)
from price_prediction.training.artifact_metadata import ArtifactMetadata

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionResult:
    predictions: list[float]
    target_column: str
    model_name: str
    model_version: str
    target_transform: str


class HousePricePredictor:
    """Run predictions using a previously saved, fitted pipeline."""

    def __init__(self, artifact_path: str) -> None:
        self.model, self.metadata = load_artifact(artifact_path)

        self.expected_features = (
            self.metadata.numeric_features
            + self.metadata.categorical_features
        )

    def predict(self, data: pd.DataFrame) -> PredictionResult:
        """Predict house prices for one or more records."""
        self._validate_input(data)

        try:
            # The fitted pipeline applies the same transformations
            # and preprocessing learned during training.
            raw_predictions = self.model.predict(
                data.loc[:, list(self.expected_features)]
            )

            predictions = inverse_transform_target(
                raw_predictions,
                method=self.metadata.target_transform,
            )
        except Exception:
            logger.exception("House-price prediction failed.")
            raise

        if not np.isfinite(predictions).all():
            raise ValueError("Model produced non-finite predictions.")

        result = PredictionResult(
            predictions=[float(value) for value in predictions],
            target_column=self.metadata.target_column,
            model_name=self.metadata.model_name,
            model_version=self.metadata.model_version,
            target_transform=self.metadata.target_transform,
        )

        logger.info(
            "Prediction completed: model=%s version=%s rows=%d",
            result.model_name,
            result.model_version,
            len(result.predictions),
        )
        return result

    def _validate_input(self, data: pd.DataFrame) -> None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        if data.empty:
            raise ValueError("Input data cannot be empty.")

        if not data.columns.is_unique:
            raise ValueError("Input contains duplicate column names.")

        missing = set(self.expected_features) - set(data.columns)
        if missing:
            raise ValueError(
                f"Missing required features: {sorted(missing)}"
            )

        # Extra columns are permitted but ignored. Only model features
        # are passed to the fitted pipeline.
        selected = data.loc[:, list(self.expected_features)]

        numeric = selected.loc[
            :, list(self.metadata.numeric_features)
        ] if self.metadata.numeric_features else pd.DataFrame(index=data.index)

        if not numeric.empty:
            converted = numeric.apply(pd.to_numeric, errors="coerce")
            invalid = numeric.notna() & converted.isna()
            if invalid.any().any():
                raise ValueError("Numeric features contain non-numeric values.")

            if np.isinf(converted.to_numpy(dtype=float)).any():
                raise ValueError("Numeric features cannot contain infinity.")

        categorical = selected.loc[
            :, list(self.metadata.categorical_features)
        ] if self.metadata.categorical_features else pd.DataFrame(index=data.index)

        # Missing categorical values are allowed here because the fitted
        # preprocessing pipeline handles them using its trained imputer.
        if not categorical.empty:
            for column in categorical.columns:
                non_missing = categorical[column].dropna()
                if not non_missing.map(
                    lambda value: isinstance(value, (str, int, float, bool))
                ).all():
                    raise ValueError(
                        f"Unsupported categorical values in {column!r}."
                    )