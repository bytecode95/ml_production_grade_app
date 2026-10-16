"""Application-level house-price prediction service."""

import logging
from typing import Any

import pandas as pd

from price_prediction.inference.predictor import HousePricePredictor

logger = logging.getLogger(__name__)


class PredictionService:
    """Coordinate request data and the loaded inference model."""

    def __init__(self, predictor: HousePricePredictor) -> None:
        self._predictor = predictor

    def predict(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Predict prices for validated feature records."""
        if not records:
            raise ValueError("At least one house record is required.")

        dataframe = pd.DataFrame.from_records(records)
        result = self._predictor.predict(dataframe)

        return {
            "predictions": [
                {"predicted_price": price}
                for price in result.predictions
            ],
            "model_name": result.model_name,
            "model_version": result.model_version,
            "target": result.target_column,
        }

    def get_model_info(self) -> dict[str, Any]:
        """Return safe, non-sensitive model metadata."""
        metadata = self._predictor.metadata

        return {
            "model_name": metadata.model_name,
            "model_version": metadata.model_version,
            "target_column": metadata.target_column,
            "numeric_features": list(metadata.numeric_features),
            "categorical_features": list(
                metadata.categorical_features
            ),
            "target_transform": metadata.target_transform,
            "created_at": metadata.created_at,
        }