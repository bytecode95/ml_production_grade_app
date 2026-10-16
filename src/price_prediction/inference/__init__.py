"""Model artifact loading and prediction interfaces."""

from price_prediction.inference.predictor import (
    HousePricePredictor,
    PredictionResult,
)

__all__ = ["HousePricePredictor", "PredictionResult"]