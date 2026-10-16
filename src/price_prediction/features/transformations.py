
"""Deterministic feature engineering for housing-price prediction."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from price_prediction.domain.schemas import FeatureSchema
from price_prediction.features.preprocessing import build_preprocessor


class HouseAgeFeature(BaseEstimator, TransformerMixin):
    """Add derived housing features without modifying the input DataFrame."""

    def __init__(self, reference_year: int = 2026) -> None:
        self.reference_year = reference_year

    def fit(self, X: pd.DataFrame, y=None):
        """No data-dependent parameters need to be learned."""
        self._validate_input(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the input DataFrame with engineered features added."""
        self._validate_input(X)

        X = X.copy()

        X["HouseAge"] = self.reference_year - X["Year Built"]

        X["TotalSF"] = (
            X["Gr Liv Area"] + X["Total Bsmt SF"].fillna(0)
        )

        X["QualitySF"] = X["Overall Qual"] * X["Gr Liv Area"]

        lot_area = X["Lot Area"]
        if (lot_area.dropna() < 0).any():
            raise ValueError("Lot Area cannot contain negative values.")

        X["LogLotArea"] = np.log1p(lot_area)

        return X

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        required_columns = {
            "Year Built",
            "Gr Liv Area",
            "Total Bsmt SF",
            "Overall Qual",
            "Lot Area",
        }
        missing = required_columns - set(X.columns)

        if missing:
            raise ValueError(
                f"Missing columns required for feature engineering: "
                f"{sorted(missing)}"
            )


def build_sklearn_pipeline(
    schema: FeatureSchema,
    model,
    reference_year: int = 2026,
) -> Pipeline:
    """Compose feature engineering, preprocessing and model estimation."""
    return Pipeline(
        [
            (
                "features",
                HouseAgeFeature(reference_year=reference_year),
            ),
            ("prep", build_preprocessor(schema)),
            ("model", model),
        ]
    )
