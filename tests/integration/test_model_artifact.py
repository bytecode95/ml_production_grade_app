"""Integration tests for model artifact persistence."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from price_prediction.domain.schemas import FeatureSchema
from price_prediction.features.transformations import build_sklearn_pipeline
from price_prediction.inference.artifact import save_artifact
from price_prediction.inference.predictor import HousePricePredictor
from price_prediction.training.artifact_metadata import ArtifactMetadata


@pytest.fixture
def schema() -> FeatureSchema:
    return FeatureSchema(
        target="SalePrice",
        numeric_features=["Gr Liv Area", "Overall Qual"],
        categorical_features=["Neighborhood"],
    )


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "Gr Liv Area": [1000, 1200, 1500, 1800, 2000, 2200],
            "Overall Qual": [4, 5, 6, 7, 8, 9],
            "Neighborhood": ["A", "A", "B", "B", "C", "C"],
        }
    )
    y = pd.Series([100000, 125000, 160000, 195000, 230000, 270000])
    return X, y


def test_saved_artifact_preserves_predictions(
    tmp_path,
    schema,
    training_data,
):
    X_train, y_train = training_data

    pipeline = build_sklearn_pipeline(
        schema=schema,
        model=Ridge(),
        reference_year=2026,
    )
    pipeline.fit(X_train, y_train)

    sample = pd.DataFrame(
        {
            "Gr Liv Area": [1400, 1900],
            "Overall Qual": [6, 8],
            "Neighborhood": ["B", "C"],
        }
    )

    expected = pipeline.predict(sample)

    metadata = ArtifactMetadata(
        model_name="ridge",
        model_version="test-001",
        target_column=schema.target,
        numeric_features=tuple(schema.numeric_features),
        categorical_features=tuple(schema.categorical_features),
        metrics={"mae": 12000.0, "rmse": 15000.0, "r2": 0.90, "mape": 8.0},
    )

    artifact_path = tmp_path / "house_price_model.joblib"
    save_artifact(pipeline, metadata, artifact_path)

    predictor = HousePricePredictor(str(artifact_path))
    actual = predictor.predict(sample)

    np.testing.assert_allclose(
        actual.predictions,
        expected,
        rtol=1e-10,
        atol=1e-8,
    )
    assert actual.target_column == "SalePrice"
    assert actual.model_version == "test-001"