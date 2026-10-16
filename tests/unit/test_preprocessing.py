
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from price_prediction.domain.schemas import FeatureSchema
from price_prediction.features.preprocessing import build_preprocessor


def test_preprocessor_handles_missing_and_unseen_categories():
    schema = FeatureSchema(
        numeric_features=["area"],
        categorical_features=["neighborhood"],
    )

    train = pd.DataFrame(
        {
            "area": [1000.0, np.nan, 1500.0, 2000.0],
            "neighborhood": ["A", "B", "A", "B"],
        }
    )

    validation = pd.DataFrame(
        {
            "area": [1200.0, np.nan],
            "neighborhood": ["C", "A"],
        }
    )

    preprocessor = build_preprocessor(schema)

    assert isinstance(preprocessor, object)

    train_matrix = preprocessor.fit_transform(train)
    validation_matrix = preprocessor.transform(validation)

    assert train_matrix.shape[0] == len(train)
    assert validation_matrix.shape[0] == len(validation)
    assert train_matrix.shape[1] == validation_matrix.shape[1]

    assert np.isfinite(train_matrix).all()
    assert np.isfinite(validation_matrix).all()


def test_preprocessor_can_be_used_inside_pipeline():
    schema = FeatureSchema(
        numeric_features=["area"],
        categorical_features=["neighborhood"],
    )

    df = pd.DataFrame(
        {
            "area": [1000.0, 1200.0, 1400.0, 1600.0],
            "neighborhood": ["A", "B", "A", "B"],
        }
    )

    pipeline = Pipeline([("prep", build_preprocessor(schema))])
    matrix = pipeline.fit_transform(df)

    assert matrix.shape[0] == len(df)
    assert np.isfinite(matrix).all()
