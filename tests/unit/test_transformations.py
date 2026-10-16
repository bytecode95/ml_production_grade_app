
import numpy as np
import pandas as pd
import pytest

from price_prediction.features.transformations import HouseAgeFeature


@pytest.fixture
def raw_df():
    return pd.DataFrame(
        {
            "Year Built": [2000, 1990],
            "Gr Liv Area": [1500.0, 1800.0],
            "Total Bsmt SF": [800.0, np.nan],
            "Overall Qual": [8, 7],
            "Lot Area": [6000.0, 8000.0],
        }
    )


def test_engineered_features_are_correct(raw_df, caplog):
    import logging

    with caplog.at_level(logging.INFO):
        result = HouseAgeFeature(reference_year=2026).fit_transform(raw_df)

    assert result["HouseAge"].tolist() == [26, 36]
    assert result["TotalSF"].tolist() == [2300.0, 1800.0]
    assert result["QualitySF"].tolist() == [12000.0, 12600.0]

    assert np.isclose(result["LogLotArea"].iloc[0], np.log1p(6000.0))
    assert "HouseAge" not in raw_df.columns
    assert len(result) == len(raw_df)


def test_transform_does_not_mutate_input(raw_df):
    original = raw_df.copy(deep=True)

    HouseAgeFeature().fit_transform(raw_df)

    pd.testing.assert_frame_equal(raw_df, original)


def test_missing_required_column_raises(raw_df):
    raw_df = raw_df.drop(columns=["Lot Area"])

    with pytest.raises(ValueError, match="Missing columns"):
        HouseAgeFeature().fit_transform(raw_df)


def test_negative_lot_area_raises(raw_df):
    raw_df.loc[0, "Lot Area"] = -1

    with pytest.raises(ValueError, match="Lot Area"):
        HouseAgeFeature().fit_transform(raw_df)
