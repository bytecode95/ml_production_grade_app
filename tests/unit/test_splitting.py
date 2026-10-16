
import logging

import pandas as pd
import pytest

from price_prediction.data.splitting import split_dataset


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "Gr Liv Area": range(100, 200),
            "Overall Qual": [5, 6, 7, 8, 9] * 20,
            "Neighborhood": ["NAmes", "CollgCr"] * 50,
            "SalePrice": range(100_000, 200_000, 1_000),
        }
    )


def test_split_sizes_and_target_separation(sample_df, caplog):
    with caplog.at_level(logging.INFO):
        splits = split_dataset(sample_df)

    assert len(splits.X_train) + len(splits.X_val) + len(splits.X_test) == len(sample_df)
    assert len(splits.y_train) == len(splits.X_train)
    assert len(splits.y_val) == len(splits.X_val)
    assert len(splits.y_test) == len(splits.X_test)

    assert "SalePrice" not in splits.X_train.columns
    assert "SalePrice" not in splits.X_val.columns
    assert "SalePrice" not in splits.X_test.columns

    assert "Dataset split completed" in caplog.text


def test_split_is_reproducible(sample_df):
    first = split_dataset(sample_df, random_state=42)
    second = split_dataset(sample_df, random_state=42)

    pd.testing.assert_frame_equal(first.X_train, second.X_train)
    pd.testing.assert_series_equal(first.y_test, second.y_test)


def test_partitions_do_not_overlap(sample_df):
    splits = split_dataset(sample_df)

    train_idx = set(splits.X_train.index)
    val_idx = set(splits.X_val.index)
    test_idx = set(splits.X_test.index)

    assert train_idx.isdisjoint(val_idx)
    assert train_idx.isdisjoint(test_idx)
    assert val_idx.isdisjoint(test_idx)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"test_size": 0},
        {"test_size": 1},
        {"val_size": 0},
        {"test_size": 0.6, "val_size": 0.4},
    ],
)
def test_invalid_split_configuration_raises(sample_df, kwargs):
    with pytest.raises(ValueError):
        split_dataset(sample_df, **kwargs)


def test_missing_target_raises(sample_df):
    with pytest.raises(ValueError, match="Target column"):
        split_dataset(sample_df.drop(columns=["SalePrice"]))


def test_missing_target_values_raise(sample_df):
    sample_df.loc[0, "SalePrice"] = None

    with pytest.raises(ValueError, match="missing values"):
        split_dataset(sample_df)
