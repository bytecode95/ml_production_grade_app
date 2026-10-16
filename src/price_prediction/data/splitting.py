
"""Deterministic dataset splitting for ML training and evaluation."""

import logging
import math
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSplits:
    """Container for train, validation and test partitions."""

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame

    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def split_dataset(
    df: pd.DataFrame,
    target: str = "SalePrice",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> DatasetSplits:
    """Split a dataset into train, validation and test partitions.

    Args:
        df: Validated dataset containing features and target.
        target: Name of the prediction target column.
        test_size: Fraction of the complete dataset reserved for testing.
        val_size: Fraction of the complete dataset reserved for validation.
        random_state: Seed used to make splitting reproducible.

    Returns:
        DatasetSplits containing separate feature DataFrames and target Series.

    Raises:
        TypeError: If df is not a Pandas DataFrame.
        ValueError: If the dataset or split configuration is invalid.
    """
    # 1. Validate input.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset cannot be empty.")

    if not df.columns.is_unique:
        raise ValueError("Dataset contains duplicate column names.")

    if not isinstance(target, str) or not target.strip():
        raise ValueError("Target column name must be a non-empty string.")

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' is missing from the dataset."
        )

    if df[target].isna().any():
        raise ValueError(
            f"Target column '{target}' contains missing values."
        )

    # Reject invalid ratios, including NaN and infinity.
    if (
        isinstance(test_size, bool)
        or not isinstance(test_size, (int, float))
        or not math.isfinite(test_size)
        or not 0 < test_size < 1
    ):
        raise ValueError("test_size must be a finite number between 0 and 1.")

    if (
        isinstance(val_size, bool)
        or not isinstance(val_size, (int, float))
        or not math.isfinite(val_size)
        or not 0 < val_size < 1
    ):
        raise ValueError("val_size must be a finite number between 0 and 1.")

    if test_size + val_size >= 1:
        raise ValueError(
            "The sum of test_size and val_size must be less than 1."
        )

    if (
        isinstance(random_state, bool)
        or not isinstance(random_state, int)
    ):
        raise ValueError("random_state must be an integer.")

    # 2. Separate features and target.
    X = df.drop(columns=[target])
    y = df[target]

    if X.shape[1] == 0:
        raise ValueError("Dataset must contain at least one feature column.")

    logger.info(
        "Starting dataset split: rows=%d, features=%d, target=%s, "
        "validation_fraction=%.3f, test_fraction=%.3f, random_state=%d",
        len(df),
        X.shape[1],
        target,
        val_size,
        test_size,
        random_state,
    )

    # 3. Split off the final test set.
    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
        )

        # val_size is relative to the entire dataset.
        relative_val_size = val_size / (1.0 - test_size)

        # 4. Split the remaining rows into train and validation sets.
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=relative_val_size,
            random_state=random_state,
            shuffle=True,
        )

    except ValueError:
        logger.exception(
            "Dataset splitting failed: rows=%d, test_size=%.3f, "
            "val_size=%.3f",
            len(df),
            test_size,
            val_size,
        )
        raise

    # 5. Log the final partition sizes.
    logger.info(
        "Dataset split completed: total_rows=%d, train=%d, "
        "validation=%d, test=%d",
        len(df),
        len(X_train),
        len(X_val),
        len(X_test),
    )

    # 6. Return features and targets separately.
    return DatasetSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )
