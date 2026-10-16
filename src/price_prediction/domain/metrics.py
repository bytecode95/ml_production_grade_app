
"""Regression evaluation metrics."""

import logging
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegressionMetrics:
    """Metrics expressed in the original target units where applicable."""

    mae: float
    rmse: float
    r2: float
    mape: float | None

    def to_dict(self) -> dict[str, float | None]:
        """Return metrics in a serialization-friendly format."""
        return asdict(self)


def calculate_metrics(
    y_true,
    y_pred,
) -> RegressionMetrics:
    """Calculate MAE, RMSE, R2 and MAPE.

    MAPE is expressed as a percentage. Rows with zero actual targets
    are excluded from MAPE only; all rows are used for other metrics.

    Raises:
        ValueError: If arrays are empty, mismatched or non-finite.
    """
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)

    if actual.size == 0:
        raise ValueError("Cannot calculate metrics for empty targets.")

    if actual.shape != predicted.shape:
        raise ValueError(
            "Actual and predicted values must have the same shape."
        )

    if not np.isfinite(actual).all():
        raise ValueError("Actual targets contain NaN or infinite values.")

    if not np.isfinite(predicted).all():
        raise ValueError("Predictions contain NaN or infinite values.")

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    r2 = float(r2_score(actual, predicted))

    nonzero_mask = actual != 0
    mape = None

    if nonzero_mask.any():
        mape = float(
            np.mean(
                np.abs(
                    (actual[nonzero_mask] - predicted[nonzero_mask])
                    / actual[nonzero_mask]
                )
            )
            * 100.0
        )

    metrics = RegressionMetrics(
        mae=mae,
        rmse=rmse,
        r2=r2,
        mape=mape,
    )

    logger.info("Regression metrics calculated: %s", metrics.to_dict())

    return metrics
