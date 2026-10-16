"""Target transformations shared by training and inference."""

from typing import Literal

import numpy as np
import numpy.typing as npt

TargetTransformName = Literal["identity", "log1p"]


def transform_target(
    values: npt.ArrayLike,
    method: TargetTransformName = "identity",
) -> np.ndarray:
    """Transform target values before model training."""
    array = np.asarray(values, dtype=float)

    if not np.isfinite(array).all():
        raise ValueError("Target values must be finite.")

    if method == "identity":
        return array

    if method == "log1p":
        if (array < 0).any():
            raise ValueError("log1p requires target values >= 0.")
        return np.log1p(array)

    raise ValueError(f"Unsupported target transform: {method}")


def inverse_transform_target(
    values: npt.ArrayLike,
    method: TargetTransformName = "identity",
) -> np.ndarray:
    """Convert model outputs back to the original target scale."""
    array = np.asarray(values, dtype=float)

    if not np.isfinite(array).all():
        raise ValueError("Predictions must be finite.")

    if method == "identity":
        return array

    if method == "log1p":
        return np.expm1(array)

    raise ValueError(f"Unsupported target transform: {method}")