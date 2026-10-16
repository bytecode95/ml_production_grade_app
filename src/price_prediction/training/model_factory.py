
"""Factory for supported regression estimators."""

import logging
from typing import Any

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


def create_model(
    name: str,
    *,
    random_state: int = 42,
    params: dict[str, Any] | None = None,
):
    """Create an unfitted regression estimator.

    Supported names:
        baseline, ridge, random_forest, extra_trees,
        hist_gradient_boosting
    """
    params = dict(params or {})

    factories = {
        "baseline": lambda: DummyRegressor(strategy="mean", **params),
        "ridge": lambda: Ridge(**params),
        "random_forest": lambda: RandomForestRegressor(
            random_state=random_state, n_jobs=-1, **params
        ),
        "extra_trees": lambda: ExtraTreesRegressor(
            random_state=random_state, n_jobs=-1, **params
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(
            random_state=random_state, **params
        ),
    }

    if name not in factories:
        raise ValueError(
            f"Unsupported model '{name}'. "
            f"Supported models: {sorted(factories)}"
        )

    model = factories[name]()

    logger.info(
        "Created unfitted model: name=%s type=%s",
        name,
        type(model).__name__,
    )

    return model
