
"""Data-quality checks and configurable training gates."""

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from price_prediction.data.schemas import DEFAULT_TARGET_COLUMN
from price_prediction.domain.exceptions import QualityCheckError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataQualityConfig:
    """Thresholds used to decide whether data is fit for training."""

    min_rows: int = 20
    max_duplicate_rows: int = 0
    max_target_missing_pct: float = 0.0
    max_feature_missing_pct: float = 0.10
    require_positive_target: bool = True

    def __post_init__(self) -> None:
        if self.min_rows < 1:
            raise ValueError("min_rows must be at least 1")

        if self.max_duplicate_rows < 0:
            raise ValueError("max_duplicate_rows cannot be negative")

        for name, value in (
            ("max_target_missing_pct", self.max_target_missing_pct),
            ("max_feature_missing_pct", self.max_feature_missing_pct),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass
class DataQualityReport:
    row_count: int
    column_count: int
    duplicate_rows: int
    target_missing_pct: float
    feature_missing_pct: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict:
        return asdict(self) | {"passed": self.passed}



def run_quality_checks(
    df: pd.DataFrame,
    target: str = DEFAULT_TARGET_COLUMN,
    config: DataQualityConfig | None = None,
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> DataQualityReport:
    """Evaluate data quality and reject datasets that fail the gate."""

    config = config or DataQualityConfig()

    if df.empty:
        raise QualityCheckError(
            "Cannot run quality checks on an empty dataset."
        )

    if target not in df.columns:
        raise QualityCheckError(f"Target column {target!r} is missing.")

    # Default to all non-target columns when no feature contract is supplied.
    features = (
        list(feature_columns)
        if feature_columns is not None
        else [column for column in df.columns if column != target]
    )

    if target in features:
        raise QualityCheckError(
            "Target column must not be included in feature_columns."
        )

    missing_features = set(features) - set(df.columns)
    if missing_features:
        raise QualityCheckError(
            f"Missing model features: {sorted(missing_features)}"
        )

    failures: list[str] = []
    duplicate_rows = int(df.duplicated().sum())

    if len(df) < config.min_rows:
        failures.append(
            f"Row count {len(df)} is below minimum {config.min_rows}."
        )

    if duplicate_rows > config.max_duplicate_rows:
        failures.append(
            f"Duplicate rows {duplicate_rows} exceed "
            f"maximum {config.max_duplicate_rows}."
        )

    target_series = df[target]
    target_missing_pct = float(target_series.isna().mean())

    if target_missing_pct > config.max_target_missing_pct:
        failures.append(
            f"Target missing percentage {target_missing_pct:.2%} "
            f"exceeds {config.max_target_missing_pct:.2%}."
        )

    target_numeric = pd.to_numeric(target_series, errors="coerce")

    if target_numeric.isna().any():
        failures.append(
            f"Target column {target!r} contains missing or non-numeric values."
        )
    elif not np.isfinite(target_numeric.to_numpy(dtype=float)).all():
        failures.append(
            f"Target column {target!r} contains non-finite values."
        )
    elif (
        config.require_positive_target
        and (target_numeric <= 0).any()
    ):
        failures.append(
            f"Target column {target!r} contains non-positive values."
        )

    feature_missing_pct: dict[str, float] = {}

    for column in features:
        series = df[column]
        missing_pct = float(series.isna().mean())
        feature_missing_pct[column] = missing_pct

        if missing_pct > config.max_feature_missing_pct:
            failures.append(
                f"Feature {column!r} missing percentage "
                f"{missing_pct:.2%} exceeds "
                f"{config.max_feature_missing_pct:.2%}."
            )

        if pd.api.types.is_numeric_dtype(series):
            values = series.dropna().to_numpy(dtype=float)

            if not np.isfinite(values).all():
                failures.append(
                    f"Feature {column!r} contains non-finite numeric values."
                )

    report = DataQualityReport(
        row_count=len(df),
        column_count=len(df.columns),
        duplicate_rows=duplicate_rows,
        target_missing_pct=target_missing_pct,
        feature_missing_pct=feature_missing_pct,
        failures=failures,
    )

    if not report.passed:
        logger.error(
            "Data-quality gate failed: %d issue(s): %s",
            len(failures),
            failures,
        )
        raise QualityCheckError(
            f"Data-quality gate failed: {failures}"
        )

    logger.info(
        "Data-quality checks passed: rows=%d, features_checked=%d",
        len(df),
        len(features),
    )

    return report
