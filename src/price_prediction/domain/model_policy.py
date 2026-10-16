
"""Configurable acceptance policy for regression models."""

from dataclasses import dataclass

from price_prediction.domain.metrics import RegressionMetrics


@dataclass(frozen=True)
class ModelPolicy:
    """Required model quality thresholds.

    Set a threshold to None to disable that particular threshold.
    """

    max_mae: float | None = None
    max_rmse: float | None = None
    min_r2: float | None = None
    max_mape: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_mae", "max_rmse", "max_mape"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative.")

        if self.min_r2 is not None and not -1 <= self.min_r2 <= 1:
            raise ValueError("min_r2 must be between -1 and 1.")

        if all(
            getattr(self, name) is None
            for name in ("max_mae", "max_rmse", "min_r2", "max_mape")
        ):
            raise ValueError("At least one model acceptance threshold is required.")

    def check(
        self,
        metrics: RegressionMetrics,
    ) -> list[str]:
        """Return all reasons a candidate violates the policy."""
        failures: list[str] = []

        if self.max_mae is not None and metrics.mae > self.max_mae:
            failures.append(
                f"MAE {metrics.mae:.4f} exceeds maximum {self.max_mae:.4f}"
            )

        if self.max_rmse is not None and metrics.rmse > self.max_rmse:
            failures.append(
                f"RMSE {metrics.rmse:.4f} exceeds maximum {self.max_rmse:.4f}"
            )

        if self.min_r2 is not None and metrics.r2 < self.min_r2:
            failures.append(
                f"R2 {metrics.r2:.4f} is below minimum {self.min_r2:.4f}"
            )

        if self.max_mape is not None:
            if metrics.mape is None:
                failures.append("MAPE is unavailable because all targets are zero.")
            elif metrics.mape > self.max_mape:
                failures.append(
                    f"MAPE {metrics.mape:.4f}% exceeds "
                    f"maximum {self.max_mape:.4f}%"
                )

        return failures
