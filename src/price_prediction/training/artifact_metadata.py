"""Metadata contract for trained model artifacts."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
import json


@dataclass(frozen=True)
class ArtifactMetadata:
    model_name: str
    model_version: str
    target_column: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    metrics: dict[str, float | None]
    target_transform: str = "identity"
    dataset_version: str = "unknown"
    code_version: str = "unknown"
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty.")
        if not self.model_version.strip():
            raise ValueError("model_version cannot be empty.")
        if not self.target_column.strip():
            raise ValueError("target_column cannot be empty.")
        if self.target_transform not in {"identity", "log1p"}:
            raise ValueError("Unsupported target_transform.")
        if not self.numeric_features and not self.categorical_features:
            raise ValueError("At least one feature is required.")

        feature_names = self.numeric_features + self.categorical_features
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("Feature names must be unique.")

        if not self.created_at:
            object.__setattr__(
                self,
                "created_at",
                datetime.now(timezone.utc).isoformat(),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metadata."""
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)