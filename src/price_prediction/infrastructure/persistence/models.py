"""SQLAlchemy persistence models."""

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Float, Index, JSON, String, func
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import Mapped, mapped_column

from price_prediction.infrastructure.persistence.database import Base


class PredictionRecord(Base):
    """Persisted prediction history record."""

    __tablename__ = "prediction_history"

    __table_args__ = (
        Index("ix_prediction_history_created_at", "created_at"),
        Index("ix_prediction_history_model_version", "model_version"),
    )

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )

    model_name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    target_column: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    predicted_price: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    input_features: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )