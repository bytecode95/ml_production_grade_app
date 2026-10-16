"""Repository for prediction history."""

from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from price_prediction.infrastructure.persistence.models import (
    PredictionRecord,
)


class PredictionRepository:
    """Persist and retrieve prediction history."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(
        self,
        *,
        model_name: str,
        model_version: str,
        target_column: str,
        predicted_price: float,
        input_features: dict,
    ) -> PredictionRecord:
        """Save one prediction and commit the transaction."""
        record = PredictionRecord(
            model_name=model_name,
            model_version=model_version,
            target_column=target_column,
            predicted_price=predicted_price,
            input_features=input_features,
        )

        try:
            self._session.add(record)
            self._session.commit()
            self._session.refresh(record)
        except Exception:
            self._session.rollback()
            raise

        return record

    def get_by_id(self, prediction_id: UUID) -> PredictionRecord | None:
        """Retrieve one prediction by UUID."""
        return self._session.get(PredictionRecord, prediction_id)

    def list_recent(
        self,
        *,
        limit: int = 20,
        offset: int = 0,
        model_version: str | None = None,
    ) -> list[PredictionRecord]:
        """Return recent predictions with pagination."""
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100.")

        if offset < 0:
            raise ValueError("offset cannot be negative.")

        statement = select(PredictionRecord)

        if model_version is not None:
            statement = statement.where(
                PredictionRecord.model_version == model_version
            )

        statement = (
            statement
            .order_by(PredictionRecord.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        return list(self._session.scalars(statement).all())

    def delete_before(self, cutoff: datetime) -> int:
        """Delete records older than a retention cutoff."""
        from sqlalchemy import delete

        statement = delete(PredictionRecord).where(
            PredictionRecord.created_at < cutoff
        )

        try:
            result = self._session.execute(statement)
            self._session.commit()
            return result.rowcount or 0
        except Exception:
            self._session.rollback()
            raise