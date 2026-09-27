"""Persistence model.

Five tables, intentionally small:

``applications``        current state of each submitted application (aggregate
                        root; carries the optimistic-lock ``version`` column).
``applicant_snapshots`` immutable point-in-time copy of the external applicant
                        data used for the decision (explainability + audit).
``idempotency_records`` request de-duplication keyed by the client supplied
                        ``Idempotency-Key`` so a retried POST returns the exact
                        same response instead of creating a second application.
``outbox_events``       transactional outbox: decision events are written in the
                        same transaction as the decision itself and published by
                        a background dispatcher with retries.
``audit_events``        append-only timeline of state transitions used by the
                        ``/audit`` endpoint to explain *how* a decision was
                        reached.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from suretyseven.clock import to_iso, utcnow


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class ApplicationStatus(StrEnum):
    """Lifecycle of an application.

    ``PROCESSING``  accepted and persisted, enrichment/decision not finished.
    ``APPROVED``/``REFERRED``/``DECLINED``  final underwriting outcomes.
    ``PENDING_RETRY``  retryable dependency failure; the reconciler picks it up
                       again (bounded by ``max_processing_attempts``).
    ``FAILED``      permanent failure; needs a human fix or a new application
                    (e.g. the applicant does not exist upstream).
    """

    PROCESSING = "PROCESSING"
    APPROVED = "APPROVED"
    REFERRED = "REFERRED"
    DECLINED = "DECLINED"
    PENDING_RETRY = "PENDING_RETRY"
    FAILED = "FAILED"

    @property
    def is_final(self) -> bool:
        return self in {
            ApplicationStatus.APPROVED,
            ApplicationStatus.REFERRED,
            ApplicationStatus.DECLINED,
        }

    @classmethod
    def from_decision(cls, decision: Decision) -> ApplicationStatus:
        return {
            Decision.APPROVE: cls.APPROVED,
            Decision.REFER: cls.REFERRED,
            Decision.DECLINE: cls.DECLINED,
        }[decision]


class Decision(StrEnum):
    """Underwriting outcome produced by the scoring model."""

    APPROVE = "APPROVE"
    REFER = "REFER"
    DECLINE = "DECLINE"


class OutboxStatus(StrEnum):
    PENDING = "PENDING"
    IN_FLIGHT = "IN_FLIGHT"
    PUBLISHED = "PUBLISHED"
    DEAD_LETTER = "DEAD_LETTER"


class ApplicantSnapshot(Base):
    """Applicant data returned by the external API at decision time."""

    __tablename__ = "applicant_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    application_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("applications.id", ondelete="CASCADE"), index=True
    )
    applicant_id: Mapped[str] = mapped_column(String(64), index=True)
    annual_revenue: Mapped[Decimal] = mapped_column(Numeric(16, 2))
    years_in_business: Mapped[int] = mapped_column(Integer)
    credit_score: Mapped[int] = mapped_column(Integer)
    existing_exposure: Mapped[Decimal] = mapped_column(Numeric(16, 2))
    source: Mapped[str] = mapped_column(String(32), default="applicant-api")
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    application: Mapped[Application] = relationship(back_populates="snapshots")

    def to_dict(self) -> dict[str, Any]:
        return {
            "annualRevenue": float(self.annual_revenue),
            "yearsInBusiness": self.years_in_business,
            "creditScore": self.credit_score,
            "existingExposure": float(self.existing_exposure),
            "fetchedAt": to_iso(self.fetched_at),
        }



class Application(Base):
    __tablename__ = "applications"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_applications_idempotency_key"),
        Index("ix_applications_status_next_attempt", "status", "next_attempt_at"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    applicant_id: Mapped[str] = mapped_column(String(64), index=True)
    bond_type: Mapped[str] = mapped_column(String(32))
    bond_amount: Mapped[Decimal] = mapped_column(Numeric(16, 2))
    effective_date: Mapped[date] = mapped_column(Date)
    obligee_name: Mapped[str] = mapped_column(String(200))

    status: Mapped[ApplicationStatus] = mapped_column(
        Enum(ApplicationStatus, native_enum=False, length=16, validate_strings=True),
        default=ApplicationStatus.PROCESSING,
        index=True,
    )
    decision: Mapped[Decision | None] = mapped_column(
        Enum(Decision, native_enum=False, length=16, validate_strings=True), nullable=True
    )
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    score_model_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    scoring_breakdown: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, nullable=True)
    decision_reasons: Mapped[list[str] | None] = mapped_column(JSON, nullable=True)

    failure_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    failure_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    processing_attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    correlation_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    request_payload: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    snapshots: Mapped[list[ApplicantSnapshot]] = relationship(
        back_populates="application", cascade="all, delete-orphan", lazy="selectin"
    )
    outbox_events: Mapped[list[OutboxEvent]] = relationship(
        back_populates="application", cascade="all, delete-orphan", lazy="selectin"
    )

    @property
    def latest_snapshot(self) -> ApplicantSnapshot | None:
        return self.snapshots[-1] if self.snapshots else None

    def to_summary(self) -> dict[str, Any]:
        return {
            "applicationId": self.id,
            "applicantId": self.applicant_id,
            "status": self.status.value,
            "decision": self.decision.value if self.decision else None,
            "score": self.score,
            "createdAt": to_iso(self.created_at),
            "updatedAt": to_iso(self.updated_at),
        }


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    request_fingerprint: Mapped[str] = mapped_column(String(64))
    application_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("applications.id", ondelete="CASCADE"), index=True
    )
    response_status_code: Mapped[int] = mapped_column(Integer)
    response_body: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )


class OutboxEvent(Base):
    """Reliable downstream notification record (transactional outbox pattern)."""

    __tablename__ = "outbox_events"
    __table_args__ = (
        # Only one event of a given type per application may ever exist: the
        # second line of defence against duplicate downstream notifications.
        UniqueConstraint("application_id", "event_type", name="uq_outbox_app_type"),
        Index("ix_outbox_status_next_attempt", "status", "next_attempt_at"),
    )

    id: Mapped[str] = mapped_column(String(48), primary_key=True)
    application_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("applications.id", ondelete="CASCADE"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    status: Mapped[OutboxStatus] = mapped_column(
        Enum(OutboxStatus, native_enum=False, length=16, validate_strings=True),
        default=OutboxStatus.PENDING,
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    claimed_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    application: Mapped[Application] = relationship(back_populates="outbox_events")


class AuditEvent(Base):
    """Append-only timeline explaining every state transition."""

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    application_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("applications.id", ondelete="CASCADE"), index=True
    )
    event_type: Mapped[str] = mapped_column(String(64))
    from_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    to_status: Mapped[str | None] = mapped_column(String(16), nullable=True)
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "eventType": self.event_type,
            "fromStatus": self.from_status,
            "toStatus": self.to_status,
            "detail": self.detail or {},
            "correlationId": self.correlation_id,
            "createdAt": to_iso(self.created_at),
        }
