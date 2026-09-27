"""Transactional outbox for downstream notifications.

Why an outbox: the decision and the "please tell the downstream system" intent
must be committed atomically.  If we published inside the request transaction
and the process died between commit and publish, the decision would exist but
the downstream system would never hear about it.  If we published first and the
commit then failed, we would have announced a decision that does not exist.

Delivery guarantees
-------------------
* ``UNIQUE (application_id, event_type)`` - at most one decision event per
  application, regardless of how many retries/workers race.
* Claim by compare-and-swap - a worker may only work on an event it claimed, so
  two replicas cannot publish the same row simultaneously (safe at N replicas).
* **At-least-once** delivery.  The event carries a stable ``eventId`` and the
  consumer de-duplicates on it, which makes the end-to-end effect
  effectively-once for idempotent consumers.
* Bounded retries with exponential backoff; after ``max_attempts`` the event
  goes to ``DEAD_LETTER``, is logged loudly, counted in metrics and can be
  requeued by an operator (``POST /applications/{id}/retry``).
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

from sqlalchemy import and_, or_, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from suretyseven.clock import utcnow
from suretyseven.downstream import DeliveryReceipt, DownstreamNotifier
from suretyseven.errors import DownstreamDeliveryError
from suretyseven.logging_config import get_logger
from suretyseven.metrics import Metrics
from suretyseven.models import Application, OutboxEvent, OutboxStatus
from suretyseven.schemas import DecisionEventPayload

logger = get_logger(__name__)

DECISION_EVENT = "APPLICATION_DECISIONED"


def new_event_id() -> str:
    return f"EVT-{uuid.uuid4().hex[:20]}"


def build_decision_event(application: Application) -> DecisionEventPayload:
    """Project a decided application onto the downstream event contract."""
    if application.decision is None or application.score is None:
        raise ValueError("cannot build a decision event for an undecided application")
    return DecisionEventPayload(
        event_id=new_event_id(),
        application_id=application.id,
        applicant_id=application.applicant_id,
        bond_type=application.bond_type,
        decision=application.decision,
        status=application.status,
        score=application.score,
        score_model_version=application.score_model_version or "unknown",
        occurred_at=utcnow(),
        correlation_id=application.correlation_id,
    )


def enqueue_decision_event(session: Session, application: Application) -> OutboxEvent:
    """Insert the decision event in the caller's transaction (idempotent)."""
    existing = session.scalar(
        select(OutboxEvent).where(
            OutboxEvent.application_id == application.id,
            OutboxEvent.event_type == DECISION_EVENT,
        )
    )
    if existing is not None:
        return existing

    event = build_decision_event(application)
    row = OutboxEvent(
        id=event.event_id,
        application_id=application.id,
        event_type=DECISION_EVENT,
        payload=event.model_dump(mode="json", by_alias=True),
        status=OutboxStatus.PENDING,
        next_attempt_at=utcnow(),
    )
    try:
        # Savepoint: if another replica inserted the same event a moment ago the
        # unique constraint fires, we roll back only this insert and reuse theirs.
        with session.begin_nested():
            session.add(row)
    except IntegrityError:
        logger.info(
            "outbox_event_already_exists",
            extra={"applicationId": application.id, "eventType": DECISION_EVENT},
        )
        reloaded = session.scalar(
            select(OutboxEvent).where(
                OutboxEvent.application_id == application.id,
                OutboxEvent.event_type == DECISION_EVENT,
            )
        )
        return reloaded if reloaded is not None else row
    return row


def requeue_dead_letters(session: Session, application_id: str) -> int:
    """Operator escape hatch: make failed notifications deliverable again."""
    result = cast(
        "CursorResult[Any]",
        session.execute(
            update(OutboxEvent)
            .where(
                OutboxEvent.application_id == application_id,
                OutboxEvent.status == OutboxStatus.DEAD_LETTER,
            )
            .values(
                status=OutboxStatus.PENDING,
                attempts=0,
                next_attempt_at=utcnow(),
                last_error=None,
                claimed_by=None,
                claimed_at=None,
            )
        ),
    )
    return int(result.rowcount or 0)



@dataclass(slots=True)
class DispatchStats:
    claimed: int = 0
    published: int = 0
    duplicates: int = 0
    failed: int = 0
    dead_lettered: int = 0


class OutboxDispatcher:
    """Publishes due outbox events, one short transaction at a time."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        notifier: DownstreamNotifier,
        *,
        batch_size: int = 25,
        max_attempts: int = 5,
        backoff_base_seconds: float = 1.0,
        backoff_max_seconds: float = 60.0,
        stale_claim_seconds: float = 60.0,
        metrics: Metrics | None = None,
        worker_id: str | None = None,
        random_value=random.random,
    ) -> None:
        self._session_factory = session_factory
        self._notifier = notifier
        self._batch_size = batch_size
        self._max_attempts = max_attempts
        self._backoff_base = backoff_base_seconds
        self._backoff_max = backoff_max_seconds
        self._stale_claim = stale_claim_seconds
        self._metrics = metrics
        self._worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self._random = random_value

    # ---------------------------------------------------------------- internals
    def _backoff(self, attempts: int) -> float:
        base = min(self._backoff_max, self._backoff_base * (2 ** max(attempts - 1, 0)))
        return base * (0.5 + 0.5 * self._random())  # jitter avoids lockstep retries

    def _claimable_ids(self, now, limit: int) -> list[str]:
        """Rows to try next: pending, or in-flight but abandoned by a dead worker."""
        stale_cutoff = now - timedelta(seconds=self._stale_claim)
        statement = (
            select(OutboxEvent.id)
            .where(
                OutboxEvent.next_attempt_at <= now,
                or_(
                    OutboxEvent.status == OutboxStatus.PENDING,
                    and_(
                        OutboxEvent.status == OutboxStatus.IN_FLIGHT,
                        OutboxEvent.claimed_at.is_not(None),
                        OutboxEvent.claimed_at <= stale_cutoff,
                    ),
                ),
            )
            .order_by(OutboxEvent.created_at)
            .limit(limit)
        )
        with self._session_factory() as session:
            return list(session.scalars(statement))

    def _claim(self, event_id: str, now) -> bool:
        """Compare-and-swap claim; only the winner is allowed to publish."""
        claimable = or_(
            OutboxEvent.status == OutboxStatus.PENDING,
            and_(
                OutboxEvent.status == OutboxStatus.IN_FLIGHT,
                OutboxEvent.claimed_at.is_not(None),
                OutboxEvent.claimed_at <= now - timedelta(seconds=self._stale_claim),
            ),
        )
        with self._session_factory() as session:
            result = cast(
                "CursorResult[Any]",
                session.execute(
                    update(OutboxEvent)
                    .where(OutboxEvent.id == event_id, claimable)
                    .values(
                        status=OutboxStatus.IN_FLIGHT,
                        claimed_by=self._worker_id,
                        claimed_at=now,
                    )
                ),
            )
            session.commit()
            return bool(result.rowcount)

    def _load_payload(self, event_id: str) -> DecisionEventPayload | None:
        with self._session_factory() as session:
            row = session.get(OutboxEvent, event_id)
            if row is None or row.status is OutboxStatus.PUBLISHED:
                return None
            return DecisionEventPayload.model_validate(row.payload)

    def _mark_published(self, event_id: str, receipt: DeliveryReceipt) -> None:
        with self._session_factory() as session:
            row = session.get(OutboxEvent, event_id)
            if row is None:
                return
            event_type = row.event_type
            row.status = OutboxStatus.PUBLISHED
            row.attempts = row.attempts + 1
            row.published_at = utcnow()
            row.last_error = None
            row.claimed_by = None
            row.claimed_at = None
            session.commit()
        if self._metrics is not None:
            self._metrics.outbox_events.labels(
                event_type=event_type, status=OutboxStatus.PUBLISHED.value
            ).inc()
        logger.info(
            "outbox_event_published",
            extra={
                "eventId": event_id,
                "eventType": event_type,
                "duplicate": receipt.duplicate,
                "statusCode": receipt.status_code,
            },
        )

    def _mark_failure(self, event_id: str, error: str) -> bool:
        """Record a failed attempt. Returns True when the event is dead-lettered."""
        with self._session_factory() as session:
            row = session.get(OutboxEvent, event_id)
            if row is None:
                return False
            attempts = row.attempts + 1
            dead = attempts >= self._max_attempts
            row.attempts = attempts
            row.last_error = error[:2000]
            row.claimed_by = None
            row.claimed_at = None
            if dead:
                row.status = OutboxStatus.DEAD_LETTER
            else:
                row.status = OutboxStatus.PENDING
                row.next_attempt_at = utcnow() + timedelta(seconds=self._backoff(attempts))
            event_type = row.event_type
            session.commit()

        if self._metrics is not None:
            status = OutboxStatus.DEAD_LETTER.value if dead else OutboxStatus.PENDING.value
            self._metrics.outbox_events.labels(event_type=event_type, status=status).inc()
        log = logger.error if dead else logger.warning
        log(
            "outbox_event_delivery_failed",
            extra={
                "eventId": event_id,
                "eventType": event_type,
                "attempts": attempts,
                "deadLettered": dead,
                "error": error,
            },
        )
        return dead

    # ------------------------------------------------------------- public API
    def dispatch_due(self, limit: int | None = None) -> DispatchStats:
        """Publish every due event (bounded by the batch size)."""
        stats = DispatchStats()
        now = utcnow()
        for event_id in self._claimable_ids(now, limit or self._batch_size):
            if not self._claim(event_id, now):
                continue  # another replica got there first
            stats.claimed += 1

            # Network calls happen outside any database transaction.
            payload = self._load_payload(event_id)
            if payload is None:
                continue
            try:
                receipt = self._notifier.publish(payload)
            except DownstreamDeliveryError as exc:
                if self._mark_failure(event_id, str(exc)):
                    stats.dead_lettered += 1
                stats.failed += 1
            else:
                self._mark_published(event_id, receipt)
                stats.published += 1
                if receipt.duplicate:
                    stats.duplicates += 1
        return stats

