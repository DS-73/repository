"""Application service - the use cases of the system.

Processing model
----------------
``POST /applications`` runs the pipeline *synchronously but in phases*, so the
caller gets an answer while the database never holds a transaction open across a
network call:

1. **tx1** validate + persist ``PROCESSING`` application + idempotency record.
2. **no tx** call the Applicant API (timeouts/retries/circuit breaker).
3. **tx2** persist snapshot, score, decision, audit entries and the outbox event
   that notifies the downstream system - all atomically.

If the process dies anywhere in between, the application stays ``PROCESSING``
and the reconciler (``worker.py``) retries it, which is why the design tolerates
crashes.  Dependency failures never surface as a 5xx to the broker: the
application is parked in ``PENDING_RETRY`` (retryable) or ``FAILED``
(permanent) with a machine readable ``failure.code``, and ``GET`` explains it.
"""

from __future__ import annotations

import hashlib
import json
import random
import uuid
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from suretyseven.clock import utcnow
from suretyseven.errors import (
    ApplicationNotFoundError,
    ApplicationNotRetryableError,
    BusinessValidationError,
    ExternalDependencyError,
    IdempotencyConflictError,
    ServiceError,
)
from suretyseven.external_client import ApplicantApiClient
from suretyseven.logging_config import APPLICATION_ID, get_logger
from suretyseven.metrics import Metrics
from suretyseven.models import (
    ApplicantSnapshot,
    Application,
    ApplicationStatus,
    AuditEvent,
    Decision,
    IdempotencyRecord,
)
from suretyseven.outbox import enqueue_decision_event, requeue_dead_letters
from suretyseven.schemas import (
    CreateApplicationRequest,
    serialize_application,
)
from suretyseven.scoring import ScoreModelConfig, evaluate

logger = get_logger(__name__)

MAX_IDEMPOTENCY_KEY_LENGTH = 128


def new_application_id() -> str:
    return f"APP-{uuid.uuid4().hex[:12].upper()}"


def fingerprint_of(request: CreateApplicationRequest) -> str:
    """Stable hash of the request body, used to detect key misuse."""
    canonical = json.dumps(
        request.model_dump(mode="json", by_alias=True), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class CreateResult:
    """Outcome of ``POST /applications``."""

    application: Application
    replayed: bool
    status_code: int
    idempotency_key: str


@dataclass(slots=True)
class ReconcileStats:
    retried: int = 0
    decided: int = 0
    failed: int = 0
    skipped: int = 0



class ApplicationService:
    """All state changes for an application live here."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        applicant_client: ApplicantApiClient,
        score_model: ScoreModelConfig,
        *,
        max_processing_attempts: int = 5,
        stale_processing_seconds: float = 60.0,
        retry_backoff_base_seconds: float = 5.0,
        retry_backoff_max_seconds: float = 300.0,
        duplicate_window_seconds: int = 600,
        metrics: Metrics | None = None,
        random_value=random.random,
    ) -> None:
        self._sessions = session_factory
        self._client = applicant_client
        self._score_model = score_model
        self._max_attempts = max_processing_attempts
        self._stale_seconds = stale_processing_seconds
        self._retry_base = retry_backoff_base_seconds
        self._retry_max = retry_backoff_max_seconds
        self._duplicate_window = duplicate_window_seconds
        self._metrics = metrics
        self._random = random_value

    @property
    def score_model(self) -> ScoreModelConfig:
        return self._score_model

    # ------------------------------------------------------------------ reads
    def get_application(self, application_id: str) -> Application:
        with self._sessions() as session:
            application = session.get(Application, application_id)
            if application is None:
                raise ApplicationNotFoundError(f"application '{application_id}' does not exist")
            return application

    def list_applications(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status: ApplicationStatus | None = None,
        applicant_id: str | None = None,
    ) -> tuple[list[Application], int]:
        filters = []
        if status is not None:
            filters.append(Application.status == status)
        if applicant_id is not None:
            filters.append(Application.applicant_id == applicant_id)
        with self._sessions() as session:
            total = session.scalar(
                select(func.count()).select_from(Application).where(*filters)
            )
            rows = list(
                session.scalars(
                    select(Application)
                    .where(*filters)
                    .order_by(Application.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                )
            )
        return rows, int(total or 0)

    def get_audit_trail(self, application_id: str) -> list[AuditEvent]:
        with self._sessions() as session:
            if session.get(Application, application_id) is None:
                raise ApplicationNotFoundError(f"application '{application_id}' does not exist")
            return list(
                session.scalars(
                    select(AuditEvent)
                    .where(AuditEvent.application_id == application_id)
                    .order_by(AuditEvent.id)
                )
            )

    # ----------------------------------------------------------------- create
    def create_application(
        self,
        request: CreateApplicationRequest,
        *,
        idempotency_key: str | None = None,
        correlation_id: str | None = None,
    ) -> CreateResult:
        """Accept an application, then run the decision pipeline."""
        fingerprint = fingerprint_of(request)
        key = self._effective_key(idempotency_key, fingerprint)

        with self._sessions() as session:
            replay = self._find_replay(session, key, fingerprint)
            if replay is not None:
                return replay

            application = Application(
                id=new_application_id(),
                applicant_id=request.applicant_id,
                bond_type=request.bond_type.value,
                bond_amount=request.bond_amount,
                effective_date=request.effective_date,
                obligee_name=request.obligee.name,
                status=ApplicationStatus.PROCESSING,
                processing_attempts=0,
                correlation_id=correlation_id,
                idempotency_key=key,
                request_payload=request.model_dump(mode="json", by_alias=True),
            )
            session.add(application)
            session.add(
                IdempotencyRecord(
                    key=key,
                    request_fingerprint=fingerprint,
                    application_id=application.id,
                    response_status_code=201,
                    response_body={"status": ApplicationStatus.PROCESSING.value},
                )
            )
            self._audit(
                session,
                application,
                event_type="APPLICATION_ACCEPTED",
                from_status=None,
                to_status=ApplicationStatus.PROCESSING.value,
                detail={"idempotencyKeySource": "client" if idempotency_key else "auto"},
                correlation_id=correlation_id,
            )
            try:
                session.commit()
            except IntegrityError:
                # Another request with the same key won the race.
                session.rollback()
                replay = self._find_replay(session, key, fingerprint)
                if replay is None:  # pragma: no cover - defensive
                    raise
                return replay

        if self._metrics is not None:
            self._metrics.applications_created.labels(replayed="false").inc()
        logger.info(
            "application_accepted",
            extra={
                "applicationId": application.id,
                "applicantId": application.applicant_id,
                "bondType": application.bond_type,
                "correlationId": correlation_id,
            },
        )

        processed = self.process_application(application.id, correlation_id=correlation_id)
        self._store_response_snapshot(key, processed)
        return CreateResult(
            application=processed, replayed=False, status_code=201, idempotency_key=key
        )

    # -------------------------------------------------------------- pipeline
    def process_application(
        self, application_id: str, *, correlation_id: str | None = None
    ) -> Application:
        """Run (or re-run) enrichment + scoring + decision for one application.

        Never raises for dependency problems: the failure is recorded on the
        application so operators and the reconciler can act on it.
        """
        token = APPLICATION_ID.set(application_id)
        try:
            prepared = self._begin_attempt(application_id)
            if prepared is None:
                return self.get_application(application_id)  # already final/given up
            application, attempt, effective_correlation = prepared

            try:
                data = self._client.fetch_applicant(
                    application.applicant_id, correlation_id=effective_correlation
                )
            except ExternalDependencyError as exc:
                return self._record_dependency_failure(
                    application_id, exc, attempt=attempt, correlation_id=effective_correlation
                )
            except Exception as exc:  # pragma: no cover - unexpected client bug
                logger.exception("applicant_lookup_unexpected_error")
                return self._record_dependency_failure(
                    application_id,
                    ServiceError(f"unexpected error contacting the Applicant API: {exc}"),
                    attempt=attempt,
                    correlation_id=effective_correlation,
                )

            return self._apply_decision(
                application_id, data, attempt=attempt, correlation_id=effective_correlation
            )
        finally:
            APPLICATION_ID.reset(token)

    # ------------------------------------------------------------- internals
    def _effective_key(self, idempotency_key: str | None, fingerprint: str) -> str:
        """Client key, or a content-addressed key inside the dedupe window."""
        if idempotency_key is not None:
            key = idempotency_key.strip()
            if not key:
                raise BusinessValidationError("Idempotency-Key must not be blank")
            if len(key) > MAX_IDEMPOTENCY_KEY_LENGTH:
                raise BusinessValidationError(
                    f"Idempotency-Key must be at most {MAX_IDEMPOTENCY_KEY_LENGTH} characters"
                )
            return key
        # No key supplied: still protect against the common "client retried the
        # exact same request" case, but only inside a bounded window so a genuine
        # re-application later is not silently swallowed.
        bucket = int(utcnow().timestamp() // max(self._duplicate_window, 1))
        return f"auto:{fingerprint}:{bucket}"

    def _find_replay(
        self, session: Session, key: str, fingerprint: str
    ) -> CreateResult | None:
        record = session.get(IdempotencyRecord, key)
        if record is None:
            return None
        if record.request_fingerprint != fingerprint:
            if self._metrics is not None:
                self._metrics.idempotency_conflicts.inc()
            logger.warning(
                "idempotency_key_reused_with_different_body",
                extra={"idempotencyKey": key, "applicationId": record.application_id},
            )
            raise IdempotencyConflictError(
                "Idempotency-Key was already used with a different request body",
                detail={"applicationId": record.application_id},
            )
        application = session.get(Application, record.application_id)
        if application is None:  # pragma: no cover - defensive
            raise ApplicationNotFoundError(f"application '{record.application_id}' does not exist")
        if self._metrics is not None:
            self._metrics.applications_created.labels(replayed="true").inc()
        logger.info(
            "application_replayed",
            extra={"applicationId": application.id, "idempotencyKey": key},
        )
        # We deliberately return the *current* state of the resource rather than a
        # byte-identical copy of the first response: the resource is mutable
        # (PENDING_RETRY -> APPROVED) and a stale body would mislead a retrying
        # client.  The original response is still recorded on the idempotency row.
        return CreateResult(
            application=application, replayed=True, status_code=200, idempotency_key=key
        )

    def _store_response_snapshot(self, key: str, application: Application) -> None:
        """Keep an audit copy of what the client saw (best effort)."""
        with self._sessions() as session:
            record = session.get(IdempotencyRecord, key)
            if record is None:
                return
            record.response_body = serialize_application(application).model_dump(
                mode="json", by_alias=True
            )
            session.commit()

    def _begin_attempt(
        self, application_id: str
    ) -> tuple[Application, int, str | None] | None:
        """Claim a processing attempt; returns None when there is nothing to do."""
        with self._sessions() as session:
            application = session.get(Application, application_id)
            if application is None:
                raise ApplicationNotFoundError(f"application '{application_id}' does not exist")
            if application.status.is_final:
                return None
            if application.processing_attempts >= self._max_attempts:
                self._mark_failed(
                    session,
                    application,
                    code="PROCESSING_RETRY_EXHAUSTED",
                    message=f"gave up after {application.processing_attempts} processing attempts",
                    correlation_id=application.correlation_id,
                )
                session.commit()
                return None
            application.processing_attempts += 1
            application.status = ApplicationStatus.PROCESSING
            application.next_attempt_at = None
            attempt = application.processing_attempts
            correlation_id = application.correlation_id
            session.commit()
            return application, attempt, correlation_id



    def _apply_decision(
        self,
        application_id: str,
        data: Any,
        *,
        attempt: int,
        correlation_id: str | None,
    ) -> Application:
        with self._sessions() as session:
            application = session.get(Application, application_id)
            if application is None:  # pragma: no cover - defensive
                raise ApplicationNotFoundError(f"application '{application_id}' does not exist")
            if application.status.is_final:
                return application

            session.add(
                ApplicantSnapshot(
                    application_id=application.id,
                    applicant_id=data.applicant_id,
                    annual_revenue=data.annual_revenue,
                    years_in_business=data.years_in_business,
                    credit_score=data.credit_score,
                    existing_exposure=data.existing_exposure,
                    source="applicant-api",
                    fetched_at=utcnow(),
                )
            )

            result = evaluate(
                self._score_model,
                bond_amount=Decimal(application.bond_amount),
                annual_revenue=Decimal(data.annual_revenue),
                years_in_business=data.years_in_business,
                credit_score=data.credit_score,
                existing_exposure=Decimal(data.existing_exposure),
            )

            from_status = application.status.value
            application.score = result.score
            application.decision = Decision(result.decision)
            application.score_model_version = result.model_version
            application.scoring_breakdown = result.breakdown
            application.decision_reasons = result.reasons
            application.status = ApplicationStatus.from_decision(application.decision)
            application.decided_at = utcnow()
            application.next_attempt_at = None
            application.failure_code = None
            application.failure_message = None
            application.version += 1
            application.updated_at = utcnow()

            self._audit(
                session,
                application,
                event_type="APPLICANT_DATA_FETCHED",
                from_status=from_status,
                to_status=from_status,
                detail={
                    "attempt": attempt,
                    "creditScore": data.credit_score,
                    "yearsInBusiness": data.years_in_business,
                    "annualRevenue": float(data.annual_revenue),
                    "existingExposure": float(data.existing_exposure),
                },
                correlation_id=correlation_id,
            )
            self._audit(
                session,
                application,
                event_type="SCORING_COMPLETED",
                from_status=from_status,
                to_status=application.status.value,
                detail={
                    "score": result.score,
                    "decision": result.decision,
                    "modelVersion": result.model_version,
                    "breakdown": result.breakdown,
                    "reasons": result.reasons,
                    "missingMetrics": result.missing_metrics,
                },
                correlation_id=correlation_id,
            )
            event = enqueue_decision_event(session, application)
            self._audit(
                session,
                application,
                event_type="DECISION_EVENT_ENQUEUED",
                from_status=application.status.value,
                to_status=application.status.value,
                detail={"eventId": event.id, "eventType": event.event_type},
                correlation_id=correlation_id,
            )
            session.commit()

        if self._metrics is not None:
            self._metrics.decisions.labels(decision=result.decision).inc()
            self._metrics.processing_outcomes.labels(outcome="decided").inc()
        logger.info(
            "application_decided",
            extra={
                "applicationId": application_id,
                "decision": result.decision,
                "score": result.score,
                "modelVersion": result.model_version,
                "attempt": attempt,
            },
        )
        return self.get_application(application_id)

    def _record_dependency_failure(
        self,
        application_id: str,
        error: Exception,
        *,
        attempt: int,
        correlation_id: str | None,
    ) -> Application:
        """Park the application and remember why; never leak a 5xx to the broker."""
        retryable = bool(getattr(error, "retryable", False))
        code = getattr(error, "code", "INTERNAL_ERROR")
        scheduled = retryable and attempt < self._max_attempts
        delay = self._retry_delay(attempt) if scheduled else None

        with self._sessions() as session:
            application = session.get(Application, application_id)
            if application is None:  # pragma: no cover - defensive
                raise ApplicationNotFoundError(f"application '{application_id}' does not exist")
            if application.status.is_final:
                return application

            application.failure_code = code
            application.failure_message = str(error)[:2000]
            application.updated_at = utcnow()
            if scheduled:
                application.status = ApplicationStatus.PENDING_RETRY
                application.next_attempt_at = utcnow() + timedelta(seconds=delay or 0)
                event_type = "PROCESSING_RETRY_SCHEDULED"
                detail = {
                    "attempt": attempt,
                    "maxAttempts": self._max_attempts,
                    "errorCode": code,
                    "retryInSeconds": round(delay or 0, 2),
                }
            else:
                self._mark_failed(
                    session,
                    application,
                    code=code,
                    message=str(error),
                    correlation_id=correlation_id,
                    audit=False,
                )
                event_type = "PROCESSING_FAILED"
                detail = {"attempt": attempt, "errorCode": code, "retryable": retryable}
            self._audit(
                session,
                application,
                event_type=event_type,
                from_status=ApplicationStatus.PROCESSING.value,
                to_status=application.status.value,
                detail=detail,
                correlation_id=correlation_id,
            )
            session.commit()

        if self._metrics is not None:
            self._metrics.processing_outcomes.labels(
                outcome="retry_scheduled" if scheduled else "failed"
            ).inc()
        logger.warning(
            "application_processing_failed",
            extra={
                "applicationId": application_id,
                "errorCode": code,
                "attempt": attempt,
                "retryScheduled": scheduled,
                "error": str(error),
            },
        )
        return self.get_application(application_id)

    def _retry_delay(self, attempt: int) -> float:
        base = min(self._retry_max, self._retry_base * (2 ** max(attempt - 1, 0)))
        return base * (0.5 + 0.5 * self._random())

    def _mark_failed(
        self,
        session: Session,
        application: Application,
        *,
        code: str,
        message: str,
        correlation_id: str | None = None,
        audit: bool = True,
    ) -> None:
        from_status = application.status.value
        application.status = ApplicationStatus.FAILED
        application.failure_code = code
        application.failure_message = message[:2000]
        application.next_attempt_at = None
        application.updated_at = utcnow()
        if audit:
            self._audit(
                session,
                application,
                event_type="PROCESSING_FAILED",
                from_status=from_status,
                to_status=application.status.value,
                detail={"errorCode": code, "message": message[:500]},
                correlation_id=correlation_id or application.correlation_id,
            )

    def _audit(
        self,
        session: Session,
        application: Application,
        *,
        event_type: str,
        from_status: str | None,
        to_status: str | None,
        detail: dict[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        session.add(
            AuditEvent(
                application_id=application.id,
                event_type=event_type,
                from_status=from_status,
                to_status=to_status,
                detail=detail or {},
                correlation_id=correlation_id or application.correlation_id,
            )
        )


    # ------------------------------------------------- operator / reconciler
    def retry_application(self, application_id: str) -> Application:
        """Manual retry for a parked or failed application (operator action).

        An application that already has a final decision is never re-scored, but
        its *notification* may be sitting in the dead letter queue after a
        downstream outage.  In that case the operator action requeues the
        notification so the downstream system can still learn the decision.
        """
        notification_requeued = False
        with self._sessions() as session:
            application = session.get(Application, application_id)
            if application is None:
                raise ApplicationNotFoundError(f"application '{application_id}' does not exist")
            if application.status.is_final:
                requeued = requeue_dead_letters(session, application_id)
                if requeued == 0:
                    raise ApplicationNotRetryableError(
                        "application already has a final decision; submit a corrected "
                        "application instead of retrying this one",
                        detail={"status": application.status.value},
                    )
                notification_requeued = True
                application.updated_at = utcnow()
                self._audit(
                    session,
                    application,
                    event_type="DEAD_LETTER_REQUEUED",
                    from_status=application.status.value,
                    to_status=application.status.value,
                    detail={"eventsRequeued": requeued},
                    correlation_id=application.correlation_id,
                )
                session.commit()
            else:
                requeued = requeue_dead_letters(session, application_id)
                application.processing_attempts = 0
                application.status = ApplicationStatus.PROCESSING
                application.next_attempt_at = None
                application.failure_code = None
                application.failure_message = None
                application.updated_at = utcnow()
                self._audit(
                    session,
                    application,
                    event_type="MANUAL_RETRY_REQUESTED",
                    from_status=None,
                    to_status=ApplicationStatus.PROCESSING.value,
                    detail={"deadLettersRequeued": requeued},
                    correlation_id=application.correlation_id,
                )
                session.commit()
        if notification_requeued:
            logger.info(
                "dead_letters_requeued",
                extra={"applicationId": application_id},
            )
            return self.get_application(application_id)
        logger.info("application_manual_retry", extra={"applicationId": application_id})
        return self.process_application(application_id)

    def due_application_ids(self, *, limit: int = 50) -> list[str]:
        """Applications the reconciler should look at.

        Two populations: retryable failures whose backoff has elapsed, and rows
        stuck in ``PROCESSING`` (a crashed request, or one that exceeded the
        stale threshold while a dependency hung).
        """
        now = utcnow()
        stale_before = now - timedelta(seconds=self._stale_seconds)
        # The attempt bound applies to *retry due* rows only: a stale PROCESSING
        # row must still be picked up when attempts are exhausted, otherwise it
        # would sit in PROCESSING forever instead of being failed terminally by
        # ``_begin_attempt`` (PROCESSING_RETRY_EXHAUSTED).
        retry_due = and_(
            Application.status == ApplicationStatus.PENDING_RETRY,
            Application.next_attempt_at.is_not(None),
            Application.next_attempt_at <= now,
            Application.processing_attempts < self._max_attempts,
        )
        stale_processing = and_(
            Application.status == ApplicationStatus.PROCESSING,
            Application.updated_at <= stale_before,
        )
        statement = (
            select(Application.id)
            .where(or_(retry_due, stale_processing))
            .order_by(Application.updated_at)
            .limit(limit)
        )
        with self._sessions() as session:
            return list(session.scalars(statement))

    def reconcile_once(self, *, limit: int = 50) -> ReconcileStats:
        """Re-drive parked/stale applications; safe to run concurrently."""
        stats = ReconcileStats()
        for application_id in self.due_application_ids(limit=limit):
            try:
                application = self.process_application(application_id)
            except ApplicationNotFoundError:  # pragma: no cover - deleted mid-flight
                stats.skipped += 1
                continue
            if application.status.is_final:
                stats.decided += 1
            elif application.status is ApplicationStatus.FAILED:
                stats.failed += 1
            elif application.status is ApplicationStatus.PENDING_RETRY:
                stats.retried += 1
            else:
                stats.skipped += 1
        if stats.retried or stats.decided or stats.failed:
            logger.info(
                "reconciler_run",
                extra={
                    "retried": stats.retried,
                    "decided": stats.decided,
                    "failed": stats.failed,
                },
            )
        return stats

