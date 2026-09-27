"""HTTP contracts (request/response models).

camelCase on the wire (as in the assignment brief), snake_case in Python, via a
pydantic alias generator.  Requests use ``extra="forbid"`` so typos and
unexpected fields fail loudly instead of being silently dropped - important when
the caller is an automated broker integration.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, field_validator
from pydantic.alias_generators import to_camel

from suretyseven.clock import to_iso, utcnow
from suretyseven.models import Application, ApplicationStatus, Decision

#: Serialise datetimes as ISO-8601 UTC with a ``Z`` suffix on the wire.
UtcDateTime = Annotated[
    datetime, PlainSerializer(to_iso, return_type=str, when_used="json")
]

#: Money values are stored as NUMERIC(16,2) but emitted as JSON numbers.
Money = Annotated[Decimal, PlainSerializer(float, return_type=float, when_used="json")]

# Business guard rails (documented in README "Validation rules").
MIN_BOND_AMOUNT = Decimal("1000")
MAX_BOND_AMOUNT = Decimal("100000000")
MAX_EFFECTIVE_DATE_PAST_DAYS = 365
MAX_EFFECTIVE_DATE_FUTURE_DAYS = 730


class BondType(StrEnum):
    """Bond products supported by this iteration of the service."""

    CONTRACT = "CONTRACT"
    PERFORMANCE = "PERFORMANCE"
    PAYMENT = "PAYMENT"
    SUPPLY = "SUPPLY"
    MAINTENANCE = "MAINTENANCE"
    LICENSE_PERMIT = "LICENSE_PERMIT"
    OTHER = "OTHER"


class Obligee(BaseModel):
    """The party the bond protects."""

    model_config = ConfigDict(
        alias_generator=to_camel, populate_by_name=True, extra="forbid"
    )

    name: str = Field(min_length=1, max_length=200)

    @field_validator("name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if len(cleaned) < 2:
            raise ValueError("obligee.name must contain at least 2 characters")
        return cleaned


class CreateApplicationRequest(BaseModel):
    """POST /applications body (see the assignment brief)."""

    model_config = ConfigDict(
        alias_generator=to_camel, populate_by_name=True, extra="forbid"
    )

    applicant_id: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
    bond_type: BondType
    bond_amount: Decimal = Field(ge=MIN_BOND_AMOUNT, le=MAX_BOND_AMOUNT)
    effective_date: date
    obligee: Obligee

    @field_validator("bond_amount")
    @classmethod
    def _two_decimals(cls, value: Decimal) -> Decimal:
        exponent = value.as_tuple().exponent
        if not isinstance(exponent, int):  # NaN/infinity cannot slip through
            raise ValueError("bondAmount must be a finite number")
        if exponent < -2:
            raise ValueError("bondAmount must not have more than 2 decimal places")
        return value.quantize(Decimal("0.01"))

    @field_validator("effective_date")
    @classmethod
    def _within_window(cls, value: date) -> date:
        today = utcnow().date()
        if value < today - timedelta(days=MAX_EFFECTIVE_DATE_PAST_DAYS):
            raise ValueError(
                f"effectiveDate must not be more than {MAX_EFFECTIVE_DATE_PAST_DAYS} days in the past"
            )
        if value > today + timedelta(days=MAX_EFFECTIVE_DATE_FUTURE_DAYS):
            raise ValueError(
                "effectiveDate must not be more than "
                f"{MAX_EFFECTIVE_DATE_FUTURE_DAYS} days in the future"
            )
        return value


class ExternalApplicantData(BaseModel):
    """Contract of the external Applicant API (also validates its mock)."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="ignore",  # upstream may add fields; we must not break
    )

    applicant_id: str = Field(min_length=1, max_length=64)
    annual_revenue: Decimal = Field(ge=0, le=Decimal("1000000000000"))
    years_in_business: int = Field(ge=0, le=200)
    credit_score: int = Field(ge=0, le=850)
    existing_exposure: Decimal = Field(ge=0)


class ScoreRuleOutcome(BaseModel):
    """One line of the score card, kept for explainability."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    rule: str
    metric: str
    description: str = ""
    observed: float | None = None
    points: int
    applied: bool
    note: str | None = None


class ApplicantDataOut(BaseModel):
    """External applicant data used for the decision (from the snapshot)."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    annual_revenue: Money
    years_in_business: int
    credit_score: int
    existing_exposure: Money
    fetched_at: UtcDateTime


class ApplicationFailure(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    code: str
    message: str
    retryable: bool


class NotificationState(BaseModel):
    """Outbox status so an operator can see if the downstream system knows."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    event_type: str
    status: str
    attempts: int
    last_error: str | None = None
    published_at: UtcDateTime | None = None


class ApplicationResponse(BaseModel):
    """GET /applications/{id} and POST /applications response."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    application_id: str
    applicant_id: str
    bond_type: str
    bond_amount: Money
    effective_date: date
    obligee: Obligee

    status: ApplicationStatus
    decision: Decision | None = None
    score: int | None = None
    score_model_version: str | None = None
    score_breakdown: list[ScoreRuleOutcome] | None = None
    decision_reasons: list[str] | None = None
    applicant: ApplicantDataOut | None = None
    failure: ApplicationFailure | None = None

    notification: NotificationState | None = None
    correlation_id: str | None = None
    created_at: UtcDateTime
    updated_at: UtcDateTime
    decided_at: UtcDateTime | None = None


def serialize_application(application: Application) -> ApplicationResponse:
    """Project the ORM aggregate onto the API contract."""
    snapshot = application.latest_snapshot
    failure = None
    if application.failure_code:
        failure = ApplicationFailure(
            code=application.failure_code,
            message=application.failure_message or "",
            retryable=application.status
            in (ApplicationStatus.PENDING_RETRY, ApplicationStatus.PROCESSING),
        )

    notification = None
    if application.outbox_events:
        latest = application.outbox_events[-1]
        notification = NotificationState(
            event_type=latest.event_type,
            status=latest.status.value,
            attempts=latest.attempts,
            last_error=latest.last_error,
            published_at=latest.published_at,
        )

    return ApplicationResponse(
        application_id=application.id,
        applicant_id=application.applicant_id,
        bond_type=application.bond_type,
        bond_amount=application.bond_amount,
        effective_date=application.effective_date,
        obligee=Obligee(name=application.obligee_name),
        status=application.status,
        decision=application.decision,
        score=application.score,
        score_model_version=application.score_model_version,
        score_breakdown=(
            [ScoreRuleOutcome.model_validate(row) for row in application.scoring_breakdown]
            if application.scoring_breakdown
            else None
        ),
        decision_reasons=application.decision_reasons,
        applicant=(
            ApplicantDataOut(
                annual_revenue=snapshot.annual_revenue,
                years_in_business=snapshot.years_in_business,
                credit_score=snapshot.credit_score,
                existing_exposure=snapshot.existing_exposure,
                fetched_at=snapshot.fetched_at,
            )
            if snapshot
            else None
        ),
        failure=failure,
        notification=notification,
        correlation_id=application.correlation_id,
        created_at=application.created_at,
        updated_at=application.updated_at,
        decided_at=application.decided_at,
    )




class AuditEventResponse(BaseModel):
    """One entry of the application timeline."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    event_type: str
    from_status: str | None = None
    to_status: str | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None
    created_at: UtcDateTime


class AuditTrailResponse(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    application_id: str
    events: list[AuditEventResponse]


class ApplicationListResponse(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    items: list[ApplicationResponse]
    total: int
    limit: int
    offset: int


class DecisionEventPayload(BaseModel):
    """Message published to the downstream system when a decision is final."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    event_id: str
    event_type: str = "APPLICATION_DECISIONED"
    schema_version: int = 1
    application_id: str
    applicant_id: str
    bond_type: str
    decision: Decision
    status: ApplicationStatus
    score: int
    score_model_version: str
    occurred_at: UtcDateTime = Field(default_factory=utcnow)
    correlation_id: str | None = None


class ErrorBody(BaseModel):
    code: str
    message: str
    detail: dict[str, Any] | None = None
    correlation_id: str | None = Field(default=None, alias="correlationId")


class ErrorResponse(BaseModel):
    error: ErrorBody


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, str]
