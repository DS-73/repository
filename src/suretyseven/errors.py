"""Domain level errors.

The service deliberately translates infrastructure failures into a small set of
domain errors carrying a stable ``code`` and a ``retryable`` flag.  The HTTP
layer maps them to status codes, the worker uses ``retryable`` to decide whether
an application should be parked in ``PENDING_RETRY`` or failed permanently.
"""

from __future__ import annotations

from typing import Any


class ServiceError(Exception):
    """Base class for all errors raised by this service."""

    code = "INTERNAL_ERROR"
    http_status = 500
    retryable = False

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.detail:
            payload["detail"] = self.detail
        return payload


# --------------------------------------------------------------------- domain
class ApplicationNotFoundError(ServiceError):
    code = "APPLICATION_NOT_FOUND"
    http_status = 404


class IdempotencyConflictError(ServiceError):
    """Same ``Idempotency-Key`` was reused with a different request body."""

    code = "IDEMPOTENCY_KEY_REUSED"
    http_status = 409


class ApplicationNotRetryableError(ServiceError):
    code = "APPLICATION_NOT_RETRYABLE"
    http_status = 409


class BusinessValidationError(ServiceError):
    """The payload is syntactically valid but breaks a business rule."""

    code = "BUSINESS_VALIDATION_FAILED"
    http_status = 422


# -------------------------------------------------- external dependency errors
class ExternalDependencyError(ServiceError):
    """Base class for failures talking to the Applicant API."""

    code = "EXTERNAL_DEPENDENCY_ERROR"
    http_status = 502
    dependency = "applicant-api"


class ExternalTimeoutError(ExternalDependencyError):
    code = "EXTERNAL_TIMEOUT"
    http_status = 504
    retryable = True


class ExternalUnavailableError(ExternalDependencyError):
    """Connection errors, 5xx and 429 responses."""

    code = "EXTERNAL_UNAVAILABLE"
    http_status = 503
    retryable = True


class ExternalMalformedResponseError(ExternalDependencyError):
    """The dependency answered but the body could not be trusted."""

    code = "EXTERNAL_MALFORMED_RESPONSE"
    http_status = 502
    retryable = True


class ExternalNotFoundError(ExternalDependencyError):
    """The applicant does not exist - retrying will not help."""

    code = "APPLICANT_NOT_FOUND"
    http_status = 404
    retryable = False


class CircuitOpenError(ExternalDependencyError):
    """The circuit breaker is open; fail fast instead of hammering the API."""

    code = "CIRCUIT_OPEN"
    http_status = 503
    retryable = True


# ---------------------------------------------------------------- downstream
class DownstreamDeliveryError(ServiceError):
    code = "DOWNSTREAM_DELIVERY_FAILED"
    http_status = 502
    retryable = True
    dependency = "downstream"
