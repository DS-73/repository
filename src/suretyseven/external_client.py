"""Resilient HTTP client for the external Applicant API.

Reliability features (the brief explicitly asks how we behave when this
dependency misbehaves):

* **Per-attempt timeout** - a slow dependency must not hold a request thread.
* **Bounded retries with exponential backoff and full jitter** - a blip is
  absorbed, a thundering herd after an outage is not.
* **Total deadline** - retries can never exceed the caller's patience; we stop
  retrying once the deadline would be crossed and surface the last error.
* **Circuit breaker** - after N consecutive failures we fail fast (and let the
  reconciler retry later) instead of queueing requests behind a dead dependency.
* **Strict response validation** - a 200 with garbage, a wrong ``applicantId``
  or a schema mismatch is treated as a failure, never silently scored.

Errors are mapped to the domain errors in :mod:`suretyseven.errors`, each of
which knows whether it is retryable.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock
from urllib.parse import quote

import httpx
from pydantic import ValidationError

from suretyseven.errors import (
    CircuitOpenError,
    ExternalDependencyError,
    ExternalMalformedResponseError,
    ExternalNotFoundError,
    ExternalTimeoutError,
    ExternalUnavailableError,
)
from suretyseven.logging_config import get_logger
from suretyseven.metrics import Metrics
from suretyseven.schemas import ExternalApplicantData

logger = get_logger(__name__)

DEPENDENCY = "applicant-api"


class CircuitState(StrEnum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry budget for a dependency call."""

    max_attempts: int = 3
    base_delay_seconds: float = 0.2
    max_delay_seconds: float = 2.0
    total_deadline_seconds: float = 6.0

    def delay_for(self, attempt: int) -> float:
        return min(self.max_delay_seconds, self.base_delay_seconds * (2 ** (attempt - 1)))


class CircuitBreaker:
    """Thread-safe circuit breaker (closed -> open -> half-open)."""

    def __init__(
        self,
        *,
        failure_threshold: int,
        reset_timeout_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._reset_timeout = reset_timeout_seconds
        self._clock = clock
        self._lock = Lock()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at: float | None = None
        self._probe_in_flight = False

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    def allow(self) -> bool:
        """Return True if a call may proceed (may transition OPEN -> HALF_OPEN)."""
        with self._lock:
            if self._state is CircuitState.CLOSED:
                return True
            assert self._opened_at is not None
            if self._clock() - self._opened_at < self._reset_timeout:
                return False
            if self._probe_in_flight:
                return False
            self._state = CircuitState.HALF_OPEN
            self._probe_in_flight = True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._opened_at = None
            self._probe_in_flight = False

    def record_failure(self) -> None:
        with self._lock:
            self._probe_in_flight = False
            if self._state is CircuitState.HALF_OPEN:
                self._trip()
                return
            self._failures += 1
            if self._failures >= self._failure_threshold:
                self._trip()

    def _trip(self) -> None:
        self._state = CircuitState.OPEN
        self._opened_at = self._clock()


class ApplicantApiClient:
    """Fetches applicant enrichment data with retries and a circuit breaker."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 2.0,
        policy: RetryPolicy | None = None,
        breaker: CircuitBreaker | None = None,
        client: httpx.Client | None = None,
        metrics: Metrics | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._policy = policy or RetryPolicy()
        self._breaker = breaker or CircuitBreaker(
            failure_threshold=5, reset_timeout_seconds=15.0, clock=clock
        )
        self._owns_client = client is None
        # A dedicated client keeps connection pooling (and TLS sessions) warm.
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(timeout_seconds), follow_redirects=False
        )
        self._metrics = metrics
        self._sleep = sleeper
        self._clock = clock
        self._jitter = jitter

    # ------------------------------------------------------------- public API
    @property
    def circuit_state(self) -> CircuitState:
        return self._breaker.state

    def fetch_applicant(
        self, applicant_id: str, *, correlation_id: str | None = None
    ) -> ExternalApplicantData:
        """Return enrichment data for ``applicant_id`` or raise a domain error."""
        url = f"{self._base_url}/external/applicants/{quote(applicant_id, safe='')}"
        headers = {"Accept": "application/json"}
        if correlation_id:
            headers["X-Correlation-ID"] = correlation_id

        deadline = self._clock() + self._policy.total_deadline_seconds
        attempts = 0
        last_error: ExternalDependencyError | None = None

        while attempts < self._policy.max_attempts:
            attempts += 1
            if not self._breaker.allow():
                raise CircuitOpenError(
                    "circuit breaker is open for the Applicant API",
                    detail={"attempts": attempts, "dependency": DEPENDENCY},
                )

            started = self._clock()
            try:
                response = self._client.get(
                    url,
                    headers=headers,
                    timeout=min(self._timeout, max(deadline - started, 0.01)),
                )
            except httpx.TimeoutException as exc:
                last_error = ExternalTimeoutError(
                    f"Applicant API timed out after {self._timeout}s",
                    detail={"attempts": attempts, "dependency": DEPENDENCY},
                )
                self._record_transport_failure(last_error, started, exc)
            except httpx.HTTPError as exc:  # connection reset, DNS, TLS, ...
                last_error = ExternalUnavailableError(
                    "Applicant API is unreachable",
                    detail={"attempts": attempts, "dependency": DEPENDENCY, "error": str(exc)},
                )
                self._record_transport_failure(last_error, started, exc)
            else:
                self._observe_latency(started)
                try:
                    return self._handle_response(response, applicant_id, attempts)
                except ExternalDependencyError as exc:
                    if not exc.retryable:
                        raise
                    last_error = exc

            if attempts >= self._policy.max_attempts:
                break
            delay = self._jitter(0.0, self._policy.delay_for(attempts))
            if self._clock() + delay >= deadline:
                logger.warning(
                    "external_call_deadline_exceeded",
                    extra={"dependency": DEPENDENCY, "attempts": attempts},
                )
                break
            logger.info(
                "external_call_retry",
                extra={
                    "dependency": DEPENDENCY,
                    "attempt": attempts,
                    "delaySeconds": round(delay, 3),
                    "error": last_error.code if last_error else None,
                },
            )
            self._sleep(delay)

        if last_error is None:  # defensive: loop always sets an error before exiting
            raise ExternalDependencyError("Applicant API call failed without a recorded error")
        last_error.detail["attempts"] = attempts
        raise last_error

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    # --------------------------------------------------------------- internals
    def _handle_response(
        self, response: httpx.Response, applicant_id: str, attempts: int
    ) -> ExternalApplicantData:
        status = response.status_code
        if status == httpx.codes.OK:
            try:
                payload = response.json()
            except ValueError as exc:
                raise self._malformed(f"body is not valid JSON: {exc}", attempts) from exc
            try:
                data = ExternalApplicantData.model_validate(payload)
            except ValidationError as exc:
                raise self._malformed(f"body failed schema validation: {exc.errors()}", attempts)
            if data.applicant_id != applicant_id:
                raise self._malformed(
                    f"upstream returned applicantId '{data.applicant_id}' for '{applicant_id}'",
                    attempts,
                )
            self._breaker.record_success()
            self._count_outcome("success")
            return data

        if status == httpx.codes.NOT_FOUND:
            # A healthy (but negative) answer: not a dependency failure and not
            # retryable - the applicant simply does not exist.
            self._breaker.record_success()
            self._count_outcome("not_found")
            raise ExternalNotFoundError(
                f"applicant '{applicant_id}' was not found by the Applicant API",
                detail={"attempts": attempts, "dependency": DEPENDENCY},
            )

        if status == httpx.codes.TOO_MANY_REQUESTS or status >= 500:
            self._breaker.record_failure()
            self._count_outcome(f"http_{status}")
            raise ExternalUnavailableError(
                f"Applicant API returned {status}",
                detail={"attempts": attempts, "dependency": DEPENDENCY, "status": status},
            )

        # Any other 4xx means our request was wrong; retrying cannot help.
        self._breaker.record_success()
        self._count_outcome(f"http_{status}")
        raise ExternalDependencyError(
            f"Applicant API rejected the request with {status}",
            detail={"attempts": attempts, "dependency": DEPENDENCY, "status": status},
        )

    def _malformed(self, reason: str, attempts: int) -> ExternalMalformedResponseError:
        self._breaker.record_failure()
        self._count_outcome("malformed")
        return ExternalMalformedResponseError(
            f"Applicant API returned an unusable response: {reason}",
            detail={"attempts": attempts, "dependency": DEPENDENCY},
        )

    def _record_transport_failure(
        self, error: ExternalDependencyError, started: float, exc: Exception
    ) -> None:
        self._breaker.record_failure()
        self._observe_latency(started)
        self._count_outcome(
            "timeout" if isinstance(exc, httpx.TimeoutException) else "unavailable"
        )
        logger.warning(
            "external_call_failed",
            extra={"dependency": DEPENDENCY, "errorCode": error.code, "error": str(exc)},
        )

    def _observe_latency(self, started: float) -> None:
        if self._metrics is not None:
            self._metrics.external_latency.labels(dependency=DEPENDENCY).observe(
                max(self._clock() - started, 0.0)
            )
            self._metrics.circuit_open.labels(dependency=DEPENDENCY).set(
                1 if self._breaker.state is CircuitState.OPEN else 0
            )

    def _count_outcome(self, outcome: str) -> None:
        if self._metrics is not None:
            self._metrics.external_calls.labels(dependency=DEPENDENCY, outcome=outcome).inc()

