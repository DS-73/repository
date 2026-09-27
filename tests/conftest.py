"""Shared test fixtures.

Tests never touch the network and never need Docker: the Applicant API is
replaced by an ``httpx.MockTransport`` double (so failures such as timeouts can
be produced deterministically), the downstream system by a recording notifier,
and the database by an in-memory SQLite instance.  The mock *services* are still
tested - in ``test_mock_services.py`` - because they are part of the deliverable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from suretyseven.api import create_app
from suretyseven.config import Settings
from suretyseven.db import Database
from suretyseven.downstream import DeliveryReceipt
from suretyseven.errors import DownstreamDeliveryError
from suretyseven.external_client import ApplicantApiClient, CircuitBreaker, RetryPolicy
from suretyseven.metrics import Metrics
from suretyseven.schemas import DecisionEventPayload

TEST_API_KEY = "test-api-key"


# --------------------------------------------------------------------- doubles
APPLICANT_PROFILES: dict[str, dict[str, Any]] = {
    "COMP-123": {
        "annualRevenue": 12_000_000,
        "yearsInBusiness": 8,
        "creditScore": 760,
        "existingExposure": 1_500_000,
    },
    "COMP-WEAK": {
        "annualRevenue": 800_000,
        "yearsInBusiness": 2,
        "creditScore": 620,
        "existingExposure": 400_000,
    },
    "COMP-MIDDLE": {
        "annualRevenue": 4_000_000,
        "yearsInBusiness": 3,
        "creditScore": 715,
        "existingExposure": 1_200_000,
    },
}


@dataclass
class FakeApplicantApi:
    """Configurable stand-in for the external Applicant API."""

    mode: str = "ok"  # ok | timeout | server_error | malformed | not_found | mismatch
    failures_remaining: int | None = None  # "fail N times, then succeed"
    status_code: int = 500
    body: str = "{not-json"
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def response_for(self, applicant_id: str) -> httpx.Response:
        profile = self.profiles.get(applicant_id, APPLICANT_PROFILES.get(applicant_id))
        if profile is None:
            profile = {
                "annualRevenue": 5_000_000,
                "yearsInBusiness": 6,
                "creditScore": 730,
                "existingExposure": 500_000,
            }
        payload = {"applicantId": applicant_id, **profile}
        match self.mode:
            case "timeout":
                raise httpx.ReadTimeout("simulated timeout", request=None)  # type: ignore[arg-type]
            case "server_error":
                return httpx.Response(self.status_code, json={"error": "boom"})
            case "malformed":
                return httpx.Response(
                    200, content=self.body, headers={"content-type": "application/json"}
                )
            case "not_found":
                return httpx.Response(404, json={"error": "not found"})
            case "mismatch":
                return httpx.Response(200, json={**payload, "applicantId": "SOMEONE-ELSE"})
            case _:
                return httpx.Response(200, json=payload)

    def handler(self, request: httpx.Request) -> httpx.Response:
        applicant_id = request.url.path.rsplit("/", 1)[-1]
        self.calls.append(applicant_id)
        if self.failures_remaining is not None and self.failures_remaining > 0:
            self.failures_remaining -= 1
            return httpx.Response(self.status_code, json={"error": "temporary blip"})
        return self.response_for(applicant_id)

    def http_client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def recover(self) -> None:
        self.mode = "ok"
        self.failures_remaining = 0


@dataclass
class RecordingNotifier:
    """In-memory downstream system, optionally failing on demand."""

    events: list[DecisionEventPayload] = field(default_factory=list)
    seen_ids: set[str] = field(default_factory=set)
    failures_remaining: int = 0
    duplicates: int = 0
    notify_calls: int = 0

    def fail_next(self, times: int = 1, status: int = 500) -> None:
        self.failures_remaining = times

    def publish(self, event: DecisionEventPayload) -> DeliveryReceipt:
        self.notify_calls += 1
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise DownstreamDeliveryError("simulated downstream failure")
        duplicate = event.event_id in self.seen_ids
        if duplicate:
            self.duplicates += 1
        else:
            self.seen_ids.add(event.event_id)
            self.events.append(event)
        return DeliveryReceipt(accepted=True, duplicate=duplicate, status_code=202)


# -------------------------------------------------------------------- fixtures
@pytest.fixture
def settings() -> Settings:
    return Settings(
        environment="test",
        database_url="sqlite://",  # in-memory; StaticPool keeps it alive
        require_auth=True,
        api_key=TEST_API_KEY,
        log_json=False,
        log_level="WARNING",
        worker_enabled=False,
        applicant_api_base_url="http://applicant.test",
        applicant_api_timeout_seconds=0.25,
        applicant_api_max_attempts=2,
        applicant_api_backoff_base_seconds=0.01,
        applicant_api_backoff_max_seconds=0.01,
        applicant_api_total_deadline_seconds=1.0,
        circuit_breaker_failure_threshold=3,
        circuit_breaker_reset_seconds=30.0,
        downstream_transport="log",
        outbox_max_attempts=3,
        outbox_backoff_base_seconds=0.01,
        outbox_backoff_max_seconds=0.01,
        processing_backoff_base_seconds=0.01,
        processing_backoff_max_seconds=0.01,
        max_processing_attempts=3,
        stale_processing_seconds=300.0,
        rate_limit_requests=1000,
        rate_limit_window_seconds=60.0,
        max_request_body_bytes=8192,
    )


@pytest.fixture
def applicant_api() -> FakeApplicantApi:
    return FakeApplicantApi()


@pytest.fixture
def notifier() -> RecordingNotifier:
    return RecordingNotifier()


@pytest.fixture
def database(settings: Settings) -> Iterator[Database]:
    db = Database(settings.database_url)
    yield db
    db.dispose()


@pytest.fixture
def metrics() -> Metrics:
    return Metrics()


@pytest.fixture
def make_applicant_client(
    settings: Settings, applicant_api: FakeApplicantApi, metrics: Metrics
):
    """Factory for an ``ApplicantApiClient`` bound to the in-process fake API."""

    def _make(client_settings: Settings | None = None) -> ApplicantApiClient:
        config = client_settings or settings
        return ApplicantApiClient(
            config.applicant_api_base_url,
            timeout_seconds=config.applicant_api_timeout_seconds,
            policy=RetryPolicy(
                max_attempts=config.applicant_api_max_attempts,
                base_delay_seconds=config.applicant_api_backoff_base_seconds,
                max_delay_seconds=config.applicant_api_backoff_max_seconds,
                total_deadline_seconds=config.applicant_api_total_deadline_seconds,
            ),
            breaker=CircuitBreaker(
                failure_threshold=config.circuit_breaker_failure_threshold,
                reset_timeout_seconds=config.circuit_breaker_reset_seconds,
            ),
            client=applicant_api.http_client(),
            metrics=metrics,
            sleeper=lambda _seconds: None,  # keep tests fast and deterministic
            jitter=lambda low, high: 0.0,
        )

    return _make


@pytest.fixture
def app(
    settings: Settings,
    database: Database,
    make_applicant_client,
    notifier: RecordingNotifier,
    metrics: Metrics,
) -> FastAPI:
    return create_app(
        settings,
        database=database,
        applicant_client=make_applicant_client(),
        notifier=notifier,
        metrics=metrics,
        enable_workers=False,
    )



@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture
def sample_payload() -> dict[str, Any]:
    return {
        "applicantId": "COMP-123",
        "bondType": "CONTRACT",
        "bondAmount": 500000,
        "effectiveDate": "2026-10-01",
        "obligee": {"name": "ABC Construction LLC"},
    }


@pytest.fixture
def post_application(
    client: TestClient, auth_headers: dict[str, str]
) -> Callable[..., httpx.Response]:
    """POST /applications helper (returns the raw response)."""

    def _post(
        payload: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        request_headers = {**auth_headers, **(headers or {})}
        if idempotency_key:
            request_headers["Idempotency-Key"] = idempotency_key
        return client.post("/applications", json=payload, headers=request_headers)

    return _post


@pytest.fixture
def run_job(app: FastAPI) -> Callable[[str], Any]:
    """Run one background job synchronously (the scheduler is disabled in tests)."""

    def _run(name: str) -> Any:
        return app.state.runtime.scheduler.run_job(name)

    return _run


@pytest.fixture
def drain(run_job: Callable[[str], Any]) -> Callable[[int], None]:
    """Run the outbox dispatcher + reconciler until the system is quiet."""

    def _drain(rounds: int = 4) -> None:
        for _ in range(rounds):
            run_job("outbox_dispatch")
            run_job("reconciler")

    return _drain

