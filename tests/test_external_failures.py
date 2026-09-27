"""Dependency failures: timeouts, bad responses, circuit breaker, graceful parking.

Requirement: the service must never return a misleading decision when the
Applicant API is unavailable - it parks the application in ``PENDING_RETRY``
with a machine readable failure code, and ``GET`` explains what happened.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from suretyseven.models import Application, ApplicationStatus, OutboxEvent


def get_one(app: FastAPI, application_id: str) -> Application:
    with app.state.runtime.database.session_factory() as session:
        application = session.scalar(
            select(Application).where(Application.id == application_id)
        )
        assert application is not None
        return application


def outbox_events(app: FastAPI, application_id: str) -> list[OutboxEvent]:
    with app.state.runtime.database.session_factory() as session:
        return list(
            session.scalars(
                select(OutboxEvent).where(OutboxEvent.application_id == application_id)
            )
        )


def _force_due(app: FastAPI, application_id: str) -> None:
    """Make a parked application's backoff already elapsed."""
    with app.state.runtime.database.session_factory() as session:
        application = session.get(Application, application_id)
        assert application is not None
        application.next_attempt_at = datetime.now(UTC) - timedelta(seconds=1)
        session.commit()


def _force_status(
    app: FastAPI,
    application_id: str,
    *,
    status: ApplicationStatus,
    attempts: int | None = None,
) -> None:
    """Simulate a row left behind by a crashed request."""
    with app.state.runtime.database.session_factory() as session:
        application = session.get(Application, application_id)
        assert application is not None
        application.status = status
        application.updated_at = datetime.now(UTC) - timedelta(seconds=400)
        if attempts is not None:
            application.processing_attempts = attempts
        session.commit()


def _audit_events(
    client: TestClient, auth_headers: dict[str, str], application_id: str
) -> list[str]:
    response = client.get(f"/applications/{application_id}/audit", headers=auth_headers)
    assert response.status_code == 200
    return [entry["eventType"] for entry in response.json()["events"]]


def test_timeout_parks_the_application_instead_of_faking_a_decision(
    applicant_api, post_application, sample_payload: dict[str, Any]
) -> None:
    applicant_api.mode = "timeout"

    response = post_application(sample_payload)
    assert response.status_code == 201  # the request itself succeeded...
    body = response.json()
    assert body["status"] == "PENDING_RETRY"
    assert body["decision"] is None  # ...but no decision was invented
    assert body["score"] is None
    assert body["failure"]["code"] == "EXTERNAL_TIMEOUT"
    assert body["failure"]["retryable"] is True


def test_malformed_response_is_parked_not_trusted(
    applicant_api, post_application, sample_payload: dict[str, Any]
) -> None:
    applicant_api.mode = "malformed"
    body = post_application(sample_payload).json()
    assert body["status"] == "PENDING_RETRY"
    assert body["failure"]["code"] == "EXTERNAL_MALFORMED_RESPONSE"
    assert body["decision"] is None


def test_server_error_is_parked_not_trusted(
    applicant_api, post_application, sample_payload: dict[str, Any]
) -> None:
    applicant_api.mode = "server_error"
    body = post_application(sample_payload).json()
    assert body["status"] == "PENDING_RETRY"
    assert body["failure"]["code"] == "EXTERNAL_UNAVAILABLE"
    assert body["decision"] is None


def test_transient_blip_recovers_within_the_same_request(
    applicant_api, post_application, sample_payload: dict[str, Any]
) -> None:
    """Retry policy: one failure then success -> the client still gets a decision."""
    applicant_api.failures_remaining = 1

    response = post_application(sample_payload)
    assert response.status_code == 201
    assert response.json()["status"] == "APPROVED"
    assert response.json()["decision"] == "APPROVE"
    assert applicant_api.calls == ["COMP-123", "COMP-123"]


def test_unknown_applicant_fails_permanently_immediately(
    applicant_api, post_application, sample_payload: dict[str, Any]
) -> None:
    """Retrying a 404 cannot help: the application fails with an explanation."""
    applicant_api.mode = "not_found"

    body = post_application(sample_payload).json()
    assert body["status"] == "FAILED"
    assert body["failure"]["code"] == "APPLICANT_NOT_FOUND"
    assert body["failure"]["retryable"] is False
    assert body["decision"] is None


def test_parked_application_is_listed_for_operators(
    client: TestClient, auth_headers: dict[str, str], applicant_api, post_application, sample_payload
) -> None:
    applicant_api.mode = "timeout"
    created = post_application(sample_payload).json()

    listing = client.get("/applications", params={"status": "PENDING_RETRY"}, headers=auth_headers)
    assert listing.status_code == 200
    items = listing.json()["items"]
    assert [item["applicationId"] for item in items] == [created["applicationId"]]
    assert items[0]["failure"]["code"] == "EXTERNAL_TIMEOUT"


# ------------------------------------------------------------------- reconciler
def test_reconciler_retries_a_parked_application_once_backoff_elapses(
    app: FastAPI,
    client: TestClient,
    auth_headers: dict[str, str],
    applicant_api,
    post_application,
    sample_payload: dict[str, Any],
) -> None:
    applicant_api.mode = "timeout"
    created = post_application(sample_payload).json()
    assert created["status"] == "PENDING_RETRY"

    applicant_api.recover()
    _force_due(app, created["applicationId"])
    stats = app.state.runtime.service.reconcile_once()
    assert stats.decided == 1

    body = client.get(f"/applications/{created['applicationId']}", headers=auth_headers).json()
    assert body["status"] == "APPROVED"
    assert body["decision"] == "APPROVE"
    assert body["failure"] is None


def test_gives_up_after_max_processing_attempts(
    app: FastAPI,
    client: TestClient,
    auth_headers: dict[str, str],
    applicant_api,
    post_application,
    sample_payload: dict[str, Any],
) -> None:
    """Persistent failures park, retry, and finally fail terminally."""
    applicant_api.mode = "server_error"  # max_processing_attempts = 3 in tests
    created = post_application(sample_payload).json()
    assert created["status"] == "PENDING_RETRY"

    for _ in range(2):  # attempts 2 and 3
        _force_due(app, created["applicationId"])
        app.state.runtime.service.reconcile_once()

    final = client.get(f"/applications/{created['applicationId']}", headers=auth_headers).json()
    assert final["status"] == "FAILED"
    # depends on whether the breaker opened first, but the code is always the
    # terminal explanation the operator sees
    assert final["failure"]["code"] in ("EXTERNAL_UNAVAILABLE", "CIRCUIT_OPEN")
    assert final["failure"]["retryable"] is False
    assert final["decision"] is None

    # no outbox event was produced: a failed application never claims a decision
    assert final["notification"] is None
    events = _audit_events(client, auth_headers, created["applicationId"])
    assert "PROCESSING_FAILED" in events


def test_stale_processing_row_is_recovered_by_the_reconciler(
    app: FastAPI,
    client: TestClient,
    auth_headers: dict[str, str],
    applicant_api,
    post_application,
    sample_payload: dict[str, Any],
) -> None:
    """Crash recovery: a request that died in PROCESSING gets re-driven."""
    applicant_api.mode = "timeout"
    created = post_application(sample_payload).json()
    _force_status(app, created["applicationId"], status=ApplicationStatus.PROCESSING)

    applicant_api.recover()
    app.state.runtime.service.reconcile_once()

    body = client.get(f"/applications/{created['applicationId']}", headers=auth_headers).json()
    assert body["status"] == "APPROVED"


def test_stale_processing_row_with_exhausted_attempts_fails_terminally(
    app: FastAPI,
    client: TestClient,
    auth_headers: dict[str, str],
    applicant_api,
    post_application,
    sample_payload: dict[str, Any],
) -> None:
    """Not picked up forever: exhausted attempts are failed, not re-driven."""
    applicant_api.mode = "server_error"
    created = post_application(sample_payload).json()
    _force_status(
        app,
        created["applicationId"],
        status=ApplicationStatus.PROCESSING,
        attempts=3,  # == max_processing_attempts
    )

    stats = app.state.runtime.service.reconcile_once()
    assert stats.failed == 1

    body = client.get(f"/applications/{created['applicationId']}", headers=auth_headers).json()
    assert body["status"] == "FAILED"
    assert body["failure"]["code"] == "PROCESSING_RETRY_EXHAUSTED"


# ------------------------------------------------------------- circuit breaker
def test_circuit_breaker_short_circuits_instead_of_hammering_the_api(
    applicant_api, post_application, sample_payload: dict[str, Any]
) -> None:
    """threshold = 3: after three upstream failures further calls are skipped."""
    applicant_api.mode = "server_error"

    for amount in (500_000, 500_001, 500_002):
        body = post_application({**sample_payload, "bondAmount": amount}).json()
        assert body["status"] == "PENDING_RETRY"
        assert body["decision"] is None

    # two client attempts for the first application, one more for the second
    # (the breaker then opened), and no HTTP calls at all for the third
    assert len(applicant_api.calls) == 3
    assert body["failure"]["code"] == "CIRCUIT_OPEN"
    assert body["failure"]["retryable"] is True


# ---------------------------------------------------------------- manual retry
def test_manual_retry_recovers_a_parked_application(
    client: TestClient,
    auth_headers: dict[str, str],
    applicant_api,
    post_application,
    sample_payload: dict[str, Any],
) -> None:
    applicant_api.mode = "timeout"
    created = post_application(sample_payload).json()
    assert created["status"] == "PENDING_RETRY"

    applicant_api.recover()
    response = client.post(
        f"/applications/{created['applicationId']}/retry", headers=auth_headers
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "APPROVED"
    assert body["failure"] is None


def test_manual_retry_of_a_decided_application_is_rejected(
    client: TestClient, auth_headers: dict[str, str], post_application, sample_payload
) -> None:
    created = post_application(sample_payload).json()
    assert created["status"] == "APPROVED"

    response = client.post(
        f"/applications/{created['applicationId']}/retry", headers=auth_headers
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "APPLICATION_NOT_RETRYABLE"

