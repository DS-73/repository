"""Cross-cutting HTTP surface: auth, validation, ops endpoints, abuse controls.

These tests exercise the middleware and exception handlers rather than the
underwriting domain (covered by the other test modules).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from suretyseven.ratelimit import SlidingWindowRateLimiter


# ------------------------------------------------------------------- security
def test_missing_api_key_is_rejected(client: TestClient) -> None:
    response = client.post("/applications", json={})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHENTICATED"
    assert response.headers["WWW-Authenticate"] == "ApiKey"


def test_wrong_api_key_is_rejected(client: TestClient) -> None:
    response = client.post("/applications", json={}, headers={"X-API-Key": "nope"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHENTICATED"


def test_ops_endpoints_are_unauthenticated(client: TestClient) -> None:
    assert client.get("/health").status_code == 200

    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["checks"]["database"] == "ok"

    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert "ss_http_requests_total" in metrics.text


# -------------------------------------------------------------- validation
def test_invalid_payload_is_422_without_echoing_input(
    client: TestClient, auth_headers: dict[str, str], sample_payload: dict[str, Any]
) -> None:
    response = client.post(
        "/applications", json={**sample_payload, "bondAmount": -5}, headers=auth_headers
    )
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "REQUEST_VALIDATION_FAILED"
    errors = error["detail"]["errors"]
    assert errors
    # 'input' is stripped so a rejected payload value is never echoed back
    assert all("input" not in entry for entry in errors)


def test_list_limit_beyond_the_cap_is_rejected(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.get("/applications", params={"limit": 500}, headers=auth_headers)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "REQUEST_VALIDATION_FAILED"


def test_unknown_application_is_a_structured_404(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.get("/applications/APP-DOES-NOT-EXIST", headers=auth_headers)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "APPLICATION_NOT_FOUND"

    retry = client.post("/applications/APP-DOES-NOT-EXIST/retry", headers=auth_headers)
    assert retry.status_code == 404
    assert retry.json()["error"]["code"] == "APPLICATION_NOT_FOUND"


def test_unknown_route_is_a_structured_404(client: TestClient) -> None:
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


# ---------------------------------------------------------------- middleware
def test_correlation_id_is_echoed_or_generated(
    client: TestClient, auth_headers: dict[str, str], sample_payload: dict[str, Any]
) -> None:
    response = client.post(
        "/applications",
        json=sample_payload,
        headers={**auth_headers, "X-Correlation-ID": "req-mine-123"},
    )
    assert response.headers["X-Correlation-ID"] == "req-mine-123"
    assert float(response.headers["X-Response-Time-Ms"]) >= 0

    auto = client.get("/health")
    assert auto.headers["X-Correlation-ID"].startswith("req-")


def test_oversized_body_is_rejected_before_the_handlers(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    response = client.post(
        "/applications",
        content=b"{" + b"x" * 9_000 + b"}",
        headers={**auth_headers, "Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_rate_limiter_returns_429_with_retry_after(app: FastAPI, client: TestClient) -> None:
    # small window for the test; principal here is the client host (no API key)
    app.state.runtime.rate_limiter = SlidingWindowRateLimiter(limit=2, window_seconds=60)

    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 200

    limited = client.get("/health")
    assert limited.status_code == 429
    assert limited.json()["error"]["code"] == "RATE_LIMITED"
    assert int(limited.headers["Retry-After"]) >= 1

    # a different principal, so the metrics scrape is not itself throttled
    metrics = client.get("/metrics", headers={"X-API-Key": "observer"}).text
    assert "ss_rate_limited_requests_total" in metrics


# --------------------------------------------------------------------- ops
def test_root_reports_the_service_identity(client: TestClient) -> None:
    body = client.get("/").json()
    assert body["service"]
    assert body["version"]
    assert body["docs"] == "/docs"


def test_metrics_reflect_traffic_and_decisions(
    client: TestClient, auth_headers: dict[str, str], post_application, sample_payload
) -> None:
    post_application(sample_payload)
    text = client.get("/metrics").text
    assert "ss_applications_created_total" in text
    assert "ss_application_decisions_total" in text
    assert 'method="POST"' in text


def test_audit_trail_is_a_non_empty_ordered_timeline(
    client: TestClient, auth_headers: dict[str, str], post_application, sample_payload
) -> None:
    created = post_application(sample_payload).json()
    response = client.get(
        f"/applications/{created['applicationId']}/audit", headers=auth_headers
    )
    assert response.status_code == 200
    events = response.json()["events"]
    assert events, "every application must have an audit trail"
    names = [entry["eventType"] for entry in events]
    assert names[0] == "APPLICATION_ACCEPTED"
    assert "SCORING_COMPLETED" in names
    for entry in events:
        assert set(entry) >= {"eventType", "fromStatus", "toStatus", "detail"}
