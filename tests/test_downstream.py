"""Downstream transport: HTTP callback + in-process fallback.

These tests use ``httpx.MockTransport`` so no sockets are opened but the real
request/response handling (status mapping, headers, error translation) runs.
"""

from __future__ import annotations

import httpx
import pytest

from suretyseven.downstream import (
    HttpDownstreamNotifier,
    LoggingDownstreamNotifier,
    build_notifier,
)
from suretyseven.errors import DownstreamDeliveryError
from suretyseven.schemas import DecisionEventPayload


def _event(event_id: str = "EVT-1") -> DecisionEventPayload:
    return DecisionEventPayload.model_validate(
        {
            "eventId": event_id,
            "applicationId": "APP-1",
            "applicantId": "COMP-123",
            "bondType": "performance",
            "decision": "APPROVE",
            "status": "APPROVED",
            "score": 82,
            "scoreModelVersion": "v1",
            "correlationId": "req-abc",
        }
    )


def _notifier(handler) -> HttpDownstreamNotifier:
    return HttpDownstreamNotifier(
        "https://downstream.example", client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_accepted_delivery_sends_the_documented_contract() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["idempotency_key"] = request.headers.get("Idempotency-Key")
        seen["correlation"] = request.headers.get("X-Correlation-ID")
        seen["body"] = request.read()
        return httpx.Response(202, json={"accepted": True, "duplicate": False})

    receipt = _notifier(handler).publish(_event())

    assert receipt.accepted is True
    assert receipt.duplicate is False
    assert receipt.status_code == 202
    assert seen["path"] == "/events"
    assert seen["idempotency_key"] == "EVT-1"
    assert seen["correlation"] == "req-abc"
    body = seen["body"]
    assert isinstance(body, bytes)
    assert b'"eventId"' in body and b'"EVT-1"' in body


def test_duplicate_flag_from_the_consumer_marks_the_receipt() -> None:
    receipt = _notifier(lambda request: httpx.Response(200, json={"duplicate": True})).publish(
        _event()
    )
    assert receipt.accepted is True
    assert receipt.duplicate is True


def test_conflict_is_treated_as_a_delivered_duplicate() -> None:
    receipt = _notifier(lambda request: httpx.Response(409, json={"error": "seen"})).publish(
        _event()
    )
    assert receipt.accepted is True
    assert receipt.duplicate is True
    assert receipt.status_code == 409


@pytest.mark.parametrize("status", [400, 401, 413, 500, 503])
def test_rejected_delivery_raises_a_retryable_error(status: int) -> None:
    notifier = _notifier(lambda request: httpx.Response(status, json={"error": "nope"}))
    with pytest.raises(DownstreamDeliveryError) as excinfo:
        notifier.publish(_event())
    assert excinfo.value.detail["status"] == status


def test_network_failure_is_wrapped_not_leaked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(DownstreamDeliveryError, match="downstream delivery failed"):
        _notifier(handler).publish(_event())


def test_unparsable_success_body_is_tolerated() -> None:
    receipt = _notifier(lambda request: httpx.Response(202, content=b"<html>")).publish(_event())
    assert receipt.accepted is True
    assert receipt.detail == {}


def test_build_notifier_selects_the_transport() -> None:
    log = build_notifier(transport="log", url="", timeout_seconds=1.0)
    assert isinstance(log, LoggingDownstreamNotifier)

    http = build_notifier(transport="http", url="https://x.example", timeout_seconds=1.0)
    assert isinstance(http, HttpDownstreamNotifier)
    http.close()


def test_logging_notifier_records_the_event() -> None:
    notifier = LoggingDownstreamNotifier()
    receipt = notifier.publish(_event())
    assert receipt.accepted is True
    assert [event.event_id for event in notifier.received] == ["EVT-1"]
