"""Transactional outbox: publish, retry, dead-letter, operator requeue.

The decision and the "tell the downstream system" intent are committed together;
publication happens later from the outbox.  Notifications are at-least-once with
a stable ``eventId`` so a retrying dispatcher cannot confuse the consumer.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import FastAPI
from sqlalchemy import func, select

from suretyseven.models import OutboxEvent, OutboxStatus


def _event(app: FastAPI, application_id: str) -> OutboxEvent:
    with app.state.runtime.database.session_factory() as session:
        row = session.scalar(
            select(OutboxEvent).where(OutboxEvent.application_id == application_id)
        )
        assert row is not None
        return row


def _event_count(app: FastAPI, application_id: str) -> int:
    with app.state.runtime.database.session_factory() as session:
        return int(
            session.scalar(
                select(func.count())
                .select_from(OutboxEvent)
                .where(OutboxEvent.application_id == application_id)
            )
        )


def _force_due(app: FastAPI, application_id: str) -> None:
    with app.state.runtime.database.session_factory() as session:
        row = session.scalar(
            select(OutboxEvent).where(OutboxEvent.application_id == application_id)
        )
        assert row is not None
        row.next_attempt_at = datetime.now(UTC) - timedelta(seconds=1)
        session.commit()


def _force_stale_claim(app: FastAPI, application_id: str) -> None:
    """Simulate a dispatcher that died after claiming but before publishing."""
    with app.state.runtime.database.session_factory() as session:
        row = session.scalar(
            select(OutboxEvent).where(OutboxEvent.application_id == application_id)
        )
        assert row is not None
        row.status = OutboxStatus.IN_FLIGHT
        row.claimed_by = "dead-worker"
        row.claimed_at = datetime.now(UTC) - timedelta(seconds=400)
        session.commit()


def test_decision_event_is_enqueued_with_the_decision(
    app: FastAPI, notifier, post_application, sample_payload: dict[str, Any]
) -> None:
    """Same transaction: a decision always has its intent recorded."""
    body = post_application(sample_payload).json()
    assert body["status"] == "APPROVED"

    row = _event(app, body["applicationId"])
    assert row.event_type == "APPLICATION_DECISIONED"
    assert row.status is OutboxStatus.PENDING
    assert row.payload["applicationId"] == body["applicationId"]
    assert row.payload["decision"] == "APPROVE"

    notification = body["notification"]
    assert notification["eventType"] == "APPLICATION_DECISIONED"
    assert notification["status"] == "PENDING"
    assert notification["attempts"] == 0

    # not dispatched yet: the job has not run
    assert notifier.events == []


def test_dispatcher_publishes_and_marks_the_event(
    app: FastAPI, client, auth_headers, notifier, post_application, sample_payload
) -> None:
    body = post_application(sample_payload).json()
    app.state.runtime.scheduler.run_job("outbox_dispatch")

    assert len(notifier.events) == 1
    assert notifier.events[0].application_id == body["applicationId"]

    row = _event(app, body["applicationId"])
    assert row.status is OutboxStatus.PUBLISHED
    assert row.published_at is not None
    assert row.attempts == 1
    assert row.last_error is None

    view = client.get(f"/applications/{body['applicationId']}", headers=auth_headers).json()
    assert view["notification"]["status"] == "PUBLISHED"
    assert view["notification"]["publishedAt"] is not None


def test_dispatch_never_publishes_the_same_decision_twice(
    app: FastAPI, notifier, post_application, sample_payload
) -> None:
    body = post_application(sample_payload).json()

    app.state.runtime.scheduler.run_job("outbox_dispatch")
    app.state.runtime.scheduler.run_job("outbox_dispatch")
    app.state.runtime.scheduler.run_job("outbox_dispatch")

    assert len(notifier.events) == 1
    assert notifier.duplicates == 0
    assert _event_count(app, body["applicationId"]) == 1


def test_downstream_outage_never_blocks_the_decision(
    app: FastAPI, notifier, post_application, sample_payload
) -> None:
    """Requirement: notification failure must not fail the application."""
    notifier.fail_next(50)

    body = post_application(sample_payload).json()
    assert body["status"] == "APPROVED"
    assert body["decision"] == "APPROVE"

    app.state.runtime.scheduler.run_job("outbox_dispatch")
    row = _event(app, body["applicationId"])
    assert row.status is OutboxStatus.PENDING  # will be retried, not lost
    assert row.attempts == 1
    assert row.last_error is not None


def test_failed_delivery_is_retried_with_backoff(
    app: FastAPI, notifier, post_application, sample_payload
) -> None:
    """Bounded retries: attempts grow and the next attempt is scheduled."""
    notifier.fail_next(50)
    body = post_application(sample_payload).json()

    app.state.runtime.scheduler.run_job("outbox_dispatch")
    row = _event(app, body["applicationId"])
    assert row.status is OutboxStatus.PENDING
    assert row.next_attempt_at is not None
    # SQLite hands timestamps back naive (UTC)
    assert row.next_attempt_at > datetime.now(UTC).replace(tzinfo=None) - timedelta(
        seconds=1
    )

    _force_due(app, body["applicationId"])
    app.state.runtime.scheduler.run_job("outbox_dispatch")
    row = _event(app, body["applicationId"])
    assert row.attempts == 2
    assert row.status is OutboxStatus.PENDING


def test_exhausted_retries_dead_letter_the_event(
    app: FastAPI, notifier, post_application, sample_payload
) -> None:
    """outbox_max_attempts = 3: the event stops retrying and is parked."""
    notifier.fail_next(50)
    body = post_application(sample_payload).json()

    for _ in range(3):
        _force_due(app, body["applicationId"])
        stats = app.state.runtime.scheduler.run_job("outbox_dispatch")

    row = _event(app, body["applicationId"])
    assert row.status is OutboxStatus.DEAD_LETTER
    assert row.attempts == 3
    assert row.last_error is not None
    assert stats.dead_lettered == 1


def test_operator_retry_requeues_a_dead_letter_and_delivery_recovers(
    app: FastAPI,
    client,
    auth_headers,
    notifier,
    post_application,
    sample_payload,
) -> None:
    """Decision is final, but its notification is recoverable by an operator."""
    notifier.fail_next(50)
    body = post_application(sample_payload).json()
    for _ in range(3):
        _force_due(app, body["applicationId"])
        app.state.runtime.scheduler.run_job("outbox_dispatch")
    assert _event(app, body["applicationId"]).status is OutboxStatus.DEAD_LETTER

    # downstream is healthy again
    notifier.fail_next(0)

    response = client.post(
        f"/applications/{body['applicationId']}/retry", headers=auth_headers
    )
    assert response.status_code == 202
    assert response.json()["status"] == "APPROVED"  # decision was not re-scored

    row = _event(app, body["applicationId"])
    assert row.status is OutboxStatus.PENDING
    assert row.attempts == 0
    assert row.last_error is None

    app.state.runtime.scheduler.run_job("outbox_dispatch")
    assert len(notifier.events) == 1
    assert _event(app, body["applicationId"]).status is OutboxStatus.PUBLISHED


def test_stale_in_flight_claim_is_recovered(
    app: FastAPI, notifier, post_application, sample_payload
) -> None:
    """A worker that died mid-dispatch does not strand the event."""
    body = post_application(sample_payload).json()
    _force_stale_claim(app, body["applicationId"])

    stats = app.state.runtime.scheduler.run_job("outbox_dispatch")
    assert stats.published == 1

    row = _event(app, body["applicationId"])
    assert row.status is OutboxStatus.PUBLISHED
    assert len(notifier.events) == 1


def test_parked_application_produces_no_notification(
    applicant_api, post_application, sample_payload
) -> None:
    """No decision -> no event: the downstream system never hears a maybe."""
    applicant_api.mode = "timeout"
    body = post_application(sample_payload).json()
    assert body["status"] == "PENDING_RETRY"
    assert body["notification"] is None
