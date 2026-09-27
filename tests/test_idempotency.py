"""Idempotency: replay, conflict, automatic de-duplication, race safety."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from suretyseven.models import Application, IdempotencyRecord

REPLAY_HEADER = "Idempotent-Replay"


def application_count(app: FastAPI) -> int:
    with app.state.runtime.database.session_factory() as session:
        return int(session.scalar(select(func.count()).select_from(Application)) or 0)


def test_explicit_key_replays_the_original_submission(
    post_application, sample_payload: dict[str, Any]
) -> None:
    first = post_application(sample_payload, idempotency_key="key-abc")
    assert first.status_code == 201
    assert REPLAY_HEADER not in first.headers

    second = post_application(sample_payload, idempotency_key="key-abc")
    # A replay is a success, not a new resource: 200 + explicit replay marker.
    assert second.status_code == 200
    assert second.headers[REPLAY_HEADER] == "true"
    assert second.json() == first.json()


def test_replayed_request_does_not_re_score(
    app: FastAPI, applicant_api, post_application, sample_payload: dict[str, Any]
) -> None:
    post_application(sample_payload, idempotency_key="once-only")
    post_application(sample_payload, idempotency_key="once-only")

    assert applicant_api.calls == ["COMP-123"]  # one lookup, not two
    assert application_count(app) == 1


def test_same_key_different_body_is_a_conflict(
    post_application, sample_payload: dict[str, Any]
) -> None:
    created = post_application(sample_payload, idempotency_key="shared")
    assert created.status_code == 201

    changed = {**sample_payload, "bondAmount": 999999}
    response = post_application(changed, idempotency_key="shared")
    assert response.status_code == 409
    error = response.json()["error"]
    assert error["code"] == "IDEMPOTENCY_KEY_REUSED"
    assert error["detail"]["applicationId"] == created.json()["applicationId"]


def test_key_is_compared_after_trimming(
    post_application, sample_payload: dict[str, Any]
) -> None:
    """A whitespace-padded key is the same key: clients get one application."""
    first = post_application(sample_payload, idempotency_key="  padded  ")
    second = post_application(sample_payload, idempotency_key="padded")
    assert first.json()["applicationId"] == second.json()["applicationId"]
    assert second.headers[REPLAY_HEADER] == "true"


def test_blank_and_oversized_keys_are_rejected(
    post_application, sample_payload: dict[str, Any]
) -> None:
    blank = post_application(sample_payload, idempotency_key="   ")
    assert blank.status_code == 422
    assert blank.json()["error"]["code"] == "BUSINESS_VALIDATION_FAILED"

    too_long = post_application(sample_payload, idempotency_key="k" * 200)
    assert too_long.status_code == 422
    assert too_long.json()["error"]["code"] == "BUSINESS_VALIDATION_FAILED"


def test_identical_body_without_a_key_is_de_duplicated_in_the_window(
    post_application, sample_payload: dict[str, Any], app: FastAPI
) -> None:
    """A retried request that lost its header still cannot create duplicates."""
    first = post_application(sample_payload)
    second = post_application(sample_payload)
    assert first.status_code == 201
    assert second.status_code == 200
    assert second.json()["applicationId"] == first.json()["applicationId"]
    assert application_count(app) == 1


def test_different_bodies_without_keys_are_all_accepted(
    post_application, sample_payload: dict[str, Any], app: FastAPI
) -> None:
    post_application(sample_payload)
    post_application({**sample_payload, "bondAmount": 250000})
    post_application({**sample_payload, "bondAmount": 777000})
    assert application_count(app) == 3


def test_expired_window_stops_suppressing_identical_requests(
    app: FastAPI,
    post_application,
    sample_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """De-duplication is time bounded: later, the same body is a new application."""
    import suretyseven.service as service_module

    clock = {"now": datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)}
    monkeypatch.setattr(service_module, "utcnow", lambda: clock["now"])
    window = app.state.runtime.service._duplicate_window

    first = post_application(sample_payload)
    assert first.status_code == 201

    clock["now"] += timedelta(seconds=window * 2)
    second = post_application(sample_payload)
    assert second.status_code == 201
    assert second.json()["applicationId"] != first.json()["applicationId"]


def test_idempotency_record_keeps_the_first_response(
    app: FastAPI, post_application, sample_payload: dict[str, Any]
) -> None:
    post_application(sample_payload, idempotency_key="snapshot")
    with app.state.runtime.database.session_factory() as session:
        record = session.get(IdempotencyRecord, "snapshot")
    assert record is not None
    assert record.response_status_code == 201
    assert record.response_body["status"] == "APPROVED"
    assert record.application_id == record.response_body["applicationId"]


def test_concurrent_identical_requests_create_one_application(
    settings,
    applicant_api,
    notifier,
    metrics,
    make_applicant_client,
    auth_headers: dict[str, str],
    sample_payload: dict[str, Any],
    tmp_path,
) -> None:
    """The unique index - not application logic - is the last line of defence.

    Uses a file-backed SQLite database so threads get separate connections, then
    races eight identical submissions with the same Idempotency-Key.
    """
    from suretyseven.api import create_app
    from suretyseven.db import Database

    race_settings = settings.model_copy(
        update={"database_url": f"sqlite:///{tmp_path / 'race.db'}"}
    )
    database = Database(race_settings.database_url)
    try:
        race_app = create_app(
            race_settings,
            database=database,
            applicant_client=make_applicant_client(race_settings),
            notifier=notifier,
            metrics=metrics,
            enable_workers=False,
        )
        with TestClient(race_app) as race_client:

            def submit(_index: int) -> tuple[int, str]:
                response = race_client.post(
                    "/applications",
                    json=sample_payload,
                    headers={**auth_headers, "Idempotency-Key": "race"},
                )
                return response.status_code, response.json()["applicationId"]

            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(submit, range(8)))
    finally:
        database.dispose()

    # Every caller is told the request was accepted: the winner gets 201, whoever
    # loses the race replays the winner's in-flight/completed resource with 200.
    assert {status for status, _ in results} <= {200, 201}
    assert len({application_id for _, application_id in results}) == 1
    assert applicant_api.calls.count("COMP-123") == 1

