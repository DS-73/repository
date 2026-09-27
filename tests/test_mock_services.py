"""Tests for the mock services that ship with the deliverable.

The mocks are part of the brief (fault injection for the Applicant API and a
de-duplicating downstream consumer), so they are tested directly rather than
only through the double in ``conftest``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from suretyseven.mocks.applicant_app import app as applicant_app
from suretyseven.mocks.downstream_app import app as downstream_app

PATH = "/external/applicants/"


@pytest.fixture
def applicant_client() -> Iterator[TestClient]:
    with TestClient(applicant_app) as client:
        client.post("/_control/reset")
        yield client
        client.post("/_control/reset")


@pytest.fixture
def downstream_client() -> Iterator[TestClient]:
    with TestClient(downstream_app) as client:
        client.post("/_control/reset")
        yield client
        client.post("/_control/reset")


# ------------------------------------------------------------- applicant mock
def test_known_applicant_returns_the_canned_profile(applicant_client: TestClient) -> None:
    response = applicant_client.get(f"{PATH}COMP-123")
    assert response.status_code == 200
    body = response.json()
    assert body["applicantId"] == "COMP-123"
    assert body["creditScore"] == 760
    assert body["yearsInBusiness"] == 8


def test_suffixes_trigger_the_documented_fault_modes(applicant_client: TestClient) -> None:
    assert applicant_client.get(f"{PATH}COMP-A-500").status_code == 500
    assert applicant_client.get(f"{PATH}COMP-A-404").status_code == 404
    assert applicant_client.get(f"{PATH}COMP-A-429").status_code == 429

    malformed = applicant_client.get(f"{PATH}COMP-A-MALFORMED")
    assert malformed.status_code == 200
    with pytest.raises(json.JSONDecodeError):
        json.loads(malformed.content)

    mismatched = applicant_client.get(f"{PATH}COMP-A-MISMATCH")
    assert mismatched.status_code == 200
    assert mismatched.json()["applicantId"] == "SOMEONE-ELSE"


def test_control_api_can_fail_n_times_then_succeed(applicant_client: TestClient) -> None:
    registered = applicant_client.post(
        "/_control/applicants/COMP-SCRIPTED",
        json={"mode": "server_error", "times": 2, "statusCode": 503},
    )
    assert registered.status_code == 200
    assert registered.json() == {
        "applicantId": "COMP-SCRIPTED",
        "mode": "server_error",
        "registered": True,
    }

    statuses = [
        applicant_client.get(f"{PATH}COMP-SCRIPTED").status_code for _ in range(3)
    ]
    assert statuses == [503, 503, 200]

    log = applicant_client.get("/_control/requests").json()
    assert log["count"] == 3


def test_query_parameter_forces_a_mode_for_one_call(applicant_client: TestClient) -> None:
    assert applicant_client.get(f"{PATH}COMP-123?mode=server_error").status_code == 500
    assert applicant_client.get(f"{PATH}COMP-123").status_code == 200


def test_unknown_applicant_gets_a_deterministic_profile(applicant_client: TestClient) -> None:
    first = applicant_client.get(f"{PATH}COMP-UNKNOWN").json()
    second = applicant_client.get(f"{PATH}COMP-UNKNOWN").json()
    assert first == second
    assert first["applicantId"] == "COMP-UNKNOWN"
    assert 600 <= first["creditScore"] <= 850


def test_reset_clears_registered_faults(applicant_client: TestClient) -> None:
    applicant_client.post("/_control/applicants/COMP-BROKEN", json={"mode": "server_error"})
    assert applicant_client.get(f"{PATH}COMP-BROKEN").status_code == 500
    applicant_client.post("/_control/reset")
    assert applicant_client.get(f"{PATH}COMP-BROKEN").status_code == 200
    assert applicant_client.get("/_control/requests").json()["count"] == 1


# ------------------------------------------------------------ downstream mock
EVENT: dict[str, Any] = {
    "eventId": "EVT-TEST-1",
    "eventType": "APPLICATION_DECISIONED",
    "applicationId": "APP-TEST-1",
    "decision": "APPROVE",
    "score": 82,
}


def test_event_is_accepted_then_de_duplicated_by_event_id(
    downstream_client: TestClient,
) -> None:
    first = downstream_client.post("/events", json=EVENT)
    assert first.status_code == 202
    assert first.json()["accepted"] is True
    assert first.json()["duplicate"] is False

    second = downstream_client.post("/events", json=EVENT)
    assert second.status_code == 200
    assert second.json()["duplicate"] is True

    view = downstream_client.get("/_control/events").json()
    assert view["uniqueEvents"] == 1
    assert view["deliveries"] == 2
    assert view["duplicates"] == 1
    assert view["events"][0]["applicationId"] == "APP-TEST-1"


def test_injected_fault_then_recovery_makes_the_retry_succeed(
    downstream_client: TestClient,
) -> None:
    downstream_client.post("/_control/faults", json={"mode": "server_error", "times": 1})

    assert downstream_client.post("/events", json=EVENT).status_code == 500
    assert downstream_client.post("/events", json=EVENT).status_code == 202

    view = downstream_client.get("/_control/events").json()
    assert view["uniqueEvents"] == 1  # the failed delivery stored nothing


def test_downstream_reset_forgets_everything(downstream_client: TestClient) -> None:
    downstream_client.post("/events", json=EVENT)
    downstream_client.post("/_control/reset")
    view = downstream_client.get("/_control/events").json()
    assert view["uniqueEvents"] == 0
    assert view["deliveries"] == 0
