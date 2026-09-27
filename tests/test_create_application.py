"""End-to-end behaviour of ``POST /applications`` and ``GET /applications/{id}``."""

from __future__ import annotations

from fastapi.testclient import TestClient

AUTH = {"X-API-Key": "test-api-key"}


def test_create_application_returns_decision(post_application) -> None:
    response = post_application(
        {
            "applicantId": "COMP-123",
            "bondType": "CONTRACT",
            "bondAmount": 500000,
            "effectiveDate": "2026-10-01",
            "obligee": {"name": "ABC Construction LLC"},
        }
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["applicationId"].startswith("APP-")
    assert body["status"] == "APPROVED"
    assert body["decision"] == "APPROVE"
    assert body["score"] == 100
    assert body["scoreModelVersion"]
    assert body["obligee"] == {"name": "ABC Construction LLC"}
    assert body["bondAmount"] == 500000
    assert body["createdAt"].endswith("Z")
    assert body["failure"] is None
    # Explainability: the breakdown says which rules fired.
    applied = {item["rule"] for item in body["scoreBreakdown"] if item["applied"]}
    assert applied == {
        "CREDIT_EXCELLENT",
        "TENURE_ESTABLISHED",
        "BOND_SMALL_VS_REVENUE",
        "EXPOSURE_LOW",
    }
    assert response.headers["Location"] == f"/applications/{body['applicationId']}"


def test_get_application_matches_create(client: TestClient, post_application) -> None:
    created = post_application(
        {
            "applicantId": "COMP-123",
            "bondType": "CONTRACT",
            "bondAmount": 500000,
            "effectiveDate": "2026-10-01",
            "obligee": {"name": "ABC Construction LLC"},
        }
    ).json()

    fetched = client.get(f"/applications/{created['applicationId']}", headers=AUTH)
    assert fetched.status_code == 200
    assert fetched.json()["status"] == created["status"]
    assert fetched.json()["score"] == created["score"]


def test_weak_applicant_is_scored_and_declined(post_application) -> None:
    response = post_application(
        {
            "applicantId": "COMP-WEAK",
            "bondType": "PERFORMANCE",
            "bondAmount": 400000,
            "effectiveDate": "2026-10-01",
            "obligee": {"name": "City of Springfield"},
        }
    )
    body = response.json()
    # 400k bond on 800k revenue = 50% -> 10 pts; 2 years -> 10; 620 credit -> 5;
    # exposure 400k on 800k = 50% -> 5 => 30 -> DECLINE
    assert body["score"] == 30
    assert body["decision"] == "DECLINE"
    assert body["status"] == "DECLINED"


def test_middle_applicant_is_referred(post_application) -> None:
    response = post_application(
        {
            "applicantId": "COMP-MIDDLE",
            "bondType": "PAYMENT",
            "bondAmount": 300000,
            "effectiveDate": "2026-10-01",
            "obligee": {"name": "Springfield Schools"},
        }
    )
    body = response.json()
    # 715 credit -> 20; 3 years -> 10; 300k/4M = 7.5% -> 30; 1.2M/4M = 30% -> 5 => 65
    assert body["score"] == 65
    assert body["decision"] == "REFER"
    assert body["status"] == "REFERRED"


def test_unknown_application_returns_404(client: TestClient) -> None:
    response = client.get("/applications/APP-DOES-NOT-EXIST", headers=AUTH)
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "APPLICATION_NOT_FOUND"
    assert body["error"]["correlationId"]


def test_audit_trail_explains_the_decision(client: TestClient, post_application) -> None:
    created = post_application(
        {
            "applicantId": "COMP-123",
            "bondType": "CONTRACT",
            "bondAmount": 500000,
            "effectiveDate": "2026-10-01",
            "obligee": {"name": "ABC Construction LLC"},
        }
    ).json()

    audit = client.get(f"/applications/{created['applicationId']}/audit", headers=AUTH)
    assert audit.status_code == 200
    events = audit.json()["events"]
    assert [event["eventType"] for event in events] == [
        "APPLICATION_ACCEPTED",
        "APPLICANT_DATA_FETCHED",
        "SCORING_COMPLETED",
        "DECISION_EVENT_ENQUEUED",
    ]
    scoring = next(event for event in events if event["eventType"] == "SCORING_COMPLETED")
    assert scoring["detail"]["score"] == 100
    assert scoring["detail"]["modelVersion"]
    assert scoring["toStatus"] == "APPROVED"
