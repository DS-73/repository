"""Mock external Applicant API with fault injection.

The brief requires the external dependency to simulate success, timeout, 5xx,
malformed payloads and slow responses.  Three ways to trigger a scenario:

1. **Applicant id suffixes** (handy for curl demos, no setup needed):
   ``COMP-A-500`` answers 500, ``COMP-A-TIMEOUT`` hangs, ``COMP-A-MALFORMED``
   returns a broken body, ``COMP-A-SLOW`` answers after a long delay,
   ``COMP-A-404`` does not exist, ``COMP-A-429`` is rate limited and
   ``COMP-A-MISMATCH`` answers with the wrong ``applicantId``.
2. **Query parameter** ``?mode=timeout`` for one-off manual testing.
3. **Control API** ``POST /_control/applicants/{id}`` - used by the automated
   tests to script "fail twice, then succeed".  A registered fault wins over a
   suffix / query parameter.

The mock is deliberately dumb (in-memory, single process) but it is a real HTTP
service, so the client under test exercises real sockets, timeouts and status
codes.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Lock
from typing import Any

from fastapi import FastAPI, Header, Query, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from suretyseven.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


class FaultMode(StrEnum):
    OK = "ok"
    TIMEOUT = "timeout"
    SERVER_ERROR = "server_error"
    UNAVAILABLE = "unavailable"
    MALFORMED = "malformed"
    SLOW = "slow"
    NOT_FOUND = "not_found"
    RATE_LIMITED = "rate_limited"
    MISMATCHED_ID = "mismatched_id"


SUFFIX_MODES: dict[str, FaultMode] = {
    "TIMEOUT": FaultMode.TIMEOUT,
    "500": FaultMode.SERVER_ERROR,
    "503": FaultMode.UNAVAILABLE,
    "MALFORMED": FaultMode.MALFORMED,
    "SLOW": FaultMode.SLOW,
    "404": FaultMode.NOT_FOUND,
    "429": FaultMode.RATE_LIMITED,
    "MISMATCH": FaultMode.MISMATCHED_ID,
}

#: A couple of canned companies so demos/tests have predictable numbers.
KNOWN_APPLICANTS: dict[str, dict[str, Any]] = {
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
    "COMP-ZEROREVENUE": {
        "annualRevenue": 0,
        "yearsInBusiness": 12,
        "creditScore": 780,
        "existingExposure": 0,
    },
}


@dataclass
class Fault:
    """Scripted behaviour for one applicant id."""

    mode: FaultMode = FaultMode.OK
    times: int | None = None  # None -> forever, N -> next N calls only
    delay_seconds: float = 5.0
    data: dict[str, Any] | None = None
    status_code: int = 500
    body: str = "{not-json"


@dataclass
class Behaviour:
    faults: dict[str, Fault] = field(default_factory=dict)
    requests: list[dict[str, Any]] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)

    def set(self, applicant_id: str, fault: Fault) -> None:
        with self.lock:
            self.faults[applicant_id] = fault

    def clear(self, applicant_id: str | None = None) -> None:
        with self.lock:
            if applicant_id is None:
                self.faults.clear()
                self.requests.clear()
            else:
                self.faults.pop(applicant_id, None)

    def next_fault(self, applicant_id: str, override: FaultMode | None) -> Fault | None:
        with self.lock:
            if override is not None:
                return Fault(mode=override)
            fault = self.faults.get(applicant_id)
            if fault is None:
                return None
            if fault.times is not None:
                if fault.times <= 0:
                    return None
                fault.times -= 1
                if fault.times <= 0:
                    self.faults.pop(applicant_id, None)
            return fault

    def record(self, entry: dict[str, Any]) -> None:
        with self.lock:
            self.requests.append(entry)

    def snapshot(self) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.requests)


BEHAVIOUR = Behaviour()


def synthetic_profile(applicant_id: str) -> dict[str, Any]:
    """Deterministic pseudo-random company data derived from the id."""
    seed = int(hashlib.sha256(applicant_id.encode()).hexdigest()[:8], 16)
    rnd = random.Random(seed)
    return {
        "annualRevenue": float(rnd.choice([1_000_000, 3_500_000, 12_000_000, 25_000_000])),
        "yearsInBusiness": rnd.randint(1, 20),
        "creditScore": rnd.choice([620, 665, 715, 745, 760, 812]),
        "existingExposure": float(rnd.choice([100_000, 500_000, 2_000_000, 6_000_000])),
    }


# ------------------------------------------------------------------ contracts
class FaultRequest(BaseModel):
    mode: FaultMode = FaultMode.OK
    times: int | None = Field(default=None, ge=0)
    delaySeconds: float = 5.0
    statusCode: int = 500
    data: dict[str, Any] | None = None


class FaultResponse(BaseModel):
    applicantId: str | None = None
    mode: FaultMode | None = None
    registered: bool = False


configure_logging()
app = FastAPI(
    title="Mock External Applicant API",
    description="Stand-in for the upstream applicant data provider, with fault injection.",
    version="0.1.0",
)


def _resolve_fault(applicant_id: str, mode: FaultMode | None) -> Fault | None:
    registered = BEHAVIOUR.next_fault(applicant_id, None)
    if registered is not None:
        return registered
    if mode is not None:
        return Fault(mode=mode)
    suffix = mode_from_suffix(applicant_id)
    if suffix is not None:
        return Fault(mode=suffix)
    return None


@app.get("/external/applicants/{applicant_id}")
def get_applicant(
    applicant_id: str,
    response: Response,
    mode: FaultMode | None = Query(default=None),
    x_correlation_id: str | None = Header(default=None, alias="X-Correlation-ID"),
) -> Response:
    """Return applicant enrichment data, or simulate a failure."""
    fault = _resolve_fault(applicant_id, mode)
    BEHAVIOUR.record(
        {
            "applicantId": applicant_id,
            "mode": (fault.mode.value if fault else FaultMode.OK.value),
            "correlationId": x_correlation_id,
            "at": time.time(),
        }
    )
    logger.info(
        "mock_applicant_lookup",
        extra={
            "applicantId": applicant_id,
            "mode": fault.mode.value if fault else "ok",
            "correlationId": x_correlation_id,
        },
    )

    if fault is None or fault.mode is FaultMode.OK:
        return JSONResponse(
            content={"applicantId": applicant_id, **profile_for(applicant_id)}
        )

    match fault.mode:
        case FaultMode.TIMEOUT | FaultMode.SLOW:
            time.sleep(fault.delay_seconds)
            return JSONResponse(content={"applicantId": applicant_id, **profile_for(applicant_id)})
        case FaultMode.NOT_FOUND:
            return JSONResponse(status_code=404, content={"error": "applicant not found"})
        case FaultMode.RATE_LIMITED:
            return JSONResponse(
                status_code=429,
                content={"error": "too many requests"},
                headers={"Retry-After": "1"},
            )
        case FaultMode.UNAVAILABLE:
            return JSONResponse(status_code=503, content={"error": "upstream unavailable"})
        case FaultMode.SERVER_ERROR:
            return JSONResponse(
                status_code=fault.status_code, content={"error": "internal upstream error"}
            )
        case FaultMode.MALFORMED:
            return Response(content=fault.body, media_type="application/json", status_code=200)
        case FaultMode.MISMATCHED_ID:
            return JSONResponse(
                content={"applicantId": "SOMEONE-ELSE", **profile_for(applicant_id)}
            )

    raise AssertionError("unreachable")  # pragma: no cover


def profile_for(applicant_id: str) -> dict[str, Any]:
    if applicant_id in KNOWN_APPLICANTS:
        return dict(KNOWN_APPLICANTS[applicant_id])
    if applicant_id.upper().endswith("ZEROREVENUE"):
        return dict(KNOWN_APPLICANTS["COMP-ZEROREVENUE"])
    return synthetic_profile(applicant_id)


def mode_from_suffix(applicant_id: str) -> FaultMode | None:
    parts = applicant_id.upper().rsplit("-", 1)
    if len(parts) != 2:
        return None
    return SUFFIX_MODES.get(parts[1])


# --------------------------------------------------------------- control API
@app.post("/_control/applicants/{applicant_id}", response_model=FaultResponse)
def set_fault(applicant_id: str, payload: FaultRequest) -> FaultResponse:
    """Script the behaviour of one applicant id (used by tests and demos)."""
    BEHAVIOUR.set(
        applicant_id,
        Fault(
            mode=payload.mode,
            times=payload.times,
            delay_seconds=payload.delaySeconds,
            data=payload.data,
            status_code=payload.statusCode,
        ),
    )
    return FaultResponse(applicantId=applicant_id, mode=payload.mode, registered=True)


@app.delete("/_control/applicants/{applicant_id}", response_model=FaultResponse)
def clear_fault(applicant_id: str) -> FaultResponse:
    BEHAVIOUR.clear(applicant_id)
    return FaultResponse(applicantId=applicant_id, registered=False)


@app.post("/_control/reset")
def reset() -> dict[str, Any]:
    BEHAVIOUR.clear()
    return {"status": "reset"}


@app.get("/_control/requests")
def list_requests() -> dict[str, Any]:
    """What the mock has been asked for - lets tests assert call counts."""
    requests = BEHAVIOUR.snapshot()
    return {"count": len(requests), "requests": requests}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "mock-applicant-api"}

