"""Mock downstream system that consumes ``APPLICATION_DECISIONED`` events.

It demonstrates the consumer side of the reliability story:

* **De-duplication by ``eventId``** - a re-delivered event (at-least-once
  delivery) is recognised and answered with ``duplicate: true`` instead of being
  processed twice.  Two claims in the response keep the whole pipeline
  effectively-once.
* **Fault injection** - the control API can make it fail (5xx), time out or hang
  so the outbox retry/back-off/dead-letter path can be exercised end to end.
* **Inspection** - ``/_control/events`` lists what it has accepted so tests can
  assert exactly-once behaviour.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from threading import Lock
from typing import Any

from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from suretyseven.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


class DeliveryFault(StrEnum):
    OK = "ok"
    SERVER_ERROR = "server_error"
    TIMEOUT = "timeout"


@dataclass
class Store:
    events: dict[str, dict[str, Any]] = field(default_factory=dict)
    deliveries: int = 0
    duplicates: int = 0
    fault: DeliveryFault = DeliveryFault.OK
    fault_times: int | None = None
    delay_seconds: float = 5.0
    lock: Lock = field(default_factory=Lock)

    def record(self, event_id: str, payload: dict[str, Any], key: str | None) -> bool:
        """Store the event; returns True when it is a duplicate."""
        with self.lock:
            self.deliveries += 1
            if event_id in self.events:
                self.duplicates += 1
                return True
            self.events[event_id] = {"payload": payload, "idempotencyKey": key}
            return False

    def next_fault(self) -> DeliveryFault:
        with self.lock:
            fault = self.fault
            if fault is not DeliveryFault.OK and self.fault_times is not None:
                self.fault_times -= 1
                if self.fault_times <= 0:
                    self.fault = DeliveryFault.OK
                    self.fault_times = None
            return fault

    def reset(self) -> None:
        with self.lock:
            self.events.clear()
            self.deliveries = 0
            self.duplicates = 0
            self.fault = DeliveryFault.OK
            self.fault_times = None


STORE = Store()

configure_logging()
app = FastAPI(
    title="Mock Downstream System",
    description="Consumes APPLICATION_DECISIONED events, de-duplicating on eventId.",
    version="0.1.0",
)


class DeliveryFaultRequest(BaseModel):
    mode: DeliveryFault = DeliveryFault.OK
    times: int | None = Field(default=None, ge=0)
    delaySeconds: float = 5.0


@app.post("/events")
async def receive_event(
    request: Request,
    response: Response,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    x_correlation_id: str | None = Header(default=None, alias="X-Correlation-ID"),
) -> Response:
    """Accept a decision event (HTTP callback contract of the outbox)."""
    payload = await request.json()
    event_id = payload.get("eventId") or idempotency_key or "unknown"
    fault = STORE.next_fault()
    logger.info(
        "mock_downstream_received",
        extra={
            "eventId": event_id,
            "eventType": payload.get("eventType"),
            "applicationId": payload.get("applicationId"),
            "decision": payload.get("decision"),
            "score": payload.get("score"),
            "correlationId": x_correlation_id,
            "idempotencyKey": idempotency_key,
        },
    )

    if fault is DeliveryFault.TIMEOUT:
        time.sleep(STORE.delay_seconds)
    if fault is DeliveryFault.SERVER_ERROR:
        return JSONResponse(status_code=500, content={"error": "downstream boom"})

    duplicate = STORE.record(event_id, payload, idempotency_key)
    if duplicate:
        logger.warning("mock_downstream_duplicate_event", extra={"eventId": event_id})
        return JSONResponse(status_code=200, content={"accepted": True, "duplicate": True})
    return JSONResponse(
        status_code=202,
        content={
            "accepted": True,
            "duplicate": False,
            "eventId": event_id,
            "receivedAt": time.time(),
        },
    )



@app.post("/_control/faults")
def set_fault(payload: DeliveryFaultRequest) -> dict[str, Any]:
    """Make the consumer fail, so outbox retries can be observed end to end."""
    STORE.fault = payload.mode
    STORE.fault_times = payload.times
    STORE.delay_seconds = payload.delaySeconds
    return {"mode": STORE.fault.value, "times": STORE.fault_times}


@app.get("/_control/events")
def list_events() -> dict[str, Any]:
    """Everything accepted so far (de-duplicated view)."""
    return {
        "uniqueEvents": len(STORE.events),
        "deliveries": STORE.deliveries,
        "duplicates": STORE.duplicates,
        "events": [
            {
                "eventId": event_id,
                "applicationId": entry["payload"].get("applicationId"),
                "decision": entry["payload"].get("decision"),
                "score": entry["payload"].get("score"),
                "idempotencyKey": entry["idempotencyKey"],
            }
            for event_id, entry in STORE.events.items()
        ],
    }


@app.post("/_control/reset")
def reset() -> dict[str, str]:
    STORE.reset()
    return {"status": "reset"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "mock-downstream"}
