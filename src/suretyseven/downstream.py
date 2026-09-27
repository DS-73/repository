"""Downstream notification transport.

Choice (justified in ``docs/architecture.md``): the decision event is delivered
by an **HTTP callback to a downstream endpoint** driven by the transactional
outbox.  For a single team owning both sides, HTTP + a persistent outbox gives
at-least-once delivery, retries, back-pressure and an inspectable delivery state
without operating a broker.  Swapping in Kafka/SNS later is a matter of
implementing :class:`DownstreamNotifier` - the rest of the service is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from suretyseven.errors import DownstreamDeliveryError
from suretyseven.logging_config import get_logger
from suretyseven.metrics import Metrics
from suretyseven.schemas import DecisionEventPayload

logger = get_logger(__name__)

DELIVERY_PATH = "/events"


@dataclass(slots=True)
class DeliveryReceipt:
    """What the downstream system told us about the delivery."""

    accepted: bool
    duplicate: bool = False
    status_code: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)


class DownstreamNotifier(Protocol):
    """Port implemented by any downstream transport."""

    def publish(self, event: DecisionEventPayload) -> DeliveryReceipt:  # pragma: no cover
        ...


class HttpDownstreamNotifier:
    """POSTs decision events to ``{base_url}/events``."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 3.0,
        client: httpx.Client | None = None,
        metrics: Metrics | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + DELIVERY_PATH
        self._timeout = timeout_seconds
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=httpx.Timeout(timeout_seconds))
        self._metrics = metrics

    def publish(self, event: DecisionEventPayload) -> DeliveryReceipt:
        headers = {
            "Content-Type": "application/json",
            # The consumer de-duplicates on this header (and on eventId in body).
            "Idempotency-Key": event.event_id,
        }
        if event.correlation_id:
            headers["X-Correlation-ID"] = event.correlation_id
        try:
            response = self._client.post(
                self._url,
                json=event.model_dump(mode="json", by_alias=True),
                headers=headers,
                timeout=self._timeout,
            )
        except httpx.HTTPError as exc:
            raise DownstreamDeliveryError(
                f"downstream delivery failed: {exc}", detail={"eventId": event.event_id}
            ) from exc

        if response.status_code in (200, 201, 202):
            body = _safe_json(response)
            duplicate = bool(body.get("duplicate", False))
            if self._metrics is not None:
                self._metrics.outbox_attempts.labels(
                    result="duplicate" if duplicate else "delivered"
                ).inc()
            return DeliveryReceipt(
                accepted=True,
                duplicate=duplicate,
                status_code=response.status_code,
                detail=body,
            )

        if response.status_code == 409:
            # The consumer already has this event -> treat as delivered.
            if self._metrics is not None:
                self._metrics.outbox_attempts.labels(result="duplicate").inc()
            return DeliveryReceipt(
                accepted=True,
                duplicate=True,
                status_code=response.status_code,
                detail=_safe_json(response),
            )

        if self._metrics is not None:
            self._metrics.outbox_attempts.labels(result=f"http_{response.status_code}").inc()
        raise DownstreamDeliveryError(
            f"downstream rejected the event with {response.status_code}",
            detail={"eventId": event.event_id, "status": response.status_code},
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


class LoggingDownstreamNotifier:
    """In-process notifier: records the event and accepts it.

    Used when ``SS_DOWNSTREAM_TRANSPORT=log`` (unit tests, or a local run without
    the downstream mock) so the service still reaches a terminal state.
    """

    def __init__(self, metrics: Metrics | None = None) -> None:
        self._metrics = metrics
        self.received: list[DecisionEventPayload] = []

    def publish(self, event: DecisionEventPayload) -> DeliveryReceipt:
        logger.info(
            "downstream_event_published",
            extra={
                "eventId": event.event_id,
                "eventType": event.event_type,
                "applicationId": event.application_id,
                "decision": event.decision.value,
                "score": event.score,
            },
        )
        self.received.append(event)
        if self._metrics is not None:
            self._metrics.outbox_attempts.labels(result="delivered").inc()
        return DeliveryReceipt(accepted=True, status_code=202)


def build_notifier(
    *,
    transport: str,
    url: str,
    timeout_seconds: float,
    metrics: Metrics | None = None,
    client: httpx.Client | None = None,
) -> DownstreamNotifier:
    """Factory used by the app wiring (keeps ``httpx`` out of ``api.py``)."""
    if transport == "log" or not url:
        return LoggingDownstreamNotifier(metrics=metrics)
    return HttpDownstreamNotifier(
        url, timeout_seconds=timeout_seconds, client=client, metrics=metrics
    )


def _safe_json(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}

