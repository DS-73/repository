"""Prometheus metrics.

One :class:`Metrics` instance per application (own registry) so tests can create
many apps without hitting prometheus-client's duplicate-registration guard.
Metrics are optional at runtime: instrumentation never fails a request.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


class Metrics:
    """Container for the metric families this service exposes."""

    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()

        self.http_requests = Counter(
            "ss_http_requests_total",
            "HTTP requests by method, route template and status code.",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.http_latency = Histogram(
            "ss_http_request_duration_seconds",
            "HTTP request latency.",
            ["method", "path"],
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.applications_created = Counter(
            "ss_applications_created_total",
            "Applications accepted by the create endpoint.",
            ["replayed"],
            registry=self.registry,
        )
        self.idempotency_conflicts = Counter(
            "ss_idempotency_conflicts_total",
            "Requests rejected because an Idempotency-Key was reused with a new body.",
            registry=self.registry,
        )
        self.decisions = Counter(
            "ss_application_decisions_total",
            "Final underwriting decisions.",
            ["decision"],
            registry=self.registry,
        )
        self.processing_outcomes = Counter(
            "ss_application_processing_total",
            "Outcome of a processing attempt.",
            ["outcome"],
            registry=self.registry,
        )
        self.external_calls = Counter(
            "ss_external_calls_total",
            "Calls to the external Applicant API by outcome.",
            ["dependency", "outcome"],
            registry=self.registry,
        )
        self.external_latency = Histogram(
            "ss_external_call_duration_seconds",
            "External Applicant API latency (per HTTP attempt).",
            ["dependency"],
            buckets=_LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.circuit_open = Gauge(
            "ss_external_circuit_open",
            "1 when the circuit breaker for a dependency is open.",
            ["dependency"],
            registry=self.registry,
        )
        self.outbox_events = Counter(
            "ss_outbox_events_total",
            "Outbox events by type and terminal status.",
            ["event_type", "status"],
            registry=self.registry,
        )
        self.outbox_attempts = Counter(
            "ss_outbox_dispatch_attempts_total",
            "Outbox delivery attempts by result.",
            ["result"],
            registry=self.registry,
        )
        self.worker_runs = Counter(
            "ss_worker_job_runs_total",
            "Background job executions by job and result.",
            ["job", "result"],
            registry=self.registry,
        )
        self.rate_limited = Counter(
            "ss_rate_limited_requests_total",
            "Requests rejected by the rate limiter.",
            ["principal"],
            registry=self.registry,
        )

    def render(self) -> tuple[bytes, str]:
        """Return the exposition payload and its content type."""
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
