"""Structured logging with correlation IDs and PII redaction.

Design notes
------------
* Logs are JSON so they can be shipped to any log platform without a parser.
* A ``contextvars`` based context carries ``correlation_id``/``application_id``
  so every log line emitted while handling a request (including inside the
  background worker) can be tied back to the originating call.
* A redaction filter scrubs the fields we consider sensitive (credit score,
  revenue, obligee names, API keys).  We never log raw request bodies.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

CORRELATION_ID: ContextVar[str | None] = ContextVar("correlation_id", default=None)
APPLICATION_ID: ContextVar[str | None] = ContextVar("application_id", default=None)

#: Keys whose values must never reach a log sink.
SENSITIVE_KEYS = frozenset(
    {
        "creditScore",
        "credit_score",
        "annualRevenue",
        "annual_revenue",
        "existingExposure",
        "existing_exposure",
        "obligee",
        "obligee_name",
        "obligeeName",
        "apiKey",
        "api_key",
        "x-api-key",
        "authorization",
        "ssn",
        "taxId",
        "tax_id",
    }
)

REDACTED = "***redacted***"

_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
) | {"message", "asctime", "taskName"}


def redact(value: Any) -> Any:
    """Recursively redact sensitive keys from a mapping/list structure."""
    if isinstance(value, dict):
        return {
            key: (REDACTED if key in SENSITIVE_KEYS else redact(item))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter - no third party dependency needed."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC)
            .isoformat()
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        correlation_id = CORRELATION_ID.get()
        if correlation_id:
            payload["correlationId"] = correlation_id
        application_id = APPLICATION_ID.get()
        if application_id:
            payload["applicationId"] = application_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = redact(value)
        return json.dumps(payload, default=str)


class PlainFormatter(logging.Formatter):
    """Human friendly formatter for local development (``SS_LOG_JSON=false``)."""

    def format(self, record: logging.LogRecord) -> str:
        correlation_id = CORRELATION_ID.get() or "-"
        base = (
            f"{self.formatTime(record, '%Y-%m-%dT%H:%M:%S%z')} "
            f"{record.levelname:<8} [{correlation_id}] {record.name}: {record.getMessage()}"
        )
        extras = {
            key: redact(value)
            for key, value in record.__dict__.items()
            if key not in _RESERVED and not key.startswith("_")
        }
        if extras:
            base += " " + json.dumps(extras, default=str)
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    """Install our stdout handler without disturbing other handlers.

    We tag and replace only the handler this function owns so that tools like
    pytest's ``caplog`` (which attaches its own handler) keep working.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else PlainFormatter())
    handler._ss_managed = True  # type: ignore[attr-defined]

    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not getattr(h, "_ss_managed", False)]
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Uvicorn installs its own handlers; route them through ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True
    # httpx logs every request at INFO which is too chatty for production.
    logging.getLogger("httpx").setLevel("WARNING")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
