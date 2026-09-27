"""Small time helpers so every timestamp in the system is UTC and comparable.

SQLite does not preserve timezone information, so values read back from the
database are naive.  ``ensure_utc`` normalises them to aware UTC datetimes at the
boundary, which keeps score/decision logic and JSON serialisation predictable.
"""

from __future__ import annotations

from datetime import UTC, datetime

__all__ = ["ensure_utc", "to_iso", "utcnow"]


def utcnow() -> datetime:
    """Timezone aware current UTC time."""
    return datetime.now(UTC)


def ensure_utc(value: datetime | None) -> datetime | None:
    """Return ``value`` as an aware UTC datetime (``None`` passes through)."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def to_iso(value: datetime | None) -> str | None:
    """Serialise a datetime as ISO-8601 with a ``Z`` suffix."""
    normalised = ensure_utc(value)
    if normalised is None:
        return None
    return normalised.isoformat().replace("+00:00", "Z")
