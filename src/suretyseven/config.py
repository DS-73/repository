"""Environment driven configuration (12-factor).

Every knob is configurable through an ``SS_`` prefixed environment variable so
the same image can run locally, in CI and in a container platform without code
changes.  Defaults are chosen so ``pytest`` and ``uvicorn`` work out of the box.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from suretyseven import __version__


class Settings(BaseSettings):
    """Runtime configuration for the API, the workers and the mocks."""

    model_config = SettingsConfigDict(
        env_prefix="SS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------ service
    service_name: str = "suretyseven-underwriting"
    environment: str = "local"
    version: str = __version__
    log_level: str = "INFO"
    log_json: bool = True

    # --------------------------------------------------------------- persistence
    database_url: str = "sqlite:///./data/suretyseven.db"
    db_echo: bool = False

    # -------------------------------------------------- external applicant API
    applicant_api_base_url: str = "http://localhost:8001"
    applicant_api_timeout_seconds: float = 2.0
    applicant_api_max_attempts: int = 3
    applicant_api_backoff_base_seconds: float = 0.2
    applicant_api_backoff_max_seconds: float = 2.0
    applicant_api_total_deadline_seconds: float = 6.0
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_reset_seconds: float = 15.0

    # ------------------------------------------------------- downstream system
    downstream_url: str = "http://localhost:8002"
    downstream_transport: str = "http"  # "http" | "log"
    downstream_timeout_seconds: float = 3.0
    outbox_batch_size: int = 25
    outbox_poll_seconds: float = 1.0
    outbox_max_attempts: int = 5
    outbox_backoff_base_seconds: float = 1.0
    outbox_backoff_max_seconds: float = 60.0

    # ------------------------------------------------------- background workers
    worker_enabled: bool = True
    worker_tick_seconds: float = 0.5
    reconciler_interval_seconds: float = 5.0
    stale_processing_seconds: float = 60.0
    max_processing_attempts: int = 5
    processing_backoff_base_seconds: float = 5.0
    processing_backoff_max_seconds: float = 300.0

    # ---------------------------------------------------------- idempotency
    #: Without a client supplied Idempotency-Key we still de-duplicate identical
    #: bodies inside this window (protects against blind client retries).
    duplicate_window_seconds: int = 600

    # ------------------------------------------------------------ security/abuse
    require_auth: bool = False
    api_key: str = "dev-api-key"
    rate_limit_requests: int = 120
    rate_limit_window_seconds: float = 60.0
    max_request_body_bytes: int = 65_536

    # ------------------------------------------------------------------ scoring
    scoring_config_path: Path | None = None

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()

    @field_validator("downstream_transport")
    @classmethod
    def _known_transport(cls, value: str) -> str:
        allowed = {"http", "log"}
        if value not in allowed:
            raise ValueError(f"downstream_transport must be one of {sorted(allowed)}")
        return value

    @model_validator(mode="after")
    def _check_ranges(self) -> Settings:
        positive = {
            "applicant_api_timeout_seconds": self.applicant_api_timeout_seconds,
            "applicant_api_total_deadline_seconds": self.applicant_api_total_deadline_seconds,
            "outbox_max_attempts": self.outbox_max_attempts,
            "outbox_poll_seconds": self.outbox_poll_seconds,
            "worker_tick_seconds": self.worker_tick_seconds,
            "max_processing_attempts": self.max_processing_attempts,
            "processing_backoff_base_seconds": self.processing_backoff_base_seconds,
            "duplicate_window_seconds": self.duplicate_window_seconds,
            "rate_limit_requests": self.rate_limit_requests,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be > 0, got {value}")
        if self.require_auth and not self.api_key:
            raise ValueError("SS_API_KEY must be set when SS_REQUIRE_AUTH=true")
        return self

    # ------------------------------------------------------------------ helpers
    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    def sqlite_path(self) -> Path | None:
        """Return the on-disk sqlite path (``None`` for in-memory databases)."""
        if not self.is_sqlite:
            return None
        if "///" in self.database_url:
            target = self.database_url.split("///", 1)[-1]
        else:
            target = self.database_url[len("sqlite://") :]
        if target in {"", ":memory:"} or ":memory:" in target:
            return None
        return Path(target)

    def ensure_runtime_dirs(self) -> None:
        """Create directories that the runtime expects to exist."""
        path = self.sqlite_path()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings instance."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached settings (used by tests)."""
    get_settings.cache_clear()
