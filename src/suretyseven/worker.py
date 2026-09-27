"""Background jobs: outbox dispatch and reconciliation.

Why a background thread instead of Celery/RQ/Kafka consumers?  For a single
service with one Postgres/SQLite database, a thread that runs two idempotent,
claim-based jobs keeps the operational surface tiny (``docker compose up`` and
nothing else to run).  Nothing in the job bodies assumes a single process, so
moving them into separate worker deployments later is a copy/paste job; the
outbox rows are the queue.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from suretyseven.logging_config import get_logger
from suretyseven.metrics import Metrics

logger = get_logger(__name__)


@dataclass(slots=True)
class Job:
    """A named, periodic unit of work."""

    name: str
    interval_seconds: float
    run: Callable[[], Any]
    last_run: float = field(default=0.0, init=False)

    def due(self, now: float) -> bool:
        return (now - self.last_run) >= self.interval_seconds


class BackgroundScheduler:
    """Runs jobs on a single daemon thread."""

    def __init__(
        self,
        jobs: list[Job],
        *,
        tick_seconds: float = 0.5,
        metrics: Metrics | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._jobs = {job.name: job for job in jobs}
        self._tick = tick_seconds
        self._metrics = metrics
        self._clock = clock
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def job_names(self) -> list[str]:
        return list(self._jobs)

    def start(self) -> None:
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, name="ss-scheduler", daemon=True)
        self._thread.start()
        logger.info("scheduler_started", extra={"jobs": self.job_names})

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("scheduler_stopped")

    def run_job(self, name: str) -> Any:
        """Run one job immediately (used by tests and by the API's admin paths)."""
        job = self._jobs[name]
        try:
            result = job.run()
        except Exception as exc:  # never let a job kill the scheduler
            logger.exception("scheduler_job_failed", extra={"job": name, "error": str(exc)})
            if self._metrics is not None:
                self._metrics.worker_runs.labels(job=name, result="error").inc()
            raise
        finally:
            job.last_run = self._clock()
        if self._metrics is not None:
            self._metrics.worker_runs.labels(job=name, result="ok").inc()
        return result

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            now = self._clock()
            for job in self._jobs.values():
                if self._stop_event.is_set():
                    break
                if not job.due(now):
                    continue
                try:
                    self.run_job(job.name)
                except Exception:  # already logged; keep the loop alive
                    continue
            self._stop_event.wait(self._tick)


def build_jobs(
    *,
    dispatcher: Any,
    service: Any,
    outbox_interval_seconds: float,
    reconciler_interval_seconds: float,
) -> list[Job]:
    """Wire the two system jobs onto a scheduler."""

    def dispatch() -> Any:
        return dispatcher.dispatch_due()

    def reconcile() -> Any:
        return service.reconcile_once()

    return [
        Job(name="outbox_dispatch", interval_seconds=outbox_interval_seconds, run=dispatch),
        Job(name="reconciler", interval_seconds=reconciler_interval_seconds, run=reconcile),
    ]
