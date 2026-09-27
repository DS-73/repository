"""Background scheduler: job wiring, manual runs, failure isolation, lifecycle."""

from __future__ import annotations

import time

from suretyseven.metrics import Metrics
from suretyseven.worker import BackgroundScheduler, Job, build_jobs


class _Clock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value


def test_build_jobs_wires_the_two_background_jobs() -> None:
    class FakeDispatcher:
        def dispatch_due(self) -> str:
            return "dispatched"

    class FakeService:
        def reconcile_once(self) -> str:
            return "reconciled"

    jobs = build_jobs(
        dispatcher=FakeDispatcher(),
        service=FakeService(),
        outbox_interval_seconds=1.5,
        reconciler_interval_seconds=7.0,
    )
    assert [job.name for job in jobs] == ["outbox_dispatch", "reconciler"]
    assert jobs[0].interval_seconds == 1.5
    assert jobs[1].interval_seconds == 7.0
    assert jobs[0].run() == "dispatched"
    assert jobs[1].run() == "reconciled"


def test_job_is_due_only_after_its_interval_elapses() -> None:
    job = Job(name="x", interval_seconds=10.0, run=lambda: None)
    assert job.due(now=100.0) is True  # never ran
    job.last_run = 100.0
    assert job.due(now=105.0) is False
    assert job.due(now=110.0) is True


def test_run_job_returns_the_result_and_records_metrics() -> None:
    metrics = Metrics()
    clock = _Clock()
    scheduler = BackgroundScheduler(
        [Job(name="j", interval_seconds=1.0, run=lambda: 42)],
        metrics=metrics,
        clock=clock,
    )

    assert scheduler.run_job("j") == 42
    assert scheduler._jobs["j"].last_run == clock.value
    assert scheduler.job_names == ["j"]


def test_a_failing_job_is_reported_but_does_not_silently_swallow() -> None:
    metrics = Metrics()

    def boom() -> None:
        raise RuntimeError("job exploded")

    scheduler = BackgroundScheduler([Job(name="bad", interval_seconds=1.0, run=boom)], metrics=metrics)

    try:
        scheduler.run_job("bad")
    except RuntimeError:
        pass
    else:
        raise AssertionError("run_job must re-raise so callers can see the failure")

    payload, _ = metrics.render()
    assert b'ss_worker_job_runs_total{job="bad",result="error"}' in payload


def test_unknown_job_name_is_an_error() -> None:
    scheduler = BackgroundScheduler([Job(name="j", interval_seconds=1.0, run=lambda: None)])
    try:
        scheduler.run_job("nope")
    except KeyError:
        pass
    else:
        raise AssertionError("unknown jobs must not silently do nothing")


def test_scheduler_runs_jobs_in_the_background_until_stopped() -> None:
    runs: list[str] = []

    scheduler = BackgroundScheduler(
        [Job(name="j", interval_seconds=0.0, run=lambda: runs.append("tick"))],
        tick_seconds=0.01,
    )
    assert scheduler.running is False

    scheduler.start()
    scheduler.start()  # idempotent
    assert scheduler.running is True

    deadline = time.monotonic() + 2.0
    while not runs and time.monotonic() < deadline:
        time.sleep(0.01)
    scheduler.stop()

    assert runs, "the background loop must have executed the job"
    assert scheduler.running is False
