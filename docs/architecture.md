# Architecture — SuretySeven underwriting service

## 1. Shape of the system

```
                 ┌────────────────────────────────────────────────────────┐
  broker ───────►│ FastAPI (api.py)                                       │
  (X-API-Key,    │  middleware: rate limit, size gate, correlation id,    │
   idempotency)  │            metrics, error translation                  │
                 │  router: /applications, /audit, /retry, /health …      │
                 └───────────────┬────────────────────────────────────────┘
                                 │ ApplicationService (service.py)  ← single
                                 │                  author of state changes
        ┌────────────────────────┼─────────────────────────┬───────────────┐
        ▼                        ▼                         ▼               ▼
  SQLAlchemy (db.py)     ApplicantApiClient         scoring.py       outbox.py
  applications,          (external_client.py)       rules + bands    decision
  outbox_events,         timeout / retries /                       events in
  audit_events           circuit breaker                           same tx
        ▲                        ▲                                        │
        │                        │                                        ▼
  worker.py (background)   mock: suretyseven.mocks/            DownstreamNotifier
   ├─ reconciler: re-drive        applicant_app                (downstream.py)
   │  parked + stale rows         (fault injection)             HTTP POST /events
   └─ outbox_dispatch: publish                                (or "log" transport)
      due events
```

Everything that can fail outside our process (enrichment, notification) is
behind a port (`ApplicantApiClient`, `DownstreamNotifier`) so tests replace it
with an in-process double and the mocks are real HTTP services for demos.

## 2. Request path (happy case)

1. Middleware attaches/generates `X-Correlation-ID`, throttles by principal,
   rejects oversized bodies, times the request and records metrics.
2. `POST /applications` validates the body, de-duplicates on `Idempotency-Key`
   (or an identical body inside `duplicate_window_seconds`) and **commits the
   application row before any network call**.
3. Same request synchronously runs `process_application`: enrich → score →
   decide → audit + outbox enqueue **in one transaction**, then returns the
   final representation (`201`, or `200` on replay).
4. The background dispatcher publishes the outbox row; the HTTP response never
   waits for the downstream system.

The synchronous call gives brokers an immediate answer; if it cannot be given
(retryable failure) the row parks and the reconciler finishes the work later.
The API contract is identical either way: the client sees `PENDING_RETRY` and
polls `GET /applications/{id}`.

## 3. External dependency strategy

**Timeouts first.** The Applicant API call has a per-attempt timeout
(`SS_APPLICANT_API_TIMEOUT_SECONDS`), a bounded retry budget with jittered
exponential backoff, and a total deadline so a request cannot hang forever.

**Circuit breaker.** Consecutive failures trip
`SS_CIRCUIT_BREAKER_FAILURE_THRESHOLD` failures; while open, calls are skipped
immediately (fast `CIRCUIT_OPEN` failure) instead of hammering a sick upstream.
A half-open probe decides when to close again. This is why a bad upstream costs
the caller latency only for the first few attempts.

**Classification, not just retrying.**

| Symptom | Handling |
| --- | --- |
| timeout / connect error / 5xx / malformed body | retryable → `PENDING_RETRY` with backoff |
| `404 applicant` | permanent → `FAILED` immediately (`APPLICANT_NOT_FOUND`) |
| applicant id echoed back wrongly | data-integrity failure, not retried blindly |
| breaker open | retryable → parked, no HTTP call made |

Every outcome is written to `audit_events` with `failure.code`,
`retryable` and a message, so `GET /applications/{id}` and the audit trail
answer "why is this not decided?" without log access.

**Reconciler.** `due_application_ids()` selects (a) `PENDING_RETRY` rows whose
`next_attempt_at` has elapsed and (b) `PROCESSING` rows stale beyond
`stale_processing_seconds` — i.e. rows orphaned by a crashed request. Processing
is bounded by `max_processing_attempts`; exhausted rows become terminal
`FAILED / PROCESSING_RETRY_EXHAUSTED` rather than looping forever.

## 4. Notifications: transactional outbox

Publishing inside the request would either risk announcing a decision that
never commits, or losing the announcement if the process dies between commit
and publish. So:

1. `enqueue_decision_event(session, application)` inserts into `outbox_events`
   **in the caller's transaction** — `UNIQUE (application_id, event_type)` makes
   it idempotent and savepoint-safe under races.
2. `OutboxDispatcher` claims rows by compare-and-swap (`PENDING → IN_FLIGHT`,
   atomic update guarded on status) so N replicas cannot double-publish; a
   claim older than `stale_processing_seconds` is treated as abandoned and
   re-claimed (crash recovery).
3. Delivery is **at-least-once** with a stable `eventId` (also sent as
   `Idempotency-Key`); the consumer de-duplicates, making the end-to-end effect
   effectively-once. Failures re-queue with jittered backoff; after
   `outbox_max_attempts` the row is `DEAD_LETTER`, logged and metered.
4. Operators recover dead letters with `POST /applications/{id}/retry`: the
   decision stays final, only the notification is requeued (audited as
   `DEAD_LETTER_REQUEUED`).

A downstream outage therefore never affects underwriting: the decision, the
API response and the audit trail are already durable.

## 5. Idempotency

Surety brokers retry aggressively, so `POST /applications` is safe to replay:

* **Keyed**: with `Idempotency-Key`, the first call stores the key, a fingerprint
  of the body and the resulting application in `idempotency_records` (the
  application row also carries the key under a unique constraint, so a racing
  duplicate loses with an `IntegrityError` and gets the winner's answer). A
  replay with the same body returns the stored result (`200` +
  `Idempotent-Replay: true`); the same key with a *different* body is a client
  bug → `409 IDEMPOTENCY_KEY_REUSED`.
* **Unkeyed**: a key is derived from the body fingerprint, so identical bodies
  inside `SS_DUPLICATE_WINDOW_SECONDS` are de-duplicated too — protecting
  against blind client retries that forgot the header.

Replays never re-run scoring, never call the Applicant API twice and never
create a second outbox event.

## 6. Scoring

`scoring.py` turns the enriched applicant into a score **band** plus a decision
using explicit, configurable rules (thresholds in code/config, versioned via
`score_model_version`):

* strong credit + revenue coverage → `APPROVE`
* thin history, low revenue, high exposure → `DECLINE`
* anything explainable but borderline → `REFER` (human underwriter)

The decision, score, band and the contributing factors are stored on the row
and exposed by the API, so every outcome is auditable. The scoring module is
pure (no I/O), which is why `test_scoring.py` can assert boundaries
exhaustively without fixtures.

## 7. Data model

| Table | Role |
| --- | --- |
| `applications` | one row per submission: status, score, decision, failure, retry budget, `idempotency_key` (unique) |
| `idempotency_records` | key → body fingerprint → application binding used for replays and conflict detection |
| `audit_events` | append-only timeline (`from_status`, `to_status`, `detail`, `correlation_id`), ordered by id |
| `outbox_events` | pending notifications: status, attempts, backoff, claim owner, `last_error`, receipt |

SQLite by default (zero-setup for the exercise); the schema and session
handling are plain SQLAlchemy 2.0 and work unchanged on Postgres by setting
`SS_DATABASE_URL`.

## 8. Operations

* **Metrics** (`/metrics`, Prometheus): request counts/latency, decisions,
  processing outcomes, external call outcomes + latency, breaker state,
  outbox attempts/statuses, worker runs, rate-limited requests.
* **Logs**: single-line JSON with `correlationId` threaded through every hop,
  so one id joins API log → service decision → outbox delivery.
* **Probes**: `/health` (liveness) and `/readyz` (DB reachability).
* **Background work** (`worker.py`): one scheduler with short ticks; jobs are
  `reconciler` and `outbox_dispatch`, each with its own interval and metrics.
  `run_job(name)` also lets tests and an ops shell trigger them deterministically.

## 9. Trade-offs (honest list)

| Decision | Why | Cost |
| --- | --- | --- |
| SQLite + in-process workers | zero-dependency demo, tests are fast | single-writer; Postgres + one worker per replica for production |
| Synchronous decision in the POST | simple broker contract, immediate answer | a slow-but-working upstream lengthens the response (bounded by the deadline) |
| HTTP callback + outbox instead of a broker | one less system to run; at-least-once with de-dup is enough | consumers must de-duplicate on `eventId` |
| Static API key + in-process rate limit | demonstrates authn + throttling without an IdP | per-replica state; put it at the gateway (Redis) in production |
| Hand-rolled rules for scoring | deterministic and explainable, matches a take-home scope | replaceable by a model service behind the same interface |

Production checklist: Postgres, secrets from a vault, authn/z at the edge,
distributed rate limiting, OpenTelemetry traces on top of the correlation id,
and a dead-letter dashboard/alert on `ss_outbox_events_total{status="DEAD_LETTER"}`.

