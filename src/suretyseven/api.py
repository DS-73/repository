"""HTTP layer: routing, middleware, error envelope, app wiring.

Deliberate choices
------------------
* **Synchronous endpoints.**  DB and HTTP clients are sync; FastAPI runs ``def``
  endpoints in a worker thread, which keeps the code simple and is well within
  the throughput this problem needs.  Swapping in async drivers later does not
  change the service layer.
* **Every response carries a correlation id** (``X-Correlation-ID``), which is
  also propagated to the Applicant API and the downstream event so one id
  follows a request across the whole system.
* **One error envelope** ``{"error": {"code", "message", "detail",
  "correlationId"}}`` for every failure, including 422 validation failures, so
  broker integrations can branch on a stable ``code``.
"""

from __future__ import annotations

import secrets
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Header, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from suretyseven import __version__
from suretyseven.config import Settings, get_settings
from suretyseven.db import Database
from suretyseven.downstream import DownstreamNotifier, build_notifier
from suretyseven.errors import ServiceError
from suretyseven.external_client import ApplicantApiClient, CircuitBreaker, RetryPolicy
from suretyseven.logging_config import (
    APPLICATION_ID,
    CORRELATION_ID,
    configure_logging,
    get_logger,
)
from suretyseven.metrics import Metrics
from suretyseven.models import ApplicationStatus
from suretyseven.outbox import OutboxDispatcher
from suretyseven.ratelimit import SlidingWindowRateLimiter
from suretyseven.schemas import (
    ApplicationListResponse,
    ApplicationResponse,
    AuditEventResponse,
    AuditTrailResponse,
    CreateApplicationRequest,
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    ReadinessResponse,
    serialize_application,
)
from suretyseven.scoring import ScoreModelConfig, load_score_model
from suretyseven.service import ApplicationService
from suretyseven.worker import BackgroundScheduler, build_jobs

logger = get_logger(__name__)

CORRELATION_HEADER = "X-Correlation-ID"
REPLAY_HEADER = "Idempotent-Replay"


@dataclass(slots=True)
class Runtime:
    """All wired collaborators; stored on ``app.state`` (see ``create_app``)."""

    settings: Settings
    database: Database
    applicant_client: ApplicantApiClient
    notifier: DownstreamNotifier
    service: ApplicationService
    dispatcher: OutboxDispatcher
    scheduler: BackgroundScheduler
    metrics: Metrics
    rate_limiter: SlidingWindowRateLimiter


def get_runtime(request: Request) -> Runtime:
    return request.app.state.runtime


def get_service(request: Request) -> ApplicationService:
    return get_runtime(request).service


# ------------------------------------------------------------------ security
def require_api_key(
    request: Request,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> str:
    """Static API-key auth.

    Assumption (documented in the README): brokers are authenticated at the
    platform edge with a real identity provider; for this exercise a shared
    static key is enough to demonstrate that endpoints are not anonymous and
    that abusive callers can be throttled per principal.  Admin endpoints such
    as retry would additionally require an ``underwriter`` scope in production.
    """
    settings = get_runtime(request).settings
    if not settings.require_auth:
        return "anonymous"
    if not x_api_key:
        raise _auth_error("missing X-API-Key header")
    if not secrets.compare_digest(x_api_key, settings.api_key):
        raise _auth_error("invalid API key")
    return x_api_key


def _auth_error(message: str) -> StarletteHTTPException:
    return StarletteHTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=message,
        headers={"WWW-Authenticate": "ApiKey"},
    )


# ----------------------------------------------------------------- middleware
def _error_payload(
    code: str,
    message: str,
    *,
    detail: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    body = ErrorBody(code=code, message=message, detail=detail, correlationId=correlation_id)
    return ErrorResponse(error=body).model_dump(mode="json", by_alias=True)


def install_middleware(app: FastAPI) -> None:
    """One middleware so ordering is explicit and easy to reason about."""

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        runtime: Runtime = request.app.state.runtime
        correlation_id = request.headers.get(CORRELATION_HEADER) or f"req-{uuid.uuid4().hex}"
        token = CORRELATION_ID.set(correlation_id)
        request.state.correlation_id = correlation_id
        has_api_key = request.headers.get("X-API-Key") is not None
        principal = request.headers.get("X-API-Key") or (
            request.client.host if request.client else "unknown"
        )
        started = time.perf_counter()
        response: Response

        try:
            gate = _gate(request, runtime, principal, correlation_id)
            if gate is not None:
                response = gate
            else:
                response = await call_next(request)
        except Exception as exc:  # last-resort guard: never leak a stack trace
            logger.exception("unhandled_request_error", extra={"httpPath": request.url.path})
            response = _json_error(
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                "INTERNAL_ERROR",
                "unexpected server error",
                correlation_id,
                detail={"errorType": type(exc).__name__},
            )
        finally:
            duration = time.perf_counter() - started
            route = request.scope.get("route")
            path_label = getattr(route, "path", request.url.path)
            runtime.metrics.http_latency.labels(
                method=request.method, path=path_label
            ).observe(duration)
            runtime.metrics.http_requests.labels(
                method=request.method, path=path_label, status=str(response.status_code)
            ).inc()
            logger.info(
                "http_request",
                extra={
                    "httpMethod": request.method,
                    "httpPath": request.url.path,
                    "httpStatus": response.status_code,
                    "durationMs": round(duration * 1000, 2),
                    "principalType": "api_key" if has_api_key else "ip",
                },
            )
            CORRELATION_ID.reset(token)

        response.headers[CORRELATION_HEADER] = correlation_id
        response.headers["X-Response-Time-Ms"] = f"{duration * 1000:.1f}"
        return response


def _gate(
    request: Request, runtime: Runtime, principal: str, correlation_id: str
) -> JSONResponse | None:
    """Reject oversized or abusive requests before they reach the handlers."""
    settings = runtime.settings

    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > settings.max_request_body_bytes:
        return _json_error(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            "PAYLOAD_TOO_LARGE",
            f"request body exceeds {settings.max_request_body_bytes} bytes",
            correlation_id,
        )

    allowed, retry_after = runtime.rate_limiter.check(principal)
    if not allowed:
        runtime.metrics.rate_limited.labels(
            principal="api_key" if request.headers.get("X-API-Key") else "ip"
        ).inc()
        response = _json_error(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "RATE_LIMITED",
            "too many requests, slow down",
            correlation_id,
        )
        response.headers["Retry-After"] = str(int(retry_after) + 1)
        return response
    return None


# --------------------------------------------------------------------- routes
def build_router() -> APIRouter:
    router = APIRouter()

    @router.post(
        "/applications",
        response_model=ApplicationResponse,
        status_code=status.HTTP_201_CREATED,
        summary="Submit a bond application and run the underwriting pipeline",
        responses={200: {"description": "Idempotent replay of an earlier submission"}},
    )
    def create_application(
        payload: CreateApplicationRequest,
        request: Request,
        response: Response,
        service: Annotated[ApplicationService, Depends(get_service)],
        _principal: Annotated[str, Depends(require_api_key)],
        idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
    ) -> ApplicationResponse:
        result = service.create_application(
            payload,
            idempotency_key=idempotency_key,
            correlation_id=getattr(request.state, "correlation_id", None),
        )
        response.status_code = result.status_code
        response.headers["Location"] = f"/applications/{result.application.id}"
        if result.replayed:
            response.headers[REPLAY_HEADER] = "true"
        return serialize_application(result.application)

    @router.get(
        "/applications/{application_id}",
        response_model=ApplicationResponse,
        summary="Current status, score and decision of an application",
    )
    def get_application(
        application_id: str,
        service: Annotated[ApplicationService, Depends(get_service)],
        _principal: Annotated[str, Depends(require_api_key)],
    ) -> ApplicationResponse:
        return serialize_application(service.get_application(application_id))

    @router.get(
        "/applications/{application_id}/audit",
        response_model=AuditTrailResponse,
        summary="Timeline explaining how the decision was reached",
    )
    def get_application_audit(
        application_id: str,
        service: Annotated[ApplicationService, Depends(get_service)],
        _principal: Annotated[str, Depends(require_api_key)],
    ) -> AuditTrailResponse:
        events = service.get_audit_trail(application_id)
        return AuditTrailResponse(
            application_id=application_id,
            events=[AuditEventResponse.model_validate(event.to_dict()) for event in events],
        )

    @router.post(
        "/applications/{application_id}/retry",
        response_model=ApplicationResponse,
        status_code=status.HTTP_202_ACCEPTED,
        summary="Operator action: retry a parked or failed application",
    )
    def retry_application(
        application_id: str,
        service: Annotated[ApplicationService, Depends(get_service)],
        _principal: Annotated[str, Depends(require_api_key)],
    ) -> ApplicationResponse:
        return serialize_application(service.retry_application(application_id))

    @router.get(
        "/applications",
        response_model=ApplicationListResponse,
        summary="Operator view: recent applications",
    )
    def list_applications(
        service: Annotated[ApplicationService, Depends(get_service)],
        _principal: Annotated[str, Depends(require_api_key)],
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
        status_filter: Annotated[ApplicationStatus | None, Query(alias="status")] = None,
        applicant_id: Annotated[str | None, Query(alias="applicantId", max_length=64)] = None,
    ) -> ApplicationListResponse:
        rows, total = service.list_applications(
            limit=limit, offset=offset, status=status_filter, applicant_id=applicant_id
        )
        return ApplicationListResponse(
            items=[serialize_application(row) for row in rows],
            total=total,
            limit=limit,
            offset=offset,
        )

    return router


def build_ops_router() -> APIRouter:
    """Liveness/readiness/metrics - intentionally unauthenticated."""

    router = APIRouter()

    @router.get("/", include_in_schema=False)
    def root(request: Request) -> dict[str, Any]:
        settings = get_runtime(request).settings
        return {
            "service": settings.service_name,
            "version": __version__,
            "environment": settings.environment,
            "docs": "/docs",
            "openapi": "/openapi.json",
        }

    @router.get("/health", response_model=HealthResponse, summary="Liveness probe")
    def health(request: Request) -> HealthResponse:
        settings = get_runtime(request).settings
        return HealthResponse(
            status="ok", version=settings.version, environment=settings.environment
        )

    @router.get("/readyz", response_model=ReadinessResponse, summary="Readiness probe")
    def readyz(request: Request, response: Response) -> ReadinessResponse:
        runtime = get_runtime(request)
        checks = {"database": "ok" if runtime.database.is_ready() else "unavailable"}
        healthy = all(value == "ok" for value in checks.values())
        if not healthy:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(status="ok" if healthy else "degraded", checks=checks)

    @router.get("/metrics", summary="Prometheus metrics", include_in_schema=False)
    def metrics(request: Request) -> PlainTextResponse:
        payload, content_type = get_runtime(request).metrics.render()
        return PlainTextResponse(content=payload, media_type=content_type)

    return router




def _json_error(
    status_code: int,
    code: str,
    message: str,
    correlation_id: str | None,
    *,
    detail: dict[str, Any] | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=_error_payload(code, message, detail=detail, correlation_id=correlation_id),
        headers={CORRELATION_HEADER: correlation_id} if correlation_id else None,
    )


# --------------------------------------------------------- error translation
def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def _service_error(request: Request, exc: ServiceError) -> JSONResponse:
        correlation_id = getattr(request.state, "correlation_id", None)
        log = logger.warning if exc.http_status < 500 else logger.error
        log(
            "request_failed",
            extra={
                "errorCode": exc.code,
                "httpStatus": exc.http_status,
                "applicationId": APPLICATION_ID.get(),
            },
        )
        return _json_error(
            exc.http_status, exc.code, exc.message, correlation_id, detail=exc.detail or None
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        correlation_id = getattr(request.state, "correlation_id", None)
        # Strip 'input'/'ctx' so responses (and any logs built from them) never
        # echo payload values back.
        errors = [
            {key: value for key, value in error.items() if key in {"loc", "msg", "type"}}
            for error in exc.errors()
        ]
        return _json_error(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "REQUEST_VALIDATION_FAILED",
            "request body failed validation",
            correlation_id,
            detail={"errors": errors},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        correlation_id = getattr(request.state, "correlation_id", None)
        return_code = {
            401: "UNAUTHENTICATED",
            403: "FORBIDDEN",
            404: "NOT_FOUND",
            405: "METHOD_NOT_ALLOWED",
            429: "RATE_LIMITED",
        }.get(exc.status_code, "HTTP_ERROR")
        response = _json_error(
            exc.status_code,
            return_code,
            str(exc.detail) if exc.detail else "request failed",
            correlation_id,
        )
        if exc.headers:
            response.headers.update(exc.headers)
        return response

    @app.exception_handler(Exception)
    async def _unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception")
        return _json_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "INTERNAL_ERROR",
            "unexpected server error",
            getattr(request.state, "correlation_id", None),
            detail={"errorType": type(exc).__name__},
        )




# ------------------------------------------------------------------- factory
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: create schema, start/stop background jobs."""
    runtime: Runtime = app.state.runtime
    runtime.database.create_schema()
    if app.state.enable_workers:
        runtime.scheduler.start()
    logger.info(
        "service_started",
        extra={
            "environment": runtime.settings.environment,
            "version": __version__,
            "workers": app.state.enable_workers,
            "scoreModelVersion": runtime.service.score_model.version,
        },
    )
    try:
        yield
    finally:
        runtime.scheduler.stop()
        runtime.applicant_client.close()
        close = getattr(runtime.notifier, "close", None)
        if callable(close):
            close()
        runtime.database.dispose()
        logger.info("service_stopped")


def create_app(
    settings: Settings | None = None,
    *,
    database: Database | None = None,
    applicant_client: ApplicantApiClient | None = None,
    notifier: DownstreamNotifier | None = None,
    score_model: ScoreModelConfig | None = None,
    metrics: Metrics | None = None,
    enable_workers: bool | None = None,
) -> FastAPI:
    """Build the API.  Every collaborator can be injected for tests."""
    settings = settings or get_settings()
    settings.ensure_runtime_dirs()
    configure_logging(settings.log_level, settings.log_json)

    metrics = metrics or Metrics()
    database = database or Database(settings.database_url, echo=settings.db_echo)
    applicant_client = applicant_client or ApplicantApiClient(
        settings.applicant_api_base_url,
        timeout_seconds=settings.applicant_api_timeout_seconds,
        policy=RetryPolicy(
            max_attempts=settings.applicant_api_max_attempts,
            base_delay_seconds=settings.applicant_api_backoff_base_seconds,
            max_delay_seconds=settings.applicant_api_backoff_max_seconds,
            total_deadline_seconds=settings.applicant_api_total_deadline_seconds,
        ),
        breaker=CircuitBreaker(
            failure_threshold=settings.circuit_breaker_failure_threshold,
            reset_timeout_seconds=settings.circuit_breaker_reset_seconds,
        ),
        metrics=metrics,
    )
    notifier = notifier or build_notifier(
        transport=settings.downstream_transport,
        url=settings.downstream_url,
        timeout_seconds=settings.downstream_timeout_seconds,
        metrics=metrics,
    )
    score_model = score_model or load_score_model(settings.scoring_config_path)

    service = ApplicationService(
        database.session_factory,
        applicant_client,
        score_model,
        max_processing_attempts=settings.max_processing_attempts,
        stale_processing_seconds=settings.stale_processing_seconds,
        retry_backoff_base_seconds=settings.processing_backoff_base_seconds,
        retry_backoff_max_seconds=settings.processing_backoff_max_seconds,
        duplicate_window_seconds=settings.duplicate_window_seconds,
        metrics=metrics,
    )
    dispatcher = OutboxDispatcher(
        database.session_factory,
        notifier,
        batch_size=settings.outbox_batch_size,
        max_attempts=settings.outbox_max_attempts,
        backoff_base_seconds=settings.outbox_backoff_base_seconds,
        backoff_max_seconds=settings.outbox_backoff_max_seconds,
        stale_claim_seconds=settings.stale_processing_seconds,
        metrics=metrics,
    )
    scheduler = BackgroundScheduler(
        build_jobs(
            dispatcher=dispatcher,
            service=service,
            outbox_interval_seconds=settings.outbox_poll_seconds,
            reconciler_interval_seconds=settings.reconciler_interval_seconds,
        ),
        tick_seconds=settings.worker_tick_seconds,
        metrics=metrics,
    )
    runtime = Runtime(
        settings=settings,
        database=database,
        applicant_client=applicant_client,
        notifier=notifier,
        service=service,
        dispatcher=dispatcher,
        scheduler=scheduler,
        metrics=metrics,
        rate_limiter=SlidingWindowRateLimiter(
            limit=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        ),
    )

    app = FastAPI(
        title="SuretySeven - Surety Bond Underwriting Service",
        description=(
            "Accepts bond applications, enriches them from an external Applicant API, "
            "scores them deterministically and notifies a downstream system."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.state.runtime = runtime
    app.state.enable_workers = settings.worker_enabled if enable_workers is None else enable_workers
    install_middleware(app)
    install_exception_handlers(app)
    app.include_router(build_ops_router())
    app.include_router(build_router())
    return app
