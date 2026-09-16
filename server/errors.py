import logging
import sys
import uuid
from contextvars import ContextVar

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class UpstreamError(HTTPException):
    def __init__(self, code: str, detail: str, request_id: str) -> None:
        super().__init__(status_code=502, detail=detail)
        self.code = code
        self.request_id = request_id


_AUTH_NAMES = {"AuthenticationError", "PermissionDeniedError"}
_RATE_NAMES = {"RateLimitError"}
_TIMEOUT_NAMES = {"APITimeoutError"}
_CONN_NAMES = {"APIConnectionError", "ConnectionError"}
_BAD_REQUEST_NAMES = {"BadRequestError", "UnprocessableEntityError"}
_DB_NAMES = {"OperationalError", "DBAPIError", "DisconnectionError"}
_VECTOR_NAMES = {"UnexpectedResponse", "ResponseHandlingException", "VectorStoreError"}

# Marks a provider 4xx as a billing/credit problem rather than a malformed request
# (Anthropic 400: "Your credit balance is too low ... Plans & Billing"; OpenAI 429:
# code "insufficient_quota", "... check your plan and billing details"; OpenAI 429 on
# embeddings: code "credit_balance_exhausted", "You have no credits remaining ...").
_BILLING_MARKERS = (
    "credit balance",
    "credit_balance_exhausted",
    "no credits remaining",
    "insufficient_quota",
    "billing",
    "purchase credits",
    "payment required",
)
_PROVIDER_MESSAGE_LIMIT = 300


def _provider_message(exc: BaseException) -> str:
    # anthropic/openai SDK errors carry the parsed response in `body`; prefer its
    # message over str(exc), which prepends "Error code: NNN - {raw dict}".
    message = ""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            message = error["message"]
        elif isinstance(body.get("message"), str):
            message = body["message"]
    message = message or str(exc).strip() or type(exc).__name__
    if len(message) > _PROVIDER_MESSAGE_LIMIT:
        message = message[: _PROVIDER_MESSAGE_LIMIT - 1] + "…"
    return message


def _is_billing(exc: BaseException) -> bool:
    haystack = f"{_provider_message(exc)} {exc}".lower()
    return any(marker in haystack for marker in _BILLING_MARKERS)


def _billing_result(exc: BaseException) -> tuple[str, str]:
    return (
        "provider_billing",
        f"Provider rejected the request due to a billing or credit issue: {_provider_message(exc)}",
    )


def _classify_one(exc: BaseException) -> tuple[str, str]:
    name = type(exc).__name__
    module = getattr(type(exc), "__module__", "") or ""
    status = getattr(exc, "status_code", None)

    if name in _AUTH_NAMES or status in (401, 403):
        return (
            "provider_auth_failed",
            "Provider rejected the request (authentication). "
            "Check your LLM provider API key on the Configuration page.",
        )
    is_provider_4xx = name in _BAD_REQUEST_NAMES or name in _RATE_NAMES or status in (400, 402, 422, 429)
    if is_provider_4xx and _is_billing(exc):
        return _billing_result(exc)
    if name in _RATE_NAMES or status == 429:
        return ("provider_rate_limited", "Provider rate limit hit. Retry shortly.")
    if name in _TIMEOUT_NAMES or isinstance(exc, TimeoutError):
        return ("provider_timeout", "Provider timed out. Retry shortly.")
    if name in _CONN_NAMES or (isinstance(status, int) and status >= 500):
        return ("provider_unavailable", "Provider is unreachable or returned a server error.")
    if name in _BAD_REQUEST_NAMES or status in (400, 422):
        return ("provider_bad_request", f"Provider rejected the request: {_provider_message(exc)}")
    if name in _DB_NAMES:
        return ("datastore_unavailable", "The memory database is unreachable.")
    if name in _VECTOR_NAMES or module.startswith("qdrant_client"):
        return ("vector_store_unavailable", "The vector store is unreachable or returned an error.")
    return ("unknown", "Upstream provider error.")


def classify(exc: BaseException | None) -> tuple[str, str]:
    # Walk the cause/context chain so wrapped provider errors still classify correctly.
    seen: set[int] = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        result = _classify_one(current)
        if result[0] != "unknown":
            return result
        current = current.__cause__ or current.__context__
    # Wrappers raised without a cause (e.g. mem0.exceptions.LLMError) only carry the
    # provider text in their message: still surface billing problems distinctly.
    if exc is not None and _is_billing(exc):
        return _billing_result(exc)
    return ("unknown", "Upstream provider error.")


def new_request_id() -> str:
    return uuid.uuid4().hex[:8]


def upstream_error() -> UpstreamError:
    exc = sys.exc_info()[1]
    code, message = classify(exc)
    rid = request_id_var.get()
    logging.exception("Upstream provider error (code=%s)", code)
    return UpstreamError(code=code, detail=message, request_id=rid)


async def upstream_error_handler(_: Request, exc: UpstreamError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "code": exc.code, "request_id": exc.request_id},
        headers={"X-Request-ID": exc.request_id},
    )


def install_request_id_logging() -> None:
    old_factory = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.request_id = request_id_var.get()
        return record

    logging.setLogRecordFactory(factory)
