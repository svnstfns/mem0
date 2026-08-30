"""Unit tests for the upstream-provider error mapping (errors.py).

The classifier matches on type names and attributes, not on the provider SDKs, so the
provider exceptions are modeled as minimal stand-ins shaped like anthropic/openai errors.
"""

from errors import upstream_error


class _ProviderError(Exception):
    def __init__(self, message: str, status_code: int | None = None, body: dict | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class BadRequestError(_ProviderError):
    """Shaped like anthropic.BadRequestError; classification matches on the type name."""


class RateLimitError(_ProviderError):
    """Shaped like openai.RateLimitError."""


class AuthenticationError(_ProviderError):
    """Shaped like anthropic.AuthenticationError."""


class LLMError(Exception):
    """Shaped like mem0.exceptions.LLMError (plain wrapper, no status_code)."""


ANTHROPIC_BILLING_MESSAGE = (
    "Your credit balance is too low to access the Anthropic API. "
    "Please go to Plans & Billing to upgrade or purchase credits."
)
ANTHROPIC_BILLING_BODY = {
    "type": "error",
    "error": {"type": "invalid_request_error", "message": ANTHROPIC_BILLING_MESSAGE},
    "request_id": "req_011CeYbhTbVovZyyfb3KzMLe",
}


def _map(exc: Exception):
    try:
        raise exc
    except Exception:
        return upstream_error()


def test_billing_400_maps_to_provider_billing():
    err = _map(BadRequestError(f"Error code: 400 - {ANTHROPIC_BILLING_BODY}", 400, ANTHROPIC_BILLING_BODY))
    assert err.code == "provider_billing"
    assert ANTHROPIC_BILLING_MESSAGE in err.detail
    assert err.status_code == 502


def test_billing_400_wrapped_in_llm_error_maps_to_provider_billing():
    # mem0's _add_to_vector_store raises LLMError(f"LLM extraction failed: {e}") from e.
    try:
        try:
            raise BadRequestError(f"Error code: 400 - {ANTHROPIC_BILLING_BODY}", 400, ANTHROPIC_BILLING_BODY)
        except BadRequestError as e:
            raise LLMError(f"LLM extraction failed: {e}") from e
    except LLMError:
        err = upstream_error()
    assert err.code == "provider_billing"
    assert ANTHROPIC_BILLING_MESSAGE in err.detail


def test_llm_error_without_cause_still_detects_billing():
    err = _map(LLMError(f"LLM extraction failed: Error code: 400 - {ANTHROPIC_BILLING_BODY}"))
    assert err.code == "provider_billing"
    assert "credit balance is too low" in err.detail


def test_insufficient_quota_429_maps_to_provider_billing():
    body = {"message": "You exceeded your current quota, please check your plan and billing details.",
            "code": "insufficient_quota"}
    err = _map(RateLimitError(f"Error code: 429 - {body}", 429, body))
    assert err.code == "provider_billing"
    assert "exceeded your current quota" in err.detail


def test_malformed_400_passes_provider_message_through():
    body = {"type": "error", "error": {"type": "invalid_request_error",
                                       "message": "messages: at least one message is required"}}
    err = _map(BadRequestError(f"Error code: 400 - {body}", 400, body))
    assert err.code == "provider_bad_request"
    assert "at least one message is required" in err.detail


def test_bad_request_without_body_uses_str():
    err = _map(BadRequestError("max_tokens must be positive", 400))
    assert err.code == "provider_bad_request"
    assert "max_tokens must be positive" in err.detail


def test_provider_message_is_truncated():
    err = _map(BadRequestError("x" * 5000, 400))
    assert err.code == "provider_bad_request"
    assert len(err.detail) < 400


def test_plain_rate_limit_stays_rate_limited():
    body = {"error": {"message": "Too many requests, slow down."}}
    err = _map(RateLimitError(f"Error code: 429 - {body}", 429, body))
    assert err.code == "provider_rate_limited"


def test_auth_error_stays_auth_failed():
    err = _map(AuthenticationError("Error code: 401 - invalid x-api-key", 401))
    assert err.code == "provider_auth_failed"


def test_unrelated_wrapper_stays_unknown():
    err = _map(LLMError("LLM extraction failed: something odd"))
    assert err.code == "unknown"
