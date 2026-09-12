"""Unit tests for the keyword fallback of POST /search (search_fallback.py).

As in test_errors.py, provider exceptions are minimal stand-ins shaped like the openai SDK errors;
the Memory instance is a fake that records how it was searched.
"""

import logging

import pytest
from errors import request_id_var, upstream_error
from search_fallback import search_with_keyword_fallback

from mem0.exceptions import EmbeddingError, VectorStoreError


class _ProviderError(Exception):
    def __init__(self, message: str, status_code: int | None = None, body: dict | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class RateLimitError(_ProviderError):
    """Shaped like openai.RateLimitError."""


class APIConnectionError(_ProviderError):
    """Shaped like openai.APIConnectionError."""


class APITimeoutError(_ProviderError):
    """Shaped like openai.APITimeoutError."""


class AuthenticationError(_ProviderError):
    """Shaped like openai.AuthenticationError."""


class BadRequestError(_ProviderError):
    """Shaped like openai.BadRequestError."""


class InternalServerError(_ProviderError):
    """Shaped like openai.InternalServerError."""


class OperationalError(Exception):
    """Shaped like psycopg.OperationalError."""


# Message as surfaced by search_memories on 2026-09-12 (429 credit_balance_exhausted on /v1/embeddings).
OPENAI_NO_CREDITS_MESSAGE = (
    "You have no credits remaining. Add credits to continue using the API at "
    "https://platform.openai.com/settings/organization/billing/."
)
OPENAI_NO_CREDITS_BODY = {"error": {"message": OPENAI_NO_CREDITS_MESSAGE, "code": "credit_balance_exhausted"}}

FILTERS = {"user_id": "sst", "kind": "decision"}


class FakeMemory:
    """Fails the embedding step with `embed_error` and the keyword-only search with `keyword_error`."""

    def __init__(self, embed_error: Exception | None = None, keyword_error: Exception | None = None) -> None:
        self.embed_error = embed_error
        self.keyword_error = keyword_error
        self.calls: list[dict] = []

    def search(self, *, query, filters, keyword_only=False, **params):
        self.calls.append({"query": query, "filters": filters, "keyword_only": keyword_only, **params})
        if keyword_only:
            if self.keyword_error is not None:
                raise self.keyword_error
            return {"results": [{"id": "m1", "memory": "keyword hit", "score": 0.2}]}
        if self.embed_error is not None:
            # Same wrapping as mem0's _search_vector_store.
            try:
                raise self.embed_error
            except Exception as e:
                raise EmbeddingError(f"Query embedding failed: {e}") from e
        return {"results": [{"id": "m1", "memory": "semantic hit", "score": 0.8}]}


def _no_credits() -> RateLimitError:
    return RateLimitError(f"Error code: 429 - {OPENAI_NO_CREDITS_BODY}", 429, OPENAI_NO_CREDITS_BODY)


def test_healthy_embedder_returns_semantic_results_unflagged():
    memory = FakeMemory()
    response = search_with_keyword_fallback(memory, "deploy", FILTERS, top_k=5)
    assert response == {"results": [{"id": "m1", "memory": "semantic hit", "score": 0.8}]}
    assert [call["keyword_only"] for call in memory.calls] == [False]


def test_exhausted_credit_falls_back_to_keyword_search_and_flags_it(caplog):
    memory = FakeMemory(embed_error=_no_credits())
    token = request_id_var.set("2e46c352")
    try:
        with caplog.at_level(logging.WARNING):
            response = search_with_keyword_fallback(memory, "deploy", FILTERS, top_k=5, threshold=0.35)
    finally:
        request_id_var.reset(token)

    assert response["results"] == [{"id": "m1", "memory": "keyword hit", "score": 0.2}]
    degraded = response["degraded"]
    assert degraded["search_mode"] == "keyword"
    assert degraded["code"] == "provider_billing"
    assert OPENAI_NO_CREDITS_MESSAGE in degraded["detail"]
    assert degraded["request_id"] == "2e46c352"
    # The retry keeps query, filters and params and only switches the mode.
    assert memory.calls[1] == {**memory.calls[0], "keyword_only": True}
    assert memory.calls[1]["threshold"] == 0.35
    assert any(
        record.levelno == logging.WARNING and "provider_billing" in record.getMessage() for record in caplog.records
    )


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (RateLimitError("Error code: 429 - Too many requests", 429), "provider_rate_limited"),
        (APIConnectionError("Connection error."), "provider_unavailable"),
        (InternalServerError("Error code: 503 - overloaded", 503), "provider_unavailable"),
        (APITimeoutError("Request timed out."), "provider_timeout"),
        (AuthenticationError("Error code: 401 - Incorrect API key provided", 401), "provider_auth_failed"),
    ],
)
def test_provider_outages_fall_back(error, code):
    memory = FakeMemory(embed_error=error)
    response = search_with_keyword_fallback(memory, "deploy", FILTERS)
    assert response["degraded"]["code"] == code
    assert [call["keyword_only"] for call in memory.calls] == [False, True]


def test_bad_request_is_not_masked_by_the_fallback():
    memory = FakeMemory(embed_error=BadRequestError("Error code: 400 - input too long", 400))
    with pytest.raises(EmbeddingError):
        search_with_keyword_fallback(memory, "deploy", FILTERS)
    assert len(memory.calls) == 1


def test_unclassified_embedder_bug_is_not_masked_by_the_fallback():
    memory = FakeMemory(embed_error=ValueError("embedding dimension mismatch"))
    with pytest.raises(EmbeddingError):
        search_with_keyword_fallback(memory, "deploy", FILTERS)
    assert len(memory.calls) == 1


def test_failure_outside_the_embedder_propagates_without_fallback():
    class BrokenStoreMemory(FakeMemory):
        def search(self, **kwargs):
            super().search(**kwargs)
            raise OperationalError("server closed the connection unexpectedly")

    memory = BrokenStoreMemory()
    with pytest.raises(OperationalError):
        search_with_keyword_fallback(memory, "deploy", FILTERS)
    assert len(memory.calls) == 1


def test_failing_keyword_search_maps_to_vector_store_unavailable():
    memory = FakeMemory(
        embed_error=_no_credits(),
        keyword_error=VectorStoreError("Keyword-only search failed or is not supported by this vector store."),
    )
    try:
        search_with_keyword_fallback(memory, "deploy", FILTERS)
    except Exception:
        err = upstream_error()
    else:
        pytest.fail("keyword search failure was swallowed")
    assert err.code == "vector_store_unavailable"
    assert err.status_code == 502
