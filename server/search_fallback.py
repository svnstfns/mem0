"""Keyword fallback for POST /search when the embedding provider fails.

The query embedding is created before any retrieval, so an embedder outage (exhausted credit,
rate limit, provider down) used to fail every search. On provider-side failures the search is
repeated without an embedding (full-text matches only), the response carries a `degraded`
object and a warning is logged, so the quality loss stays visible instead of silent.
"""

import logging
from typing import Any

from errors import classify, request_id_var

from mem0.exceptions import EmbeddingError

# provider_bad_request is deliberately missing: a rejected request is a caller or configuration
# bug, and falling back would hide it.
KEYWORD_FALLBACK_CODES = frozenset(
    {"provider_auth_failed", "provider_billing", "provider_rate_limited", "provider_timeout", "provider_unavailable"}
)


def search_with_keyword_fallback(memory: Any, query: str, filters: dict[str, Any], **params: Any) -> dict[str, Any]:
    try:
        return memory.search(query=query, filters=filters, **params)
    except EmbeddingError as exc:
        code, detail = classify(exc)
        if code not in KEYWORD_FALLBACK_CODES:
            raise
        logging.warning("Query embedding failed (code=%s), serving keyword-only search results: %s", code, detail)
        response = memory.search(query=query, filters=filters, keyword_only=True, **params)
        response["degraded"] = {
            "search_mode": "keyword",
            "code": code,
            "detail": detail,
            "request_id": request_id_var.get(),
        }
        return response
