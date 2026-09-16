"""LLM and embedder defaults resolved from the environment.

Upstream hard-codes both providers to OpenAI and lets a persisted /configure override
switch them at runtime, which is invisible in the deployment and drifts. Here the
environment is the single source: OpenAI keeps upstream's defaults, any other provider
must name its model explicitly, and the embedder additionally its dimensions, because
the vector store's column width is derived from them - a silent 1536 default would
only surface later as failing pgvector inserts.
"""

from typing import Any, Dict, Mapping, Tuple

OPENAI_DEFAULT_LLM_MODEL = "gpt-5-mini"
OPENAI_DEFAULT_EMBEDDER_MODEL = "text-embedding-3-small"
OPENAI_DEFAULT_DIMS = 1536
OLLAMA_DEFAULT_BASE_URL = "http://ollama:11434"
# The ollama client sets no httpx timeout at all, so an unresponsive runtime would hang /search
# forever. 60s leaves room for a cold model load into VRAM and still fails fast enough for the
# keyword fallback to take over.
OLLAMA_DEFAULT_EMBEDDER_TIMEOUT = 60.0
# A generation legitimately takes far longer than an embedding, and ollama answers chat() in a
# single blocking response, so this timeout has to cover the cold model load into VRAM plus the
# whole generation. 300s does that and still fails instead of hanging an add forever - and unlike
# a failed embedding there is no keyword fallback to catch it.
OLLAMA_DEFAULT_LLM_TIMEOUT = 300.0
API_KEY_VARS = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}


def _provider(env: Mapping[str, str], var: str, bundled: Tuple[str, ...]) -> str:
    provider = env.get(var, "openai")
    if provider not in bundled:
        raise RuntimeError(f"{var}='{provider}' is not bundled in this image (bundled: {', '.join(bundled)}).")
    return provider


def llm_from_env(env: Mapping[str, str], bundled: Tuple[str, ...]) -> Dict[str, Any]:
    """Return the mem0 llm config block."""
    provider = _provider(env, "MEM0_DEFAULT_LLM_PROVIDER", bundled)
    model = env.get("MEM0_DEFAULT_LLM_MODEL")
    if provider == "openai":
        model = model or OPENAI_DEFAULT_LLM_MODEL
    elif not model:
        raise RuntimeError(f"LLM provider '{provider}' needs MEM0_DEFAULT_LLM_MODEL.")

    config: Dict[str, Any] = {"model": model}
    if provider in API_KEY_VARS:
        config["api_key"] = env.get(API_KEY_VARS[provider])
    if provider == "openai":
        config["temperature"] = 0.2
    if env.get("MEM0_LLM_MAX_TOKENS"):
        config["max_tokens"] = int(env["MEM0_LLM_MAX_TOKENS"])
    if provider == "ollama":
        config["ollama_base_url"] = env.get("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL)
        config["ollama_timeout"] = float(env.get("MEM0_LLM_TIMEOUT") or OLLAMA_DEFAULT_LLM_TIMEOUT)
    return {"provider": provider, "config": config}


def embedder_from_env(env: Mapping[str, str], bundled: Tuple[str, ...]) -> Tuple[Dict[str, Any], int]:
    """Return the mem0 embedder config block and the vector dimensions it produces."""
    provider = _provider(env, "MEM0_DEFAULT_EMBEDDER_PROVIDER", bundled)
    model = env.get("MEM0_DEFAULT_EMBEDDER_MODEL")
    dims = env.get("MEM0_EMBEDDING_DIMS")
    if provider == "openai":
        model = model or OPENAI_DEFAULT_EMBEDDER_MODEL
        dims = dims or str(OPENAI_DEFAULT_DIMS)
    elif not model or not dims:
        raise RuntimeError(f"Embedder provider '{provider}' needs MEM0_DEFAULT_EMBEDDER_MODEL and MEM0_EMBEDDING_DIMS.")

    config: Dict[str, Any] = {"model": model, "embedding_dims": int(dims)}
    if provider in API_KEY_VARS:
        config["api_key"] = env.get(API_KEY_VARS[provider])
    if provider == "ollama":
        config["ollama_base_url"] = env.get("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL)
        config["ollama_timeout"] = float(env.get("MEM0_EMBEDDER_TIMEOUT") or OLLAMA_DEFAULT_EMBEDDER_TIMEOUT)
    return {"provider": provider, "config": config}, int(dims)
