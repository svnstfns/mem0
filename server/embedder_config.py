"""Embedder defaults resolved from the environment.

The vector store's column width must match what the embedder produces, so both are
derived here from one set of variables. OpenAI keeps upstream's defaults; any other
provider must name its model and dimensions explicitly, because a silent 1536 default
would only surface later as failing pgvector inserts.
"""

from typing import Any, Dict, Mapping, Tuple

OPENAI_DEFAULT_MODEL = "text-embedding-3-small"
OPENAI_DEFAULT_DIMS = 1536
OLLAMA_DEFAULT_BASE_URL = "http://ollama:11434"


def embedder_from_env(env: Mapping[str, str], bundled: Tuple[str, ...]) -> Tuple[Dict[str, Any], int]:
    """Return the mem0 embedder config and the vector dimensions it produces."""
    provider = env.get("MEM0_DEFAULT_EMBEDDER_PROVIDER", "openai")
    if provider not in bundled:
        raise RuntimeError(
            f"MEM0_DEFAULT_EMBEDDER_PROVIDER='{provider}' is not bundled in this image "
            f"(bundled: {', '.join(bundled)})."
        )

    model = env.get("MEM0_DEFAULT_EMBEDDER_MODEL")
    dims = env.get("MEM0_EMBEDDING_DIMS")
    if provider == "openai":
        model = model or OPENAI_DEFAULT_MODEL
        dims = dims or str(OPENAI_DEFAULT_DIMS)
    elif not model or not dims:
        raise RuntimeError(
            f"Embedder provider '{provider}' needs MEM0_DEFAULT_EMBEDDER_MODEL and MEM0_EMBEDDING_DIMS."
        )

    config: Dict[str, Any] = {"model": model, "embedding_dims": int(dims)}
    if provider == "openai":
        config["api_key"] = env.get("OPENAI_API_KEY")
    elif provider == "ollama":
        config["ollama_base_url"] = env.get("OLLAMA_BASE_URL", OLLAMA_DEFAULT_BASE_URL)
    return {"provider": provider, "config": config}, int(dims)
