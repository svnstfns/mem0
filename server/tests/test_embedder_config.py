"""Unit tests for the env-driven embedder defaults (embedder_config.py)."""

import pytest

from embedder_config import embedder_from_env

BUNDLED = ("openai", "gemini", "ollama")
OLLAMA_ENV = {
    "MEM0_DEFAULT_EMBEDDER_PROVIDER": "ollama",
    "MEM0_DEFAULT_EMBEDDER_MODEL": "bge-m3",
    "MEM0_EMBEDDING_DIMS": "1024",
    "OLLAMA_BASE_URL": "http://ollama:11434",
}


def test_no_env_keeps_upstream_openai_defaults() -> None:
    config, dims = embedder_from_env({"OPENAI_API_KEY": "sk-test"}, BUNDLED)

    assert config == {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small", "embedding_dims": 1536, "api_key": "sk-test"},
    }
    assert dims == 1536


def test_ollama_gets_base_url_and_no_openai_key() -> None:
    config, dims = embedder_from_env({**OLLAMA_ENV, "OPENAI_API_KEY": "sk-test"}, BUNDLED)

    assert config == {
        "provider": "ollama",
        "config": {"model": "bge-m3", "embedding_dims": 1024, "ollama_base_url": "http://ollama:11434"},
    }
    assert dims == 1024


def test_ollama_base_url_defaults_to_compose_service_name() -> None:
    env = {k: v for k, v in OLLAMA_ENV.items() if k != "OLLAMA_BASE_URL"}

    config, _ = embedder_from_env(env, BUNDLED)

    assert config["config"]["ollama_base_url"] == "http://ollama:11434"


@pytest.mark.parametrize("missing", ["MEM0_DEFAULT_EMBEDDER_MODEL", "MEM0_EMBEDDING_DIMS"])
def test_non_openai_provider_requires_model_and_dims(missing: str) -> None:
    env = {k: v for k, v in OLLAMA_ENV.items() if k != missing}

    with pytest.raises(RuntimeError, match="needs MEM0_DEFAULT_EMBEDDER_MODEL and MEM0_EMBEDDING_DIMS"):
        embedder_from_env(env, BUNDLED)


def test_unbundled_provider_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="not bundled"):
        embedder_from_env({**OLLAMA_ENV, "MEM0_DEFAULT_EMBEDDER_PROVIDER": "huggingface"}, BUNDLED)
