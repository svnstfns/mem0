"""Unit tests for the env-driven provider defaults (provider_config.py)."""

import pytest

from provider_config import embedder_from_env, llm_from_env

LLM_BUNDLED = ("openai", "anthropic", "gemini")
EMBEDDER_BUNDLED = ("openai", "gemini", "ollama")
OLLAMA_ENV = {
    "MEM0_DEFAULT_EMBEDDER_PROVIDER": "ollama",
    "MEM0_DEFAULT_EMBEDDER_MODEL": "bge-m3",
    "MEM0_EMBEDDING_DIMS": "1024",
    "OLLAMA_BASE_URL": "http://ollama:11434",
}
ANTHROPIC_ENV = {
    "MEM0_DEFAULT_LLM_PROVIDER": "anthropic",
    "MEM0_DEFAULT_LLM_MODEL": "claude-opus-5",
    "ANTHROPIC_API_KEY": "sk-ant-test",
    "OPENAI_API_KEY": "sk-test",
}


def test_llm_no_env_keeps_upstream_openai_defaults() -> None:
    assert llm_from_env({"OPENAI_API_KEY": "sk-test"}, LLM_BUNDLED) == {
        "provider": "openai",
        "config": {"model": "gpt-5-mini", "api_key": "sk-test", "temperature": 0.2},
    }


def test_llm_anthropic_uses_its_own_key_and_no_sampling_params() -> None:
    assert llm_from_env({**ANTHROPIC_ENV, "MEM0_LLM_MAX_TOKENS": "16000"}, LLM_BUNDLED) == {
        "provider": "anthropic",
        "config": {"model": "claude-opus-5", "api_key": "sk-ant-test", "max_tokens": 16000},
    }


def test_llm_non_openai_provider_requires_model() -> None:
    env = {k: v for k, v in ANTHROPIC_ENV.items() if k != "MEM0_DEFAULT_LLM_MODEL"}

    with pytest.raises(RuntimeError, match="needs MEM0_DEFAULT_LLM_MODEL"):
        llm_from_env(env, LLM_BUNDLED)


def test_llm_unbundled_provider_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="not bundled"):
        llm_from_env({"MEM0_DEFAULT_LLM_PROVIDER": "ollama", "MEM0_DEFAULT_LLM_MODEL": "x"}, LLM_BUNDLED)


def test_embedder_no_env_keeps_upstream_openai_defaults() -> None:
    config, dims = embedder_from_env({"OPENAI_API_KEY": "sk-test"}, EMBEDDER_BUNDLED)

    assert config == {
        "provider": "openai",
        "config": {"model": "text-embedding-3-small", "embedding_dims": 1536, "api_key": "sk-test"},
    }
    assert dims == 1536


def test_embedder_ollama_gets_base_url_and_no_openai_key() -> None:
    config, dims = embedder_from_env({**OLLAMA_ENV, "OPENAI_API_KEY": "sk-test"}, EMBEDDER_BUNDLED)

    assert config == {
        "provider": "ollama",
        "config": {
            "model": "bge-m3",
            "embedding_dims": 1024,
            "ollama_base_url": "http://ollama:11434",
            "ollama_timeout": 60.0,
        },
    }
    assert dims == 1024


def test_embedder_ollama_base_url_defaults_to_compose_service_name() -> None:
    env = {k: v for k, v in OLLAMA_ENV.items() if k != "OLLAMA_BASE_URL"}

    config, _ = embedder_from_env(env, EMBEDDER_BUNDLED)

    assert config["config"]["ollama_base_url"] == "http://ollama:11434"


@pytest.mark.parametrize("missing", ["MEM0_DEFAULT_EMBEDDER_MODEL", "MEM0_EMBEDDING_DIMS"])
def test_embedder_non_openai_provider_requires_model_and_dims(missing: str) -> None:
    env = {k: v for k, v in OLLAMA_ENV.items() if k != missing}

    with pytest.raises(RuntimeError, match="needs MEM0_DEFAULT_EMBEDDER_MODEL and MEM0_EMBEDDING_DIMS"):
        embedder_from_env(env, EMBEDDER_BUNDLED)


def test_embedder_unbundled_provider_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="not bundled"):
        embedder_from_env({**OLLAMA_ENV, "MEM0_DEFAULT_EMBEDDER_PROVIDER": "huggingface"}, EMBEDDER_BUNDLED)


def test_embedder_ollama_timeout_defaults_to_60s() -> None:
    # The ollama client sets no httpx timeout on its own; without this a hung runtime
    # would hang /search instead of failing over to keyword-only search.
    config, _ = embedder_from_env(OLLAMA_ENV, EMBEDDER_BUNDLED)

    assert config["config"]["ollama_timeout"] == 60.0


def test_embedder_ollama_timeout_is_overridable() -> None:
    config, _ = embedder_from_env({**OLLAMA_ENV, "MEM0_EMBEDDER_TIMEOUT": "15"}, EMBEDDER_BUNDLED)

    assert config["config"]["ollama_timeout"] == 15.0


def test_embedder_openai_gets_no_ollama_timeout() -> None:
    config, _ = embedder_from_env({"OPENAI_API_KEY": "sk-test", "MEM0_EMBEDDER_TIMEOUT": "15"}, EMBEDDER_BUNDLED)

    assert "ollama_timeout" not in config["config"]
