"""Keyword-only search: the path the server falls back to when the query embedding fails."""

from types import SimpleNamespace

import pytest

from mem0.exceptions import EmbeddingError, VectorStoreError
from mem0.memory.main import AsyncMemory, Memory


def _build_memory(mocker, memory_cls):
    mocker.patch("mem0.utils.factory.EmbedderFactory.create", mocker.MagicMock())
    mocker.patch(
        "mem0.utils.factory.VectorStoreFactory.create", side_effect=[mocker.MagicMock(), mocker.MagicMock()]
    )
    mocker.patch("mem0.utils.factory.LlmFactory.create", mocker.MagicMock())
    mocker.patch("mem0.memory.storage.SQLiteManager", mocker.MagicMock())
    mocker.patch("mem0.memory.main.SQLiteManager", mocker.MagicMock())
    mocker.patch("mem0.memory.main.MEM0_TELEMETRY", False)
    mocker.patch("mem0.memory.main.capture_event")
    # Keep spaCy out of unit tests: no model download, deterministic terms.
    mocker.patch("mem0.memory.main.lemmatize_for_bm25", side_effect=lambda text: text.lower())
    mocker.patch("mem0.memory.main.extract_entities", return_value=[])
    memory = memory_cls()
    memory.api_version = "v1.1"
    memory.vector_store = mocker.MagicMock()
    memory.db = mocker.MagicMock()
    return memory


def _hit(mem_id, score, **payload):
    return SimpleNamespace(id=mem_id, score=score, payload={"data": f"memory {mem_id}", "user_id": "sst", **payload})


def test_keyword_only_ranks_full_text_matches_without_embedding(mocker):
    memory = _build_memory(mocker, Memory)
    memory.vector_store.keyword_search.return_value = [
        _hit("weak", 0.1),
        _hit("strong", 1.4, kind="decision"),
    ]

    response = memory.search(
        "Docker Deploy", filters={"user_id": "sst"}, top_k=5, threshold=0.9, keyword_only=True
    )

    memory.embedding_model.embed.assert_not_called()
    memory.vector_store.search.assert_not_called()
    memory.vector_store.keyword_search.assert_called_once_with(
        query="docker deploy", top_k=60, filters={"user_id": "sst"}, match_any=True
    )
    results = response["results"]
    # Raw full-text ranks, best first; the semantic threshold (0.9) is not applied to them.
    assert [(r["id"], r["score"]) for r in results] == [("strong", 1.4), ("weak", 0.1)]
    assert results[0]["metadata"] == {"kind": "decision"}
    assert results[0]["user_id"] == "sst"


def test_keyword_only_drops_expired_memories_and_respects_top_k(mocker):
    memory = _build_memory(mocker, Memory)
    memory.vector_store.keyword_search.return_value = [
        _hit("expired", 2.0, expiration_date="2000-01-01"),
        _hit("best", 1.0),
        _hit("second", 0.5),
    ]

    results = memory.search("deploy", filters={"user_id": "sst"}, top_k=1, keyword_only=True)["results"]

    assert [r["id"] for r in results] == ["best"]


def test_keyword_only_explain_reports_the_keyword_score(mocker):
    memory = _build_memory(mocker, Memory)
    memory.vector_store.keyword_search.return_value = [_hit("m1", 0.7)]

    results = memory.search("deploy", filters={"user_id": "sst"}, explain=True, keyword_only=True)["results"]

    assert results[0]["score_details"] == {"bm25_score": 0.7, "final_score": 0.7}


def test_keyword_only_fails_loudly_when_keyword_search_is_unavailable(mocker):
    # pgvector returns None when the full-text query fails; the base store returns None when unsupported.
    memory = _build_memory(mocker, Memory)
    memory.vector_store.keyword_search.return_value = None

    with pytest.raises(VectorStoreError):
        memory.search("deploy", filters={"user_id": "sst"}, keyword_only=True)


def test_query_embedding_failure_raises_embedding_error_with_provider_cause(mocker):
    memory = _build_memory(mocker, Memory)
    provider_error = RuntimeError("Error code: 429 - You have no credits remaining.")
    memory.embedding_model.embed.side_effect = provider_error

    with pytest.raises(EmbeddingError) as excinfo:
        memory.search("deploy", filters={"user_id": "sst"})

    assert excinfo.value.__cause__ is provider_error
    memory.vector_store.search.assert_not_called()


@pytest.mark.asyncio
async def test_async_keyword_only_ranks_full_text_matches_without_embedding(mocker):
    memory = _build_memory(mocker, AsyncMemory)
    memory.vector_store.keyword_search.return_value = [_hit("weak", 0.1), _hit("strong", 1.4)]

    response = await memory.search("deploy", filters={"user_id": "sst"}, threshold=0.9, keyword_only=True)

    memory.embedding_model.embed.assert_not_called()
    memory.vector_store.search.assert_not_called()
    assert memory.vector_store.keyword_search.call_args.kwargs["match_any"] is True
    assert [r["id"] for r in response["results"]] == ["strong", "weak"]


@pytest.mark.asyncio
async def test_async_query_embedding_failure_raises_embedding_error_with_provider_cause(mocker):
    memory = _build_memory(mocker, AsyncMemory)
    provider_error = RuntimeError("Error code: 429 - You have no credits remaining.")
    memory.embedding_model.embed.side_effect = provider_error

    with pytest.raises(EmbeddingError) as excinfo:
        await memory.search("deploy", filters={"user_id": "sst"})

    assert excinfo.value.__cause__ is provider_error
    memory.vector_store.search.assert_not_called()
