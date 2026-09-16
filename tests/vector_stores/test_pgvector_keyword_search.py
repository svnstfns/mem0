"""pgvector keyword_search: default all-terms matching and the match_any mode behind keyword-only search."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from mem0.vector_stores.pgvector import PGVector


@pytest.fixture
def store():
    with patch("mem0.vector_stores.pgvector.PSYCOPG_VERSION", 3), patch("mem0.vector_stores.pgvector.ConnectionPool"):
        pgvector = PGVector(
            dbname="test_db",
            collection_name="memories",
            embedding_model_dims=3,
            user="test_user",
            password="test_pass",
            host="localhost",
            port=5432,
            diskann=False,
            hnsw=False,
        )
    pgvector._collection_ensured = True
    cursor = MagicMock()
    cursor.fetchall.return_value = []
    with patch.object(PGVector, "_get_cursor") as get_cursor:
        get_cursor.return_value.__enter__.return_value = cursor
        yield pgvector, cursor


def _executed(cursor):
    (statement, params), _ = cursor.execute.call_args
    return str(statement), params


def test_default_matches_all_terms(store):
    pgvector, cursor = store
    pgvector.keyword_search("docker deploy", top_k=5)

    statement, params = _executed(cursor)
    assert "plainto_tsquery('simple', %s)" in statement
    assert params == ("docker deploy", "docker deploy", 5)


def test_match_any_ors_the_query_terms(store):
    pgvector, cursor = store
    pgvector.keyword_search("docker deploy nas", top_k=60, match_any=True)

    statement, params = _executed(cursor)
    assert "plainto_tsquery" not in statement
    assert "to_tsquery('simple', %s)" in statement
    assert params == ("'docker' | 'deploy' | 'nas'", "'docker' | 'deploy' | 'nas'", 60)


def test_match_any_reduces_tsquery_syntax_to_quoted_terms(store):
    pgvector, cursor = store
    pgvector.keyword_search("it's a' | !b & (c:*)", match_any=True)

    _, params = _executed(cursor)
    assert params[0] == "'it' | 's' | 'a' | 'b' | 'c'"


def test_match_any_without_terms_skips_the_query(store):
    pgvector, cursor = store
    assert pgvector.keyword_search("?! --", match_any=True) == []
    cursor.execute.assert_not_called()


def test_match_any_keeps_filters_and_maps_rows(store):
    pgvector, cursor = store
    cursor.fetchall.return_value = [("0b0e3f38-6f2e-4a5e-9d0e-6f7e8c1a2b3c", 1.25, {"data": "deploy via script"})]

    results = pgvector.keyword_search("deploy", top_k=3, filters={"user_id": "sst"}, match_any=True)

    statement, params = _executed(cursor)
    assert "AND" in statement
    assert params[0] == params[1] == "'deploy'"
    assert "sst" in params[2:-1]
    assert params[-1] == 3
    assert [(r.id, r.score, r.payload) for r in results] == [
        ("0b0e3f38-6f2e-4a5e-9d0e-6f7e8c1a2b3c", 1.25, {"data": "deploy via script"})
    ]


def test_query_failure_returns_none_and_logs_a_warning(store, caplog):
    pgvector, cursor = store
    cursor.execute.side_effect = RuntimeError("relation memories does not exist")

    with caplog.at_level(logging.WARNING, logger="mem0.vector_stores.pgvector"):
        assert pgvector.keyword_search("deploy", match_any=True) is None

    assert any("Keyword search failed" in record.getMessage() for record in caplog.records)
