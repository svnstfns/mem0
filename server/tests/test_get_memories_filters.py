"""Tests for the metadata filters of GET /memories (kind, scope, project, glossary).

main.py is loaded with a patched Memory.from_config, the way tests/test_server_auth.py does it;
the Memory instance is a MagicMock that records which filters the handler asked for.
"""

import importlib
import os
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed")

from fastapi.testclient import TestClient

ENV = {"AUTH_DISABLED": "true", "ADMIN_API_KEY": "", "OPENAI_API_KEY": "fake-key"}


@pytest.fixture
def memory():
    instance = MagicMock()
    instance.get_all.return_value = {"results": []}
    instance.vector_store.list.return_value = [[]]
    return instance


@pytest.fixture
def client(memory, monkeypatch):
    with patch.dict(os.environ, ENV), patch("mem0.Memory.from_config", return_value=memory):
        import auth
        import main

        importlib.reload(auth)
        importlib.reload(main)
    # The request-log middleware opens a Postgres session in a worker thread; there is none here.
    monkeypatch.setattr(main, "_persist_request_log", lambda *args: None)
    return TestClient(main.app)


def _get_all_filters(memory):
    memory.get_all.assert_called_once()
    return memory.get_all.call_args.kwargs["filters"]


def test_entity_only_request_passes_no_metadata_filters(client, memory):
    resp = client.get("/memories", params={"user_id": "sst"})

    assert resp.status_code == 200
    assert _get_all_filters(memory) == {"user_id": "sst"}


def test_glossary_rows_are_selected_server_side(client, memory):
    resp = client.get(
        "/memories", params={"user_id": "sst", "kind": "terminology", "glossary": "true", "top_k": 5000}
    )

    assert resp.status_code == 200
    assert _get_all_filters(memory) == {"user_id": "sst", "kind": "terminology", "glossary": True}
    assert memory.get_all.call_args.kwargs["top_k"] == 5000


def test_glossary_is_forwarded_as_a_bool(client, memory):
    client.get("/memories", params={"user_id": "sst", "glossary": "true"})

    assert _get_all_filters(memory)["glossary"] is True


def test_glossary_false_is_a_filter_not_an_omission(client, memory):
    client.get("/memories", params={"user_id": "sst", "glossary": "false"})

    assert _get_all_filters(memory)["glossary"] is False


def test_scope_and_project_filters(client, memory):
    client.get("/memories", params={"user_id": "sst", "scope": "project", "project": "memory-server"})

    assert _get_all_filters(memory) == {"user_id": "sst", "scope": "project", "project": "memory-server"}


def test_empty_metadata_param_is_ignored(client, memory):
    client.get("/memories", params={"user_id": "sst", "kind": ""})

    assert _get_all_filters(memory) == {"user_id": "sst"}


def test_non_boolean_glossary_is_rejected(client, memory):
    resp = client.get("/memories", params={"user_id": "sst", "glossary": "maybe"})

    assert resp.status_code == 422
    memory.get_all.assert_not_called()


def test_admin_listing_applies_metadata_filters(client, memory):
    resp = client.get("/memories", params={"kind": "terminology", "glossary": "true", "top_k": 10})

    assert resp.status_code == 200
    memory.get_all.assert_not_called()
    memory.vector_store.list.assert_called_once_with(filters={"kind": "terminology", "glossary": True}, top_k=10)


def test_admin_listing_without_metadata_params_is_unfiltered(client, memory):
    resp = client.get("/memories")

    assert resp.status_code == 200
    memory.vector_store.list.assert_called_once_with(filters=None, top_k=5000)
