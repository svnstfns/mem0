from unittest.mock import Mock, patch

import pytest

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.ollama import OllamaConfig
from mem0.llms.ollama import OllamaLLM
from mem0.utils.factory import LlmFactory


@pytest.fixture
def mock_ollama_client():
    with patch("mem0.llms.ollama.Client") as mock_ollama:
        mock_client = Mock()
        mock_client.list.return_value = {"models": [{"name": "llama3.1:70b"}]}
        mock_ollama.return_value = mock_client
        yield mock_client


def test_generate_response_without_tools(mock_ollama_client):
    config = OllamaConfig(model="llama3.1:70b", temperature=0.7, max_tokens=100, top_p=1.0)
    llm = OllamaLLM(config)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]

    mock_response = {"message": {"content": "I'm doing well, thank you for asking!"}}
    mock_ollama_client.chat.return_value = mock_response

    response = llm.generate_response(messages)

    mock_ollama_client.chat.assert_called_once_with(
        model="llama3.1:70b", messages=messages, options={"temperature": 0.7, "num_predict": 100, "top_p": 1.0}
    )
    assert response == "I'm doing well, thank you for asking!"


def test_generate_response_with_tools_passes_tools_to_client(mock_ollama_client):
    """Tools should be forwarded to ollama client.chat()."""
    config = OllamaConfig(model="llama3.1:70b", temperature=0.1, max_tokens=100, top_p=1.0)
    llm = OllamaLLM(config)
    messages = [{"role": "user", "content": "Extract entities from: Alice works at UCSD"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_entities",
                "description": "Extract entities",
                "parameters": {"type": "object", "properties": {"entities": {"type": "array"}}},
            },
        }
    ]

    mock_response = {
        "message": {
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "extract_entities",
                        "arguments": {"entities": [{"name": "Alice"}, {"name": "UCSD"}]},
                    }
                }
            ],
        }
    }
    mock_ollama_client.chat.return_value = mock_response

    response = llm.generate_response(messages, tools=tools)

    # Verify tools were passed to client.chat
    call_kwargs = mock_ollama_client.chat.call_args
    assert "tools" in call_kwargs.kwargs or (len(call_kwargs.args) > 0 and "tools" in call_kwargs[1])
    assert call_kwargs[1]["tools"] == tools

    # Verify tool_calls were parsed correctly
    assert response["tool_calls"] == [
        {"name": "extract_entities", "arguments": {"entities": [{"name": "Alice"}, {"name": "UCSD"}]}}
    ]


def test_generate_response_with_tools_no_tool_calls_in_response(mock_ollama_client):
    """When model returns content without tool_calls, tool_calls should be empty list."""
    config = OllamaConfig(model="llama3.1:70b", temperature=0.1, max_tokens=100, top_p=1.0)
    llm = OllamaLLM(config)
    messages = [{"role": "user", "content": "Hello"}]
    tools = [{"type": "function", "function": {"name": "noop", "parameters": {}}}]

    mock_response = {"message": {"content": "I cannot use tools for this.", "tool_calls": []}}
    mock_ollama_client.chat.return_value = mock_response

    response = llm.generate_response(messages, tools=tools)

    assert response["content"] == "I cannot use tools for this."
    assert response["tool_calls"] == []


def test_generate_response_with_tools_string_arguments(mock_ollama_client):
    """When tool_call arguments come as JSON string, they should be parsed."""
    config = OllamaConfig(model="llama3.1:70b", temperature=0.1, max_tokens=100, top_p=1.0)
    llm = OllamaLLM(config)
    messages = [{"role": "user", "content": "test"}]
    tools = [{"type": "function", "function": {"name": "test_fn", "parameters": {}}}]

    mock_response = {
        "message": {
            "content": "",
            "tool_calls": [
                {"function": {"name": "test_fn", "arguments": '{"key": "value"}'}}
            ],
        }
    }
    mock_ollama_client.chat.return_value = mock_response

    response = llm.generate_response(messages, tools=tools)

    assert response["tool_calls"] == [{"name": "test_fn", "arguments": {"key": "value"}}]


def test_parse_response_with_tools_object_style(mock_ollama_client):
    """Test _parse_response with object-style response (non-dict)."""
    config = OllamaConfig(model="llama3.1:70b")
    llm = OllamaLLM(config)

    # Simulate object-style response
    mock_fn = Mock()
    mock_fn.name = "extract"
    mock_fn.arguments = {"entities": ["Alice"]}

    mock_tool_call = Mock()
    mock_tool_call.function = mock_fn

    mock_message = Mock()
    mock_message.content = ""
    mock_message.tool_calls = [mock_tool_call]

    mock_response = Mock()
    mock_response.message = mock_message

    tools = [{"type": "function", "function": {"name": "extract"}}]
    result = llm._parse_response(mock_response, tools)

    assert result["tool_calls"] == [{"name": "extract", "arguments": {"entities": ["Alice"]}}]


def test_chat_client_gets_the_configured_timeout():
    with patch("mem0.llms.ollama.Client") as mock_ollama:
        OllamaLLM(
            OllamaConfig(model="llama3.1:70b", ollama_base_url="http://ollama:11434", ollama_timeout=300.0)
        )

        mock_ollama.assert_called_once_with(host="http://ollama:11434", timeout=300.0)


def test_chat_client_without_configured_timeout_keeps_waiting():
    with patch("mem0.llms.ollama.Client") as mock_ollama:
        OllamaLLM(OllamaConfig(model="llama3.1:70b"))

        mock_ollama.assert_called_once_with(host=None, timeout=None)


def test_base_config_timeout_survives_the_conversion_to_ollama_config():
    """OllamaLLM rebuilds a BaseLlmConfig field by field - the timeout must not be dropped there."""
    with patch("mem0.llms.ollama.Client") as mock_ollama:
        OllamaLLM(BaseLlmConfig(model="llama3.1:70b", ollama_timeout=300.0))

        mock_ollama.assert_called_once_with(host=None, timeout=300.0)


def test_factory_forwards_the_timeout_to_the_ollama_client():
    """LlmFactory does its own BaseLlmConfig -> OllamaConfig conversion, with the same risk."""
    with patch("mem0.llms.ollama.Client") as mock_ollama:
        LlmFactory.create("ollama", BaseLlmConfig(model="llama3.1:70b", ollama_timeout=300.0))

        mock_ollama.assert_called_once_with(host=None, timeout=300.0)
