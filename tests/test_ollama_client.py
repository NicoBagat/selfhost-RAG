"""
tests/test_ollama_client.py

Unit tests for app.ollama_client.OllamaClient.

All tests are fully offline: the inner ollama.Client is replaced with a
MagicMock immediately after construction, so no running Ollama instance
or network access is required.

Coverage
--------
- OllamaClient.from_config()   — field mapping, type coercion, defaults
- health_check()               — success, connection failure, missing model
- chat()                       — text response, tool-call response, errors
- chat_stream()                — token yield, empty-token skip, mid-stream error
- embed()                      — single / batch input, empty response, model used
- _parse_response()            — content vs tool_calls precedence
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import ollama
import pytest

from app.ollama_client import (
    ChatResponse,
    OllamaClient,
    OllamaConnectionError,
    OllamaModelError,
    OllamaResponseError,
    ToolCall,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client() -> OllamaClient:
    """Return an OllamaClient whose inner ollama.Client is a MagicMock."""
    c = OllamaClient(
        host="http://localhost:11434",
        model="gemma3:4b",
        embed_model="nomic-embed-text",
        temperature=0.2,
        context_window=8192,
    )
    c._client = MagicMock()
    return c


def _model_stub(tag: str) -> SimpleNamespace:
    return SimpleNamespace(model=tag)


def _chat_stub(
    content: str = "",
    tool_calls: list | None = None,
    model: str = "gemma3:4b",
    prompt_eval_count: int = 10,
    eval_count: int = 20,
) -> SimpleNamespace:
    msg = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    return SimpleNamespace(
        message=msg,
        model=model,
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
    )


def _tool_call_stub(name: str, arguments: dict) -> SimpleNamespace:
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


# ---------------------------------------------------------------------------
# from_config
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_all_fields_populated(self):
        cfg = {
            "host": "http://localhost:11434",
            "model": "gemma3:4b",
            "embed_model": "nomic-embed-text",
            "temperature": "0.1",
            "context_window": "4096",
        }
        c = OllamaClient.from_config(cfg)
        assert c.host == "http://localhost:11434"
        assert c.model == "gemma3:4b"
        assert c.embed_model == "nomic-embed-text"
        assert c.temperature == 0.1
        assert c.context_window == 4096

    def test_temperature_and_context_defaults(self):
        cfg = {
            "host": "http://localhost:11434",
            "model": "gemma3:4b",
            "embed_model": "nomic-embed-text",
        }
        c = OllamaClient.from_config(cfg)
        assert c.temperature == 0.2
        assert c.context_window == 8192

    def test_temperature_coerced_to_float(self):
        cfg = {
            "host": "http://localhost:11434",
            "model": "m",
            "embed_model": "e",
            "temperature": "0.5",
        }
        c = OllamaClient.from_config(cfg)
        assert isinstance(c.temperature, float)

    def test_context_window_coerced_to_int(self):
        cfg = {
            "host": "http://localhost:11434",
            "model": "m",
            "embed_model": "e",
            "context_window": "16384",
        }
        c = OllamaClient.from_config(cfg)
        assert isinstance(c.context_window, int)
        assert c.context_window == 16384


# ---------------------------------------------------------------------------
# health_check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_returns_list_of_available_tags(self):
        c = _make_client()
        c._client.list.return_value = SimpleNamespace(
            models=[_model_stub("gemma3:4b"), _model_stub("nomic-embed-text")]
        )
        result = c.health_check()
        assert "gemma3:4b" in result
        assert "nomic-embed-text" in result

    def test_raises_connection_error_on_unreachable_host(self):
        c = _make_client()
        c._client.list.side_effect = ConnectionRefusedError("refused")
        with pytest.raises(OllamaConnectionError):
            c.health_check()

    def test_raises_model_error_when_generation_model_absent(self):
        c = _make_client()
        c._client.list.return_value = SimpleNamespace(
            models=[_model_stub("nomic-embed-text")]
        )
        with pytest.raises(OllamaModelError, match="gemma3:4b"):
            c.health_check()

    def test_raises_model_error_when_embed_model_absent(self):
        c = _make_client()
        c._client.list.return_value = SimpleNamespace(
            models=[_model_stub("gemma3:4b")]
        )
        with pytest.raises(OllamaModelError, match="nomic-embed-text"):
            c.health_check()

    def test_prefix_match_accepts_variant_tags(self):
        # "gemma3:4b" config should match "gemma3:4b-instruct" in the list
        c = _make_client()
        c._client.list.return_value = SimpleNamespace(
            models=[
                _model_stub("gemma3:4b-instruct"),
                _model_stub("nomic-embed-text:latest"),
            ]
        )
        result = c.health_check()
        assert len(result) == 2


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


class TestChat:
    def test_text_response_populates_content(self):
        c = _make_client()
        c._client.chat.return_value = _chat_stub(content="Hello!")
        resp = c.chat([{"role": "user", "content": "hi"}])
        assert isinstance(resp, ChatResponse)
        assert resp.content == "Hello!"
        assert resp.tool_calls == []

    def test_token_counts_captured(self):
        c = _make_client()
        c._client.chat.return_value = _chat_stub(
            content="x", prompt_eval_count=42, eval_count=7
        )
        resp = c.chat([{"role": "user", "content": "hi"}])
        assert resp.prompt_eval_count == 42
        assert resp.eval_count == 7

    def test_model_tag_captured(self):
        c = _make_client()
        c._client.chat.return_value = _chat_stub(content="x", model="gemma3:4b")
        resp = c.chat([{"role": "user", "content": "hi"}])
        assert resp.model == "gemma3:4b"

    def test_tool_call_response_populates_tool_calls(self):
        c = _make_client()
        tc = _tool_call_stub("read_note", {"path": "foo.md"})
        c._client.chat.return_value = _chat_stub(tool_calls=[tc])
        resp = c.chat(
            [{"role": "user", "content": "open foo"}],
            tools=[{"function": {"name": "read_note"}}],
        )
        assert len(resp.tool_calls) == 1
        assert isinstance(resp.tool_calls[0], ToolCall)
        assert resp.tool_calls[0].name == "read_note"
        assert resp.tool_calls[0].arguments == {"path": "foo.md"}

    def test_multiple_tool_calls_all_captured(self):
        c = _make_client()
        tcs = [
            _tool_call_stub("search", {"query": "foo"}),
            _tool_call_stub("read_note", {"path": "bar.md"}),
        ]
        c._client.chat.return_value = _chat_stub(tool_calls=tcs)
        resp = c.chat([{"role": "user", "content": "hi"}])
        assert len(resp.tool_calls) == 2
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[1].name == "read_note"

    def test_tools_kwarg_forwarded_when_provided(self):
        c = _make_client()
        c._client.chat.return_value = _chat_stub(content="ok")
        tools = [{"function": {"name": "search"}}]
        c.chat([{"role": "user", "content": "hi"}], tools=tools)
        call_kwargs = c._client.chat.call_args.kwargs
        assert call_kwargs["tools"] == tools

    def test_no_tools_kwarg_when_tools_is_none(self):
        c = _make_client()
        c._client.chat.return_value = _chat_stub(content="ok")
        c.chat([{"role": "user", "content": "hi"}])
        call_kwargs = c._client.chat.call_args.kwargs
        assert "tools" not in call_kwargs

    def test_stream_is_always_false(self):
        c = _make_client()
        c._client.chat.return_value = _chat_stub(content="ok")
        c.chat([{"role": "user", "content": "hi"}])
        call_kwargs = c._client.chat.call_args.kwargs
        assert call_kwargs["stream"] is False

    def test_ollama_response_error_raises_ollama_response_error(self):
        c = _make_client()
        c._client.chat.side_effect = ollama.ResponseError("bad")
        with pytest.raises(OllamaResponseError):
            c.chat([{"role": "user", "content": "hi"}])

    def test_connection_error_raises_ollama_connection_error(self):
        c = _make_client()
        c._client.chat.side_effect = ConnectionRefusedError("refused")
        with pytest.raises(OllamaConnectionError):
            c.chat([{"role": "user", "content": "hi"}])


# ---------------------------------------------------------------------------
# chat_stream
# ---------------------------------------------------------------------------


class TestChatStream:
    @staticmethod
    def _chunk(token: str | None) -> SimpleNamespace:
        return SimpleNamespace(message=SimpleNamespace(content=token))

    def test_yields_tokens_in_order(self):
        c = _make_client()
        c._client.chat.return_value = iter([
            self._chunk("Hello"),
            self._chunk(", "),
            self._chunk("world"),
        ])
        tokens = list(c.chat_stream([{"role": "user", "content": "hi"}]))
        assert tokens == ["Hello", ", ", "world"]

    def test_empty_string_tokens_skipped(self):
        c = _make_client()
        c._client.chat.return_value = iter([
            self._chunk("Hello"),
            self._chunk(""),
            self._chunk("!"),
        ])
        tokens = list(c.chat_stream([{"role": "user", "content": "hi"}]))
        assert tokens == ["Hello", "!"]

    def test_none_tokens_skipped(self):
        c = _make_client()
        c._client.chat.return_value = iter([
            self._chunk("Hi"),
            self._chunk(None),
            self._chunk("!"),
        ])
        tokens = list(c.chat_stream([{"role": "user", "content": "hi"}]))
        assert tokens == ["Hi", "!"]

    def test_mid_stream_error_raises_connection_error(self):
        c = _make_client()

        def _bad():
            yield self._chunk("start")
            raise ConnectionResetError("dropped")

        c._client.chat.return_value = _bad()
        with pytest.raises(OllamaConnectionError):
            list(c.chat_stream([{"role": "user", "content": "hi"}]))

    def test_stream_flag_is_true(self):
        c = _make_client()
        c._client.chat.return_value = iter([])
        list(c.chat_stream([{"role": "user", "content": "hi"}]))
        call_kwargs = c._client.chat.call_args.kwargs
        assert call_kwargs["stream"] is True


# ---------------------------------------------------------------------------
# embed
# ---------------------------------------------------------------------------


class TestEmbed:
    def test_single_string_returns_flat_vector(self):
        c = _make_client()
        c._client.embed.return_value = SimpleNamespace(embeddings=[[0.1, 0.2, 0.3]])
        result = c.embed("hello")
        assert result == [0.1, 0.2, 0.3]

    def test_list_input_returns_list_of_vectors(self):
        c = _make_client()
        c._client.embed.return_value = SimpleNamespace(
            embeddings=[[0.1, 0.2], [0.3, 0.4]]
        )
        result = c.embed(["hello", "world"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_batch_preserves_order(self):
        c = _make_client()
        vecs = [[float(i)] for i in range(5)]
        c._client.embed.return_value = SimpleNamespace(embeddings=vecs)
        result = c.embed([f"text_{i}" for i in range(5)])
        assert result == vecs

    def test_empty_embeddings_raises_response_error(self):
        c = _make_client()
        c._client.embed.return_value = SimpleNamespace(embeddings=[])
        with pytest.raises(OllamaResponseError):
            c.embed("hello")

    def test_embed_uses_configured_embed_model(self):
        c = _make_client()
        c._client.embed.return_value = SimpleNamespace(embeddings=[[0.0]])
        c.embed("hello")
        call_kwargs = c._client.embed.call_args.kwargs
        assert call_kwargs["model"] == "nomic-embed-text"

    def test_connection_error_raises_ollama_connection_error(self):
        c = _make_client()
        c._client.embed.side_effect = ConnectionRefusedError("refused")
        with pytest.raises(OllamaConnectionError):
            c.embed("hello")

    def test_ollama_response_error_raises_ollama_response_error(self):
        c = _make_client()
        c._client.embed.side_effect = ollama.ResponseError("bad")
        with pytest.raises(OllamaResponseError):
            c.embed("hello")


# ---------------------------------------------------------------------------
# _parse_response (internal, tested directly)
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_content_only_response(self):
        c = _make_client()
        raw = _chat_stub(content="answer")
        resp = c._parse_response(raw)
        assert resp.content == "answer"
        assert resp.tool_calls == []

    def test_tool_calls_only_response(self):
        c = _make_client()
        tc = _tool_call_stub("my_tool", {"x": 1})
        raw = _chat_stub(tool_calls=[tc])
        resp = c._parse_response(raw)
        assert resp.tool_calls[0].name == "my_tool"
        assert resp.tool_calls[0].arguments == {"x": 1}

    def test_none_content_coerced_to_empty_string(self):
        c = _make_client()
        raw = _chat_stub(content=None)
        resp = c._parse_response(raw)
        assert resp.content == ""

    def test_arguments_dict_is_plain_dict(self):
        """arguments must be a plain dict, not a MagicMock or custom type."""
        c = _make_client()
        tc = _tool_call_stub("tool", {"key": "value"})
        raw = _chat_stub(tool_calls=[tc])
        resp = c._parse_response(raw)
        assert type(resp.tool_calls[0].arguments) is dict
