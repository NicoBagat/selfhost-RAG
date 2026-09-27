"""
app/ollama_client.py

Wrapper around the Ollama Python client for two distinct operations:
  - chat()        blocking inference with optional tool definitions
  - chat_stream() generator yielding text tokens (no tool support)
  - embed()       dense vector embedding via nomic-embed-text

Design notes:
  - All tool-call responses are returned as complete (non-streaming) messages.
    Streaming is reserved for pure text responses where no tool call is
    expected. This avoids the complexity of reconstructing tool-call JSON
    from a stream, which is unreliable at 4B parameter scale.
  - Temperature and context window are set per-request via Ollama's `options`
    dict, not at client construction, to allow future per-request overrides.
  - The ollama library raises ollama.ResponseError for HTTP-level errors and
    httpx.ConnectError for connection failures. Both are caught and re-raised
    as domain-specific exceptions defined here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Generator, overload

import ollama

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain exceptions
# ---------------------------------------------------------------------------


class OllamaError(Exception):
    """Base class for all OllamaClient errors."""


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama host is unreachable."""


class OllamaModelError(OllamaError):
    """Raised when a required model is not available on the Ollama instance."""


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an unexpected or malformed response."""


# ---------------------------------------------------------------------------
# Response data structures
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """A single tool invocation emitted by the model."""

    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    """
    Normalised response from a chat() call.

    Exactly one of `content` or `tool_calls` will be non-empty per response.
    The orchestrator must check `tool_calls` first before consuming `content`.
    """

    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = ""
    prompt_eval_count: int = 0   # tokens consumed by the prompt
    eval_count: int = 0          # tokens generated


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OllamaClient:
    """
    Thin, typed wrapper over ollama.Client.

    Parameters
    ----------
    host : str
        Full base URL of the Ollama instance, e.g. "http://localhost:11434".
    model : str
        Tag of the generation model, e.g. "gemma3:4b".
    embed_model : str
        Tag of the embedding model, e.g. "nomic-embed-text".
    temperature : float
        Sampling temperature for generation. Keep low (≤ 0.3) for tool use.
    context_window : int
        Maximum context length in tokens (num_ctx). Must not exceed the
        model's trained context limit.
    """

    def __init__(
        self,
        host: str,
        model: str,
        embed_model: str,
        temperature: float = 0.2,
        context_window: int = 8192,
    ) -> None:
        self.host = host
        self.model = model
        self.embed_model = embed_model
        self.temperature = temperature
        self.context_window = context_window
        self._client = ollama.Client(host=host)

    @classmethod
    def from_config(cls, config: dict) -> OllamaClient:
        """
        Construct from the `ollama` section of settings.yaml.

        Expected keys: host, model, embed_model, temperature, context_window.
        """
        return cls(
            host=config["host"],
            model=config["model"],
            embed_model=config["embed_model"],
            temperature=float(config.get("temperature", 0.2)),
            context_window=int(config.get("context_window", 8192)),
        )

    # ------------------------------------------------------------------
    # Health / model verification
    # ------------------------------------------------------------------

    def health_check(self) -> list[str]:
        """
        Verify the Ollama host is reachable and return the list of locally
        available model tags.

        Raises
        ------
        OllamaConnectionError
            If the host cannot be reached.
        OllamaModelError
            If either `self.model` or `self.embed_model` is not in the list.
        """
        try:
            response = self._client.list()
        except Exception as exc:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self.host}: {exc}"
            ) from exc

        available: list[str] = [m.model for m in response.models]
        logger.debug("Ollama models available: %s", available)

        for required in (self.model, self.embed_model):
            if not any(tag.startswith(required.split(":")[0]) for tag in available):
                raise OllamaModelError(
                    f"Model '{required}' not found on Ollama instance at {self.host}. "
                    f"Run: ollama pull {required}"
                )

        return available

    # ------------------------------------------------------------------
    # Chat — blocking, with optional tool definitions
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        """
        Send a chat request and return a complete ChatResponse.

        Always non-streaming. Use chat_stream() for streaming text output
        when no tool calls are expected.

        Parameters
        ----------
        messages : list[dict]
            Conversation history in Ollama/OpenAI message format:
            [{"role": "user"|"assistant"|"system"|"tool", "content": str}, ...]
        tools : list[dict] | None
            Optional tool definitions in OpenAI function-calling schema.
            When provided, the model may return tool_calls instead of content.

        Returns
        -------
        ChatResponse
            Populated with either content or tool_calls (check tool_calls first).

        Raises
        ------
        OllamaConnectionError
            If the host is unreachable during the request.
        OllamaResponseError
            If the response cannot be parsed into a ChatResponse.
        """
        options = {
            "temperature": self.temperature,
            "num_ctx": self.context_window,
        }

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": options,
            "stream": False,
        }
        if tools:
            kwargs["tools"] = tools

        logger.debug(
            "chat() → model=%s  messages=%d  tools=%s",
            self.model,
            len(messages),
            [t["function"]["name"] for t in tools] if tools else "none",
        )

        try:
            raw = self._client.chat(**kwargs)
        except ollama.ResponseError as exc:
            raise OllamaResponseError(f"Ollama API error during chat: {exc}") from exc
        except Exception as exc:
            raise OllamaConnectionError(
                f"Connection error during chat ({self.host}): {exc}"
            ) from exc

        return self._parse_response(raw)

    # ------------------------------------------------------------------
    # Chat — streaming, text only
    # ------------------------------------------------------------------

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
    ) -> Generator[str, None, None]:
        """
        Stream a chat response token by token.

        Do not pass tools here. Streaming with tool calls requires
        accumulating the full response before parsing JSON, which adds
        latency with no benefit. Use chat() for tool-enabled turns.

        Parameters
        ----------
        messages : list[dict]
            Conversation history (same format as chat()).

        Yields
        ------
        str
            Successive content tokens as they arrive from Ollama.

        Raises
        ------
        OllamaConnectionError
        OllamaResponseError
        """
        options = {
            "temperature": self.temperature,
            "num_ctx": self.context_window,
        }

        logger.debug("chat_stream() → model=%s  messages=%d", self.model, len(messages))

        try:
            stream = self._client.chat(
                model=self.model,
                messages=messages,
                options=options,
                stream=True,
            )
            for chunk in stream:
                token = chunk.message.content
                if token:
                    yield token
        except ollama.ResponseError as exc:
            raise OllamaResponseError(
                f"Ollama API error during streaming chat: {exc}"
            ) from exc
        except Exception as exc:
            raise OllamaConnectionError(
                f"Connection error during streaming chat ({self.host}): {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Embed
    # ------------------------------------------------------------------

    @overload
    def embed(self, text: str) -> list[float]: ...
    @overload
    def embed(self, text: list[str]) -> list[list[float]]: ...

    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """
        Produce dense embeddings via the embedding model.

        Parameters
        ----------
        text : str | list[str]
            A single string or a batch of strings to embed.

        Returns
        -------
        list[float]
            If `text` is a str: a single embedding vector.
        list[list[float]]
            If `text` is a list: one vector per input string, preserving order.

        Raises
        ------
        OllamaConnectionError
        OllamaResponseError
        """
        single = isinstance(text, str)
        inputs: list[str] = [text] if single else text

        logger.debug(
            "embed() → model=%s  inputs=%d", self.embed_model, len(inputs)
        )

        try:
            response = self._client.embed(model=self.embed_model, input=inputs)
        except ollama.ResponseError as exc:
            raise OllamaResponseError(
                f"Ollama API error during embed: {exc}"
            ) from exc
        except Exception as exc:
            raise OllamaConnectionError(
                f"Connection error during embed ({self.host}): {exc}"
            ) from exc

        vectors: list[list[float]] = response.embeddings

        if not vectors:
            raise OllamaResponseError(
                "Embed response contained no embeddings. "
                f"Check that '{self.embed_model}' is a valid embedding model."
            )

        return vectors[0] if single else vectors

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_response(self, raw: Any) -> ChatResponse:
        """
        Convert a raw ollama.ChatResponse into our ChatResponse dataclass.

        Tool calls take precedence: if the model returns both content and
        tool_calls (unusual but possible), tool_calls are preserved and
        content is set to empty string.
        """
        msg = raw.message

        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=dict(tc.function.arguments),
                    )
                )
            logger.debug(
                "Model emitted %d tool call(s): %s",
                len(tool_calls),
                [tc.name for tc in tool_calls],
            )

        content = msg.content or ""

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            model=raw.model,
            prompt_eval_count=getattr(raw, "prompt_eval_count", 0) or 0,
            eval_count=getattr(raw, "eval_count", 0) or 0,
        )
