"""
app/orchestrator.py

Central inference loop. Wires together OllamaClient, Retriever, and
MCPClient into a single async handle_query() entry point.

Query lifecycle
---------------
1. QueryRouter classifies intent → RAG | MCP | BOTH
2. Retriever.search() fetches top-k chunks from ChromaDB  (RAG path)
3. Tool loop via OllamaClient.chat() (non-streaming, max MAX_TOOL_ITERS):
       response.tool_calls present  → dispatch to MCPClient, append result, loop
       content looks like JSON call → FLAG 1 fallback extraction, dispatch, loop
       plain text content           → loop ends, use content directly
4. OllamaClient.chat_stream() streams the final answer token-by-token
   via the on_token callback (runs in a thread; tokens are posted back to
   the asyncio event loop via call_soon_threadsafe)
5. User turn + assistant turn are appended to the conversation history.
   History is char-budget truncated before each call to stay inside the
   model's context window.

FLAG 1 — Gemma 4B tool-call reliability
----------------------------------------
Small models sometimes embed a tool-call JSON object inside the text
content field rather than using the structured tool_calls field. The
_try_extract_tool_call() helper attempts two recovery strategies before
treating the response as a plain-text answer:
  a. Parse entire content as a JSON object with "name" / "arguments" keys
  b. Regex-extract the first {...} block containing those keys

FLAG 2 — two retrieval paths
-----------------------------
QueryRouter selects the path upfront:
  RAG   → semantic search, no live note access
  MCP   → live CRUD via mcp-obsidian, no vector search
  BOTH  → RAG for context + MCP tools available for specifics/writes
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from app.mcp_client import MCPClient, MCPConnectionError, MCPToolError
from app.ollama_client import OllamaClient
from app.rag.indexer import IndexStats, Indexer
from app.rag.retriever import Retriever

logger = logging.getLogger(__name__)

MAX_TOOL_ITERS = 3

# Characters reserved for system prompt + retrieved context + tool defs.
# History is truncated so total prompt stays within this budget.
# Rough estimate: 1 token ≈ 4 chars; default context_window = 8192 tokens.
_HISTORY_BUDGET_CHARS = 16_000

_SYSTEM_PROMPT = """\
You are a personal knowledge assistant for an Obsidian vault.
You help the user search, explore, and manage their notes.

Guidelines:
- Ground your answers in the retrieved context snippets when they are relevant.
- Cite the source note filename (e.g. "Projects/foo.md") when drawing from context.
- Use tools only when you need to access a specific note by name, list files, \
or perform writes. Do not call tools when the retrieved context already answers \
the question.
- After receiving tool results, synthesise a clear final answer — do not call \
more tools unless strictly necessary.
- Be concise and direct.\
"""


# ---------------------------------------------------------------------------
# Query routing
# ---------------------------------------------------------------------------


class QueryIntent(Enum):
    RAG = auto()   # semantic search only
    MCP = auto()   # live note access / writes only
    BOTH = auto()  # RAG context + MCP tools available


_WRITE_RE = re.compile(
    r"\b(add|write|create|append|delete|remove|move|rename|update|edit)\b"
    r".*\b(note|file|page)\b"
    r"|\bto my\s+\w*\s*(note|journal|log|daily)\b",
    re.IGNORECASE,
)

_READ_RE = re.compile(
    r"\b(open|read|show|fetch|get|load|list)\b.*\b(note|file|folder|directory|dir|vault)\b"
    r"|\b(list|show)\b.*(root|top.level|first.level)"
    r"|\bnote\s+called\b|\bfile\s+called\b"
    r"|['\"][\w\s]+\.md['\"]"
    r"|\b\w+\.md\b",
    re.IGNORECASE,
)


class QueryRouter:
    """
    Lightweight rule-based intent classifier.

    No model call — classification is deterministic and instant.
    Extend _WRITE_RE / _READ_RE patterns to refine routing.
    """

    def route(self, query: str) -> QueryIntent:
        is_write = bool(_WRITE_RE.search(query))
        is_read = bool(_READ_RE.search(query))

        if is_write:
            return QueryIntent.MCP
        if is_read:
            return QueryIntent.BOTH   # fetch specific note + semantic context
        return QueryIntent.RAG        # default: semantic search


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """
    Wires OllamaClient + Retriever + MCPClient into a single async interface.

    Parameters
    ----------
    ollama : OllamaClient
    retriever : Retriever | None
        Required for RAG paths. Pass None to disable RAG (MCP-only mode).
    mcp : MCPClient | None
        Required for MCP paths. Pass None to disable live note access.
    top_k : int
        Number of chunks injected per query on the RAG path.
    history_budget_chars : int
        Maximum characters of conversation history kept in each prompt.
    """

    def __init__(
        self,
        ollama: OllamaClient,
        retriever: Retriever | None = None,
        mcp: MCPClient | None = None,
        indexer: Indexer | None = None,
        top_k: int = 5,
        history_budget_chars: int = _HISTORY_BUDGET_CHARS,
    ) -> None:
        self._ollama = ollama
        self._retriever = retriever
        self._mcp = mcp
        self._indexer = indexer
        self._top_k = top_k
        self._history_budget = history_budget_chars
        self._router = QueryRouter()
        self._history: list[dict[str, Any]] = []  # grows across turns

    @classmethod
    def from_config(
        cls,
        config: dict,
        ollama: OllamaClient,
        retriever: Retriever | None = None,
        mcp: MCPClient | None = None,
        indexer: Indexer | None = None,
    ) -> Orchestrator:
        """Construct from the full settings dict; retriever, mcp, and indexer are optional."""
        top_k = int(config.get("rag", {}).get("top_k", 5))
        return cls(ollama=ollama, retriever=retriever, mcp=mcp, indexer=indexer, top_k=top_k)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_query(
        self,
        query: str,
        on_token: Callable[[str], None],
        intent_override: QueryIntent | None = None,
    ) -> QueryIntent:
        """
        Process a user query end-to-end.

        Retrieves context, runs the tool loop if needed, then streams the
        final answer token-by-token via on_token.  Appends both turns to
        the conversation history on completion.

        Parameters
        ----------
        query : str
            Raw user input.
        on_token : Callable[[str], None]
            Called once per streamed token.  Must be thread-safe if the
            caller is a Qt UI (use Qt signals inside the callback).
        intent_override : QueryIntent | None
            When set, bypasses the QueryRouter and forces the given intent.
            Pass None (default) to use automatic routing.

        Returns
        -------
        QueryIntent
            The intent that was actually used for this query.
        """
        if intent_override is not None:
            intent = intent_override
            logger.info("Query intent: %s (manual) — %r", intent.name, query[:80])
        else:
            intent = self._router.route(query)
            logger.info("Query intent: %s (auto) — %r", intent.name, query[:80])

        # 1. RAG retrieval
        chunks: list[dict[str, Any]] = []
        if intent in (QueryIntent.RAG, QueryIntent.BOTH) and self._retriever:
            chunks = await asyncio.to_thread(
                self._retriever.search, query, top_k=self._top_k
            )
            logger.debug("Retrieved %d chunk(s)", len(chunks))

        # 2. Build initial message list
        messages = self._build_messages(query, chunks)

        # 3. Tool loop (MCP path)
        mcp_active = (
            intent in (QueryIntent.MCP, QueryIntent.BOTH)
            and self._mcp is not None
            and self._mcp.is_running
        )
        if mcp_active:
            messages = await self._run_tool_loop(messages, self._mcp.tools)  # type: ignore[union-attr]

        # 4. Stream final answer
        final_text = await self._stream_response(messages, on_token)

        # 5. Persist turns in history
        self._history.append({"role": "user", "content": query})
        self._history.append({"role": "assistant", "content": final_text})

        return intent

    def clear_history(self) -> None:
        """Reset the conversation history (start a new session)."""
        self._history.clear()
        logger.debug("Conversation history cleared")

    async def reindex(self) -> IndexStats:
        """
        Run a full incremental vault index in a thread pool, then invalidate
        the BM25 cache so the next query picks up any new content.

        Raises RuntimeError if no Indexer was provided at construction.
        """
        if self._indexer is None:
            raise RuntimeError("No Indexer configured — pass indexer= to Orchestrator")
        stats = await asyncio.to_thread(self._indexer.index_vault)
        if self._retriever is not None:
            self._retriever.invalidate_cache()
        logger.info(
            "Reindex complete — indexed=%d  skipped=%d  removed=%d  errors=%d",
            stats.indexed, stats.skipped, stats.removed, stats.errors,
        )
        return stats

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        query: str,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Assemble the full message list: system + history + user turn.

        Retrieved chunks are injected into the system prompt so the model
        sees them before any conversation history.
        """
        system_content = _SYSTEM_PROMPT
        if chunks:
            system_content += "\n\n" + self._format_context(chunks)

        truncated_history = _truncate_history(self._history, self._history_budget)

        return [
            {"role": "system", "content": system_content},
            *truncated_history,
            {"role": "user", "content": query},
        ]

    def _format_context(self, chunks: list[dict[str, Any]]) -> str:
        """Render retrieved chunks as a labelled context block."""
        lines = ["--- Retrieved context ---"]
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "unknown")
            header = chunk.get("header", "")
            label = f"{source} › {header}" if header else source
            lines.append(f"\n[{i}] {label}\n{chunk['text']}")
        lines.append("\n--- End of context ---")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool loop
    # ------------------------------------------------------------------

    async def _run_tool_loop(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Drive the agentic tool loop until the model stops calling tools or
        MAX_TOOL_ITERS is reached.

        Returns the updated message list (including all tool exchanges).
        """
        for iteration in range(MAX_TOOL_ITERS):
            response = await asyncio.to_thread(self._ollama.chat, messages, tools=tools)

            # --- structured tool_calls (normal path) ---
            if response.tool_calls:
                messages.append(_assistant_tool_msg(response.tool_calls))
                for tc in response.tool_calls:
                    result = await self._dispatch_tool(tc.name, tc.arguments)
                    messages.append({"role": "tool", "content": result})
                continue

            # --- FLAG 1 fallback: try to extract from content ---
            extracted = _try_extract_tool_call(response.content)
            if extracted:
                logger.debug(
                    "FLAG1: extracted tool call from content — %s", extracted["name"]
                )
                messages.append(_assistant_tool_msg_raw([extracted]))
                result = await self._dispatch_tool(
                    extracted["name"], extracted["arguments"]
                )
                messages.append({"role": "tool", "content": result})
                continue

            # --- plain text: tool loop ends ---
            logger.debug("Tool loop ended after %d iteration(s)", iteration)
            break

        else:
            logger.warning(
                "Tool loop reached MAX_TOOL_ITERS (%d) — proceeding to final answer",
                MAX_TOOL_ITERS,
            )

        return messages

    async def _dispatch_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call MCPClient and return the result string, logging any errors."""
        assert self._mcp is not None
        logger.debug("Dispatching tool: %s(%s)", name, arguments)
        try:
            return await self._mcp.call_tool(name, arguments)
        except MCPToolError as exc:
            logger.warning("Tool error (%s): %s", name, exc)
            return f"[Tool error: {exc}]"
        except MCPConnectionError as exc:
            logger.error("MCP connection error during tool dispatch: %s", exc)
            return f"[Connection error: {exc}]"

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def _stream_response(
        self,
        messages: list[dict[str, Any]],
        on_token: Callable[[str], None],
    ) -> str:
        """
        Run chat_stream() in a thread pool and post tokens back to the
        asyncio event loop via call_soon_threadsafe.

        Returns the complete concatenated response text for history storage.
        """
        loop = asyncio.get_running_loop()
        buffer: list[str] = []

        def _run() -> None:
            for token in self._ollama.chat_stream(messages):
                buffer.append(token)
                loop.call_soon_threadsafe(on_token, token)

        await asyncio.to_thread(_run)
        return "".join(buffer)


# ---------------------------------------------------------------------------
# Pure helpers (no class state)
# ---------------------------------------------------------------------------


def _truncate_history(
    history: list[dict[str, Any]],
    budget_chars: int,
) -> list[dict[str, Any]]:
    """
    Return the most recent history entries that fit within budget_chars.

    Walks backwards so recent turns are always preserved over older ones.
    """
    total = 0
    kept: list[dict[str, Any]] = []
    for msg in reversed(history):
        cost = len(str(msg.get("content", "")))
        if total + cost > budget_chars:
            break
        kept.insert(0, msg)
        total += cost
    return kept


def _assistant_tool_msg(tool_calls: list[Any]) -> dict[str, Any]:
    """Build an assistant message dict from a list of ToolCall dataclass instances."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": tc.name, "arguments": tc.arguments}}
            for tc in tool_calls
        ],
    }


def _assistant_tool_msg_raw(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Build an assistant message dict from FLAG 1 extracted dicts."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
            for tc in tool_calls
        ],
    }


def _try_extract_tool_call(content: str) -> dict[str, Any] | None:
    """
    FLAG 1 fallback: attempt to extract a tool call from a plain-text response.

    Strategy A — parse entire content as JSON with "name" and "arguments".
    Strategy B — regex-extract the first {...} block containing those keys.

    Returns a dict {"name": str, "arguments": dict} or None.
    """
    stripped = content.strip()

    # Strategy A: whole content is a JSON tool call
    try:
        data = json.loads(stripped)
        if isinstance(data, dict) and "name" in data and "arguments" in data:
            if isinstance(data["arguments"], dict):
                return {"name": str(data["name"]), "arguments": data["arguments"]}
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy B: walk the string tracking brace depth to extract candidate
    # JSON blocks, then try to parse each one.  The regex approach fails here
    # because arguments values are themselves objects (nested braces).
    for block in _iter_json_blocks(stripped):
        try:
            data = json.loads(block)
            if (
                isinstance(data, dict)
                and "name" in data
                and isinstance(data.get("arguments"), dict)
            ):
                return {"name": str(data["name"]), "arguments": data["arguments"]}
        except (json.JSONDecodeError, ValueError):
            continue

    return None


def _iter_json_blocks(text: str):
    """
    Yield every top-level {...} substring from text, handling nested braces.
    Used by _try_extract_tool_call Strategy B.
    """
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                yield text[start : i + 1]
                start = -1
