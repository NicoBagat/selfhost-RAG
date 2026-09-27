"""
app/mcp_client.py

MCP client layer. Manages the full lifecycle of an mcp-obsidian subprocess
over stdio transport, exposes its tool registry in Ollama/OpenAI
function-calling format, and dispatches tool calls on behalf of the
orchestrator.

Design
------
The MCP Python SDK's ClientSession is async-native. Rather than re-spawning
the subprocess on every call, an AsyncExitStack keeps both the stdio
transport and the ClientSession alive for the lifetime of the object.
This means one start() / stop() pair per application run.

Env vars
--------
OBSIDIAN_API_KEY, OBSIDIAN_HOST, and OBSIDIAN_PORT must be present in the
process environment before start() is called. The application's .env file
is the intended source; the app itself does not load it (human-managed).
from_config() passes dict(os.environ) so the subprocess receives all
variables including OBSIDIAN_* — the MCP SDK's env=None only inherits a
short allowlist that excludes custom vars. Secrets never touch settings.yaml.

Usage
-----
    client = MCPClient.from_config(config)
    async with client:
        tools  = client.tools   # inject into OllamaClient.chat()
        result = await client.call_tool("read_note", {"path": "foo.md"})
"""

from __future__ import annotations

import logging
import os
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class MCPError(Exception):
    """Base class for all MCPClient errors."""


class MCPConnectionError(MCPError):
    """Raised when the subprocess cannot start or the session is unavailable."""


class MCPToolError(MCPError):
    """Raised when mcp-obsidian returns an error result for a tool call."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MCPClient:
    """
    Async wrapper around an mcp-obsidian stdio subprocess.

    Parameters
    ----------
    command : str
        Executable to run, e.g. ``"uvx"``.
    args : list[str]
        Arguments passed to the command, e.g. ``["mcp-obsidian"]``.
    env : dict[str, str] | None
        Environment for the subprocess. ``None`` (default) inherits the
        parent process environment, which must already contain the required
        OBSIDIAN_* variables.
    timeout : float
        Seconds to wait for tool-call responses.
    """

    def __init__(
        self,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._command = command
        self._args = args
        self._env = env
        self._timeout = timeout

        self._stack = AsyncExitStack()
        self._session: ClientSession | None = None
        self._tools_cache: list[dict[str, Any]] = []

    @classmethod
    def from_config(cls, config: dict) -> MCPClient:
        """
        Construct from the ``mcp.obsidian`` section of settings.yaml.

        Passes the full ``os.environ`` snapshot so the subprocess receives
        all variables (including OBSIDIAN_* vars loaded from .env by
        load_dotenv). The MCP SDK's default env=None only inherits a short
        allowlist that excludes OBSIDIAN_API_KEY.
        """
        mcp_cfg = config["mcp"]["obsidian"]
        return cls(
            command=mcp_cfg["command"],
            args=list(mcp_cfg["args"]),
            env=dict(os.environ),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Spawn the mcp-obsidian subprocess, initialise the ClientSession,
        and cache the tool registry. Must be called before call_tool().
        """
        params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env,
        )

        try:
            read, write = await self._stack.enter_async_context(
                stdio_client(params)
            )
            self._session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )
            await self._session.initialize()
        except Exception as exc:
            await self._stack.aclose()
            self._session = None
            raise MCPConnectionError(
                f"Failed to start mcp-obsidian subprocess: {exc}"
            ) from exc

        await self._refresh_tools()
        logger.info(
            "MCPClient ready — %d tool(s): %s",
            len(self._tools_cache),
            [t["function"]["name"] for t in self._tools_cache],
        )

    async def stop(self) -> None:
        """Shut down the session and subprocess gracefully."""
        await self._stack.aclose()
        self._session = None
        self._tools_cache = []
        logger.info("MCPClient stopped")

    # ------------------------------------------------------------------
    # Tool access
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True after start() completes and before stop() is called."""
        return self._session is not None

    @property
    def tools(self) -> list[dict[str, Any]]:
        """
        Ollama/OpenAI-compatible tool definitions ready for prompt injection.
        Returns a shallow copy so callers cannot mutate the cache.
        """
        return list(self._tools_cache)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """
        Dispatch a tool call to mcp-obsidian and return the result as a
        plain string.

        Parameters
        ----------
        name : str
            Exact tool name as listed in client.tools.
        arguments : dict
            Keyword arguments matching the tool's parameter schema.

        Returns
        -------
        str
            Concatenated text content from the tool response.

        Raises
        ------
        MCPConnectionError
            If the client has not been started or the call fails at the
            transport level.
        MCPToolError
            If mcp-obsidian returns an error result (isError=True).
        """
        if self._session is None:
            raise MCPConnectionError(
                "MCPClient is not running — call start() or use 'async with'"
            )

        logger.debug("call_tool(%s, %s)", name, arguments)

        try:
            result = await self._session.call_tool(name, arguments)
        except Exception as exc:
            raise MCPConnectionError(
                f"Transport error during tool call '{name}': {exc}"
            ) from exc

        if getattr(result, "isError", False):
            error_text = _extract_text(result.content)
            raise MCPToolError(f"Tool '{name}' returned an error: {error_text}")

        text = _extract_text(result.content)
        logger.debug("call_tool(%s) → %d chars", name, len(text))
        return text

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> MCPClient:
        await self.start()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _refresh_tools(self) -> None:
        """Query the server for its current tool list and update the cache."""
        assert self._session is not None
        result = await self._session.list_tools()
        self._tools_cache = [_to_ollama_tool(t) for t in result.tools]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_ollama_tool(tool: Any) -> dict[str, Any]:
    """
    Convert an MCP Tool object to the OpenAI / Ollama function-calling schema.

    MCP ``inputSchema`` is already a JSON Schema object, so it maps directly
    to the ``parameters`` field with only a defensive ``setdefault`` guard.
    """
    schema: dict[str, Any] = dict(tool.inputSchema)
    schema.setdefault("type", "object")
    schema.setdefault("properties", {})
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": schema,
        },
    }


def _extract_text(content: list[Any]) -> str:
    """
    Flatten an MCP content list to a single string.

    Handles TextContent (.text), ImageContent / EmbeddedResource (.data),
    and any unknown type via str() fallback.
    """
    parts: list[str] = []
    for item in content:
        if hasattr(item, "text"):
            parts.append(item.text)
        elif hasattr(item, "data"):
            parts.append(f"[binary: {len(item.data)} bytes]")
        else:
            parts.append(str(item))
    return "\n".join(parts)
