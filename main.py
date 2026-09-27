"""
obsidian-rag — entry point

Usage
-----
    python main.py                   # launch the PyQt6 chat UI
    python main.py --index           # full vault index pass, then exit
    python main.py --index --watch   # full index pass, then watch for changes
    python main.py --check           # verify Ollama connection and model availability

Component graph (--ui and --index paths)
-----------------------------------------
    OllamaClient          ← settings.yaml [ollama]
        ├─ Indexer        ← settings.yaml [rag]   (--index only)
        └─ Retriever      ← settings.yaml [rag]   (--ui only)
    MCPClient             ← settings.yaml [mcp]   (--ui only)
                                                   env vars from .env
    Orchestrator          ← ollama + retriever + mcp
    ChatWindow            ← orchestrator + qasync event loop

Environment variables
---------------------
OBSIDIAN_API_KEY, OBSIDIAN_HOST, and OBSIDIAN_PORT must be present in the
process environment when the UI starts.  They are read by MCPClient which
passes env=None so the subprocess inherits the parent environment.
The project .env file is the intended source; it is human-managed and is
NOT loaded here — set it in your shell or via direnv before running.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env before anything else so OBSIDIAN_* vars are in os.environ when
# MCPClient (and any other component that reads the environment) is constructed.
# override=False means a variable already exported in the shell takes precedence.
load_dotenv(Path(__file__).parent / ".env", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent / "config" / "settings.yaml"


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


# Maps environment variable names to their dotted path in the config dict.
# Values present in .env (or the shell) override whatever is in settings.yaml.
_ENV_OVERRIDES = [
    ("OLLAMA_HOST",        ["ollama", "host"]),
    ("OLLAMA_MODEL",       ["ollama", "model"]),
    ("OLLAMA_EMBED_MODEL", ["ollama", "embed_model"]),
    ("VAULT_PATH",         ["rag", "vault_path"]),
]


def load_config(path: Path) -> dict:
    if not path.exists():
        logger.error(
            "settings.yaml not found at %s — "
            "copy config/settings.yaml.example to config/settings.yaml "
            "and fill in your values.",
            path,
        )
        sys.exit(1)
    with path.open() as f:
        config = yaml.safe_load(f)
    _apply_env_overrides(config)
    return config


def _apply_env_overrides(config: dict) -> None:
    """Overwrite config values with matching environment variables when set."""
    for var, path in _ENV_OVERRIDES:
        value = os.environ.get(var)
        if not value:
            continue
        node = config
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_check(config: dict) -> None:
    """Verify that Ollama is reachable and both required models are present."""
    from app.ollama_client import OllamaClient, OllamaConnectionError

    client = OllamaClient.from_config(config["ollama"])
    try:
        models = client.health_check()
        logger.info("Ollama reachable. Available models: %s", models)
    except OllamaConnectionError as exc:
        logger.error("Ollama connection failed: %s", exc)
        sys.exit(1)


def cmd_index(config: dict, watch: bool = False) -> None:
    """
    Crawl the vault, embed every .md file, and persist chunks to ChromaDB.

    Skips files whose SHA-256 hash has not changed since the last run.
    With watch=True (--watch flag) a filesystem observer is started after
    the initial pass so changes are picked up automatically.
    """
    from app.ollama_client import OllamaClient
    from app.rag.indexer import Indexer

    ollama = OllamaClient.from_config(config["ollama"])
    indexer = Indexer.from_config(config, ollama)

    vault = config["rag"]["vault_path"]
    logger.info("Starting vault index — vault: %s", vault)

    stats = indexer.index_vault()
    logger.info(
        "Index pass complete — indexed=%d  skipped=%d  removed=%d  errors=%d",
        stats.indexed,
        stats.skipped,
        stats.removed,
        stats.errors,
    )

    if watch:
        logger.info("Watch mode active — press Ctrl+C to stop")
        indexer.start_watching()
        try:
            # Block the main thread; watchdog runs on its own thread
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping watcher…")
        finally:
            indexer.stop_watching()


def cmd_ui(config: dict) -> None:
    """
    Launch the PyQt6 chat window with the full component graph.

    The qasync event loop merges asyncio with the Qt event loop so the
    Orchestrator's async handle_query() and the MCPClient's async
    lifecycle run without a separate thread.
    """
    try:
        import qasync
        from PyQt6.QtWidgets import QApplication
    except ImportError as exc:
        logger.error(
            "UI dependencies missing: %s — run: pip install PyQt6 qasync", exc
        )
        sys.exit(1)

    import chromadb

    from app.mcp_client import MCPClient
    from app.ollama_client import OllamaClient
    from app.orchestrator import Orchestrator
    from app.rag.indexer import Indexer
    from app.rag.retriever import Retriever
    from app.ui.chat_window import ChatWindow

    # ------------------------------------------------------------------
    # Build component graph
    # ------------------------------------------------------------------
    # One PersistentClient is shared between Indexer and Retriever so we
    # never open two SQLite handles on the same chroma_persist_dir — the
    # toolbar re-index button has both components active at once.
    ollama = OllamaClient.from_config(config["ollama"])
    chroma_client = chromadb.PersistentClient(path=config["rag"]["chroma_persist_dir"])
    retriever = Retriever.from_config(config, ollama, chroma_client=chroma_client)
    indexer = Indexer.from_config(config, ollama, chroma_client=chroma_client)
    mcp = MCPClient.from_config(config)
    orchestrator = Orchestrator.from_config(
        config, ollama=ollama, retriever=retriever, mcp=mcp, indexer=indexer
    )

    # ------------------------------------------------------------------
    # Qt + qasync setup
    # ------------------------------------------------------------------
    app = QApplication(sys.argv)
    app.setApplicationName("Obsidian RAG")

    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    ui_cfg = config.get("ui", {})
    title = ui_cfg.get("window_title", "Obsidian RAG")
    theme = ui_cfg.get("theme", "light")
    window = ChatWindow(orchestrator, title=title, theme=theme)
    window.show()

    with loop:
        loop.run_forever()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Obsidian RAG — local vault assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                 # open chat UI\n"
            "  python main.py --check         # verify Ollama connection\n"
            "  python main.py --index         # index vault once\n"
            "  python main.py --index --watch # index then watch for changes\n"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify Ollama connection")
    mode.add_argument("--index", action="store_true", help="Index vault into ChromaDB")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="After indexing, watch the vault for live re-indexing (requires --index)",
    )
    args = parser.parse_args()

    if args.watch and not args.index:
        parser.error("--watch requires --index")

    config = load_config(CONFIG_PATH)

    if args.check:
        cmd_check(config)
    elif args.index:
        cmd_index(config, watch=args.watch)
    else:
        cmd_ui(config)


if __name__ == "__main__":
    main()
