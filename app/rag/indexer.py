"""
app/rag/indexer.py

Vault crawler: walks the Obsidian vault, reads Markdown files, splits them
into header-aware chunks, embeds each chunk, and persists to ChromaDB.

Chunking strategy
-----------------
Text is first split on Markdown headers (levels 1-6). Each section carries
its header as a context prefix so retrieval results are self-contained.
Sections longer than chunk_size are further split on paragraph boundaries
(double newlines), with character-level overlap between adjacent chunks.

Incremental indexing
--------------------
Each file is SHA-256 hashed. A file whose hash matches the value stored in
its ChromaDB chunk metadata is skipped. Files removed from the vault have
their chunks pruned on the next full index run.

Watchdog
--------
Call start_watching() after index_vault() to keep the index live.
The observer re-indexes modified/created notes and removes deleted ones.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import chromadb
import frontmatter

from app.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_MIN_CHUNK_CHARS = 50


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class Chunk:
    text: str
    header: str    # nearest section header line, empty string if none
    chunk_idx: int  # position within the source file (0-based)


@dataclass
class IndexStats:
    indexed: int = 0
    skipped: int = 0
    removed: int = 0
    errors: int = 0


# ---------------------------------------------------------------------------
# Markdown splitter
# ---------------------------------------------------------------------------


class MarkdownSplitter:
    """Split a Markdown document body into header-anchored chunks."""

    def split(self, text: str, chunk_size: int = 512, overlap: int = 64) -> list[Chunk]:
        chunks: list[Chunk] = []
        idx = 0
        for header, body in self._split_sections(text):
            for chunk_text in self._chunk_section(header, body, chunk_size, overlap):
                if len(chunk_text) >= _MIN_CHUNK_CHARS:
                    chunks.append(Chunk(text=chunk_text, header=header, chunk_idx=idx))
                    idx += 1
        return chunks

    def _split_sections(self, text: str) -> list[tuple[str, str]]:
        """Return [(header_line, body), ...] in document order."""
        matches = list(_HEADER_RE.finditer(text))
        if not matches:
            return [("", text.strip())]

        sections: list[tuple[str, str]] = []

        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

        for i, m in enumerate(matches):
            header = m.group(0)
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            sections.append((header, body))

        return sections

    def _chunk_section(
        self, header: str, body: str, chunk_size: int, overlap: int
    ) -> Iterator[str]:
        prefix = f"{header}\n\n" if header else ""
        full = (prefix + body).strip()

        if len(full) <= chunk_size:
            yield full
            return

        # Flatten into units: paragraphs, further split if a single paragraph
        # exceeds chunk_size so no unit is ever larger than the target.
        raw_paras = [p.strip() for p in re.split(r"\n{2,}", body) if p.strip()]
        units: list[str] = []
        for para in raw_paras:
            if len(para) <= chunk_size:
                units.append(para)
            else:
                units.extend(_split_on_words(para, chunk_size))

        # Greedily pack units into chunks; carry overlap from body text only
        # (not from the header prefix, which is always re-prepended).
        body_so_far: list[str] = []
        body_len = 0

        for unit in units:
            added = len(unit) + (2 if body_so_far else 0)  # +2 for "\n\n"
            if body_len + added > chunk_size and body_so_far:
                yield (prefix + "\n\n".join(body_so_far)).strip()
                # Seed next chunk with tail of previous body for overlap
                tail = "\n\n".join(body_so_far)[-overlap:] if overlap else ""
                body_so_far = [tail] if tail else []
                body_len = len(tail)

            body_so_far.append(unit)
            body_len += added

        if body_so_far:
            yield (prefix + "\n\n".join(body_so_far)).strip()


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------


class Indexer:
    """
    Crawls an Obsidian vault, chunks .md files, and persists embeddings to
    ChromaDB. Incremental: unchanged files (same SHA-256) are skipped.

    Parameters
    ----------
    vault_path : str | Path
    ollama : OllamaClient
        Source of embeddings via ``ollama.embed()``.
    chroma_persist_dir : str | None
        Directory where ChromaDB stores its data. Ignored when
        ``chroma_client`` is supplied; required otherwise.
    chroma_client : chromadb.PersistentClient | None
        Pre-built persistent client. Pass this when sharing a single
        client across Indexer and Retriever to avoid opening two
        SQLite handles on the same path.
    collection_name : str
        ChromaDB collection name. Defaults to ``"obsidian"``.
    chunk_size : int
        Target character count per chunk.
    chunk_overlap : int
        Characters of overlap between adjacent chunks in long sections.
    """

    def __init__(
        self,
        vault_path: str | Path,
        ollama: OllamaClient,
        chroma_persist_dir: str | None = None,
        chroma_client: chromadb.PersistentClient | None = None,
        collection_name: str = "obsidian",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
    ) -> None:
        if chroma_client is None and chroma_persist_dir is None:
            raise ValueError("Indexer requires either chroma_client or chroma_persist_dir")

        self.vault_path = Path(vault_path).expanduser().resolve()
        self._ollama = ollama
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._splitter = MarkdownSplitter()
        self._watcher = None  # watchdog Observer, set by start_watching()

        self._chroma = chroma_client or chromadb.PersistentClient(path=chroma_persist_dir)
        self._collection = self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def from_config(
        cls,
        config: dict,
        ollama: OllamaClient,
        chroma_client: chromadb.PersistentClient | None = None,
    ) -> Indexer:
        """Construct from the ``rag`` section of settings.yaml."""
        rag = config["rag"]
        return cls(
            vault_path=rag["vault_path"],
            ollama=ollama,
            chroma_persist_dir=rag["chroma_persist_dir"],
            chroma_client=chroma_client,
            chunk_size=int(rag.get("chunk_size", 512)),
            chunk_overlap=int(rag.get("chunk_overlap", 64)),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_vault(self) -> IndexStats:
        """
        Full index pass over the vault. Skips unchanged files, removes
        chunks for deleted files. Safe to call repeatedly.
        """
        stats = IndexStats()
        md_files = list(self.vault_path.rglob("*.md"))
        current_sources = {self._rel(p) for p in md_files}

        stats.removed = self._prune_deleted(current_sources)

        for path in md_files:
            try:
                if self._index_file_if_changed(path):
                    stats.indexed += 1
                else:
                    stats.skipped += 1
            except Exception:
                logger.exception("Failed to index %s", path)
                stats.errors += 1

        logger.info(
            "Index complete — indexed=%d  skipped=%d  removed=%d  errors=%d",
            stats.indexed,
            stats.skipped,
            stats.removed,
            stats.errors,
        )
        return stats

    def index_file(self, path: str | Path) -> None:
        """Force re-index a single file regardless of its stored hash."""
        self._do_index_file(Path(path).resolve())

    def remove_file(self, path: str | Path) -> None:
        """Delete all ChromaDB chunks that belong to a given vault file."""
        rel = self._rel(Path(path).resolve())
        self._delete_chunks_for(rel)
        logger.debug("Removed chunks for %s", rel)

    # ------------------------------------------------------------------
    # Watchdog
    # ------------------------------------------------------------------

    def start_watching(self) -> None:
        """Mount a watchdog observer for live incremental re-indexing."""
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        indexer = self

        class _Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith(".md"):
                    logger.info("Vault change: %s", event.src_path)
                    try:
                        indexer.index_file(event.src_path)
                    except Exception:
                        logger.exception("Re-index failed: %s", event.src_path)

            def on_created(self, event):
                self.on_modified(event)

            def on_deleted(self, event):
                if not event.is_directory and event.src_path.endswith(".md"):
                    logger.info("Vault deletion: %s", event.src_path)
                    indexer.remove_file(event.src_path)

        self._watcher = Observer()
        self._watcher.schedule(_Handler(), str(self.vault_path), recursive=True)
        self._watcher.start()
        logger.info("Watching vault at %s", self.vault_path)

    def stop_watching(self) -> None:
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher.join()
            self._watcher = None
            logger.info("Stopped vault watcher")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _index_file_if_changed(self, path: Path) -> bool:
        """Return True if the file was (re)indexed, False if skipped."""
        rel = self._rel(path)
        new_hash = _file_hash(path)

        existing = self._collection.get(
            where={"source": rel},
            limit=1,
            include=["metadatas"],
        )
        if existing["ids"] and existing["metadatas"][0].get("file_hash") == new_hash:
            return False

        return self._do_index_file(path, rel=rel, file_hash=new_hash)

    def _do_index_file(
        self,
        path: Path,
        rel: str | None = None,
        file_hash: str | None = None,
    ) -> bool:
        """Return True if chunks were written, False if the file was empty/unparseable."""
        rel = rel or self._rel(path)
        file_hash = file_hash or _file_hash(path)

        try:
            post = frontmatter.load(str(path))
        except Exception:
            logger.warning("Frontmatter parse error in %s — skipping", path)
            return False

        body = post.content.strip()
        if not body:
            logger.debug("Empty body in %s — skipping", rel)
            return False

        chunks = self._splitter.split(body, self._chunk_size, self._chunk_overlap)
        if not chunks:
            logger.debug("No chunks from %s — skipping", rel)
            return False

        self._delete_chunks_for(rel)

        ids = [f"{rel}::chunk_{c.chunk_idx}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "source": rel,
                "header": c.header,
                "chunk_idx": c.chunk_idx,
                "file_hash": file_hash,
                "mtime": path.stat().st_mtime,
            }
            for c in chunks
        ]
        embeddings = self._ollama.embed(documents)

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.debug("Indexed %s → %d chunk(s)", rel, len(chunks))
        return True

    def _delete_chunks_for(self, rel: str) -> None:
        existing = self._collection.get(where={"source": rel}, include=[])
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])

    def _prune_deleted(self, current_sources: set[str]) -> int:
        """Remove chunks whose source file no longer exists in the vault."""
        all_data = self._collection.get(include=["metadatas"])
        dead_ids = [
            id_
            for id_, meta in zip(all_data["ids"], all_data["metadatas"])
            if meta.get("source") not in current_sources
        ]
        if dead_ids:
            self._collection.delete(ids=dead_ids)
        return len(dead_ids)

    def _rel(self, path: Path) -> str:
        """POSIX-style path relative to the vault root."""
        return path.relative_to(self.vault_path).as_posix()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_hash(path: Path) -> str:
    """SHA-256 hex digest of file contents, read in 64 KB blocks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _split_on_words(text: str, max_len: int) -> list[str]:
    """
    Split a string that exceeds max_len into word-boundary-aligned pieces.
    Used as a fallback when a single paragraph is larger than chunk_size.
    """
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_len
        if end < len(text):
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        pieces.append(text[start:end].strip())
        start = end
    return [p for p in pieces if p]
