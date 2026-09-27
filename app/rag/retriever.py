"""
app/rag/retriever.py

Hybrid retriever: dense vector search (ChromaDB cosine similarity) merged
with sparse BM25 keyword search via Reciprocal Rank Fusion (RRF).

Why hybrid?
-----------
Vector search finds semantically related content even when exact keywords
are absent. BM25 finds exact keyword matches that embeddings can miss for
proper nouns, note titles, and technical terms. RRF combines both ranked
lists without requiring score normalisation.

BM25 cache
----------
The BM25 index is built lazily on the first search() call by loading all
documents from ChromaDB into memory. For a personal Obsidian vault this is
fast and keeps peak memory low. Call invalidate_cache() after any indexing
run to force a rebuild on the next query.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import chromadb
from rank_bm25 import BM25Okapi

from app.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard constant; higher values reduce rank-position sensitivity


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokenisation for BM25 — no stemming needed for personal notes."""
    return re.findall(r"\w+", text.lower())


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class Retriever:
    """
    Hybrid retriever over a ChromaDB collection.

    Parameters
    ----------
    chroma_persist_dir : str | None
        Path to the ChromaDB persistent storage directory. Ignored when
        ``chroma_client`` is supplied; required otherwise.
    ollama : OllamaClient
        Used to embed the query for vector search.
    chroma_client : chromadb.PersistentClient | None
        Pre-built persistent client. Pass this when sharing a single
        client across Indexer and Retriever to avoid opening two
        SQLite handles on the same path.
    collection_name : str
        Must match the name used by Indexer (default ``"obsidian"``).
    n_candidates : int
        Results pulled from each retrieval method before RRF merging.
        Should be comfortably larger than the largest ``top_k`` you use.
    """

    def __init__(
        self,
        chroma_persist_dir: str | None,
        ollama: OllamaClient,
        chroma_client: chromadb.PersistentClient | None = None,
        collection_name: str = "obsidian",
        n_candidates: int = 20,
    ) -> None:
        if chroma_client is None and chroma_persist_dir is None:
            raise ValueError("Retriever requires either chroma_client or chroma_persist_dir")

        self._ollama = ollama
        self._n_candidates = n_candidates

        self._chroma = chroma_client or chromadb.PersistentClient(path=chroma_persist_dir)
        self._collection = self._chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 state — populated lazily, cleared on invalidate_cache()
        self._bm25: BM25Okapi | None = None
        self._bm25_ids: list[str] = []
        self._bm25_docs: list[str] = []

    @classmethod
    def from_config(
        cls,
        config: dict,
        ollama: OllamaClient,
        chroma_client: chromadb.PersistentClient | None = None,
    ) -> Retriever:
        """
        Construct from the ``rag`` section of settings.yaml.

        Derives n_candidates automatically from top_k (4× or at least 20).
        """
        rag = config["rag"]
        top_k = int(rag.get("top_k", 5))
        return cls(
            chroma_persist_dir=rag["chroma_persist_dir"],
            ollama=ollama,
            chroma_client=chroma_client,
            n_candidates=max(top_k * 4, 20),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Return the ``top_k`` most relevant chunks for ``query``.

        Result dicts contain:
            text      — chunk content
            source    — vault-relative file path (e.g. ``"Projects/foo.md"``)
            header    — Markdown section header that anchors this chunk
            chunk_idx — position within the source file
            score     — RRF score (higher = more relevant)

        Returns an empty list if the collection has not been indexed yet.
        Falls back to vector-only ranking if the collection is empty after
        a BM25 build attempt (should not happen in normal use).
        """
        total = self._collection.count()
        if total == 0:
            logger.warning("Collection is empty — run `python main.py --index` first")
            return []

        n = min(self._n_candidates, total)

        vec_ranked = self._vector_search(query, n)
        bm25_ranked = self._bm25_search(query, n)

        rrf_scores = _rrf(vec_ranked, bm25_ranked)
        top_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]

        if not top_ids:
            return []

        # Fetch text + metadata only for the final top_k set
        fetched = self._collection.get(
            ids=top_ids,
            include=["documents", "metadatas"],
        )

        id_to_result: dict[str, dict[str, Any]] = {
            id_: {
                "text": doc,
                "source": meta.get("source", ""),
                "header": meta.get("header", ""),
                "chunk_idx": int(meta.get("chunk_idx", 0)),
                "score": rrf_scores[id_],
            }
            for id_, doc, meta in zip(
                fetched["ids"], fetched["documents"], fetched["metadatas"]
            )
        }

        return [id_to_result[id_] for id_ in top_ids if id_ in id_to_result]

    def invalidate_cache(self) -> None:
        """
        Clear the in-memory BM25 index so it is rebuilt on the next search().
        Call this after any Indexer run that changes the collection.
        """
        self._bm25 = None
        self._bm25_ids = []
        self._bm25_docs = []
        logger.debug("BM25 cache invalidated")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _vector_search(self, query: str, n: int) -> list[tuple[str, int]]:
        """
        Query ChromaDB by cosine similarity.
        Returns [(doc_id, rank), ...], rank is 0-based best-first.
        IDs only — full text is fetched later for the final top_k only.
        """
        query_vec = self._ollama.embed(query)
        results = self._collection.query(
            query_embeddings=[query_vec],
            n_results=n,
            include=[],
        )
        return [(id_, rank) for rank, id_ in enumerate(results["ids"][0])]

    def _bm25_search(self, query: str, n: int) -> list[tuple[str, int]]:
        """
        Search the in-memory BM25 index.
        Returns [(doc_id, rank), ...], rank is 0-based best-first.
        """
        if self._bm25 is None:
            self._build_bm25()

        if not self._bm25_ids:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        top_indices = sorted(
            range(len(scores)), key=scores.__getitem__, reverse=True
        )[:n]
        return [(self._bm25_ids[i], rank) for rank, i in enumerate(top_indices)]

    def _build_bm25(self) -> None:
        """Load every document from ChromaDB and construct a BM25Okapi index."""
        logger.debug("Building BM25 index from ChromaDB collection…")
        data = self._collection.get(include=["documents"])
        self._bm25_ids = data["ids"]
        self._bm25_docs = data["documents"]

        if not self._bm25_ids:
            logger.warning("Collection is empty; BM25 index has no documents")
            return

        self._bm25 = BM25Okapi([_tokenize(doc) for doc in self._bm25_docs])
        logger.debug("BM25 index built — %d documents", len(self._bm25_ids))


# ---------------------------------------------------------------------------
# RRF helper
# ---------------------------------------------------------------------------


def _rrf(
    *ranked_lists: list[tuple[str, int]],
    k: int = _RRF_K,
) -> dict[str, float]:
    """
    Reciprocal Rank Fusion over any number of ranked lists.

    Each list is ``[(doc_id, rank), ...]`` with rank 0-based.
    Returns ``{doc_id: combined_score}`` for every unique ID.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for doc_id, rank in ranked:
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores
