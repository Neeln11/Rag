"""
retriever.py
============
Utilities to reload the persisted retrieval system after ingestion.

Separation of concerns: `ingest.py` writes to disk; `retriever.py` reads from
disk.  This lets query-time code start instantly without re-ingesting.

Usage
-----
    from retriever import load_retriever

    retriever = load_retriever()
    results = retriever.invoke("How does BM25 scoring work?")
    for doc in results:
        print(doc.metadata["source"], "—", doc.page_content[:120])
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Union

from langchain_classic.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings

# All constants come from the single shared config — never from ingest.py
# directly, so this module has no import-time dependency on the ingestion code.
from config import (
    BM25_PICKLE_PATH,
    BM25_WEIGHT,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHROMA_WEIGHT,
    EMBEDDING_MODEL_NAME,
    RETRIEVER_K,
)

logger = logging.getLogger(__name__)


def load_retriever(
    chroma_dir: str = CHROMA_PERSIST_DIR,
    bm25_path: str = BM25_PICKLE_PATH,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    k: int = RETRIEVER_K,
    chroma_weight: float = CHROMA_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
) -> EnsembleRetriever:
    """
    Reload the persisted retrieval system from disk.

    Constructs an EnsembleRetriever (ChromaDB + BM25) without re-embedding
    any documents.  ChromaDB is opened in read mode against the persisted
    collection; BM25 is deserialized from the pickled index.

    Args:
        chroma_dir:      Path to the ChromaDB persistence directory.
        bm25_path:       Path to the pickled BM25Retriever.
        embedding_model: HuggingFace model name for query embedding.
        k:               Number of results each sub-retriever returns.
        chroma_weight:   RRF weight for the ChromaDB retriever (0–1).
        bm25_weight:     RRF weight for the BM25 retriever (0–1).

    Returns:
        A ready-to-use EnsembleRetriever.

    Raises:
        FileNotFoundError: If chroma_dir or bm25_path don't exist on disk.
    """
    chroma_path = Path(chroma_dir)
    bm25_file = Path(bm25_path)

    if not chroma_path.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found: {chroma_path.resolve()}\n"
            "Run `python ingest.py --docs ./docs` first."
        )
    if not bm25_file.exists():
        raise FileNotFoundError(
            f"BM25 pickle not found: {bm25_file.resolve()}\n"
            "Run `python ingest.py --docs ./docs` first."
        )

    # ── Embedding model ────────────────────────────────────────────────────
    # Required at query time so ChromaDB can embed the query string.
    # Model weights are cached after the first ingest run.
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── ChromaDB — load existing persisted collection ──────────────────────
    # We open the collection in read mode (no from_documents, no reset).
    # This guarantees we always search the same data that was written by ingest.
    vector_store = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(chroma_path),
    )

    # ── Debug log: confirm collection name and doc count ───────────────────
    try:
        doc_count = vector_store._collection.count()
    except Exception:
        doc_count = -1
    logger.info(
        "Retriever: searching collection=%r  persist_dir=%s  total_docs=%d",
        CHROMA_COLLECTION_NAME, chroma_path.resolve(), doc_count,
    )
    print(
        f"[retriever] Searching collection={CHROMA_COLLECTION_NAME!r} | "
        f"persist_dir={chroma_path.resolve()} | total_docs={doc_count}"
    )

    chroma_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    # ── BM25 — load from pickle ────────────────────────────────────────────
    with open(bm25_file, "rb") as f:
        bm25_retriever: BM25Retriever = pickle.load(f)

    bm25_retriever.k = k

    # ── EnsembleRetriever (RRF fusion) ─────────────────────────────────────
    ensemble = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[chroma_weight, bm25_weight],
    )

    logger.info(
        "Retriever ready — ChromaDB@%.1f + BM25@%.1f | k=%d",
        chroma_weight, bm25_weight, k,
    )
    return ensemble


# ---------------------------------------------------------------------------
# Quick sanity-check demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    retriever = load_retriever()

    demo_queries = [
        "How does RecursiveCharacterTextSplitter work?",
        "What is BM25 scoring formula?",
        "BAAI bge embeddings cosine similarity",
    ]

    for query in demo_queries:
        print(f"\n{'─'*60}")
        print(f"Query: {query!r}")
        results = retriever.invoke(query)
        for i, doc in enumerate(results, 1):
            meta = doc.metadata
            preview = doc.page_content.replace("\n", " ")[:100]
            print(
                f"  [{i}] {meta.get('source', '?')} "
                f"(chunk {meta.get('chunk_index', '?')}/{meta.get('total_chunks', '?')}) "
                f"— {preview}..."
            )
