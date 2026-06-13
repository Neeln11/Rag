"""
ingest.py
=========
Production-grade document ingestion pipeline for RAG (Retrieval-Augmented Generation).

Design Goals
------------
1. **No proprietary API keys** — uses HuggingFace sentence-transformers locally.
2. **Hybrid retrieval** — combines dense vector search (ChromaDB) with sparse keyword
   search (BM25) for best-of-both-worlds retrieval quality.
3. **Rich metadata** — every chunk carries provenance info (source, position, type,
   timestamp) so downstream queries can filter by document or date.
4. **Persistence** — ChromaDB is written to disk; BM25 index is pickled so nothing
   needs to be rebuilt on the next run if the corpus hasn't changed.
5. **CLI + library** — works as `python ingest.py --docs ./docs` or as an importable
   function `ingest_documents(docs_path)` for programmatic use.

Architecture Overview
---------------------
                ┌──────────────┐
                │  /docs files │
                └──────┬───────┘
                       │ 1. Load (TextLoader / UnstructuredMarkdownLoader)
                       ▼
                ┌──────────────┐
                │  Raw Documents│
                └──────┬───────┘
                       │ 2. Split (RecursiveCharacterTextSplitter)
                       ▼
                ┌──────────────┐
                │    Chunks    │  ← enriched with metadata
                └──────┬───────┘
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
    3a. Embed + Store     3b. Build BM25
    (ChromaDB, HNSW)      (rank_bm25, pickle)
              │                 │
              └────────┬────────┘
                       │ 4. EnsembleRetriever (RRF fusion)
                       ▼
                ┌──────────────┐
                │HybridRetriever│
                └──────────────┘
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Force UTF-8 stdout on Windows
# ---------------------------------------------------------------------------
# Windows consoles default to cp1252 which cannot encode Unicode characters
# like checkmarks (\u2713).  reconfigure(encoding="utf-8") switches the stdout
# stream to UTF-8 without affecting other I/O, making all print() calls safe.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# LangChain document loaders
# ---------------------------------------------------------------------------
# TextLoader handles plain .txt files.
# UnstructuredMarkdownLoader preserves Markdown structure better than
# plain TextLoader for .md files — it understands headers, lists, and code
# blocks, which helps the downstream splitter make cleaner cuts.
from langchain_community.document_loaders import (
    BSHTMLLoader,          # HTML → requires beautifulsoup4
    CSVLoader,             # CSV  → no extra deps
    Docx2txtLoader,        # DOCX → requires docx2txt
    PyPDFLoader,           # PDF  → requires pypdf
    TextLoader,            # TXT + MD
)

# ---------------------------------------------------------------------------
# Text splitter
# ---------------------------------------------------------------------------
# RecursiveCharacterTextSplitter tries separators in order, guaranteeing that
# splits happen at the most semantically meaningful boundaries first (headers
# > paragraphs > lines > words).  This is crucial for technical Markdown where
# sections are self-contained units of knowledge.
#
# NOTE: In LangChain >= 0.2, text splitters moved to the dedicated package
# `langchain_text_splitters`.  The old `langchain.text_splitter` path was
# removed in LangChain 1.x, so we import from the correct location here.
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Embeddings — HuggingFace (local, no API key)
# ---------------------------------------------------------------------------
# We use the newer `langchain_huggingface` package (not the deprecated
# `langchain_community.embeddings.HuggingFaceEmbeddings`) which properly
# supports the latest sentence-transformers API.
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# Vector store — ChromaDB
# ---------------------------------------------------------------------------
# ChromaDB runs fully in-process with SQLite + HNSW on disk.  No separate
# server, no Docker, no API key.  `Chroma` from langchain_chroma is the
# official LangChain integration maintained by the ChromaDB team.
from langchain_chroma import Chroma

# ---------------------------------------------------------------------------
# Sparse retriever — BM25
# ---------------------------------------------------------------------------
# BM25Retriever wraps the `rank_bm25` library.  It provides exact keyword
# matching, which dense embeddings can miss for rare technical terms, model
# names, and version strings not well-represented in training data.
from langchain_community.retrievers import BM25Retriever

# ---------------------------------------------------------------------------
# Hybrid retriever — EnsembleRetriever
# ---------------------------------------------------------------------------
# EnsembleRetriever uses Reciprocal Rank Fusion (RRF) to merge ranked lists
# from multiple retrievers.  This consistently outperforms either retriever
# alone on information-retrieval benchmarks.
#
# NOTE: In LangChain 1.x, EnsembleRetriever was moved to the `langchain_classic`
# package (the successor to `langchain.retrievers` for non-LLM retrieval utilities).
from langchain_classic.retrievers import EnsembleRetriever

# ---------------------------------------------------------------------------
# Constants — imported from shared config so every module agrees on the same
# paths/names.  Do NOT hard-code these values here.
# ---------------------------------------------------------------------------
from config import (
    BM25_PICKLE_PATH,
    BM25_WEIGHT,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHROMA_WEIGHT,
    EMBEDDING_MODEL_NAME,
    RETRIEVER_K,
    SEPARATORS,
)


# ---------------------------------------------------------------------------
# Helper: Load a single document
# ---------------------------------------------------------------------------

def _load_file(file_path: Path) -> list:
    """
    Load a single file into LangChain Document objects.

    Supported formats
    -----------------
    .md   → TextLoader  (raw Markdown; splitter handles headers)
    .txt  → TextLoader
    .pdf  → PyPDFLoader  (one Document per page; requires pypdf)
    .docx → Docx2txtLoader (strips formatting; requires docx2txt)
    .html → BSHTMLLoader  (extracts visible text; requires beautifulsoup4)
    .csv  → CSVLoader     (one Document per row)
    .json → loaded via stdlib json → converted to text Document

    Args:
        file_path: Absolute or relative path to the file.

    Returns:
        List of LangChain Document objects with populated page_content.
    """
    import json as _json
    from langchain_core.documents import Document as _Document

    ext = file_path.suffix.lower()

    try:
        if ext in (".md", ".txt"):
            loader = TextLoader(str(file_path), encoding="utf-8")
            return loader.load()

        elif ext == ".pdf":
            # PyPDFLoader returns one Document per page.
            # Clean up the garbled whitespace that PDF text extraction produces
            # (e.g. "Express  Analytics" with double-spaces).
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            import re as _re
            for doc in docs:
                # Collapse multiple spaces/newlines into single space
                doc.page_content = _re.sub(r'[ \t]+', ' ', doc.page_content)
                doc.page_content = _re.sub(r'\n{3,}', '\n\n', doc.page_content)
                doc.page_content = doc.page_content.strip()
            return docs

        elif ext == ".docx":
            # Docx2txtLoader strips all formatting and returns a single Document
            loader = Docx2txtLoader(str(file_path))
            return loader.load()

        elif ext == ".html":
            # BSHTMLLoader extracts visible text, strips tags
            loader = BSHTMLLoader(str(file_path), open_encoding="utf-8")
            return loader.load()

        elif ext == ".csv":
            # CSVLoader returns one Document per row with column:value formatting
            loader = CSVLoader(str(file_path), encoding="utf-8")
            return loader.load()

        elif ext == ".json":
            # stdlib json → pretty-printed text Document (no jq dependency)
            with open(file_path, encoding="utf-8") as f:
                data = _json.load(f)
            text = _json.dumps(data, indent=2, ensure_ascii=False)
            return [_Document(page_content=text, metadata={})]

        else:
            print(f"  [SKIP] Unsupported file type: {file_path.name}")
            return []

    except Exception as exc:
        print(f"  [WARN] Failed to load {file_path.name}: {exc}")
        return []


# ---------------------------------------------------------------------------
# Helper: Detect document type from file extension
# ---------------------------------------------------------------------------

def _doc_type(file_path: Path) -> str:
    """Return a human-readable document type label for metadata."""
    mapping = {
        ".md":   "markdown",
        ".txt":  "plaintext",
        ".pdf":  "pdf",
        ".docx": "word",
        ".html": "html",
        ".csv":  "csv",
        ".json": "json",
    }
    return mapping.get(file_path.suffix.lower(), "unknown")


# ---------------------------------------------------------------------------
# Core function: ingest_documents
# ---------------------------------------------------------------------------

def ingest_documents(
    docs_path: str,
    new_files_only: list | None = None,  # kept for API compat, ignored — always full rebuild
) -> dict[str, Any]:
    """
    Full ingestion pipeline: load → split → embed → store → index.

    Args:
        docs_path: Path to the folder containing ALL documents (used for BM25
                   full rebuild and for discovering the complete corpus).
        new_files_only: Optional list of Path objects for files that were just
                        uploaded.  When provided, only these files are embedded
                        and upserted into ChromaDB (fast incremental mode).
                        BM25 is always rebuilt from the full corpus so keyword
                        search stays consistent across all documents.
                        When None, ALL files in docs_path are (re-)embedded
                        (full rebuild mode, used by the CLI).

    Returns:
        Stats dict with keys:
            total_docs, total_chunks  — full corpus counts
            new_docs, new_chunks      — counts for newly ingested files
            time_taken_s, docs_path, chroma_dir, bm25_path,
            chunk_size, chunk_overlap, embedding_model
    """
    pipeline_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 0. Validate input
    # ------------------------------------------------------------------
    docs_dir = Path(docs_path).resolve()
    if not docs_dir.exists():
        raise FileNotFoundError(f"docs_path does not exist: {docs_dir}")
    if not docs_dir.is_dir():
        raise ValueError(f"docs_path must be a directory, got: {docs_dir}")

    incremental = new_files_only is not None and len(new_files_only) > 0

    print(f"\n{'='*60}")
    print(f"  Document Ingestion Pipeline  ({'incremental' if incremental else 'full rebuild'})")
    print(f"  Docs path : {docs_dir}")
    print(f"  Timestamp : {datetime.now(timezone.utc).isoformat()}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Discover full corpus (needed for BM25 rebuild)
    # ------------------------------------------------------------------
    print("[1/5] Discovering documents...")

    supported_exts = {".md", ".txt", ".pdf", ".docx", ".html", ".csv", ".json"}
    all_file_paths = sorted(
        p for p in docs_dir.iterdir()
        if p.is_file() and p.suffix.lower() in supported_exts
    )

    if not all_file_paths:
        raise ValueError(f"No supported documents found in {docs_dir}")

    # ------------------------------------------------------------------
    # 2. Load all files and split into chunks
    # ------------------------------------------------------------------
    print("[2/5] Loading and splitting all documents...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        keep_separator=True,
        separators=SEPARATORS,
    )

    ingest_ts = datetime.now(timezone.utc).isoformat()
    all_chunks: list = []

    for fp in all_file_paths:
        fp = Path(fp)
        print(f"  Loading: {fp.name}")
        loaded = _load_file(fp)
        for doc in loaded:
            doc.metadata["source"] = fp.name
            doc.metadata["doc_type"] = _doc_type(fp)
        doc_chunks = splitter.split_documents(loaded)
        total_in_doc = len(doc_chunks)
        for idx, chunk in enumerate(doc_chunks):
            chunk.metadata.update({
                "source":           fp.name,       # ensure source survives splitting
                "doc_type":         _doc_type(fp),
                "chunk_index":      idx,
                "total_chunks":     total_in_doc,
                "ingest_timestamp": ingest_ts,
            })
        all_chunks.extend(doc_chunks)
        print(f"    -> {total_in_doc} chunk(s)")

    total_chunks = len(all_chunks)
    print(f"  [OK] {total_chunks} total chunk(s) ready\n")

    # ------------------------------------------------------------------
    # 3. Embed model
    # ------------------------------------------------------------------
    print("[3/5] Loading embedding model...")
    print(f"  Model: {EMBEDDING_MODEL_NAME}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ------------------------------------------------------------------
    # 3a. ChromaDB — ALWAYS full rebuild (delete + recreate)
    # This is the only safe approach: it guarantees ChromaDB and BM25
    # are always in sync and there are never duplicate chunks.
    # ------------------------------------------------------------------
    print(f"  Persist dir: {CHROMA_PERSIST_DIR}")

    try:
        _existing = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        _existing.delete_collection()
        print(f"  [OK] Cleared previous ChromaDB collection '{CHROMA_COLLECTION_NAME}'")
    except Exception:
        pass  # Collection didn't exist yet — fine

    vector_store = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"  [OK] Stored {total_chunks} chunks in ChromaDB\n")

    # ------------------------------------------------------------------
    # 3b. BM25 — rebuild over same all_chunks (always in sync with Chroma)
    # ------------------------------------------------------------------
    print("[4/5] Building BM25 index...")

    bm25_retriever = BM25Retriever.from_documents(all_chunks)
    bm25_retriever.k = RETRIEVER_K

    bm25_path = Path(BM25_PICKLE_PATH).resolve()
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)

    print(f"  [OK] BM25 saved — {total_chunks} chunks across {len(all_file_paths)} file(s)\n")

    # ------------------------------------------------------------------
    # 4. Assemble EnsembleRetriever
    # ------------------------------------------------------------------
    print("[5/5] Assembling EnsembleRetriever (hybrid search)...")

    chroma_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVER_K},
    )
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, bm25_retriever],
        weights=[CHROMA_WEIGHT, BM25_WEIGHT],
    )
    print(f"  [OK] EnsembleRetriever ready (ChromaDB@{CHROMA_WEIGHT} + BM25@{BM25_WEIGHT})\n")

    # ------------------------------------------------------------------
    # 5. Stats
    # ------------------------------------------------------------------
    elapsed = time.perf_counter() - pipeline_start

    stats = {
        "total_docs":      len(all_file_paths),
        "total_chunks":    total_chunks,
        "new_docs":        len(all_file_paths),
        "new_chunks":      total_chunks,
        "time_taken_s":    round(elapsed, 3),
        "docs_path":       str(docs_dir),
        "chroma_dir":      str(Path(CHROMA_PERSIST_DIR).resolve()),
        "bm25_path":       str(bm25_path),
        "chunk_size":      CHUNK_SIZE,
        "chunk_overlap":   CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL_NAME,
    }

    _print_stats(stats)

    # Return the ensemble_retriever as a bonus attribute on the stats dict so
    # callers can immediately use it without re-loading from disk.
    stats["retriever"] = ensemble_retriever

    return stats


# ---------------------------------------------------------------------------
# Helper: Pretty-print ingestion statistics
# ---------------------------------------------------------------------------

def _print_stats(stats: dict[str, Any]) -> None:
    """Format and print ingestion statistics to stdout."""
    print("=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)
    print(f"  New files embedded : {stats['new_docs']}  ({stats['new_chunks']} new chunks)")
    print(f"  Full corpus        : {stats['total_docs']} file(s), {stats['total_chunks']} chunks")
    print(f"  Time taken         : {stats['time_taken_s']:.2f}s")
    print(f"  Embedding model    : {stats['embedding_model']}")
    print(f"  ChromaDB directory : {stats['chroma_dir']}")
    print(f"  BM25 pickle path   : {stats['bm25_path']}")
    print(f"  Chunk size         : {stats['chunk_size']} chars")
    print(f"  Chunk overlap      : {stats['chunk_overlap']} chars")
    print("=" * 60)



# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest technical documents into a hybrid ChromaDB + BM25 retrieval system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py --docs ./docs
  python ingest.py --docs /absolute/path/to/docs
        """,
    )
    parser.add_argument(
        "--docs",
        required=True,
        metavar="PATH",
        help="Path to the folder containing .md / .txt documents to ingest.",
    )
    return parser


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    try:
        stats = ingest_documents(docs_path=args.docs)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n[ERROR] {exc}")
        raise SystemExit(1)
