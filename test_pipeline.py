"""
test_pipeline.py
================
Lightweight integration tests for the ingestion pipeline.

These tests use pytest and run against the actual file system (no mocking of
ChromaDB or BM25) to catch real integration issues.  They are NOT unit tests —
they validate the full end-to-end contract of ingest_documents().

Run:
    pytest test_pipeline.py -v

Note: First run downloads the BAAI/bge-small-en-v1.5 model (~130 MB) from
HuggingFace.  Subsequent runs use the local cache.
"""

from __future__ import annotations

import pickle
import shutil
import tempfile
import textwrap
import time
from pathlib import Path

import pytest

# Patch constants BEFORE importing ingest so the test uses temp directories
# instead of polluting ./chroma_db and ./bm25_index.pkl in the project root.
import ingest as _ingest_module


@pytest.fixture(scope="module")
def temp_dirs():
    """
    Create isolated temporary directories for each test module run.

    We use module scope so the embedding model is loaded only once, keeping
    the test suite fast.  Each test that modifies state should use its own
    subfolder if isolation is required.
    """
    tmp = Path(tempfile.mkdtemp(prefix="pipeline_test_"))
    docs_dir = tmp / "docs"
    chroma_dir = tmp / "chroma_db"
    bm25_path = tmp / "bm25_index.pkl"

    docs_dir.mkdir()

    # Patch module-level constants to point at our temp dirs
    original_chroma = _ingest_module.CHROMA_PERSIST_DIR
    original_bm25 = _ingest_module.BM25_PICKLE_PATH

    _ingest_module.CHROMA_PERSIST_DIR = str(chroma_dir)
    _ingest_module.BM25_PICKLE_PATH = str(bm25_path)

    # Write minimal synthetic docs
    (docs_dir / "doc_a.md").write_text(textwrap.dedent("""\
        # Introduction

        ## Overview
        This document describes the test system.

        ## Details
        The RecursiveCharacterTextSplitter splits on headers first.
        BM25 uses term frequency and inverse document frequency scoring.
    """), encoding="utf-8")

    (docs_dir / "doc_b.txt").write_text(
        "ChromaDB stores vectors persistently using HNSW indexing.\n"
        "Embedding models convert text to numerical vectors.\n"
        "Hybrid retrieval combines dense and sparse search methods.\n",
        encoding="utf-8",
    )

    yield {
        "docs_dir": docs_dir,
        "chroma_dir": chroma_dir,
        "bm25_path": bm25_path,
        "tmp": tmp,
    }

    # Restore original constants and clean up
    _ingest_module.CHROMA_PERSIST_DIR = original_chroma
    _ingest_module.BM25_PICKLE_PATH = original_bm25
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def ingestion_result(temp_dirs):
    """Run the ingestion pipeline once and cache the result for all tests."""
    from ingest import ingest_documents
    return ingest_documents(docs_path=str(temp_dirs["docs_dir"]))


# ---------------------------------------------------------------------------
# Test: stats dict has all required keys
# ---------------------------------------------------------------------------

class TestIngestionStats:
    """Verify the stats dict returned by ingest_documents()."""

    REQUIRED_KEYS = {
        "total_docs",
        "total_chunks",
        "time_taken_s",
        "docs_path",
        "chroma_dir",
        "bm25_path",
        "chunk_size",
        "chunk_overlap",
        "embedding_model",
        "retriever",
    }

    def test_stats_has_required_keys(self, ingestion_result):
        missing = self.REQUIRED_KEYS - ingestion_result.keys()
        assert not missing, f"Stats dict missing keys: {missing}"

    def test_total_docs_matches_files(self, ingestion_result, temp_dirs):
        expected = len(list(temp_dirs["docs_dir"].glob("*")))
        assert ingestion_result["total_docs"] == expected

    def test_total_chunks_positive(self, ingestion_result):
        assert ingestion_result["total_chunks"] > 0, "No chunks were created"

    def test_time_taken_is_positive_float(self, ingestion_result):
        assert isinstance(ingestion_result["time_taken_s"], float)
        assert ingestion_result["time_taken_s"] > 0

    def test_chunk_size_matches_constant(self, ingestion_result):
        from ingest import CHUNK_SIZE
        assert ingestion_result["chunk_size"] == CHUNK_SIZE

    def test_embedding_model_matches_constant(self, ingestion_result):
        from ingest import EMBEDDING_MODEL_NAME
        assert ingestion_result["embedding_model"] == EMBEDDING_MODEL_NAME


# ---------------------------------------------------------------------------
# Test: ChromaDB persistence
# ---------------------------------------------------------------------------

class TestChromaDB:
    """Verify ChromaDB was written to disk correctly."""

    def test_chroma_dir_exists(self, ingestion_result, temp_dirs):
        assert temp_dirs["chroma_dir"].exists(), "ChromaDB directory not created"

    def test_chroma_dir_not_empty(self, ingestion_result, temp_dirs):
        files = list(temp_dirs["chroma_dir"].rglob("*"))
        assert files, "ChromaDB directory is empty — nothing was persisted"

    def test_chroma_returns_results_for_known_query(self, ingestion_result):
        retriever = ingestion_result["retriever"]
        results = retriever.invoke("BM25 term frequency scoring")
        assert len(results) > 0, "Retriever returned no results"

    def test_chunk_metadata_has_required_fields(self, ingestion_result):
        retriever = ingestion_result["retriever"]
        results = retriever.invoke("ChromaDB vector storage")
        assert results, "No results returned to check metadata"
        for doc in results:
            meta = doc.metadata
            for field in ("source", "chunk_index", "total_chunks",
                          "doc_type", "ingest_timestamp"):
                assert field in meta, f"Missing metadata field: {field!r}"


# ---------------------------------------------------------------------------
# Test: BM25 pickle persistence
# ---------------------------------------------------------------------------

class TestBM25Pickle:
    """Verify BM25 was serialised and can be reloaded."""

    def test_bm25_pickle_exists(self, ingestion_result, temp_dirs):
        assert temp_dirs["bm25_path"].exists(), "BM25 pickle file not created"

    def test_bm25_pickle_is_valid(self, ingestion_result, temp_dirs):
        with open(temp_dirs["bm25_path"], "rb") as f:
            loaded = pickle.load(f)
        from langchain_community.retrievers import BM25Retriever
        assert isinstance(loaded, BM25Retriever), (
            f"Loaded object is {type(loaded)}, expected BM25Retriever"
        )

    def test_bm25_returns_results(self, ingestion_result, temp_dirs):
        with open(temp_dirs["bm25_path"], "rb") as f:
            bm25 = pickle.load(f)
        results = bm25.invoke("embedding model vectors")
        assert len(results) > 0, "Reloaded BM25 retriever returned no results"


# ---------------------------------------------------------------------------
# Test: EnsembleRetriever
# ---------------------------------------------------------------------------

class TestEnsembleRetriever:
    """Verify the hybrid retriever behaves correctly."""

    def test_ensemble_retriever_type(self, ingestion_result):
        from langchain_classic.retrievers import EnsembleRetriever
        assert isinstance(ingestion_result["retriever"], EnsembleRetriever)

    def test_ensemble_returns_unique_docs(self, ingestion_result):
        retriever = ingestion_result["retriever"]
        results = retriever.invoke("splitting text into chunks")
        contents = [r.page_content for r in results]
        # Duplicate page_content would indicate RRF deduplication failure
        assert len(contents) == len(set(contents)), "Duplicate documents in results"

    def test_ensemble_result_metadata_complete(self, ingestion_result):
        retriever = ingestion_result["retriever"]
        results = retriever.invoke("HNSW cosine similarity")
        for doc in results:
            assert "source" in doc.metadata
            assert "ingest_timestamp" in doc.metadata


# ---------------------------------------------------------------------------
# Test: Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Verify that invalid inputs raise the expected exceptions."""

    def test_nonexistent_path_raises_file_not_found(self):
        from ingest import ingest_documents
        with pytest.raises(FileNotFoundError):
            ingest_documents("/this/path/does/not/exist/at/all")

    def test_empty_directory_raises_value_error(self, tmp_path):
        from ingest import ingest_documents
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match="No supported documents"):
            ingest_documents(str(empty_dir))

    def test_file_path_instead_of_dir_raises_value_error(self, tmp_path):
        from ingest import ingest_documents
        f = tmp_path / "file.txt"
        f.write_text("hello")
        with pytest.raises(ValueError):
            ingest_documents(str(f))
