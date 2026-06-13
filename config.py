"""
config.py
=========
Single source of truth for all constants shared between ingest.py,
retriever.py, rag_pipeline.py, and main.py.

Import from here everywhere — never hard-code these values in other modules.
"""

# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

# Directory where ChromaDB persists its SQLite + HNSW files.
# Must be the same value in EVERY module that touches ChromaDB.
CHROMA_PERSIST_DIR = "./chroma_db"

# Logical namespace inside the persist dir.
# Changing this creates a brand-new empty collection.
CHROMA_COLLECTION_NAME = "technical_docs"

# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

# Pickle file for the BM25Retriever.
BM25_PICKLE_PATH = "./bm25_index.pkl"

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

# HuggingFace sentence-transformer model used at both ingest time (to build
# vectors) and query time (to embed the user's question).
# Must be identical in both places or similarity scores will be meaningless.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", " "]

# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

# RRF weights — must sum to 1.0.
CHROMA_WEIGHT = 0.6
BM25_WEIGHT = 0.4

# Top-K results returned by each sub-retriever before RRF fusion.
RETRIEVER_K = 5
