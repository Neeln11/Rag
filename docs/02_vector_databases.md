# Vector Databases in Production

Vector databases are purpose-built storage systems for high-dimensional embeddings. They
enable semantic similarity search at scale, powering applications from RAG systems to
recommendation engines.

## Why Vector Databases?

Traditional relational databases excel at exact-match queries. Vector databases solve a
different problem: finding the most *similar* items to a query embedding, measured by
cosine similarity, dot product, or Euclidean distance.

### Approximate Nearest Neighbor (ANN)

Exact nearest neighbor search is O(n) per query — unacceptable at million-scale. ANN
algorithms trade a small amount of recall for massive speed gains:

- **HNSW** (Hierarchical Navigable Small World): graph-based, very fast at query time
- **IVF** (Inverted File Index): clusters vectors; only searches relevant clusters
- **PQ** (Product Quantization): compresses vectors to reduce memory footprint

## ChromaDB

ChromaDB is an open-source, embeddable vector database designed for AI applications.

### Key Features

- **Persistent storage**: Saves to SQLite + on-disk HNSW index automatically
- **Metadata filtering**: Filter results by arbitrary JSON metadata before vector search
- **Collections**: Logical namespaces for grouping related documents
- **Zero infrastructure**: Runs in-process; no separate server required for local use

### ChromaDB Architecture

```
┌─────────────────────────────────────┐
│           ChromaDB Client           │
├─────────────────┬───────────────────┤
│  SQLite Backend │  HNSW Index       │
│  (metadata +    │  (vector search)  │
│   raw docs)     │                   │
└─────────────────┴───────────────────┘
```

### Persistence Pattern

```python
import chromadb

# Persistent client saves everything to disk automatically
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)
```

## FAISS

Facebook AI Similarity Search (FAISS) is a library for efficient similarity search.

### When to Use FAISS vs ChromaDB

| Feature | FAISS | ChromaDB |
|---------|-------|----------|
| Metadata filtering | No | Yes |
| Persistence | Manual | Automatic |
| Ease of use | Low | High |
| Raw performance | Very high | High |
| Server mode | No | Yes (HTTP) |

Use FAISS when you need maximum raw throughput and handle metadata externally.
Use ChromaDB when you need metadata filtering and ease of development.

## Pinecone

Pinecone is a managed vector database service. It handles scaling, replication, and
infrastructure automatically. Best for production workloads where operational simplicity
outweighs cost considerations.

## Metadata-Aware Retrieval

Attaching rich metadata to vectors enables powerful hybrid queries:

```python
# Store with metadata
collection.add(
    documents=["chunk text..."],
    embeddings=[[0.1, 0.2, ...]],
    metadatas=[{"source": "manual.md", "chapter": 3, "version": "2.0"}],
    ids=["chunk_001"]
)

# Filter by metadata + similarity search
results = collection.query(
    query_embeddings=[[0.15, 0.22, ...]],
    n_results=5,
    where={"version": "2.0"}  # Only search version 2.0 docs
)
```

## Embedding Models for Technical Documents

For technical documentation retrieval, embedding model choice significantly impacts quality:

- **BAAI/bge-small-en-v1.5**: Strong performance, 33M params, fast inference
- **BAAI/bge-large-en-v1.5**: Best quality, 335M params, slower
- **sentence-transformers/all-MiniLM-L6-v2**: Lightweight but less accurate for technical text
- **text-embedding-3-small**: OpenAI's cost-effective option (requires API key)

For local, offline deployments without API keys, BGE models are the recommended choice.
