# Embeddings Guide for Technical Documentation

Embedding models transform text into dense numerical vectors that capture semantic meaning.
Choosing the right model is critical for retrieval quality in RAG systems.

## What Are Embeddings?

An embedding is a vector in a high-dimensional space (e.g., 384 or 768 dimensions) where
semantically similar texts are placed close together. Mathematically:

    embed("Python exception handling") ≈ embed("try-except blocks in Python")

This proximity is measured by cosine similarity:

    cosine_similarity(a, b) = (a · b) / (|a| * |b|)

Values range from -1 (opposite meaning) to 1 (identical meaning). In practice, most
documents score between 0.2 and 0.9 for relevant pairs.

## BAAI/bge-small-en-v1.5

The BGE (BAAI General Embedding) models from the Beijing Academy of AI are among the
top-performing open-source embedding models on the MTEB benchmark.

### Architecture

- **Base model**: BERT-like transformer with 12 layers
- **Hidden dimension**: 384
- **Parameters**: 33.4M
- **Max sequence length**: 512 tokens
- **Output dimension**: 384

### Why BGE Over MiniLM for Technical Docs?

| Metric | MiniLM-L6-v2 | BGE-small-en-v1.5 |
|--------|--------------|-------------------|
| MTEB average | 56.3 | 62.2 |
| Retrieval score | 49.1 | 51.7 |
| Semantic similarity | 82.4 | 84.9 |
| Inference speed | Very fast | Fast |
| Model size | 22M params | 33M params |

BGE uses a custom instruction prefix for queries: `"Represent this sentence for searching
relevant passages: "`. This is applied at query time only, not at document ingestion time.

### Usage with LangChain

```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},   # or "cuda" for GPU
    encode_kwargs={
        "normalize_embeddings": True  # Required for cosine similarity
    }
)

# Embed a batch of documents
doc_embeddings = embeddings.embed_documents(["text1", "text2"])

# Embed a query (with BGE instruction prefix applied automatically)
query_embedding = embeddings.embed_query("How does chunking work?")
```

## Normalization

Always normalize embeddings to unit length before storing in a vector database configured
for cosine similarity. Unnormalized embeddings produce incorrect cosine scores.

LangChain's `HuggingFaceEmbeddings` with `normalize_embeddings=True` handles this automatically.

## Batching

Embedding models process text in batches for efficiency. For large corpora:

```python
# Process in batches of 32 to avoid OOM on CPU
batch_size = 32
all_embeddings = []
for i in range(0, len(texts), batch_size):
    batch = texts[i:i + batch_size]
    embeddings_batch = model.embed_documents(batch)
    all_embeddings.extend(embeddings_batch)
```

ChromaDB and LangChain handle batching internally when using `add_documents()`.

## Embedding Caching

For production systems that re-ingest documents frequently, caching embeddings prevents
redundant computation:

```python
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

store = LocalFileStore("./embedding_cache")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=base_embeddings,
    document_embedding_cache=store,
    namespace="bge-small-en-v1.5"
)
```

## Dimensionality and Storage

Each embedding dimension is a 32-bit float (4 bytes).

- BGE-small (384 dims): 384 * 4 = 1,536 bytes per embedding
- For 10,000 chunks: ~15 MB of raw embedding data
- ChromaDB adds overhead for HNSW graph: ~2–5x raw size

## Evaluation: How to Measure Retrieval Quality

After ingestion, validate retrieval with known query-answer pairs:

```python
def evaluate_retrieval(retriever, test_pairs, k=5):
    hits = 0
    for query, expected_source in test_pairs:
        results = retriever.invoke(query)
        sources = [r.metadata.get("source") for r in results[:k]]
        if expected_source in sources:
            hits += 1
    return hits / len(test_pairs)  # Hit rate @ k
```

A hit rate above 0.8 for technical documentation retrieval is a good production baseline.
