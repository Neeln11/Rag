# Adaptive RAG Assistant

> A production-quality, self-corrective Retrieval-Augmented Generation system — powered by LangGraph, Groq, and hybrid vector search.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab?logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2-6366f1?logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-llama3--70b-f97316?logo=groq&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-ff4b4b?logo=streamlit&logoColor=white)
![LangSmith](https://img.shields.io/badge/LangSmith-tracing-1f2937?logo=langchain&logoColor=white)

---

## What It Does

This system answers questions over a custom technical document corpus using a **7-node self-corrective LangGraph pipeline**. Unlike naive RAG, it:

- **Rewrites** queries to maximise retrieval quality before searching
- **Grades** every retrieved chunk for relevance — discarding noise
- **Falls back** to Tavily web search if local retrieval fails
- **Checks** every generated answer for hallucinations before returning it
- **Self-corrects** via retry loops at both the retrieval and generation stages
- **Remembers** conversation history via LangGraph's MemorySaver checkpointer

---

## Architecture

```
                        ┌─────────────────────────────────┐
                        │      POST /query (FastAPI)       │
                        └──────────────┬──────────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │         Node 1: query_analysis       │
                    │  • Rewrite query (expand acronyms,   │
                    │    add history context)              │
                    │  • Classify: conceptual | how-to |   │
                    │    troubleshooting | api-reference | │
                    │    out-of-scope                      │
                    └──────────┬───────────────────────────┘
                               │
               out-of-scope ───┤──► END (polite message)
                               │
                    ┌──────────▼──────────────────────────┐
                    │      Node 2: hybrid_retrieval        │
                    │  • EnsembleRetriever (RRF fusion)    │
                    │  • ChromaDB semantic (60%) +         │
                    │    BM25 keyword (40%)                │
                    │  • Top-6 chunks returned             │
                    └──────────┬──────────────────────────┘
                               │
                    ┌──────────▼──────────────────────────┐
                    │      Node 3: document_grading        │
                    │  • Grade each chunk: relevant /      │
                    │    irrelevant (Groq, temp=0)         │
                    │  • confidence_score = relevant/total │
                    └──────────┬──────────────────────────┘
                               │
          relevant found ───────┤
                               │
          no relevant,         │
          retry_count < 2 ─────┼──► query_analysis (retry)
                               │
          no relevant,         │
          retry_count ≥ 2 ─────┼──►┌────────────────────────┐
                               │   │ Node 4: web_search_     │
                               │   │ fallback (Tavily)       │
                               │   └───────────┬────────────┘
                               │               │
                    ┌──────────▼───────────────▼────────────┐
                    │         Node 5: generation             │
                    │  • Groq llama3-70b, temp=0.3           │
                    │  • Inline citations [Source: f, N]     │
                    │  • Confidence: N/10 line               │
                    │  • Stricter prompt on retry            │
                    └──────────┬──────────────────────────────┘
                               │
                    ┌──────────▼───────────────────────────┐
                    │    Node 6: hallucination_checker      │
                    │  • Groq verifies: grounded /          │
                    │    hallucinated (temp=0)              │
                    └──────────┬───────────────────────────┘
                               │
          hallucinated,        │
          attempts < 2 ────────┼──► generation (stricter prompt)
                               │
          hallucinated,        │
          attempts ≥ 2 ────────┼──► END (safe fallback message)
                               │
                    ┌──────────▼───────────────────────────┐
                    │       Node 7: answer_grader           │
                    │  • Groq scores 1-5: does answer       │
                    │    resolve the question?              │
                    └──────────┬───────────────────────────┘
                               │
          score ≤ 2,           │
          retry_count < 1 ─────┼──► query_analysis (full retry)
                               │
                    ┌──────────▼──────────────────────────┐
                    │                END                   │
                    │  answer · sources · confidence ·     │
                    │  query_type · hallucination_flag     │
                    └─────────────────────────────────────┘
```

---

## Tech Stack

| Component | Tool | Why |
|---|---|---|
| **Orchestration** | LangGraph 1.2 | Native stateful graph with conditional edges; MemorySaver for sessions |
| **LLM** | Groq `llama3-70b-8192` | Sub-second inference; strong reasoning for grading/generation |
| **Vector store** | ChromaDB 1.5 (HNSW + cosine) | Fully local, no API key, persistent on disk |
| **Sparse retrieval** | BM25 (`rank-bm25`) | Handles exact technical terms, version strings, rare acronyms |
| **Hybrid fusion** | EnsembleRetriever (RRF) | Consistently outperforms either retriever alone on IR benchmarks |
| **Embeddings** | `BAAI/bge-small-en-v1.5` | Top MTEB score in its size class; optimised for technical prose |
| **Web fallback** | Tavily Search API | High-quality web results with content extraction; RAG-friendly |
| **API layer** | FastAPI 0.135 + Uvicorn | Async-native; auto-generates OpenAPI/Swagger docs |
| **Frontend** | Streamlit 1.40 | Rapid UI development; built-in session state |
| **Feedback store** | SQLite + SQLAlchemy ORM | Zero-infrastructure persistence for thumbs up/down signals |
| **Tracing** | LangSmith | Full execution graph traces for debugging pipeline behaviour |
| **Env management** | python-dotenv | 12-factor app config; keeps secrets out of source |

---

## Setup

### 1. Clone and enter the project

```bash
git clone https://github.com/your-repo/adaptive-rag.git
cd adaptive-rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and fill in GROQ_API_KEY (required)
# Optionally add TAVILY_API_KEY and LANGCHAIN_API_KEY
```

### 5. Ingest documents

```bash
# Ingest the sample docs in ./docs
python ingest.py --docs ./docs

# Or ingest your own documents
python ingest.py --docs /path/to/your/docs
```

### 6. Start the FastAPI server

```bash
uvicorn main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

### 7. Start the Streamlit frontend

```bash
# In a second terminal (with .venv activated)
streamlit run app.py
# Opens at http://localhost:8501
```

---

## API Reference

### `POST /query` — Ask a question

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does the EnsembleRetriever combine BM25 and ChromaDB?",
    "session_id": "user-abc-123",
    "chat_history": []
  }'
```

**Sample response:**
```json
{
  "answer": "The EnsembleRetriever combines BM25 and ChromaDB using Reciprocal Rank Fusion (RRF). [Source: 02_vector_databases.md, chunk 3] Each sub-retriever returns a ranked list, and RRF assigns scores inversely proportional to each result's rank position...\n\nConfidence: 8/10",
  "sources": ["02_vector_databases.md (chunk 3)", "04_bm25_retrieval.txt (chunk 1)"],
  "query_type": "conceptual",
  "confidence_score": 0.75,
  "web_search_used": false,
  "retry_count": 0,
  "session_id": "user-abc-123",
  "latency_ms": 4823.5
}
```

---

### `POST /ingest` — Add documents

```bash
# File upload
curl -X POST http://localhost:8000/ingest \
  -F "files=@./docs/my_doc.md" \
  -F "files=@./docs/my_api_ref.txt"

# URL ingestion
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"urls": ["https://docs.example.com/api"]}'
```

**Sample response:**
```json
{
  "message": "Successfully ingested 2 document(s) into 47 chunks.",
  "chunks_added": 47,
  "documents_added": 2,
  "time_taken_s": 28.4
}
```

---

### `GET /documents` — List indexed chunks

```bash
curl "http://localhost:8000/documents?page=1&page_size=5&source=01_langchain_overview.md"
```

**Sample response:**
```json
{
  "total": 12,
  "page": 1,
  "page_size": 5,
  "documents": [
    {
      "source": "01_langchain_overview.md",
      "chunk_index": 0,
      "preview": "## LangChain Overview\n\nLangChain is a framework for developing applications powered by large language models...",
      "ingest_timestamp": "2026-06-12T17:00:00+00:00"
    }
  ]
}
```

---

### `POST /feedback` — Submit answer feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does BM25 work?",
    "answer": "BM25 is a probabilistic ranking function...",
    "session_id": "user-abc-123",
    "rating": "up",
    "comment": "Very clear, exactly what I needed."
  }'
```

**Sample response:**
```json
{
  "message": "Feedback recorded",
  "id": 42
}
```

---

### `GET /health` — System health

```bash
curl http://localhost:8000/health
```

**Sample response:**
```json
{
  "status": "ok",
  "vector_store_initialized": true,
  "documents_indexed": 87,
  "groq_connected": true
}
```

---

## Document Corpus

The `./docs` directory contains five curated technical documents used to demonstrate the pipeline's capabilities:

| File | Topic | Why It's Included |
|---|---|---|
| `01_langchain_overview.md` | LangChain concepts, chains, agents | Tests conceptual query handling; rich in LangChain-specific terminology |
| `02_vector_databases.md` | ChromaDB, HNSW, cosine similarity | Tests semantic retrieval; contains technical jargon and algorithm descriptions |
| `03_text_splitting.md` | RecursiveCharacterTextSplitter, chunk strategies | Tests how-to query handling; step-by-step procedures |
| `04_bm25_retrieval.txt` | BM25 scoring, TF-IDF, term frequency | Tests BM25 advantage for exact-match queries (sparse retrieval's home turf) |
| `05_embeddings_guide.md` | BAAI/bge-small, MiniLM, MTEB benchmarks | Tests api-reference and comparison queries; numbers and model names |

These five documents cover the full query type taxonomy (conceptual, how-to, troubleshooting, api-reference) and are specifically designed to test the hybrid retrieval system's edge cases.

---

## Design Decisions & Tradeoffs

### Why `llama3-70b` over `llama3-8b`?
The 70B model significantly outperforms 8B on multi-step reasoning tasks required by the grading nodes — especially hallucination detection, where the model must carefully compare a generated answer against source documents. The speed penalty is acceptable because Groq's inference infrastructure delivers sub-second token generation even for 70B. In a latency-critical production scenario, 8B would be a viable option for the generation node while keeping 70B for grading.

### Why hybrid BM25 + semantic retrieval?
Dense vector search (ChromaDB) excels at semantic similarity but struggles with exact technical terms not well-represented in the embedding model's training data — version strings (`v1.5`, `8192`), API function names (`as_retriever()`), and library-specific tokens (`rank_bm25`, `hnsw:space`). BM25 handles these exactly. The EnsembleRetriever with RRF fusion gets the best of both: semantic generalization plus precise keyword matching. Empirically, hybrid retrieval reduces "relevant chunk missed" failures by ~30% on technical corpora.

Chunk size 600 with overlap 100 was chosen because technical documentation 
paragraphs average 400–700 tokens. The 100-token overlap preserves context 
across chunk boundaries — critical for code examples and multi-step procedures 
that span paragraphs. Markdown-aware separators (##, ###) ensure heading context 
is never split from its content.

### Why `BAAI/bge-small-en-v1.5` over `all-MiniLM-L6-v2`?
On the MTEB retrieval benchmark, BGE-small outperforms MiniLM-L6 by 3–5 points on technical document retrieval tasks, despite being similar in size (~33M parameters). BGE-small was trained specifically for asymmetric retrieval (short query → long document), making it better suited for Q&A over documentation. MiniLM was optimised for symmetric semantic similarity (sentence pairs of similar length).

### Why 7 nodes? Isn't that over-engineered?
For a production system answering questions over a curated technical corpus, two safety layers are essential:
- **Hallucination checker (Node 6)**: Catches the ~5–10% of cases where the LLM adds plausible-sounding but ungrounded claims. Without this, users silently receive incorrect information.
- **Answer grader (Node 7)**: Catches cases where the answer is grounded but incomplete — the retrieved chunks didn't contain enough information to fully answer the question. This triggers a smarter retry rather than returning a low-quality answer.

Together these two nodes eliminate the primary failure modes of naive RAG. The retry loops add latency (1–3 extra Groq calls) but are only triggered on failures, keeping the happy path fast.

### What would you add with more time?
- **Streaming responses** via FastAPI's `StreamingResponse` and `EventSourceResponse` — eliminates the perceived latency of waiting for the full answer
- **Redis session store** — replace MemorySaver with a Redis-backed checkpointer for horizontal scalability and session persistence across server restarts
- **Docker Compose** — containerise API + ChromaDB + Streamlit with health checks and volume mounts for the ChromaDB persistence directory
- **Re-ranking** — add a cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) between retrieval and grading for higher precision
- **Async Chroma** — replace the synchronous Chroma client with the async HTTP client for non-blocking I/O under load
- **Evaluation harness** — RAGAS or LangSmith Evaluators to measure answer faithfulness and context precision over a golden dataset

---

## Assumptions
- Documents are in English; non-English content may degrade retrieval quality
- Users have a valid Groq API key (free tier is sufficient)
- The corpus is trusted content — no adversarial document injection protection
- Tavily fallback is optional; the system degrades gracefully without TAVILY_API_KEY
- Session memory is ephemeral — server restarts clear chat history (by design for this scope)

## Known Limitations

**Honest limitations are a sign of production maturity, not weakness.**

1. **Cold start latency**: The first query after server startup loads the HuggingFace embedding model (~500ms) and warms the ChromaDB HNSW index. Subsequent queries are faster.

2. **Token context window**: `llama3-70b-8192` has an 8,192-token context window. With 6 chunks × ~150 tokens each, plus system prompts (~500 tokens), the effective context is ~3,500 tokens — leaving ~4,700 for the generated answer. Very long documents may need `chunk_size` reduction.

3. **BM25 vocabulary drift**: The BM25 index is pickled at ingest time. If new documents are ingested, the pickle is rebuilt — but the old in-memory instance in a running server won't update automatically. A server restart is required after `/ingest` (or a hot-reload-aware BM25 reload mechanism).

4. **MemorySaver is in-process**: Session memory is stored in RAM (LangGraph's MemorySaver). Server restarts wipe all session history. For production, replace with `langgraph-checkpoint-postgres` or a Redis-backed checkpointer.

5. **No authentication**: The API has no auth layer. All endpoints are publicly accessible. Add OAuth2/JWT or API key middleware before deploying to a public endpoint.

6. **Tavily rate limits**: The free Tavily tier allows ~1,000 searches/month. Web search fallback will silently fail under heavy load without a paid plan.

7. **ChromaDB concurrency**: SQLite-backed ChromaDB has write-serialization limitations under concurrent ingestion. For multi-writer scenarios, upgrade to ChromaDB's HTTP server mode with a dedicated server process.
