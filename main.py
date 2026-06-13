"""
main.py
=======
FastAPI application wrapping the LangGraph Adaptive RAG pipeline.

Startup order matters:
  1. python-dotenv loads .env into os.environ
  2. rag_pipeline is imported (reads GROQ_API_KEY from env at import time)
  3. FastAPI app is created with middleware + routes
  4. @app.on_event("startup") checks ChromaDB health

Run:
    uvicorn main:app --reload --port 8000
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# STEP 1: Load .env FIRST — before any module that reads env vars at import
# ---------------------------------------------------------------------------
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# FastAPI & ASGI
# ---------------------------------------------------------------------------
import uvicorn
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Database (SQLite via SQLAlchemy ORM) — defined in database.py
# ---------------------------------------------------------------------------
from database import FeedbackRecord, get_db

# ---------------------------------------------------------------------------
# ChromaDB (for /documents and /health)
# ---------------------------------------------------------------------------
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# RAG pipeline & ingestion
# ---------------------------------------------------------------------------
from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    EMBEDDING_MODEL_NAME,
)
from ingest import ingest_documents
from rag_pipeline import _checkpointer, _graph, run_rag, invalidate_retriever

import rag_workflow  # noqa: F401 — canonical public alias

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
from schemas import (
    DocumentItem,
    DocumentsResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SessionHistoryResponse,
    SessionTurn,
)

# ===========================================================================
# Logging
# ===========================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("rag_api")

# ===========================================================================
# SQLAlchemy — DB layer imported from database.py
# (FeedbackRecord and get_db are already imported above)
# ===========================================================================


# ===========================================================================
# ChromaDB helper (lazy singleton for read-only operations)
# ===========================================================================

_chroma_store: Optional[Chroma] = None
_chroma_embeddings: Optional[HuggingFaceEmbeddings] = None


def _get_chroma() -> Optional[Chroma]:
    """
    Return a read-only Chroma vector store handle, or None if not initialised.
    Lazy singleton — initialised on first call.
    """
    global _chroma_store, _chroma_embeddings

    chroma_path = Path(CHROMA_PERSIST_DIR)
    if not chroma_path.exists():
        return None

    if _chroma_store is None:
        _chroma_embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _chroma_store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=_chroma_embeddings,
            persist_directory=str(chroma_path),
        )

    return _chroma_store


def _invalidate_chroma_cache() -> None:
    """Reset the Chroma singleton after ingestion so counts refresh."""
    global _chroma_store
    _chroma_store = None


def _count_chroma_docs() -> int:
    """Return the number of chunks in ChromaDB, or 0 on error."""
    try:
        store = _get_chroma()
        if store is None:
            return 0
        return store._collection.count()
    except Exception:
        return 0


# ===========================================================================
# Docs directory
# ===========================================================================

_DOCS_DIR = Path(__file__).parent / "docs"
_DOCS_DIR.mkdir(exist_ok=True)

_ALLOWED_SUFFIXES = {".md", ".txt", ".pdf", ".docx", ".html", ".csv", ".json"}

# ===========================================================================
# Lifespan (replaces deprecated @app.on_event)
# ===========================================================================


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup: validate environment and warm up ChromaDB handle."""
    logger.info("=" * 60)
    logger.info("  Adaptive RAG API — starting up")
    logger.info("=" * 60)

    # Check GROQ key
    if not os.getenv("GROQ_API_KEY"):
        logger.warning("GROQ_API_KEY is not set — LLM calls will fail!")
    else:
        logger.info("✓ GROQ_API_KEY detected")

    # Check ChromaDB
    doc_count = _count_chroma_docs()
    if doc_count == 0:
        logger.warning(
            "⚠  ChromaDB appears empty. Run `python ingest.py --docs ./docs` "
            "or POST /ingest to populate the vector store."
        )
    else:
        logger.info(f"✓ ChromaDB ready — {doc_count} chunk(s) indexed")

    # Check Tavily (optional)
    if os.getenv("TAVILY_API_KEY"):
        logger.info("✓ TAVILY_API_KEY detected — web search fallback enabled")
    else:
        logger.info("  TAVILY_API_KEY not set — web search fallback disabled")

    yield  # ← app runs here

    logger.info("Adaptive RAG API shutting down.")


# ===========================================================================
# FastAPI app
# ===========================================================================

app = FastAPI(
    title="Adaptive RAG API",
    description=(
        "Production FastAPI wrapper around a self-corrective LangGraph RAG pipeline. "
        "Supports hybrid retrieval (ChromaDB + BM25), hallucination checking, "
        "web search fallback, and conversation memory."
    ),
    version="1.0.0",
    lifespan=_lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS middleware — allow all origins in dev; tighten in production
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request logging middleware — logs method, path, status, and latency
# ---------------------------------------------------------------------------
@app.middleware("http")
async def _log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "%s %s → %d (%.1fms)",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


# ===========================================================================
# POST /query
# ===========================================================================


@app.post(
    "/query",
    response_model=QueryResponse,
    summary="Run the Adaptive RAG pipeline",
    tags=["RAG"],
)
async def query_endpoint(body: QueryRequest):
    """
    Submit a question to the LangGraph RAG pipeline.

    - Auto-generates `session_id` if not provided.
    - Returns the graded, hallucination-checked answer with sources and metadata.
    - Raises **503** if ChromaDB is empty.
    - Raises **500** on LLM or retrieval failure with a human-readable message.
    """
    # Guard: ChromaDB must have data
    if _count_chroma_docs() == 0:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Vector store is not initialised. "
                "POST /ingest to add documents first."
            ),
        )

    session_id = body.session_id or str(uuid.uuid4())
    t0 = time.perf_counter()

    try:
        result = await run_rag(
            question=body.question,
            session_id=session_id,
            chat_history=body.chat_history or [],
        )
    except RuntimeError as exc:
        # 503-style errors raised by hybrid_retrieval node
        if "503" in str(exc):
            raise HTTPException(status_code=503, detail=str(exc))
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")
    except Exception as exc:
        logger.exception("Unexpected error in RAG pipeline")
        raise HTTPException(
            status_code=500,
            detail=f"LLM or retrieval failure: {type(exc).__name__}: {exc}",
        )

    latency_ms = (time.perf_counter() - t0) * 1000

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        query_type=result["query_type"],
        confidence_score=result["confidence_score"],
        web_search_used=result["web_search_used"],
        retry_count=result["retry_count"],
        session_id=session_id,
        latency_ms=round(latency_ms, 2),
    )


# ===========================================================================
# POST /ingest
# ===========================================================================


@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest documents into the vector store",
    tags=["Ingestion"],
)
async def ingest_endpoint(
    request: Request,
    files: Optional[List[UploadFile]] = File(default=None),
    urls: Optional[str] = Form(
        default=None,
        description="JSON array of URLs as a form string (alternative to files).",
    ),
):
    """
    Add documents to the RAG knowledge base.

    **Option A — File upload** (multipart/form-data):
    Upload one or more `.md` or `.txt` files via the `files` field.

    **Option B — URL list** (multipart/form-data or JSON):
    Pass `urls` as a JSON-encoded string: `'["https://..."]'`.
    Each URL's text content is fetched and saved as a `.txt` file.

    After saving files, `ingest_documents()` rebuilds ChromaDB and the BM25 index.
    """
    saved_files: list[Path] = []

    # ── Handle file uploads ──────────────────────────────────────────────
    if files:
        for upload in files:
            if not upload.filename:
                continue
            suffix = Path(upload.filename).suffix.lower()
            if suffix not in _ALLOWED_SUFFIXES:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Unsupported file type: '{suffix}' in '{upload.filename}'. "
                        f"Accepted types: {', '.join(sorted(_ALLOWED_SUFFIXES))}"
                    ),
                )
            dest = _DOCS_DIR / upload.filename
            content = await upload.read()
            dest.write_bytes(content)
            saved_files.append(dest)
            logger.info(
                "Saved uploaded file: %s (%d bytes, type=%s)",
                dest.name, len(content), suffix,
            )

    # ── Handle URL ingestion ──────────────────────────────────────────────
    if urls:
        try:
            url_list: list[str] = json.loads(urls)
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="'urls' must be a valid JSON array string.")

        import httpx as _httpx

        async with _httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            for url in url_list:
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    # Derive filename from URL
                    url_path = url.rstrip("/").split("/")[-1] or "web_content"
                    fname = f"web_{url_path[:40]}.txt"
                    dest = _DOCS_DIR / fname
                    dest.write_text(resp.text, encoding="utf-8", errors="replace")
                    saved_files.append(dest)
                    logger.info("Fetched URL → %s (%d bytes)", fname, len(resp.text))
                except Exception as exc:
                    logger.warning("Failed to fetch %s: %s", url, exc)

    # ── Also accept raw JSON body {"urls": [...]} ─────────────────────────
    if not files and not urls:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            url_list = body.get("urls", [])
            if url_list:
                import httpx as _httpx

                async with _httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
                    for url in url_list:
                        try:
                            resp = await client.get(url)
                            resp.raise_for_status()
                            url_path = url.rstrip("/").split("/")[-1] or "web_content"
                            fname = f"web_{url_path[:40]}.txt"
                            dest = _DOCS_DIR / fname
                            dest.write_text(resp.text, encoding="utf-8", errors="replace")
                            saved_files.append(dest)
                        except Exception as exc:
                            logger.warning("Failed to fetch %s: %s", url, exc)

    if not saved_files:
        raise HTTPException(
            status_code=422,
            detail="No files or URLs provided. Send files via multipart or urls via JSON/form.",
        )

    # ── Run ingestion pipeline ────────────────────────────────────────────
    # Ingest only the newly saved files so the response counts are accurate
    # and we don't needlessly re-embed unchanged old documents.
    # We pass the list of individual file paths, not the whole docs dir.
    new_file_names = [f.name for f in saved_files]
    logger.info("Starting ingestion for %d new file(s): %s", len(new_file_names), new_file_names)

    try:
        stats = ingest_documents(str(_DOCS_DIR), new_files_only=saved_files)
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {exc}")

    # Invalidate Chroma singleton so /health and /documents see fresh counts
    _invalidate_chroma_cache()

    # Invalidate the RAG pipeline's retriever singleton so the next /query
    # reloads the BM25 index and ChromaDB from disk with the new documents.
    invalidate_retriever()

    file_list_str = ", ".join(new_file_names)
    logger.info(
        "Ingestion complete: %d new chunk(s) from %d new file(s) | total corpus: %d file(s), %d chunks",
        stats["new_chunks"], stats["new_docs"],
        stats["total_docs"], stats["total_chunks"],
    )
    return IngestResponse(
        message=(
            f"Successfully ingested {stats['new_docs']} new file(s) "
            f"({file_list_str}) — {stats['new_chunks']} new chunks added. "
            f"Total corpus: {stats['total_docs']} file(s), {stats['total_chunks']} chunks."
        ),
        chunks_added=stats["new_chunks"],
        documents_added=stats["new_docs"],
        time_taken_s=stats["time_taken_s"],
    )


# ===========================================================================
# GET /documents
# ===========================================================================


@app.get(
    "/documents",
    response_model=DocumentsResponse,
    summary="List indexed document chunks",
    tags=["Ingestion"],
)
def documents_endpoint(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)."),
    page_size: int = Query(default=20, ge=1, le=100, description="Results per page."),
    source: Optional[str] = Query(default=None, description="Filter by source filename."),
):
    """
    Return a paginated list of all document chunks stored in ChromaDB,
    with optional filtering by source filename.
    """
    store = _get_chroma()
    if store is None:
        return DocumentsResponse(total=0, page=page, page_size=page_size, documents=[])

    collection = store._collection

    try:
        # Build where clause for filtering
        where = {"source": source} if source else None

        # Fetch all matching records (ChromaDB doesn't support server-side offset)
        kwargs: dict = {"include": ["metadatas", "documents"]}
        if where:
            kwargs["where"] = where

        result = collection.get(**kwargs)
    except Exception as exc:
        logger.error("ChromaDB query error: %s", exc)
        raise HTTPException(status_code=500, detail=f"ChromaDB error: {exc}")

    metadatas = result.get("metadatas") or []
    documents_text = result.get("documents") or []
    total = len(metadatas)

    # Apply pagination
    start = (page - 1) * page_size
    end = start + page_size
    page_meta = metadatas[start:end]
    page_docs = documents_text[start:end]

    items: list[DocumentItem] = []
    for meta, text in zip(page_meta, page_docs):
        items.append(
            DocumentItem(
                source=meta.get("source", "unknown"),
                chunk_index=int(meta.get("chunk_index", 0)),
                preview=(text or "")[:100],
                ingest_timestamp=meta.get("ingest_timestamp"),
            )
        )

    return DocumentsResponse(
        total=total,
        page=page,
        page_size=page_size,
        documents=items,
    )


# ===========================================================================
# POST /feedback
# ===========================================================================


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit user feedback on an answer",
    tags=["Feedback"],
)
def feedback_endpoint(body: FeedbackRequest):
    """
    Record thumbs-up / thumbs-down feedback for a generated answer.

    Stored in a local SQLite database (`feedback.db`) via SQLAlchemy ORM.
    """
    db = get_db()
    try:
        record = FeedbackRecord(
            session_id=body.session_id,
            question=body.question,
            answer=body.answer,
            rating=body.rating,
            comment=body.comment,
        )
        db.add(record)
        db.commit()
        db.refresh(record)
        logger.info(
            "Feedback recorded — id=%d session=%s rating=%s",
            record.id,
            body.session_id,
            body.rating,
        )
        return FeedbackResponse(message="Feedback recorded", id=record.id)
    except Exception as exc:
        db.rollback()
        logger.exception("Failed to save feedback")
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")
    finally:
        db.close()


# ===========================================================================
# GET /health
# ===========================================================================


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="API health check",
    tags=["System"],
)
def health_endpoint():
    """
    Return the current health of the API.

    - `vector_store_initialized`: True if ChromaDB has at least one document.
    - `documents_indexed`: Exact chunk count in ChromaDB.
    - `groq_connected`: True if GROQ_API_KEY is present (avoids live API call).
    """
    doc_count = _count_chroma_docs()
    groq_ok = bool(os.getenv("GROQ_API_KEY"))
    vs_ok = doc_count > 0

    return HealthResponse(
        status="ok" if (vs_ok and groq_ok) else "degraded",
        vector_store_initialized=vs_ok,
        documents_indexed=doc_count,
        groq_connected=groq_ok,
    )


# ===========================================================================
# GET /sessions/{session_id}/history
# ===========================================================================


@app.get(
    "/sessions/{session_id}/history",
    response_model=SessionHistoryResponse,
    summary="Retrieve conversation history for a session",
    tags=["Sessions"],
)
async def session_history_endpoint(session_id: str):
    """
    Return the accumulated chat history for a given session from
    the LangGraph MemorySaver checkpointer.
    """
    config = {"configurable": {"thread_id": session_id}}

    try:
        state_snapshot = await _graph.aget_state(config)
    except Exception as exc:
        logger.warning("Could not retrieve session state for %s: %s", session_id, exc)
        return SessionHistoryResponse(session_id=session_id, turns=[], total_turns=0)

    if state_snapshot is None or state_snapshot.values is None:
        return SessionHistoryResponse(session_id=session_id, turns=[], total_turns=0)

    raw_history: list[dict] = state_snapshot.values.get("chat_history", [])
    turns = [
        SessionTurn(role=t.get("role", "user"), content=t.get("content", ""))
        for t in raw_history
        if isinstance(t, dict)
    ]

    return SessionHistoryResponse(
        session_id=session_id,
        turns=turns,
        total_turns=len(turns),
    )


# ===========================================================================
# CLI entry point
# ===========================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
