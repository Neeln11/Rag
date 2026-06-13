"""
schemas.py
==========
Pydantic v2 models for all FastAPI request/response contracts.

Keeping schemas in a dedicated module ensures:
- Single source of truth for data shapes used by both the API and the Streamlit
  frontend (via shared imports or the /openapi.json spec).
- Clean separation: main.py contains routing logic, schemas.py contains data
  contracts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ===========================================================================
# /query
# ===========================================================================


class QueryRequest(BaseModel):
    """Payload for POST /query."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's natural-language question.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Stable session identifier. Auto-generated if omitted.",
    )
    chat_history: Optional[list[dict]] = Field(
        default_factory=list,
        description="Prior conversation turns: [{role, content}, ...]",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "question": "How does the EnsembleRetriever combine BM25 and ChromaDB?",
            "session_id": "user-abc-123",
            "chat_history": [],
        }
    }}


class QueryResponse(BaseModel):
    """Response from POST /query."""

    answer: str
    sources: list[str]
    query_type: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    web_search_used: bool
    retry_count: int = Field(ge=0)
    session_id: str
    latency_ms: float = Field(description="End-to-end server latency in milliseconds.")


# ===========================================================================
# /ingest
# ===========================================================================


class IngestResponse(BaseModel):
    """Response from POST /ingest."""

    message: str
    chunks_added: int
    documents_added: int
    time_taken_s: float


# ===========================================================================
# /documents
# ===========================================================================


class DocumentItem(BaseModel):
    """A single document chunk entry returned by GET /documents."""

    source: str = Field(description="Source filename (e.g. 'overview.md').")
    chunk_index: int
    preview: str = Field(description="First 100 characters of the chunk text.")
    ingest_timestamp: Optional[str] = None


class DocumentsResponse(BaseModel):
    """Paginated list of indexed document chunks."""

    total: int
    page: int
    page_size: int
    documents: list[DocumentItem]


# ===========================================================================
# /feedback
# ===========================================================================


class FeedbackRequest(BaseModel):
    """Payload for POST /feedback."""

    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    session_id: str
    rating: Literal["up", "down"] = Field(
        description="Thumbs up ('up') or thumbs down ('down')."
    )
    comment: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional free-text comment from the user.",
    )

    model_config = {"json_schema_extra": {
        "example": {
            "question": "How does BM25 work?",
            "answer": "BM25 is a ranking function based on TF-IDF...",
            "session_id": "user-abc-123",
            "rating": "up",
            "comment": "Very clear explanation!",
        }
    }}


class FeedbackResponse(BaseModel):
    """Response from POST /feedback."""

    message: str
    id: int


# ===========================================================================
# /health
# ===========================================================================


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: Literal["ok", "degraded", "error"]
    vector_store_initialized: bool
    documents_indexed: int
    groq_connected: bool


# ===========================================================================
# /sessions/{session_id}/history
# ===========================================================================


class SessionTurn(BaseModel):
    """A single turn in the conversation history."""

    role: Literal["user", "assistant"]
    content: str


class SessionHistoryResponse(BaseModel):
    """Conversation history for a session."""

    session_id: str
    turns: list[SessionTurn]
    total_turns: int
