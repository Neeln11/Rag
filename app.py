"""
app.py
======
Streamlit frontend for the Adaptive RAG pipeline.

Connects to the FastAPI backend at API_BASE_URL (default: http://localhost:8000).
All API calls use httpx (sync) so the Streamlit event loop is not blocked
by asyncio complexity — Streamlit is itself single-threaded per session.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional


import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 120.0  # seconds — RAG pipeline can take 30–60s

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-repo",
        "Report a bug": "https://github.com/your-repo/issues",
        "About": "### RAG Assistant\nPowered by LangGraph + Groq + ChromaDB",
    },
)

# ---------------------------------------------------------------------------
# Custom CSS — expert design system
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* ── Palette ──
       bg-deep:    #0B0F19
       bg-surface: #111827
       bg-card:    #1A1F2E
       bg-hover:   #232839
       border:     #2A2F3E
       text-1:     #F1F3F5
       text-2:     #A0A8B8
       text-3:     #636B7E
       accent:     #10B981  (emerald-500)
       accent-dim: rgba(16,185,129,0.12)
    ── */

    /* ── App background ── */
    .stApp {
        background: #0B0F19;
        color: #F1F3F5;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #1E2433;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #F1F3F5 !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {
        color: #A0A8B8 !important;
        font-size: 13px !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #1E2433 !important;
        margin: 16px 0 !important;
    }

    /* ── Sidebar metrics ── */
    section[data-testid="stSidebar"] [data-testid="stMetric"] {
        background: #1A1F2E;
        border: 1px solid #2A2F3E;
        border-radius: 8px;
        padding: 14px 16px;
    }
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #10B981 !important;
        font-weight: 700 !important;
        font-size: 24px !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #636B7E !important;
        font-size: 10px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600 !important;
    }

    /* ── Sidebar buttons ── */
    section[data-testid="stSidebar"] button[kind="secondary"] {
        background: #1A1F2E !important;
        color: #F1F3F5 !important;
        border: 1px solid #2A2F3E !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 13px !important;
        transition: all 0.15s ease !important;
    }
    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: #232839 !important;
        border-color: #10B981 !important;
        color: #10B981 !important;
    }

    /* ── Chat: user bubble ── */
    .chat-user {
        background: #1A1F2E;
        color: #F1F3F5;
        border: 1px solid #2A2F3E;
        border-radius: 16px 16px 4px 16px;
        padding: 14px 18px;
        margin: 6px 0;
        max-width: 72%;
        margin-left: auto;
        font-size: 14px;
        line-height: 1.6;
    }

    /* ── Chat: assistant bubble ── */
    .chat-bot {
        background: transparent;
        color: #E5E7EB;
        padding: 16px 0;
        margin: 4px 0;
        font-size: 14px;
        line-height: 1.7;
        border-bottom: 1px solid #1E2433;
    }
    .chat-bot strong, .chat-bot b { color: #F1F3F5; }
    .chat-bot code {
        background: #1A1F2E;
        color: #10B981;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 12.5px;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Metadata strip ── */
    .meta-strip {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 8px;
        margin: 8px 0 4px 0;
        padding: 8px 0;
        border-top: 1px solid #1E2433;
    }

    /* ── Source chip ── */
    .src-chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: #1A1F2E;
        color: #A0A8B8;
        font-size: 11px;
        font-weight: 500;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #2A2F3E;
        font-family: 'JetBrains Mono', monospace;
    }
    .src-chip::before {
        content: '›';
        color: #10B981;
        font-weight: 700;
    }

    /* ── Query type pill ── */
    .q-pill {
        display: inline-block;
        font-size: 10px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-family: 'JetBrains Mono', monospace;
    }
    .pill-conceptual    { background: #022c22; color: #6ee7b7; }
    .pill-how-to        { background: #0c1f3d; color: #7dd3fc; }
    .pill-troubleshoot  { background: #2d0f0f; color: #fca5a5; }
    .pill-api-ref       { background: #1a0533; color: #c4b5fd; }
    .pill-oos           { background: #1A1F2E; color: #636B7E; }

    /* ── Web search ── */
    .web-chip {
        display: inline-flex;
        align-items: center;
        gap: 3px;
        background: #0c1f3d;
        color: #7dd3fc;
        font-size: 10px;
        font-weight: 600;
        padding: 3px 10px;
        border-radius: 4px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-family: 'JetBrains Mono', monospace;
    }

    /* ── Confidence meter ── */
    .conf-track {
        flex: 1;
        background: #1A1F2E;
        border-radius: 3px;
        height: 4px;
        overflow: hidden;
    }
    .conf-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }

    /* ── Feedback ── */
    div[data-testid="stHorizontalBlock"] button {
        border-radius: 6px !important;
        font-size: 14px !important;
        padding: 4px 12px !important;
        background: transparent !important;
        border: 1px solid #2A2F3E !important;
        transition: all 0.12s ease !important;
    }
    div[data-testid="stHorizontalBlock"] button:hover {
        border-color: #10B981 !important;
        background: rgba(16,185,129,0.06) !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] > div {
        border-radius: 12px !important;
        border: 1px solid #2A2F3E !important;
        background: #111827 !important;
    }
    [data-testid="stChatInput"] textarea {
        color: #F1F3F5 !important;
        font-size: 14px !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #636B7E !important;
    }

    /* ── File uploader ── */
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
        border: 1px dashed #2A2F3E !important;
        border-radius: 8px;
        background: rgba(16,185,129,0.02);
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"]:hover {
        border-color: rgba(16,185,129,0.3) !important;
    }

    /* ── Expander ── */
    section[data-testid="stSidebar"] details {
        border: 1px solid #1E2433 !important;
        border-radius: 8px !important;
        background: #1A1F2E !important;
    }
    section[data-testid="stSidebar"] details summary {
        color: #636B7E !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }

    /* ── Spinner ── */
    .stSpinner > div > div {
        border-top-color: #10B981 !important;
    }

    /* ── Code blocks in sidebar ── */
    section[data-testid="stSidebar"] code {
        background: #0B0F19 !important;
        color: #636B7E !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
        border: 1px solid #1E2433 !important;
    }

    /* ── Hide chrome ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: #2A2F3E;
        border-radius: 3px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #3A4050;
    }

    /* ── Welcome card ── */
    .welcome-card {
        text-align: center;
        padding: 80px 40px 60px 40px;
        max-width: 560px;
        margin: 0 auto;
    }
    .welcome-card h1 {
        font-size: 26px;
        font-weight: 700;
        color: #F1F3F5;
        margin: 0 0 8px 0;
        letter-spacing: -0.02em;
    }
    .welcome-card p {
        color: #636B7E;
        font-size: 14px;
        line-height: 1.6;
        margin: 0;
    }
    .welcome-hints {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-top: 32px;
        text-align: left;
    }
    .hint-card {
        background: #111827;
        border: 1px solid #1E2433;
        border-radius: 8px;
        padding: 14px 16px;
        cursor: default;
        transition: border-color 0.15s ease;
    }
    .hint-card:hover {
        border-color: #2A2F3E;
    }
    .hint-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #636B7E;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .hint-text {
        font-size: 13px;
        color: #A0A8B8;
        line-height: 1.45;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# Session state initialisation
# ===========================================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    # Each message: {role, content, metadata (optional), feedback (None|"up"|"down")}
    st.session_state.messages = []

if "ingesting" not in st.session_state:
    st.session_state.ingesting = False

# Incrementing this key forces st.file_uploader to remount with no files
# selected — the only reliable way to reset a Streamlit file-uploader widget.
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Stores per-file ingest results shown in the sidebar after rerun
if "last_ingest_results" not in st.session_state:
    st.session_state.last_ingest_results = []


# ===========================================================================
# API helpers
# ===========================================================================


def _api_get(path: str, params: dict | None = None) -> dict | None:
    """Synchronous GET against the FastAPI backend."""
    try:
        r = httpx.get(f"{API_BASE_URL}{path}", params=params, timeout=10.0)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        return {"_error": str(exc)}


def _api_post(path: str, json_body: dict | None = None, **kwargs) -> dict | None:
    """Synchronous POST against the FastAPI backend."""
    try:
        r = httpx.post(
            f"{API_BASE_URL}{path}",
            json=json_body,
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", str(exc))
        return {"_error": detail, "_status": exc.response.status_code}
    except Exception as exc:
        return {"_error": str(exc)}


def _get_health() -> dict:
    data = _api_get("/health")
    return data or {}


def _get_doc_count() -> int:
    h = _get_health()
    return h.get("documents_indexed", 0)


def _submit_feedback(
    question: str,
    answer: str,
    session_id: str,
    rating: str,
    comment: str = "",
) -> bool:
    payload = {
        "question": question,
        "answer": answer,
        "session_id": session_id,
        "rating": rating,
        "comment": comment or None,
    }
    result = _api_post("/feedback", json_body=payload)
    return "_error" not in (result or {})


# ===========================================================================
# Rendering helpers
# ===========================================================================


_PILL_MAP = {
    "conceptual": "pill-conceptual",
    "how-to": "pill-how-to",
    "troubleshooting": "pill-troubleshoot",
    "api-reference": "pill-api-ref",
    "out-of-scope": "pill-oos",
}


def _render_query_pill(query_type: str) -> str:
    css = _PILL_MAP.get(query_type, "pill-conceptual")
    return f'<span class="q-pill {css}">{query_type}</span>'


def _render_sources(sources: list[str]) -> str:
    if not sources:
        return ""
    chips = "".join(f'<span class="src-chip">{s}</span>' for s in sources)
    return f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;">{chips}</div>'


def _render_assistant_message(msg: dict, idx: int) -> None:
    """Render an assistant message with metadata and feedback controls."""
    meta = msg.get("metadata", {})
    content = msg["content"]

    # ── Message body ──────────────────────────────────────────────────
    st.markdown(
        f'<div class="chat-bot">{content}',
        unsafe_allow_html=True,
    )

    # ── Metadata strip ────────────────────────────────────────────────
    if meta:
        query_type = meta.get("query_type", "")
        sources = meta.get("sources", [])
        confidence = meta.get("confidence_score", 0.0)
        web_used = meta.get("web_search_used", False)
        latency = meta.get("latency_ms", 0.0)

        # Build inline metadata
        parts = []
        if query_type:
            parts.append(_render_query_pill(query_type))
        if web_used:
            parts.append('<span class="web-chip">web</span>')
        if latency > 0:
            parts.append(
                f'<span style="font-size:11px;color:#636B7E;'
                f'font-family:JetBrains Mono,monospace;">{latency:.0f}ms</span>'
            )

        # Confidence
        if confidence > 0:
            conf_pct = int(confidence * 100)
            bar_color = (
                "#10B981" if confidence >= 0.7
                else "#F59E0B" if confidence >= 0.4
                else "#EF4444"
            )
            parts.append(
                f'<span style="font-size:11px;color:#636B7E;'
                f'font-family:JetBrains Mono,monospace;">{conf_pct}%</span>'
                f'<div class="conf-track">'
                f'<div class="conf-fill" style="width:{conf_pct}%;background:{bar_color};"></div>'
                f'</div>'
            )

        if parts:
            st.markdown(
                f'<div class="meta-strip">{" ".join(parts)}</div>',
                unsafe_allow_html=True,
            )

        # Sources
        if sources:
            st.markdown(_render_sources(sources), unsafe_allow_html=True)

    # Close the chat-bot div
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Feedback ──────────────────────────────────────────────────────
    feedback_given = msg.get("feedback")
    if feedback_given is None:
        col_up, col_down, col_spacer = st.columns([1, 1, 14])
        with col_up:
            if st.button("👍", key=f"up_{idx}", help="Helpful"):
                if _submit_feedback(
                    question=msg.get("question", ""),
                    answer=content,
                    session_id=st.session_state.session_id,
                    rating="up",
                ):
                    st.session_state.messages[idx]["feedback"] = "up"
                    st.rerun()
        with col_down:
            if st.button("👎", key=f"down_{idx}", help="Needs work"):
                if _submit_feedback(
                    question=msg.get("question", ""),
                    answer=content,
                    session_id=st.session_state.session_id,
                    rating="down",
                ):
                    st.session_state.messages[idx]["feedback"] = "down"
                    st.rerun()
    else:
        label = "Helpful" if feedback_given == "up" else "Needs work"
        st.markdown(
            f'<div style="font-size:11px;color:#636B7E;margin:4px 0 8px 0;">'
            f'Feedback: {label}</div>',
            unsafe_allow_html=True,
        )


# ===========================================================================
# Sidebar
# ===========================================================================


with st.sidebar:
    # ── Brand ─────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="padding:20px 0 8px 0;">
            <div style="
                font-size:15px;
                font-weight:700;
                color:#F1F3F5;
                letter-spacing:-0.01em;
                display:flex;
                align-items:center;
                gap:8px;
            "><span style="
                display:inline-flex;
                align-items:center;
                justify-content:center;
                width:28px;
                height:28px;
                background:#10B981;
                border-radius:6px;
                font-size:14px;
            ">⚡</span> RAG Assistant</div>
            <div style="
                font-size:11px;
                color:#636B7E;
                margin-top:4px;
                margin-left:36px;
                letter-spacing:0.02em;
            ">LangGraph · Groq · ChromaDB</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Health status ─────────────────────────────────────────────────
    health = _get_health()
    if "_error" in health:
        st.error("API offline", icon="🔴")
        api_online = False
    else:
        api_online = True
        vs_ok = health.get("vector_store_initialized", False)
        groq_ok = health.get("groq_connected", False)
        doc_count = health.get("documents_indexed", 0)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Chunks", doc_count)
        with col_b:
            status_val = health.get("status", "error")
            indicator = "●" if status_val == "ok" else "○"
            st.metric("Status", f"{indicator} {status_val.upper()}")

        if not vs_ok:
            st.warning("No documents ingested yet.")
        if not groq_ok:
            st.error("GROQ_API_KEY missing")

    st.divider()

    # ── Document ingestion ────────────────────────────────────────────
    st.markdown("### Upload")

    # The key is an integer stored in session_state; incrementing it forces
    # Streamlit to unmount + remount the widget with zero files selected.
    uploaded = st.file_uploader(
        "Drop files here",
        type=["md", "txt", "pdf", "docx", "html", "csv", "json"],
        accept_multiple_files=True,
        help=".md · .txt · .pdf · .docx · .html · .csv · .json",
        disabled=not api_online,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    # Show results from the previous ingest run (persisted across rerun)
    for _res in st.session_state.last_ingest_results:
        if _res["ok"]:
            st.success(_res["msg"])
        else:
            st.error(_res["msg"])

    if uploaded and st.button("Ingest", disabled=not api_online, use_container_width=True):
        # Clear any previous result banners
        st.session_state.last_ingest_results = []

        # Correct MIME type per file extension — FastAPI uses the filename
        # extension for routing to the right loader, but a correct MIME type
        # avoids any Content-Type sniffing issues.
        _MIME = {
            ".pdf":  "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".html": "text/html",
            ".csv":  "text/csv",
            ".json": "application/json",
            ".md":   "text/markdown",
            ".txt":  "text/plain",
        }

        results = []
        with st.spinner("Processing..."):
            for f in uploaded:
                ext = Path(f.name).suffix.lower()
                mime = _MIME.get(ext, "application/octet-stream")
                try:
                    r = httpx.post(
                        f"{API_BASE_URL}/ingest",
                        # Each file is sent as a separate request so the
                        # response chunk-count is accurate per file.
                        files=[("files", (f.name, f.read(), mime))],
                        timeout=120.0,
                    )
                    r.raise_for_status()
                    data = r.json()
                    results.append({
                        "ok": True,
                        "msg": (
                            f"**{f.name}** — "
                            f"{data['chunks_added']} chunks, "
                            f"{data['time_taken_s']:.1f}s"
                        ),
                    })
                except httpx.HTTPStatusError as exc:
                    try:
                        detail = exc.response.json().get("detail", str(exc))
                    except Exception:
                        detail = str(exc)
                    results.append({
                        "ok": False,
                        "msg": f"**{f.name}** failed ({exc.response.status_code}): {detail}",
                    })
                except Exception as exc:
                    results.append({
                        "ok": False,
                        "msg": f"**{f.name}** failed: {exc}",
                    })

        # Persist results so they survive the rerun
        st.session_state.last_ingest_results = results

        # Reset the file uploader by bumping its key — this clears the widget
        st.session_state.uploader_key += 1

        # Rerun: sidebar will re-fetch /health → chunk count updates automatically
        st.rerun()


    st.divider()

    # ── Session ───────────────────────────────────────────────────────
    st.markdown("### Session")
    st.code(st.session_state.session_id[:18] + "…", language=None)
    if st.button("New conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # ── Model info ────────────────────────────────────────────────────
    with st.expander("Config", expanded=False):
        st.markdown(
            """
            | | |
            |---|---|
            | **LLM** | llama-3.3-70b |
            | **Provider** | Groq |
            | **Retriever** | BM25 + ChromaDB |
            | **Embeddings** | bge-small-en |
            | **Top-K** | 6 |
            """
        )


# ===========================================================================
# Main chat area
# ===========================================================================

# ── Welcome / empty state ──────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
            <h1>What can I help you find?</h1>
            <p>Ask questions about your ingested documents. I'll retrieve relevant chunks, verify my answer, and cite sources.</p>
            <div class="welcome-hints">
                <div class="hint-card">
                    <div class="hint-label">Conceptual</div>
                    <div class="hint-text">What is retrieval-augmented generation?</div>
                </div>
                <div class="hint-card">
                    <div class="hint-label">How-to</div>
                    <div class="hint-text">How do I split documents for a vector store?</div>
                </div>
                <div class="hint-card">
                    <div class="hint-label">Troubleshooting</div>
                    <div class="hint-text">Why are my embeddings returning irrelevant results?</div>
                </div>
                <div class="hint-card">
                    <div class="hint-label">API Reference</div>
                    <div class="hint-text">What parameters does BM25Retriever accept?</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Render message history ─────────────────────────────────────────────────
chat_container = st.container()

with chat_container:
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            _render_assistant_message(msg, i)

# ── Input box (always at bottom) ───────────────────────────────────────────
if not api_online:
    st.info("Start the backend first: `uvicorn main:app --reload`")
else:
    question = st.chat_input(
        placeholder="Ask a question...",
        disabled=not api_online,
    )

    if question:
        # Add user message immediately for responsive UX
        st.session_state.messages.append({"role": "user", "content": question})

        # Build chat history from session state (exclude current question)
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages[:-1]
            if m["role"] in ("user", "assistant")
        ]

        # Call the API
        with st.spinner("Thinking..."):
            payload = {
                "question": question,
                "session_id": st.session_state.session_id,
                "chat_history": chat_history,
            }
            result = _api_post("/query", json_body=payload)

        if result and "_error" not in result:
            status_code = result.get("_status")
            if status_code == 503:
                answer = "No documents ingested yet. Upload files using the sidebar."
                meta = {}
            else:
                answer = result.get("answer", "No answer returned.")
                meta = {
                    "query_type": result.get("query_type", ""),
                    "sources": result.get("sources", []),
                    "confidence_score": result.get("confidence_score", 0.0),
                    "web_search_used": result.get("web_search_used", False),
                    "retry_count": result.get("retry_count", 0),
                    "latency_ms": result.get("latency_ms", 0.0),
                }

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "metadata": meta,
                    "feedback": None,
                    "question": question,  # stored for feedback submission
                }
            )
        else:
            err = (result or {}).get("_error", "Unknown error")
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Error: {err}",
                    "metadata": {},
                    "feedback": None,
                    "question": question,
                }
            )

        st.rerun()
