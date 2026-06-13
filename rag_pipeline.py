"""
rag_pipeline.py
===============
Advanced Self-Corrective Adaptive RAG pipeline built with LangGraph.

Architecture
------------
The graph implements a 7-node self-corrective loop:

  query_analysis ──out-of-scope──► END
       │
       └─► hybrid_retrieval ──► document_grading
                                      │
                      ┌───── relevant─┘
                      │        │
                      │   no-relevant ──retry< 2──► query_analysis
                      │        │
                      │   no-relevant ──retry>=2─► web_search_fallback
                      │                                   │
                      └──────────────────────────────►  generation
                                                          │
                                                   hallucination_checker
                                                          │
                                       hallucinated+retry──► generation
                                                │
                                       hallucinated+exhausted──► END
                                                │
                                          answer_grader
                                                │
                                   poor+retry──► query_analysis
                                                │
                                              END

Usage
-----
    import asyncio
    from rag_pipeline import run_rag

    result = asyncio.run(run_rag(
        question="How does the BM25 retriever work?",
        session_id="user-abc-123",
        chat_history=[],
    ))
    print(result["answer"])

Environment Variables
---------------------
    GROQ_API_KEY         (required) — Groq API key for ChatGroq
    TAVILY_API_KEY       (required for web fallback) — Tavily search key
    LANGCHAIN_API_KEY    (optional) — enable LangSmith tracing
    LANGCHAIN_TRACING_V2 (optional) — set "true" to enable tracing
    LANGCHAIN_PROJECT    (optional) — LangSmith project name
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import os
import re
import sys
import uuid
from typing import Annotated, Literal

# ---------------------------------------------------------------------------
# Force UTF-8 stdout on Windows
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# LangSmith optional tracing — must be set BEFORE any LangChain import
# ---------------------------------------------------------------------------
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "adaptive-rag-pipeline")
    print("✓ LangSmith tracing enabled")

# ---------------------------------------------------------------------------
# LangChain / LangGraph imports
# ---------------------------------------------------------------------------
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Retriever (from existing ingestion pipeline)
# ---------------------------------------------------------------------------
from retriever import load_retriever
from config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR

# ---------------------------------------------------------------------------
# Tavily web search (optional — only used when local retrieval is exhausted)
# ---------------------------------------------------------------------------
try:
    from langchain_community.tools import TavilySearchResults

    _TAVILY_AVAILABLE = bool(os.getenv("TAVILY_API_KEY"))
except ImportError:
    _TAVILY_AVAILABLE = False

# ===========================================================================
# State Schema
# ===========================================================================


class RAGState(TypedDict):
    """Full state passed between every node in the LangGraph pipeline."""

    question: str
    rewritten_query: str
    query_type: str  # "conceptual" | "how-to" | "troubleshooting" | "api-reference" | "out-of-scope"
    documents: list[Document]
    relevant_docs: list[Document]
    answer: str
    sources: list[str]
    confidence_score: float  # 0.0–1.0
    hallucination_flag: bool
    retry_count: int
    web_search_used: bool
    chat_history: list[dict]  # [{"role": "user"|"assistant", "content": "..."}]
    session_id: str
    generation_attempts: int


# ===========================================================================
# LLM Factory
# ===========================================================================

_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not _GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY environment variable is not set.\n"
        "Export it before running: set GROQ_API_KEY=gsk_..."
    )

# Grading / classification LLM — deterministic (temp=0)
_llm_grader = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=_GROQ_API_KEY,
)

# Generation LLM — slight creativity (temp=0.3)
_llm_gen = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=_GROQ_API_KEY,
)

# ===========================================================================
# Retriever (lazy-loaded singleton so import doesn't trigger disk I/O)
# ===========================================================================

_retriever = None
_RETRIEVER_K = 6  # top-k chunks per query (override for this pipeline)


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = load_retriever(k=_RETRIEVER_K)
    return _retriever


def invalidate_retriever() -> None:
    """Reset the retriever singleton so the next query reloads from disk.

    Call this from main.py immediately after a successful /ingest so that
    newly ingested documents are visible to subsequent queries.
    """
    global _retriever
    _retriever = None
    print("[retriever] Singleton invalidated — will reload from disk on next query.")


# ===========================================================================
# Utility helpers
# ===========================================================================


def _call_grader(system_prompt: str, human_msg: str) -> str:
    """Call the grader LLM and return the stripped text response."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_msg),
    ]
    response = _llm_grader.invoke(messages)
    return response.content.strip()


def _parse_json_block(text: str) -> dict:
    """
    Extract and parse the first JSON object found in *text*.

    Handles responses where the LLM wraps JSON in markdown code fences
    (```json ... ```) or returns it bare.
    """
    # Strip markdown fences if present
    clean = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()

    # Find first {...} block
    match = re.search(r"\{.*?\}", clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Fallback: attempt to parse the whole cleaned string
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {}


def _extract_confidence(text: str) -> float:
    """
    Parse the 'Confidence: N/10' line from the generated answer and
    normalise to a 0.0–1.0 float.  Returns 0.5 if no match found.
    """
    match = re.search(r"Confidence:\s*(\d+(?:\.\d+)?)\s*/\s*10", text, re.IGNORECASE)
    if match:
        raw = float(match.group(1))
        return min(max(raw / 10.0, 0.0), 1.0)
    return 0.5


def _sources_from_docs(docs: list[Document]) -> list[str]:
    """Build a deduplicated list of 'filename.ext (chunk N)' citation strings."""
    seen: set[str] = set()
    sources: list[str] = []
    for doc in docs:
        meta = doc.metadata
        src = meta.get("source", "unknown")
        chunk = meta.get("chunk_index", "?")
        citation = f"{src} (chunk {chunk})"
        if citation not in seen:
            seen.add(citation)
            sources.append(citation)
    return sources


# ===========================================================================
# Node 1 — query_analysis
# ===========================================================================

_QUERY_ANALYSIS_SYSTEM = """\
You are a search query optimizer for a RAG retrieval system.

Given a user question and optional chat history, your tasks are:
1. Rewrite the query to improve retrieval quality:
   - Keep the CORE KEYWORDS from the original question (do NOT replace them with abstractions)
   - Expand acronyms if helpful, but keep original terms too
   - Remove filler words only ("um", "please", "like")
   - If the user mentions a specific document or file, keep that reference
   - IMPORTANT: The rewritten query will be used for BOTH vector search AND BM25 keyword search
     so concrete nouns and domain terms are more valuable than abstract paraphrases

2. Classify the query into EXACTLY ONE of these types:
   - "conceptual"      — theory, definitions, "what is", "why does"
   - "how-to"          — step-by-step instructions, "how do I", "how to"
   - "troubleshooting" — errors, failures, debugging, "not working"
   - "api-reference"   — parameters, return values, function signatures, class methods
   - "out-of-scope"    — ONLY if clearly unrelated to ANY documents (e.g. weather, sports, cooking)
                         Do NOT classify as out-of-scope just because the topic seems unusual.

Respond ONLY with valid JSON (no markdown fences, no extra text):
{
  "rewritten_query": "<improved query — keep original keywords>",
  "query_type": "<one of the 5 types above>"
}
"""


def query_analysis(state: RAGState) -> RAGState:
    """
    Node 1: Rewrite the query for better retrieval and classify its type.

    If the query is out-of-scope, sets the answer immediately so the
    conditional edge can route straight to END without further processing.
    """
    print(f"\n[Node 1] query_analysis — question: {state['question']!r}")

    # Build context string from chat history (last 4 turns max)
    history_ctx = ""
    if state.get("chat_history"):
        recent = state["chat_history"][-4:]
        history_ctx = "\n".join(
            f"{turn['role'].upper()}: {turn['content']}" for turn in recent
        )

    human_content = f"User Question: {state['question']}"
    if history_ctx:
        human_content = f"Chat History:\n{history_ctx}\n\n{human_content}"

    raw = _call_grader(_QUERY_ANALYSIS_SYSTEM, human_content)
    parsed = _parse_json_block(raw)

    rewritten = parsed.get("rewritten_query", state["question"])
    query_type = parsed.get("query_type", "conceptual")

    # Sanitise query_type to allowed values
    valid_types = {"conceptual", "how-to", "troubleshooting", "api-reference", "out-of-scope"}
    if query_type not in valid_types:
        query_type = "conceptual"

    print(f"  rewritten_query: {rewritten!r}")
    print(f"  query_type     : {query_type}")

    updates: dict = {
        "rewritten_query": rewritten,
        "query_type": query_type,
    }

    # If out-of-scope, set a polite answer now — edge will route to END
    if query_type == "out-of-scope":
        updates["answer"] = (
            "I'm a technical documentation assistant and can only help with "
            "conceptual questions, how-to guides, troubleshooting, and API "
            "reference topics related to the loaded documents. Your question "
            "appears to be outside that scope. Please rephrase or ask a "
            "different question."
        )
        updates["confidence_score"] = 1.0
        updates["hallucination_flag"] = False
        updates["sources"] = []

    return {**state, **updates}


# ===========================================================================
# Node 2 — hybrid_retrieval
# ===========================================================================


def hybrid_retrieval(state: RAGState) -> RAGState:
    """
    Node 2: Retrieve top-K chunks using the EnsembleRetriever (ChromaDB + BM25).

    Raises a RuntimeError (503-style) if ChromaDB is empty.
    """
    print(f"\n[Node 2] hybrid_retrieval — query: {state['rewritten_query']!r}")

    retriever = _get_retriever()

    # ── Debug: confirm which collection we're searching and its size ────────
    try:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        import chromadb as _chromadb
        _client = _chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        _col = _client.get_collection(CHROMA_COLLECTION_NAME)
        _count = _col.count()
        print(
            f"  [debug] Searching collection={CHROMA_COLLECTION_NAME!r} "
            f"at {CHROMA_PERSIST_DIR!r} — total_docs={_count}"
        )
    except Exception as _e:
        print(f"  [debug] Could not inspect collection: {_e}")

    docs: list[Document] = retriever.invoke(state["rewritten_query"])

    if not docs:
        # ChromaDB/BM25 returned nothing — surface a clear error
        raise RuntimeError(
            "503: Retrieval system returned 0 documents. "
            "The ChromaDB collection may be empty. "
            "Run `python ingest.py --docs ./docs` to populate it."
        )

    print(f"  Retrieved {len(docs)} chunk(s)")
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get('source', '?')
        chunk_idx = doc.metadata.get('chunk_index', '?')
        print(f"    [{i}] {src} (chunk {chunk_idx})")
    return {**state, "documents": docs}


# ===========================================================================
# Node 3 — document_grading
# ===========================================================================

_DOC_GRADE_SYSTEM = """\
You are a document relevance grader for a RAG system.

Assess whether the provided document chunk is relevant to the user's question.

A chunk is RELEVANT if ANY of the following apply:
- It directly answers the question
- It contains information that partially helps answer the question
- It mentions the same topic, concept, entity, or document being asked about
- It provides useful context for answering

A chunk is IRRELEVANT only if it is completely unrelated to the question topic.
Do NOT mark a chunk as irrelevant just because it doesn't fully answer the question.
When in doubt, mark it as RELEVANT.

Respond with EXACTLY one word, either:
  relevant
  irrelevant

No explanation, no punctuation, no JSON.
"""


def document_grading(state: RAGState) -> RAGState:
    """
    Node 3: Grade each retrieved document for relevance.

    Updates relevant_docs and confidence_score.
    retry_count is incremented here when 0 relevant docs are found and we
    haven't exhausted retries — the conditional edge handles actual routing.
    """
    print(f"\n[Node 3] document_grading — {len(state['documents'])} doc(s) to grade")

    relevant: list[Document] = []
    for i, doc in enumerate(state["documents"]):
        # Use BOTH original question AND rewritten query for grading
        # so the grader gets the best chance to match the chunk content.
        human_msg = (
            f"User's original question: {state['question']}\n"
            f"Search query used: {state['rewritten_query']}\n\n"
            f"Document Chunk:\n{doc.page_content}"
        )
        verdict = _call_grader(_DOC_GRADE_SYSTEM, human_msg).lower()
        is_relevant = verdict.startswith("relevant")
        print(f"  Chunk {i+1}: {'✓' if is_relevant else '✗'} ({verdict[:20]})")
        if is_relevant:
            relevant.append(doc)

    total = len(state["documents"])
    relevant_count = len(relevant)
    confidence = relevant_count / total if total > 0 else 0.0

    print(f"  Relevant: {relevant_count}/{total} → confidence_score={confidence:.2f}")

    updates: dict = {
        "relevant_docs": relevant,
        "confidence_score": confidence,
    }

    # If no relevant docs, pre-increment retry_count so edge logic is clean
    if relevant_count == 0:
        updates["retry_count"] = state.get("retry_count", 0) + 1
        print(f"  No relevant docs — retry_count now {updates['retry_count']}")

    return {**state, **updates}


# ===========================================================================
# Node 4 — web_search_fallback
# ===========================================================================


def web_search_fallback(state: RAGState) -> RAGState:
    """
    Node 4: Use Tavily to search the web when local retrieval is exhausted.

    Converts Tavily results into Document objects so the rest of the pipeline
    can treat them identically to ChromaDB/BM25 chunks.
    """
    print(f"\n[Node 4] web_search_fallback — query: {state['rewritten_query']!r}")

    if not _TAVILY_AVAILABLE:
        print("  WARNING: Tavily not available. Skipping web search.")
        fallback_doc = Document(
            page_content=(
                "No relevant local documentation was found and web search "
                "is not configured (set TAVILY_API_KEY to enable it)."
            ),
            metadata={"source": "system_fallback", "chunk_index": 0},
        )
        return {
            **state,
            "relevant_docs": [fallback_doc],
            "web_search_used": True,
        }

    tool = TavilySearchResults(
        max_results=4,
        api_key=os.environ["TAVILY_API_KEY"],
    )
    results = tool.invoke({"query": state["rewritten_query"]})

    web_docs: list[Document] = []
    for r in results:
        web_docs.append(
            Document(
                page_content=r.get("content", ""),
                metadata={
                    "source": r.get("url", "web_search"),
                    "chunk_index": 0,
                    "doc_type": "web_search",
                },
            )
        )

    print(f"  Web search returned {len(web_docs)} result(s)")
    return {
        **state,
        "relevant_docs": web_docs,
        "web_search_used": True,
    }


# ===========================================================================
# Node 5 — generation
# ===========================================================================

_GENERATION_SYSTEM_BASE = """\
You are a precise technical documentation assistant.

Rules you MUST follow:
1. Answer ONLY from the provided context. Do not add information that is not in the context.
2. For "how-to" queries use numbered steps; for other queries use bullet points or prose as appropriate.
3. Add inline citations after every factual claim: [Source: filename.md, chunk N]
4. End your response with a "Confidence: N/10" line (N is an integer 1-10).
5. If the context is insufficient to fully answer the question, say so explicitly — NEVER fabricate information.
6. Keep the answer concise and well-structured.
"""

_GENERATION_SYSTEM_STRICT = _GENERATION_SYSTEM_BASE + """
IMPORTANT: A previous answer was flagged as containing information not grounded
in the provided context. Be even more conservative this time. If you cannot
find a direct supporting statement in the context, omit that claim entirely.
"""


def generation(state: RAGState) -> RAGState:
    """
    Node 5: Generate the final answer using the LLM with grounded context.

    Switches to a stricter system prompt on retry attempts.
    """
    attempts = state.get("generation_attempts", 0)
    print(f"\n[Node 5] generation — attempt #{attempts + 1}")

    # Choose system prompt based on whether this is a retry
    system_prompt = _GENERATION_SYSTEM_STRICT if attempts > 0 else _GENERATION_SYSTEM_BASE

    # Build context block
    docs = state.get("relevant_docs", [])
    context_parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        src = doc.metadata.get("source", "unknown")
        chunk = doc.metadata.get("chunk_index", "?")
        context_parts.append(
            f"[Context {i}] Source: {src}, chunk {chunk}\n{doc.page_content}"
        )
    context_block = "\n\n---\n\n".join(context_parts)

    human_content = (
        f"User Question: {state['question']}\n\n"
        f"Context:\n{context_block}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content),
    ]
    response = _llm_gen.invoke(messages)
    answer_text = response.content.strip()

    confidence = _extract_confidence(answer_text)
    sources = _sources_from_docs(docs)

    print(f"  Generated {len(answer_text)} chars — confidence={confidence:.2f}")

    return {
        **state,
        "answer": answer_text,
        "sources": sources,
        "confidence_score": confidence,
        "generation_attempts": attempts + 1,
    }


# ===========================================================================
# Node 6 — hallucination_checker
# ===========================================================================

_HALLUCINATION_SYSTEM = """\
You are a hallucination detection expert for a RAG system.

Your job: verify that every factual claim in the generated answer is directly
supported by the provided context documents. An answer is "hallucinated" if it
states facts, figures, or explanations that cannot be traced to any provided
context chunk.

Respond with EXACTLY one word:
  grounded
  hallucinated

No explanation, no JSON, no punctuation.
"""


def hallucination_checker(state: RAGState) -> RAGState:
    """
    Node 6: Verify that the generated answer does not fabricate information.

    Sets hallucination_flag and may increment generation_attempts for routing.
    """
    print(f"\n[Node 6] hallucination_checker")

    docs = state.get("relevant_docs", [])
    context = "\n\n".join(d.page_content for d in docs)

    human_msg = (
        f"Context Documents:\n{context}\n\n"
        f"Generated Answer:\n{state['answer']}"
    )

    verdict = _call_grader(_HALLUCINATION_SYSTEM, human_msg).lower()
    is_hallucinated = verdict.startswith("hallucinated")

    print(f"  Verdict: {verdict!r} — hallucination_flag={is_hallucinated}")

    updates: dict = {"hallucination_flag": is_hallucinated}

    if is_hallucinated and state.get("generation_attempts", 0) >= 2:
        # Exhausted retries — replace answer with safe fallback
        updates["answer"] = (
            "I was unable to generate a reliable, grounded answer for your question "
            "after multiple attempts. Please try rephrasing your question or check "
            "whether the relevant documentation has been ingested into the system."
        )
        updates["confidence_score"] = 0.0
        print("  ⚠ Hallucination retry limit reached — using safe fallback")

    return {**state, **updates}


# ===========================================================================
# Node 7 — answer_grader
# ===========================================================================

_ANSWER_GRADE_SYSTEM = """\
You are an answer quality evaluator for a RAG system.

Score the answer on a 1-5 scale based on how well it resolves the user's question:
  5 — Complete, accurate, directly addresses the question
  4 — Mostly answers the question with minor gaps
  3 — Partially answers the question
  2 — Barely relevant to the question
  1 — Does not answer the question at all

Respond ONLY with valid JSON:
{
  "score": <integer 1-5>,
  "reason": "<brief 1-sentence reason>"
}
"""


def answer_grader(state: RAGState) -> RAGState:
    """
    Node 7: Grade whether the generated answer actually resolves the question.

    Scores 1-5; if score <= 2 and we haven't retried, flags for a full retry
    via the conditional edge.
    """
    print(f"\n[Node 7] answer_grader")

    human_msg = (
        f"User Question: {state['question']}\n\n"
        f"Generated Answer:\n{state['answer']}"
    )

    raw = _call_grader(_ANSWER_GRADE_SYSTEM, human_msg)
    parsed = _parse_json_block(raw)
    score = int(parsed.get("score", 3))
    reason = parsed.get("reason", "")

    print(f"  Answer quality score: {score}/5 — {reason}")

    # Store score in confidence_score if retrieval gave 0.5 default
    # (blend: give grader score some weight)
    current_conf = state.get("confidence_score", 0.5)
    blended_conf = (current_conf + (score / 5.0)) / 2.0

    updates: dict = {
        "confidence_score": blended_conf,
    }

    # If score is poor and we haven't done a full query retry yet, signal retry
    # The edge will route back to query_analysis.
    # We piggyback on retry_count: we set it to a sentinel that the edge reads.
    if score <= 2 and state.get("retry_count", 0) < 1:
        # Set retry_count = -1 as a sentinel: "grader wants a full retry"
        # query_analysis will reset documents and relevant_docs on re-entry
        updates["retry_count"] = state.get("retry_count", 0) + 1
        updates["answer"] = ""  # Clear stale answer before retry
        print("  ⚠ Answer quality poor — flagging for full retry")

    return {**state, **updates}


# ===========================================================================
# Conditional edge functions
# ===========================================================================


def _route_after_query_analysis(state: RAGState) -> Literal["hybrid_retrieval", "__end__"]:
    """After query_analysis: short-circuit to END for out-of-scope queries."""
    if state.get("query_type") == "out-of-scope":
        print("  ↪ Edge: out-of-scope → END")
        return "__end__"
    print("  ↪ Edge: → hybrid_retrieval")
    return "hybrid_retrieval"


def _route_after_document_grading(
    state: RAGState,
) -> Literal["generation", "query_analysis", "web_search_fallback"]:
    """
    After document_grading:
    - Relevant docs found       → generation
    - No relevant, retry < 2    → query_analysis (retry_count was pre-incremented)
    - No relevant, retry >= 2   → web_search_fallback
    """
    relevant = state.get("relevant_docs", [])
    retry = state.get("retry_count", 0)

    if relevant:
        print("  ↪ Edge: relevant docs → generation")
        return "generation"
    elif retry < 2:
        print(f"  ↪ Edge: no relevant (retry_count={retry}) → query_analysis")
        return "query_analysis"
    else:
        print(f"  ↪ Edge: no relevant (retry_count={retry}, exhausted) → web_search_fallback")
        return "web_search_fallback"


def _route_after_hallucination_checker(
    state: RAGState,
) -> Literal["generation", "answer_grader", "__end__"]:
    """
    After hallucination_checker:
    - Grounded                        → answer_grader
    - Hallucinated + attempts < 2     → generation (retry with stricter prompt)
    - Hallucinated + attempts >= 2    → END (safe fallback already set)
    """
    is_hallucinated = state.get("hallucination_flag", False)
    attempts = state.get("generation_attempts", 0)

    if not is_hallucinated:
        print("  ↪ Edge: grounded → answer_grader")
        return "answer_grader"
    elif attempts < 2:
        print(f"  ↪ Edge: hallucinated (attempts={attempts}) → generation")
        return "generation"
    else:
        print(f"  ↪ Edge: hallucinated + exhausted → END")
        return "__end__"


def _route_after_answer_grader(
    state: RAGState,
) -> Literal["query_analysis", "__end__"]:
    """
    After answer_grader:
    - Poor quality (answer cleared + retry flagged) → query_analysis
    - Acceptable → END
    """
    answer = state.get("answer", "")
    retry = state.get("retry_count", 0)

    # If the grader cleared the answer, we're doing a full retry
    if not answer and retry > 0:
        print("  ↪ Edge: poor answer → query_analysis (full retry)")
        return "query_analysis"

    print("  ↪ Edge: answer acceptable → END")
    return "__end__"


# ===========================================================================
# Graph Assembly
# ===========================================================================


def _build_graph() -> StateGraph:
    """Construct and compile the LangGraph StateGraph."""

    builder = StateGraph(RAGState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    builder.add_node("query_analysis", query_analysis)
    builder.add_node("hybrid_retrieval", hybrid_retrieval)
    builder.add_node("document_grading", document_grading)
    builder.add_node("web_search_fallback", web_search_fallback)
    builder.add_node("generation", generation)
    builder.add_node("hallucination_checker", hallucination_checker)
    builder.add_node("answer_grader", answer_grader)

    # ── Entry ──────────────────────────────────────────────────────────────
    builder.add_edge(START, "query_analysis")

    # ── Conditional edge: after query_analysis ─────────────────────────────
    builder.add_conditional_edges(
        "query_analysis",
        _route_after_query_analysis,
        {
            "hybrid_retrieval": "hybrid_retrieval",
            "__end__": END,
        },
    )

    # ── Linear edge: retrieval → grading ───────────────────────────────────
    builder.add_edge("hybrid_retrieval", "document_grading")

    # ── Conditional edge: after document_grading ───────────────────────────
    builder.add_conditional_edges(
        "document_grading",
        _route_after_document_grading,
        {
            "generation": "generation",
            "query_analysis": "query_analysis",
            "web_search_fallback": "web_search_fallback",
        },
    )

    # ── Linear edge: web_search_fallback → generation ──────────────────────
    builder.add_edge("web_search_fallback", "generation")

    # ── Linear edge: generation → hallucination_checker ────────────────────
    builder.add_edge("generation", "hallucination_checker")

    # ── Conditional edge: after hallucination_checker ──────────────────────
    builder.add_conditional_edges(
        "hallucination_checker",
        _route_after_hallucination_checker,
        {
            "generation": "generation",
            "answer_grader": "answer_grader",
            "__end__": END,
        },
    )

    # ── Conditional edge: after answer_grader ──────────────────────────────
    builder.add_conditional_edges(
        "answer_grader",
        _route_after_answer_grader,
        {
            "query_analysis": "query_analysis",
            "__end__": END,
        },
    )

    return builder


# Compile once at module level with MemorySaver for cross-invocation session memory
_checkpointer = MemorySaver()
_graph = _build_graph().compile(checkpointer=_checkpointer)


# ===========================================================================
# Public API
# ===========================================================================


async def run_rag(
    question: str,
    session_id: str,
    chat_history: list[dict] | None = None,
) -> dict:
    """
    Run the Adaptive RAG pipeline for a single question.

    Args:
        question:     The user's natural-language question.
        session_id:   A stable identifier for this conversation session.
                      The MemorySaver checkpointer uses this to persist
                      state across multiple calls in the same conversation.
        chat_history: Optional list of prior turns:
                      [{"role": "user"|"assistant", "content": "..."}]

    Returns:
        A dict with the following keys:
        {
            "answer":           str,   # Final generated answer
            "sources":          list,  # Citation strings
            "query_type":       str,   # Classified query type
            "confidence_score": float, # 0.0–1.0 blended confidence
            "web_search_used":  bool,  # True if Tavily was invoked
            "retry_count":      int,   # Number of retrieval retries
            "hallucination_flag": bool # True if hallucination was detected
        }
    """
    if chat_history is None:
        chat_history = []

    # Initial state — all counters start at 0/False
    initial_state: RAGState = {
        "question": question,
        "rewritten_query": "",
        "query_type": "",
        "documents": [],
        "relevant_docs": [],
        "answer": "",
        "sources": [],
        "confidence_score": 0.0,
        "hallucination_flag": False,
        "retry_count": 0,
        "web_search_used": False,
        "chat_history": chat_history,
        "session_id": session_id,
        "generation_attempts": 0,
    }

    # LangGraph thread config — ties checkpointer state to this session
    config = {"configurable": {"thread_id": session_id}}

    print(f"\n{'='*70}")
    print(f"  Adaptive RAG — session: {session_id}")
    print(f"  Question: {question!r}")
    print(f"{'='*70}")

    # ainvoke runs the graph asynchronously
    final_state: RAGState = await _graph.ainvoke(initial_state, config=config)

    print(f"\n{'='*70}")
    print(f"  Pipeline complete")
    print(f"  query_type      : {final_state.get('query_type')}")
    print(f"  confidence_score: {final_state.get('confidence_score', 0):.2f}")
    print(f"  web_search_used : {final_state.get('web_search_used')}")
    print(f"  retry_count     : {final_state.get('retry_count')}")
    print(f"  hallucination   : {final_state.get('hallucination_flag')}")
    print(f"{'='*70}\n")

    return {
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
        "query_type": final_state.get("query_type", ""),
        "confidence_score": final_state.get("confidence_score", 0.0),
        "web_search_used": final_state.get("web_search_used", False),
        "retry_count": final_state.get("retry_count", 0),
        "hallucination_flag": final_state.get("hallucination_flag", False),
    }


# ===========================================================================
# CLI demo entry point
# ===========================================================================

if __name__ == "__main__":
    import asyncio

    demo_session = str(uuid.uuid4())

    demo_questions = [
        "How does the EnsembleRetriever combine BM25 and ChromaDB results?",
        "How do I ingest new documents into the pipeline?",
        "What embedding model is used and why was it chosen?",
    ]

    async def _demo() -> None:
        for q in demo_questions:
            result = await run_rag(
                question=q,
                session_id=demo_session,
            )
            print(f"\nQ: {q}")
            print(f"Type    : {result['query_type']}")
            print(f"Answer  : {result['answer'][:300]}...")
            print(f"Sources : {result['sources']}")
            print(f"Conf    : {result['confidence_score']:.2f}")
            print(f"Web?    : {result['web_search_used']}")
            print("-" * 60)

    asyncio.run(_demo())
