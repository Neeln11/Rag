"""
rag_workflow.py
===============
Canonical public module for the Adaptive RAG LangGraph pipeline.

This module re-exports everything from rag_pipeline so that:
  - External callers (main.py, app.py, tests) can use either name.
  - The project folder structure matches the target layout:

      rag-assistant/
      └── rag_workflow.py   ← this file

All implementation lives in rag_pipeline.py.  If you rename the internal
module in future, only this shim needs updating — nothing else changes.

Usage
-----
    from rag_workflow import run_rag, _graph, _checkpointer
"""

# Re-export everything the public API needs
from rag_pipeline import (  # noqa: F401
    RAGState,
    _checkpointer,
    _graph,
    run_rag,
)

__all__ = [
    "RAGState",
    "_checkpointer",
    "_graph",
    "run_rag",
]
