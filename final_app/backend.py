"""
Backend entry point.

The compiled LangGraph app is assembled in graph.py.
This module re-exports `app` so frontend.py can continue to use
`from backend import app` without change.
"""
from graph import app  # noqa: F401

__all__ = ["app"]
