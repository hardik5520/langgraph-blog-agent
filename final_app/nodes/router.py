"""
Router node — the first node in the pipeline.

Examines the topic and decides whether web research is needed before planning,
and which mode controls downstream behaviour.

Modes:
  closed_book  — evergreen content, no web search needed
  hybrid       — mostly evergreen but benefits from recent examples (45-day window)
  open_book    — volatile / news topics; workers must cite sources (7-day window)
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from llm import llm
from models import RouterDecision, State

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""


def router_node(state: State) -> dict:
    """
    Classify the topic and return routing metadata.

    Returns:
        needs_research  — whether to run the research node next
        mode            — "closed_book" | "hybrid" | "open_book"
        queries         — search queries for Tavily (empty if closed_book)
        recency_days    — lookback window used to filter evidence downstream
    """
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    # Map each mode to a concrete recency window in days
    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650  # effectively "no cutoff" for evergreen content

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }


def route_next(state: State) -> str:
    """Conditional edge: skip the research node when the topic doesn't need it."""
    return "research" if state["needs_research"] else "orchestrator"
