"""
Research node — fetches and filters web evidence via Tavily.

Only runs when the router sets needs_research=True.
Results are deduplicated by URL, and in open_book mode they are also
filtered to the recency window that the router determined.
"""
from __future__ import annotations

import os
from datetime import date, timedelta
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from llm import llm
from models import EvidenceItem, EvidencePack, State


RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""


def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    """
    Run a single Tavily query and return raw result dicts.
    Returns an empty list gracefully if the API key is missing or the call fails,
    so the pipeline can continue in degraded mode rather than crashing.
    """
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception:
        return []


def _iso_to_date(s: Optional[str]) -> Optional[date]:
    """Parse an ISO date string to a date object; returns None on any failure."""
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None


def research_node(state: State) -> dict:
    """
    Run all Tavily queries, normalise results through an LLM extractor,
    deduplicate by URL, and optionally filter by recency window.

    Returns:
        evidence — list of EvidenceItem, ready to be passed to the orchestrator
    """
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for q in queries:
        raw.extend(_tavily_search(q, max_results=6))

    if not raw:
        return {"evidence": []}

    # Use the LLM to normalise field names and deduplicate raw search results
    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke(
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(
                content=(
                    f"As-of date: {state['as_of']}\n"
                    f"Recency days: {state['recency_days']}\n\n"
                    f"Raw results:\n{raw}"
                )
            ),
        ]
    )

    # Deduplicate by URL a second time in case the LLM still emitted duplicates
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    evidence = list(dedup.values())

    # For open_book mode, drop any evidence published outside the recency window
    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [
            e for e in evidence
            if (d := _iso_to_date(e.published_at)) and d >= cutoff
        ]

    return {"evidence": evidence}
