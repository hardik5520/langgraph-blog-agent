"""
Data models for the blog writing agent.

All Pydantic schemas and the LangGraph State TypedDict live here.
Centralising types in one place avoids circular imports and provides
a single source of truth for the full data contract.
"""
from __future__ import annotations

import operator
from typing import Annotated, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Blog planning types
# ---------------------------------------------------------------------------

class Task(BaseModel):
    """A single section of the blog, assigned to one parallel worker."""
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False   # worker should ground claims in evidence
    requires_citations: bool = False  # worker should include inline citation links
    requires_code: bool = False       # worker should include at least one code snippet


class Plan(BaseModel):
    """Full blog outline produced by the orchestrator node."""
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


# ---------------------------------------------------------------------------
# Research / evidence types
# ---------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    """One piece of web evidence returned by the research node."""
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred; null if unknown
    snippet: Optional[str] = None
    source: Optional[str] = None


class EvidencePack(BaseModel):
    """Wrapper used by the LLM structured-output call in research_node."""
    evidence: List[EvidenceItem] = Field(default_factory=list)


class RouterDecision(BaseModel):
    """Output of the router node — decides research strategy before planning."""
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


# ---------------------------------------------------------------------------
# Image planning types
# ---------------------------------------------------------------------------

class ImageSpec(BaseModel):
    """Spec for one diagram/image to be generated and embedded in the blog."""
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"


class GlobalImagePlan(BaseModel):
    """LLM output from the decide_images node — markdown with placeholders + image specs."""
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Graph-wide state
# ---------------------------------------------------------------------------

class State(TypedDict):
    """
    Shared state that flows through the entire LangGraph pipeline.

    Fields are populated progressively as nodes execute:
      router → (research?) → orchestrator → workers (parallel) → reducer
    """
    topic: str

    # Routing / research phase
    mode: str            # "closed_book" | "hybrid" | "open_book"
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # Recency window — controls how old evidence is allowed to be
    as_of: str           # ISO date string, e.g. "2026-04-04"
    recency_days: int    # open_book=7, hybrid=45, closed_book=3650 (≈ no cutoff)

    # Worker outputs — operator.add merges concurrent writes into a single list
    # Each worker appends one (task_id, section_markdown) tuple
    sections: Annotated[List[tuple[int, str]], operator.add]

    # Reducer / image phase
    merged_md: str              # sections joined in order, before images
    md_with_placeholders: str   # merged_md with [[IMAGE_N]] placeholders inserted
    image_specs: List[dict]     # serialised ImageSpec list

    final: str                  # fully assembled blog markdown (also written to disk)
