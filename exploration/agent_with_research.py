from __future__ import annotations

import operator
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
load_dotenv()


# ============================================================
# SCHEMAS (PYDANTIC MODELS)
#
# These define the exact shape of data the LLM must return.
# with_structured_output() forces the LLM to fill these fields
# instead of returning free-form text — like a strict JSON form
# the LLM must fill out completely and correctly.
# ============================================================

class Task(BaseModel):
    """
    Represents ONE section of the blog post.
    The orchestrator creates one Task per section in the plan.
    Each Task is later handed to a worker node to write.
    """
    id: int                          # section number — used to sort sections in correct order at the end
    title: str                       # section heading e.g. "How Attention Works"

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should understand after this section.",
    )
    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3-6 concrete, non-overlapping subpoints the section must cover.",
    )
    target_words: int = Field(..., description="Target word count for this section (120-550).")

    tags: List[str] = Field(default_factory=list)   # e.g. ["attention", "transformers"]
    requires_research: bool = False   # if True, worker should use evidence from web search
    requires_citations: bool = False  # if True, worker must cite sources with URLs
    requires_code: bool = False       # if True, worker must include a code snippet


class Plan(BaseModel):
    """
    The full blog plan produced by the orchestrator node.
    Contains metadata about the blog + a list of Tasks (one per section).
    """
    blog_title: str
    audience: str          # e.g. "intermediate Python developers"
    tone: str              # e.g. "technical but approachable"
    blog_kind: Literal[
        "explainer", "tutorial", "news_roundup", "comparison", "system_design"
    ] = "explainer"        # controls worker behaviour (e.g. news_roundup = no tutorials)
    constraints: List[str] = Field(default_factory=list)   # e.g. ["avoid marketing language"]
    tasks: List[Task]      # one Task per section — 5-9 tasks expected


class EvidenceItem(BaseModel):
    """
    One piece of web search evidence returned by Tavily.
    Workers use these URLs when they need to cite sources.
    """
    title: str
    url: str
    published_at: Optional[str] = None   # YYYY-MM-DD if available; null if unknown
    snippet: Optional[str] = None        # short excerpt from the page
    source: Optional[str] = None         # domain/outlet name


class RouterDecision(BaseModel):
    """
    The routing decision made before planning starts.
    Determines whether web research is needed based on the topic.
    """
    needs_research: bool                 # True = run Tavily search before planning
    mode: Literal[
        "closed_book",   # purely evergreen topic — no web search needed
        "hybrid",        # mostly evergreen but benefits from fresh examples
        "open_book",     # highly time-sensitive (news, rankings, pricing) — research required
    ]
    queries: List[str] = Field(default_factory=list)  # search queries if needs_research=True


class EvidencePack(BaseModel):
    """
    Structured container for evidence items returned by the research node.
    The LLM fills this from raw Tavily results, deduplicating and cleaning as it goes.
    """
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ============================================================
# STATE
#
# The shared whiteboard for the entire graph.
# Every node reads from and/or writes to this.
# Fields are filled progressively as the graph runs:
#
#   topic          → provided by user at the start, never changes
#   mode           → set by router_node (closed_book / hybrid / open_book)
#   needs_research → set by router_node (True/False)
#   queries        → set by router_node (search queries to run)
#   evidence       → set by research_node (list of EvidenceItem from Tavily)
#   plan           → set by orchestrator_node (the full blog Plan)
#   sections       → filled by worker nodes in parallel.
#                    Each worker appends a (task_id, section_md) tuple.
#                    Uses operator.add as reducer — appends, never overwrites.
#                    Stored as tuples so reducer_node can sort by task_id
#                    to restore correct section order regardless of which
#                    worker finishes first.
#   final          → set by reducer_node (complete assembled blog Markdown)
# ============================================================

class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md) tuples
    final: str


# ============================================================
# LLM
#
# Single shared instance used by all nodes.
# gpt-4.1-mini is fast and cost-efficient — important because
# multiple workers call it simultaneously in the fan-out phase.
# ============================================================

llm = ChatOpenAI(model="gpt-4.1-mini")


# ============================================================
# NODE 1: ROUTER
#
# The very first node. Decides the research strategy BEFORE
# any planning or writing happens.
#
# Three possible modes:
#   closed_book → skip research entirely, go straight to planning
#   hybrid      → do research first, use results to enrich planning
#   open_book   → heavy research first, all claims must be cited
#
# Also generates the search queries if research is needed.
# This avoids wasting Tavily credits on evergreen topics.
# ============================================================

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3-10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""

def router_node(state: State) -> dict:
    """
    Reads the topic and decides:
      - needs_research: should we run Tavily before planning?
      - mode: which grounding mode to use (closed_book / hybrid / open_book)
      - queries: what to search for if research is needed

    Returns routing info that the route_next() function uses
    to decide which node runs next (research or orchestrator).
    """
    topic = state["topic"]
    decider = llm.with_structured_output(RouterDecision)
    decision = decider.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {topic}"),
    ])
    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
    }

def route_next(state: State) -> str:
    """
    Conditional edge function that runs after router_node.
    Returns the name of the next node to execute:
      - "research"     if the topic needs web search first
      - "orchestrator" if the topic is evergreen (skip research)
    """
    return "research" if state["needs_research"] else "orchestrator"


# ============================================================
# NODE 2: RESEARCH
#
# Only runs if router_node decided needs_research=True.
# Runs Tavily web searches for each query, then passes
# the raw results back to the LLM to clean, deduplicate,
# and structure into a list of EvidenceItem objects.
#
# The resulting evidence is passed to:
#   - orchestrator_node: to inform the planning (which sections need citations)
#   - worker_node: so workers can cite specific URLs when writing
# ============================================================

def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    tool = TavilySearch(max_results=max_results)
    results = tool.invoke({"query": query})

    # TavilySearch returns list of dicts with 'results' key
    # or directly a list — handle both formats
    if isinstance(results, dict):
        results = results.get("results", [])

    normalized: List[dict] = []
    for r in results or []:
        if isinstance(r, str):
            continue  # skip plain strings
        normalized.append({
            "title":        r.get("title") or "",
            "url":          r.get("url") or "",
            "snippet":      r.get("content") or r.get("snippet") or "",
            "published_at": r.get("published_date") or r.get("published_at"),
            "source":       r.get("source"),
        })
    return normalized

RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    """
    Runs Tavily searches for all queries generated by router_node.
    Passes raw results to the LLM to extract structured EvidenceItem objects.
    Deduplicates evidence by URL before storing in state.

    If no results are found, returns an empty evidence list —
    the orchestrator and workers handle this gracefully.
    """
    queries = state.get("queries", []) or []
    max_results = 6

    # Run all queries and collect raw results
    raw_results: List[dict] = []
    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=max_results))

    if not raw_results:
        return {"evidence": []}

    # Ask LLM to clean and structure the raw results
    extractor = llm.with_structured_output(EvidencePack)
    pack = extractor.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=f"Raw results:\n{raw_results}"),
    ])

    # Final deduplication by URL (in case LLM missed any duplicates)
    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e

    return {"evidence": list(dedup.values())}


# ============================================================
# NODE 3: ORCHESTRATOR
#
# The "manager" node. Runs after research (or directly after
# router if no research needed).
#
# Takes the topic, mode, and evidence (if any) and produces
# a full Plan — the complete blog outline with 5-9 Tasks.
#
# The mode affects how the plan is structured:
#   closed_book → evergreen content only, no evidence used
#   hybrid      → use evidence for fresh examples only
#   open_book   → all sections grounded in evidence, news style
# ============================================================

ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5-9 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3-6 bullets that are concrete, specific, and non-overlapping
  3) target word count (120-550)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    """
    Produces the full blog Plan from the topic, mode, and evidence.
    The Plan contains 5-9 Task objects — one per blog section.
    After this node, fanout() spawns one worker per Task in parallel.
    """
    planner = llm.with_structured_output(Plan)

    evidence = state.get("evidence", [])
    mode     = state.get("mode", "closed_book")

    plan = planner.invoke([
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(content=(
            f"Topic: {state['topic']}\n"
            f"Mode: {mode}\n\n"
            f"Evidence (ONLY use for fresh claims; may be empty):\n"
            f"{[e.model_dump() for e in evidence][:16]}"  # cap at 16 to avoid token overflow
        )),
    ])

    return {"plan": plan}


# ============================================================
# FAN-OUT FUNCTION
#
# Not a node — this is a conditional edge function that runs
# after orchestrator_node.
#
# Returns a list of Send() objects — one per Task in the plan.
# Each Send() spawns a SEPARATE worker_node in PARALLEL,
# passing that task's data as the payload.
#
# If the plan has 7 sections → 7 workers run simultaneously,
# each writing their own section independently.
#
# Each Send() payload is a plain dict (not State) containing
# only what that specific worker needs. Evidence is passed so
# workers can cite URLs when requires_citations=True.
# ============================================================

def fanout(state: State):
    return [
        Send("worker", {
            "task":     task.model_dump(),   # convert Pydantic → dict for Send()
            "topic":    state["topic"],
            "mode":     state["mode"],
            "plan":     state["plan"].model_dump(),
            "evidence": [e.model_dump() for e in state.get("evidence", [])],
        })
        for task in state["plan"].tasks
    ]


# ============================================================
# NODE 4: WORKER
#
# Runs in parallel — one instance per Task from the plan.
# Each worker writes exactly ONE section in Markdown.
#
# Behaviour is controlled by flags on the Task:
#   requires_code      → must include a code snippet
#   requires_citations → must cite evidence URLs inline
#   requires_research  → section is about fresh/current information
#
# The mode also controls grounding:
#   open_book → ONLY use evidence URLs, no invented facts
#   hybrid    → use evidence for fresh claims, evergreen reasoning otherwise
#   closed_book → no citations needed
#
# Returns {"sections": [(task.id, section_md)]} — a list with one
# tuple. operator.add appends this to all other workers' results
# in state["sections"]. The task.id is stored so reducer_node
# can sort sections back into the correct order at the end.
# ============================================================

WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (+-15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URLs.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""

def worker_node(payload: dict) -> dict:
    """
    Receives a single Task and writes that section in Markdown.
    Reconstructs Pydantic objects from the plain dicts sent by Send().

    Passes the full plan context (title, audience, tone, blog_kind)
    and all evidence URLs to the LLM so it can:
      - Match the overall blog style
      - Cite correct URLs when required
      - Avoid writing tutorial content in news_roundup mode

    Returns a list with one (task_id, section_md) tuple.
    operator.add appends it to state["sections"] alongside
    results from all other parallel workers.
    """
    # Reconstruct Pydantic objects from plain dicts (Send() serializes everything)
    task     = Task(**payload["task"])
    plan     = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]
    topic    = payload["topic"]
    mode     = payload.get("mode", "closed_book")

    # Format bullets as a readable list for the prompt
    bullets_text = "\n- " + "\n- ".join(task.bullets)

    # Format evidence as "title | url | date" lines (cap at 20 to avoid token overflow)
    evidence_text = ""
    if evidence:
        evidence_text = "\n".join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = llm.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=(
            f"Blog title: {plan.blog_title}\n"
            f"Audience: {plan.audience}\n"
            f"Tone: {plan.tone}\n"
            f"Blog kind: {plan.blog_kind}\n"
            f"Constraints: {plan.constraints}\n"
            f"Topic: {topic}\n"
            f"Mode: {mode}\n\n"
            f"Section title: {task.title}\n"
            f"Goal: {task.goal}\n"
            f"Target words: {task.target_words}\n"
            f"Tags: {task.tags}\n"
            f"requires_research: {task.requires_research}\n"
            f"requires_citations: {task.requires_citations}\n"
            f"requires_code: {task.requires_code}\n"
            f"Bullets:{bullets_text}\n\n"
            f"Evidence (ONLY use these URLs when citing):\n{evidence_text}\n"
        )),
    ]).content.strip()

    # Store as (task.id, markdown) tuple so reducer_node can sort by id
    # regardless of which worker finishes first
    return {"sections": [(task.id, section_md)]}


# ============================================================
# NODE 5: REDUCER
#
# Runs once after ALL workers have finished.
# By this point state["sections"] contains all (task_id, section_md)
# tuples from every worker, accumulated by operator.add.
#
# Because workers run in parallel, they may finish in any order.
# Sorting by task_id restores the original planned section order.
#
# Assembles the final Markdown blog:
#   # Blog Title
#   ## Section 1
#   ## Section 2
#   ...
#
# Saves to a .md file named after the blog title.
# ============================================================

def reducer_node(state: State) -> dict:
    """
    Sorts all worker sections by task_id (restores planned order),
    joins them into one complete blog post, and saves to a .md file.

    Sorting is critical — without it, sections would appear in
    whichever order the parallel workers happened to finish.
    """
    plan = state["plan"]

    # Sort by task_id to restore correct section order
    ordered_sections = [
        md for _, md in sorted(state["sections"], key=lambda x: x[0])
    ]

    body     = "\n\n".join(ordered_sections).strip()
    final_md = f"# {plan.blog_title}\n\n{body}\n"

    # Save to file — filename is the blog title with spaces replaced
    filename = f"{plan.blog_title}.md"
    Path(filename).write_text(final_md, encoding="utf-8")
    print(f"\n✅ Blog saved to: {Path(filename).resolve()}\n")

    return {"final": final_md}


# ============================================================
# GRAPH ASSEMBLY
#
# Wires all nodes together. The full execution flow is:
#
#   START
#     ↓
#   router_node          → decides if research is needed
#     ↓ (route_next)
#     ├─ needs_research=True  → research_node → orchestrator_node
#     └─ needs_research=False →               → orchestrator_node
#                                                    ↓ (fanout)
#                               worker worker worker worker worker   ← parallel
#                                    ↘    ↓    ↓    ↓    ↙
#                                       reducer_node
#                                            ↓
#                                           END
#
# Key graph design choices:
#   - add_conditional_edges("router", route_next, {...})
#     The dict maps return values of route_next() to node names.
#   - add_conditional_edges("orchestrator", fanout, ["worker"])
#     fanout() returns Send() objects — LangGraph spawns parallel workers.
#     ["worker"] tells LangGraph which nodes fanout can route to.
# ============================================================

g = StateGraph(State)

g.add_node("router",       router_node)
g.add_node("research",     research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker",       worker_node)
g.add_node("reducer",      reducer_node)

g.add_edge(START, "router")

# Conditional edge: route_next() returns "research" or "orchestrator"
g.add_conditional_edges(
    "router",
    route_next,
    {"research": "research", "orchestrator": "orchestrator"}
)

g.add_edge("research", "orchestrator")  # research always feeds into planning

# fanout() returns Send() objects — spawns parallel workers
g.add_conditional_edges("orchestrator", fanout, ["worker"])

g.add_edge("worker",  "reducer")  # all workers feed into reducer
g.add_edge("reducer", END)

app = g.compile()


# ============================================================
# ENTRY POINT
#
# State must be fully initialized — all keys need a starting value.
# Empty strings/lists/False/None are fine as defaults.
# sections=[] is required because operator.add needs an existing
# list to append to when the first worker result arrives.
# ============================================================

def run(topic: str):
    """
    Invoke the full graph with a topic string.
    Returns the complete state dict including out["final"]
    which contains the assembled blog Markdown.
    """
    out = app.invoke({
        "topic":          topic,
        "mode":           "",
        "needs_research": False,
        "queries":        [],
        "evidence":       [],
        "plan":           None,
        "sections":       [],   # required initial value for operator.add reducer
        "final":          "",
    })
    return out


# run("Write a blog on Open Source LLMs in 2026")
out=run("State of Multimodal LLMs in 2026")
print(out["final"])