"""
Orchestrator node — plans the blog structure.

Uses the topic, mode, and evidence (if any) to produce a Plan with 5–9 tasks.
Each task becomes one independent section written by a parallel worker.

Also contains `fanout`, the conditional edge that dispatches tasks to workers
via LangGraph's Send() API, enabling true parallel execution.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send

from llm import llm
from models import Plan, State

ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don't invent events).

Output must match Plan schema.
"""


def orchestrator_node(state: State) -> dict:
    """
    Produce a Plan from the topic, mode, and available evidence.

    For open_book mode, blog_kind is force-set to "news_roundup" as a safeguard
    in case the LLM selects an inappropriate format despite the prompt instruction.
    """
    planner = llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )

    # Hard-enforce news_roundup even if the LLM ignored the prompt instruction
    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}


def fanout(state: State):
    """
    Fan out one Send() message per task so all workers run in parallel.

    Each worker receives a self-contained payload (task + context) rather than
    the full State, keeping the worker interface minimal and avoiding unnecessary copies.
    Sections are later merged by the operator.add reducer in State.
    """
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]
