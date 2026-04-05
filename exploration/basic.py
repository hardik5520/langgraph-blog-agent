from __future__ import annotations

import operator
from typing import TypedDict, List, Annotated
from pathlib import Path

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# MODELS
# Pydantic models that define the structure of data the LLM
# will return. with_structured_output() forces the LLM to
# return data in exactly this shape — no free-form text.
# ============================================================

class Task(BaseModel):
    """Represents one section of the blog."""
    id: int
    title: str
    brief: str = Field(..., description="What this section should cover")

class Plan(BaseModel):
    """The full blog plan returned by the orchestrator."""
    blog_title: str
    tasks: List[Task]   # one Task per blog section (5-7 tasks)


# ============================================================
# STATE
# The shared memory of the entire graph. Every node reads from
# and writes to this. Think of it as a whiteboard that all
# nodes can see and update.
#
# - topic    : the blog topic the user provides (never changes)
# - plan     : the blog plan created by the orchestrator node
# - sections : list of written sections, one per worker node.
#              Uses operator.add as reducer — when multiple
#              workers return {"sections": [...]}, LangGraph
#              automatically APPENDS them all into one list
#              instead of overwriting. This is what makes
#              parallel fan-out work correctly.
# - final    : the fully assembled blog post (set by reducer node)
# ============================================================

class State(TypedDict):
    topic: str
    plan: Plan
    sections: Annotated[List[str], operator.add]  # operator.add = append, not overwrite
    final: str


# ============================================================
# LLM
# Single shared LLM instance used by all nodes.
# gpt-4.1-mini is fast and cheap — good for parallel workloads.
# ============================================================

llm = ChatOpenAI(model="gpt-4.1-mini")


# ============================================================
# NODE 1: ORCHESTRATOR
# The "manager" node. Runs once at the start.
# Takes the user's topic and asks the LLM to break it into
# 5-7 sections, returning a structured Plan object.
# Writes the plan to state so fan-out can read it.
# ============================================================

def orchestrator(state: State) -> dict:
    plan = llm.with_structured_output(Plan).invoke([
        SystemMessage(content="Create a blog plan with 5-7 sections on the following topic."),
        HumanMessage(content=f"Topic: {state['topic']}"),
    ])
    return {"plan": plan}


# ============================================================
# FAN-OUT FUNCTION
# Not a node — this is a conditional edge function that runs
# after the orchestrator. It returns a list of Send() objects,
# one per task in the plan. Each Send() spawns a separate
# worker node in PARALLEL, passing that task's data as payload.
#
# If the plan has 6 sections → 6 workers run simultaneously.
# This is what makes the blog writing fast.
# ============================================================

def fanout(state: State):
    return [
        Send("worker", {
            "task": task,
            "topic": state["topic"],
            "plan": state["plan"],
        })
        for task in state["plan"].tasks
    ]


# ============================================================
# NODE 2: WORKER
# Runs in parallel — one instance per task/section.
# Each worker receives its own task (title + brief) and writes
# that one section in Markdown.
# Returns {"sections": [section_text]} which gets appended to
# state["sections"] via the operator.add reducer.
#
# Note: payload is a plain dict (not State) because Send()
# passes custom data directly to this node.
# ============================================================

def worker(payload: dict) -> dict:
    task  = payload["task"]    # the specific section this worker handles
    topic = payload["topic"]   # full blog topic for context
    plan  = payload["plan"]    # full plan for context

    section_md = llm.invoke([
        SystemMessage(content="Write one clean Markdown section."),
        HumanMessage(content=(
            f"Blog title : {plan.blog_title}\n"
            f"Topic      : {topic}\n\n"
            f"Section    : {task.title}\n"
            f"Brief      : {task.brief}\n\n"
            "Return only the section content in Markdown. No extra commentary."
        )),
    ]).content.strip()

    # Wrap in a list — operator.add will append this to the
    # growing sections list in state as each worker finishes.
    return {"sections": [section_md]}


# ============================================================
# NODE 3: REDUCER (JOINER)
# Runs once after ALL workers have finished.
# By this point, state["sections"] contains every section
# written by every worker, concatenated by operator.add.
# This node joins them into one complete blog post,
# saves it to a .md file, and stores it in state["final"].
# ============================================================

def reducer(state: State) -> dict:
    title = state["plan"].blog_title

    # Join all sections with a blank line between each
    body = "\n\n".join(state["sections"]).strip()

    # Build the final Markdown document
    final_md = f"# {title}\n\n{body}\n"

    # Save to a file named after the blog title
    filename    = title.lower().replace(" ", "_") + ".md"
    output_path = Path(filename)
    output_path.write_text(final_md, encoding="utf-8")
    print(f"\n✅ Blog saved to: {output_path.resolve()}\n")

    return {"final": final_md}


# ============================================================
# GRAPH ASSEMBLY
# Wires all nodes together into the execution graph.
#
# Flow:
#   START
#     ↓
#   orchestrator  (creates the plan)
#     ↓
#   fanout()      (spawns N parallel workers via Send)
#   ↙ ↓ ↓ ↓ ↓ ↘
#  W1 W2 W3 W4 W5  (each writes one section simultaneously)
#   ↘ ↓ ↓ ↓ ↓ ↙
#   reducer     (joins all sections, saves file)
#     ↓
#   END
# ============================================================

g = StateGraph(State)

g.add_node("orchestrator", orchestrator)
g.add_node("worker",       worker)
g.add_node("reducer",    reducer)

g.add_edge(START, "orchestrator")

# fanout is a conditional edge — it returns Send() objects
# instead of a single node name, spawning parallel workers.
# ["worker"] tells LangGraph which nodes fanout can route to.
g.add_conditional_edges("orchestrator", fanout, ["worker"])

g.add_edge("worker",    "reducer")
g.add_edge("reducer", END)

app = g.compile()


# ============================================================
# RUN
# sections must be initialized as [] because operator.add
# needs an existing list to append to on the first worker result.
# ============================================================


result = app.invoke({
    "topic": "Write a blog on Self Attention",
    "sections": [],   # required initial value for the reducer
})

print(result["final"])