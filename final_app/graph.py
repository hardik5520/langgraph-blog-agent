"""
Graph assembly — wires all nodes into the compiled LangGraph application.

Pipeline topology:
  START
    └─ router           (classify topic; pick research strategy)
         ├─[needs_research=True]──  research  ──┐
         └─[needs_research=False]───────────────┤
                                                └─ orchestrator    (plan 5-9 sections)
                                                     └─ worker × N  (parallel writers)
                                                          └─ reducer  (merge + images)
                                                               └─ END

The reducer is itself a 3-node subgraph:
  merge_content → decide_images → generate_and_place_images
"""
from __future__ import annotations

from dotenv import load_dotenv

# Must be called before any API clients are imported/initialised
# (LLM, Tavily, and Gemini all read from environment variables)
load_dotenv()

from langgraph.graph import END, START, StateGraph

from models import State
from nodes.router import route_next, router_node
from nodes.research import research_node
from nodes.orchestrator import fanout, orchestrator_node
from nodes.worker import worker_node
from nodes.reducer import decide_images, generate_and_place_images, merge_content

# ---------------------------------------------------------------------------
# Reducer subgraph
# Compiled separately so the main graph sees the entire merge+image pipeline
# as a single atomic "reducer" node, keeping the top-level topology readable.
# ---------------------------------------------------------------------------
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# ---------------------------------------------------------------------------
# Main graph
# ---------------------------------------------------------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)  # entire reducer pipeline as one node

g.add_edge(START, "router")
g.add_conditional_edges(
    "router",
    route_next,
    {"research": "research", "orchestrator": "orchestrator"},
)
g.add_edge("research", "orchestrator")
g.add_conditional_edges("orchestrator", fanout, ["worker"])  # fan-out: one worker per task
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
