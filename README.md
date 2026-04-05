# Blog Writing Agent

A LangGraph-powered blog writing agent that researches topics on the web, plans a full blog outline, writes all sections in parallel, and optionally generates diagrams using Gemini — all from a single topic prompt.

---

## How It Works

The agent is a directed graph with five stages. Each stage is a node (or subgraph); they execute in order with the research stage being optional.

```
START
  └─ router           → decides whether web research is needed
       ├─[yes]── research ──┐
       └─[no]───────────────┤
                            └─ orchestrator   → plans 5–9 blog sections
                                 └─ worker × N  → writes sections in parallel
                                      └─ reducer  → merges, adds images, saves .md
                                           └─ END
```

### Nodes

| Node | File | What it does |
|------|------|--------------|
| **router** | `nodes/router.py` | Classifies the topic as `closed_book`, `hybrid`, or `open_book` and decides whether to run a web search before planning |
| **research** | `nodes/research.py` | Runs Tavily search queries, deduplicates results, and filters by recency window |
| **orchestrator** | `nodes/orchestrator.py` | Plans the blog: produces 5–9 tasks with goals, bullets, and word targets |
| **worker** | `nodes/worker.py` | Writes one section in Markdown; multiple workers run in parallel (one per task) |
| **reducer** | `nodes/reducer.py` | Sorts sections, decides where images help, generates them via Gemini, and writes the final `.md` file |

### Research Modes

The router picks one of three modes based on the topic:

| Mode | When | Evidence window |
|------|------|----------------|
| `closed_book` | Evergreen concepts (e.g. "how attention works") | No web search |
| `hybrid` | Mostly evergreen but references recent tools/models | 45-day lookback |
| `open_book` | Volatile / news topics (e.g. "AI news this week") | 7-day lookback; all claims must be cited |

---

## Project Structure

```
writing agent/
├── final_app/                  # Production app
│   ├── frontend.py             # Streamlit UI
│   ├── backend.py              # Entry point — re-exports `app` from graph.py
│   ├── graph.py                # Graph assembly and compilation
│   ├── models.py               # All Pydantic schemas and LangGraph State
│   ├── llm.py                  # Shared LLM instance (gpt-4.1-mini)
│   └── nodes/
│       ├── __init__.py
│       ├── router.py           # Router node
│       ├── research.py         # Research node (Tavily)
│       ├── orchestrator.py     # Orchestrator node + fanout edge
│       ├── worker.py           # Worker node (parallel section writer)
│       └── reducer.py          # Reducer subgraph (merge → images → save)
│
├── exploration/                # Iterative prototypes (not production code)
│   ├── basic.py                # v1: simple parallel fan-out, no research
│   ├── basic_with_better_prompt.py  # v2: richer prompts and task schema
│   ├── agent_with_research.py  # v3: adds Tavily research + citations
│   ├── agent_with_image.py     # v4: adds Gemini image generation
│   └── test_tavily.py          # Standalone Tavily API sanity check
│
├── outputs/                    # Sample blog outputs from previous runs
├── images/                     # Generated images (created at runtime)
├── .env                        # API keys (never commit this)
├── .gitignore
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install langgraph langchain-openai langchain-community \
            streamlit pandas python-dotenv \
            tavily-python google-genai
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...   # required for hybrid / open_book topics
GOOGLE_API_KEY=...         # required for image generation
```

All three keys are optional at startup — the agent degrades gracefully:
- No `TAVILY_API_KEY` → research node returns empty evidence (closed_book behaviour)
- No `GOOGLE_API_KEY` → image placeholders are replaced with error callout blocks

### 3. Run the app

```bash
cd final_app
streamlit run frontend.py
```

---

## Using the UI

1. **Enter a topic** in the sidebar text area (e.g. "Self-attention in Transformers" or "AI news roundup April 2026").
2. **Set the as-of date** — used by the router to determine recency windows for news topics.
3. Click **Generate Blog**.
4. Watch real-time progress in the status bar as each node executes.
5. Explore the output tabs:
   - **Plan** — blog title, audience, tone, and task breakdown table
   - **Evidence** — sources retrieved by the research node
   - **Markdown Preview** — rendered blog with inline images
   - **Images** — generated diagrams with download option
   - **Logs** — raw event stream from the graph
6. Use **Download Markdown** or **Download Bundle** (MD + images zip) to save the output.
7. Previously generated blogs appear in the sidebar under **Past blogs** for quick reload.

---

## How Parallel Writing Works

The orchestrator produces a `Plan` with N tasks. The `fanout` function in `orchestrator.py` emits one `Send()` message per task, which LangGraph dispatches to N worker nodes running concurrently. Each worker returns `(task_id, section_markdown)`. The `operator.add` reducer in `State.sections` safely merges all concurrent writes into a single list — no locking needed. The reducer then sorts by `task_id` to restore the planned reading order before assembling the final document.

---

## Exploration Code

The `exploration/` directory contains the iterative prototypes built before the final app, in order of increasing complexity:

| File | What it adds |
|------|-------------|
| `basic.py` | Baseline: orchestrator → parallel workers → reducer |
| `basic_with_better_prompt.py` | Richer task schema: section types, word count targets |
| `agent_with_research.py` | Router + Tavily research + citation grounding |
| `agent_with_image.py` | Gemini image generation in the reducer |

These are kept for reference and are not imported by the production app.

---

## Key Design Decisions

- **Structured outputs everywhere** — Pydantic models are used for all LLM outputs (`Plan`, `RouterDecision`, `GlobalImagePlan`, etc.), avoiding free-form text parsing and making schema changes explicit.
- **Self-contained worker payloads** — workers receive a plain dict rather than the full State, so they have no hidden dependencies and can run safely in parallel.
- **Reducer as a subgraph** — the three reducer nodes (`merge_content` → `decide_images` → `generate_and_place_images`) are compiled into a subgraph and attached as a single node in the main graph, keeping the top-level topology clean.
- **Graceful degradation** — missing API keys and failed image generation are handled without crashing; the pipeline always produces a usable Markdown output.
