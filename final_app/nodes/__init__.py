"""
Node functions for the blog-writing LangGraph pipeline.

Each module owns one logical stage:
  router       — classifies topic; picks closed_book / hybrid / open_book
  research     — fetches and deduplicates web evidence via Tavily
  orchestrator — plans the blog outline (tasks / sections)
  worker       — writes one blog section (runs in parallel per task)
  reducer      — merges sections, decides images, generates and embeds them
"""
