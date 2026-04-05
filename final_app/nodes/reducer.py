"""
Reducer nodes — merge worker outputs into a finished blog with images.

Three sequential nodes that form a subgraph (compiled in graph.py):
  1. merge_content              — sorts sections by task_id and joins them
  2. decide_images              — LLM decides where (if anywhere) images help
  3. generate_and_place_images  — calls Gemini, saves images, embeds in Markdown

The subgraph is attached to the main graph as a single "reducer" node,
keeping the top-level pipeline topology simple.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from llm import llm
from models import GlobalImagePlan, State

# Anchor all file I/O to the project root regardless of where streamlit is launched from.
# This file lives at:  <project_root>/final_app/nodes/reducer.py
#   .parent   → final_app/nodes/
#   .parent.parent   → final_app/
#   .parent.parent.parent → <project_root>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_OUTPUTS_DIR = _PROJECT_ROOT / "outputs"   # generated .md blogs go here
_IMAGES_DIR = _PROJECT_ROOT / "images"     # generated images go here

DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no images needed: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""


def merge_content(state: State) -> dict:
    """
    Sort sections by task_id (workers finish in non-deterministic order)
    and assemble the full Markdown document with a top-level H1 heading.
    """
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")

    # task_id order restores the planned reading flow regardless of which worker finished first
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


def decide_images(state: State) -> dict:
    """
    Ask the LLM to decide where (if anywhere) images improve the blog.
    Returns the Markdown with [[IMAGE_N]] placeholders inserted and a list
    of image specs for the next node to generate.
    """
    planner = llm.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{merged_md}"
                )
            ),
        ]
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """
    Generate an image via Gemini and return raw PNG/JPEG bytes.

    Requires:
      - pip install google-genai
      - GOOGLE_API_KEY environment variable
    """
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_ONLY_HIGH",
                )
            ],
        ),
    )

    # Handle both old and new Gemini SDK response shapes
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None

    if not parts:
        raise RuntimeError("No image content returned (safety/quota/SDK change).")

    for part in parts:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            return inline.data

    raise RuntimeError("No inline image bytes found in response.")


def _safe_slug(title: str) -> str:
    """Convert a blog title to a filesystem-safe filename slug."""
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def generate_and_place_images(state: State) -> dict:
    """
    Generate each requested image via Gemini, save to images/, and replace
    [[IMAGE_N]] placeholders with proper Markdown image syntax.

    If image generation fails for any spec, a visible error callout block is
    inserted instead so the rest of the blog remains readable and usable.
    The finished Markdown is also written to disk as <slug>.md.
    """
    plan = state["plan"]
    assert plan is not None

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = state.get("image_specs", []) or []

    # No images requested — write markdown as-is and finish
    if not image_specs:
        _OUTPUTS_DIR.mkdir(exist_ok=True)
        (_OUTPUTS_DIR / f"{_safe_slug(plan.blog_title)}.md").write_text(md, encoding="utf-8")
        return {"final": md}

    _IMAGES_DIR.mkdir(exist_ok=True)
    images_dir = _IMAGES_DIR

    for spec in image_specs:
        placeholder = spec["placeholder"]
        filename = spec["filename"]
        out_path = images_dir / filename

        # Skip generation if this image already exists (e.g. re-running the same topic)
        if not out_path.exists():
            try:
                img_bytes = _gemini_generate_image_bytes(spec["prompt"])
                out_path.write_bytes(img_bytes)
            except Exception as e:
                # Graceful fallback: replace placeholder with a visible error block
                # so the document is still usable even when image generation fails
                prompt_block = (
                    f"> **[IMAGE GENERATION FAILED]** {spec.get('caption','')}\n>\n"
                    f"> **Alt:** {spec.get('alt','')}\n>\n"
                    f"> **Prompt:** {spec.get('prompt','')}\n>\n"
                    f"> **Error:** {e}\n"
                )
                md = md.replace(placeholder, prompt_block)
                continue

        img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
        md = md.replace(placeholder, img_md)

    _OUTPUTS_DIR.mkdir(exist_ok=True)
    (_OUTPUTS_DIR / f"{_safe_slug(plan.blog_title)}.md").write_text(md, encoding="utf-8")
    return {"final": md}
