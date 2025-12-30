import os
import base64
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
load_dotenv()


VISION_MODEL_NAME = os.getenv("AZURE_VISION_MODEL", "gpt-5")

vision_llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("URL"),
    api_key=os.getenv("GPT_API"),
    api_version="2024-02-01",
    model=VISION_MODEL_NAME,
    reasoning_effort= 'medium'
)


def _encode_image_to_base64(image_path: Path) -> Optional[str]:
    """
    Return base64 string for an image file, or None if file doesn't exist.
    """
    if not image_path.exists():
        return None
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    


# ==========================================================
# 5. LLM-based semantic chunking of markdown (page-aware)
# ==========================================================



def _parse_markdown_pages(md_path: str | Path) -> List[Dict[str, Any]]:
    """
    Parse a markdown file produced by `doc_processor_with_descriptions`.

    It expects page separators of the form:

        --- PAGE 1 ---
        --- PAGE 2 ---

    Returns a list of:
    [
      {"page_number": 1, "text": "...markdown for page 1..."},
      {"page_number": 2, "text": "...markdown for page 2..."},
      ...
    ]
    """
    md_path = Path(md_path)
    text = md_path.read_text(encoding="utf-8")

    lines = text.splitlines()
    pages: List[Dict[str, Any]] = []

    page_pattern = re.compile(r"^\s*---\s*PAGE\s+(\d+)\s*---\s*$")

    current_page_num: Optional[int] = None
    current_lines: List[str] = []

    for line in lines:
        m = page_pattern.match(line)
        if m:
            # flush previous page
            if current_page_num is not None:
                pages.append(
                    {
                        "page_number": current_page_num,
                        "text": "\n".join(current_lines).strip(),
                    }
                )
                current_lines = []

            current_page_num = int(m.group(1))
        else:
            current_lines.append(line)

    # last page
    if current_page_num is not None and current_lines:
        pages.append(
            {
                "page_number": current_page_num,
                "text": "\n".join(current_lines).strip(),
            }
        )

    return pages


def _get_page_image_b64(
    doc_name: str,
    page_no: int,
    page_images_dir: Path,
) -> Optional[str]:
    """
    Return base64-encoded PNG for the given page, or None if not found.
    """
    img_path = page_images_dir / f"{doc_name}-page-{page_no}.png"
    if not img_path.exists():
        return None
    return _encode_image_to_base64(img_path)


def _chunk_single_page_with_llm(
    target_page: Dict[str, Any],
    next_page: Optional[Dict[str, Any]],
    doc_name: str,
    page_images_dir: Path,
) -> Dict[str, Any]:
    """
    Use Azure LLM (vision_llm) to chunk ONE target page.

    - target_page: {"page_number": int, "text": str}
    - next_page  : same dict or None (only for context, NOT chunked)

    Returns a dict like:
    {
      "chunk_1": {"content": "...", "refreshed_content": "...", "source": [1]},
      "chunk_2": {...},
      ...
    }
    """

    target_num: int = target_page["page_number"]
    target_text: str = target_page["text"]

    if next_page is not None:
        next_num: int = next_page["page_number"]
        next_text: str = next_page["text"]
    else:
        next_num = None
        next_text = ""

    # Encode page screenshots (if they exist)
    target_img_b64 = _get_page_image_b64(doc_name, target_num, page_images_dir)
    next_img_b64 = (
        _get_page_image_b64(doc_name, next_num, page_images_dir)
        if next_num is not None
        else None
    )

    # Big instruction text – tells the model EXACTLY how to chunk + JSON format
    instruction_text = f"""
You are an expert at splitting technical PDFs into semantically meaningful chunks
for a Retrieval-Augmented Generation system.

You are given:

1. The MARKDOWN content of a TARGET page from a PDF.
2. Optionally, the MARKDOWN content of the NEXT page (ONLY for context).
3. Screenshots of these pages (if available).

Your job is to chunk ONLY the TARGET page, but use the NEXT page and images
to understand if any text or idea clearly flows across the page boundary.

Chunking rules (VERY IMPORTANT):

- Think like a human reader. Group together text that belongs together:
  * headings with their immediate paragraphs
  * bullet lists with their label/intro
  * figures/tables with their captions and descriptive bullets
- Prefer coherent, medium-sized chunks that are individually understandable
  , but semantic coherence is more important than size.
- The NEXT page is ONLY for understanding cross-page flow. DO NOT create chunks
  that are purely from the NEXT page.
- If a chunk on the TARGET page clearly continues onto the NEXT page
  (for example, a sentence or list broken at the page break), you may bring
  the minimal necessary continuation text from the NEXT page into that chunk
  and then set its "source" metadata to [TARGET_PAGE, NEXT_PAGE].
- Otherwise, use "source": [TARGET_PAGE] for that chunk.
- The "source" field MUST ALWAYS be a LIST of integers (page numbers),
  even if it is a single page.

Output format (STRICT):

Return ONLY a single JSON object. No markdown formatting, no comments, no prose.

The JSON must look like this:

{{
  "chunk_1": {{
    "content": "original-ish text for this chunk from the target page (plus any minimal continuation if needed).",
    "refreshed_content": "the same information, but rephrased meaning fully it should give meaning as a single entity for better embeddings; do NOT drop technical details.",
    "source": [{target_num}]  OR  [{target_num}, {next_num if next_num is not None else "SECOND_PAGE_NUMBER"}]
  }},
  "chunk_2": {{
    "content": "...",
    "refreshed_content": "...",
    "source": [ ... ]
  }}
  // add as many chunks as needed
}}

TARGET_PAGE_NUMBER = {target_num}

TARGET_PAGE_MARKDOWN:
{target_text}

NEXT_PAGE_NUMBER = {next_num if next_num is not None else "null"}

NEXT_PAGE_MARKDOWN:
{next_text if next_text else "NONE"}
""".strip()

    # Build multimodal content for HumanMessage
    human_content: List[Dict[str, Any]] = [
        {"type": "text", "text": instruction_text}
    ]

    if target_img_b64 is not None:
        human_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{target_img_b64}"
                },
            }
        )

    if next_img_b64 is not None:
        human_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{next_img_b64}"
                },
            }
        )

    messages = [
        SystemMessage(
            content=(
                "You are a precise document chunking engine. "
                "You MUST respond with a single valid JSON object only."
            )
        ),
        HumanMessage(content=human_content),
    ]

    resp = vision_llm.invoke(messages)
    raw = resp.content.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        # If model ever slips, it's useful to see what it produced
        raise ValueError(
            f"LLM did not return valid JSON for page {target_num}.\nRaw output:\n{raw}"
        ) from e

    return data

def chunk_markdown_with_llm(
    md_path: str | Path,
    page_images_dir: str | Path | None = None,
    json_path : str | Path | None = None,
) -> Dict[str, Any]:
    """
    High-level helper.

    Usage:
        chunks = chunk_markdown_with_llm("scratch/sample_AIAYN_with_descriptions.md")

    It will:
      - Parse the markdown into pages using `--- PAGE n ---` markers.
      - For each page n, call Azure LLM with:
            * page n markdown (TARGET)
            * page n+1 markdown (REFERENCE ONLY)
            * screenshots for page n and n+1 (if they exist)
      - The LLM returns per-page chunks in the required JSON format.
      - This function merges all chunks into ONE dict:

        {
          "chunk_1": { "content": "...", "refreshed_content": "...", "source": [1] },
          "chunk_2": { ... },
          ...
        }
    """

    md_path = Path(md_path)
    if page_images_dir is None:
        # default: same parent as markdown, folder "page_images"
        page_images_dir = md_path.parent / "page_images"

    if json_path is None:
        # default: same parent as markdown, folder "page_images"
        json_path = md_path.parent / "chunks.json"

    page_images_dir = Path(page_images_dir)

    # Derive original doc_name from markdown name: sample_AIAYN_with_descriptions.md -> sample_AIAYN
    stem = md_path.stem
    if stem.endswith("_with_descriptions"):
        doc_name = stem[: -len("_with_descriptions")]
    else:
        doc_name = stem

    pages = _parse_markdown_pages(md_path)

    global_chunks: Dict[str, Any] = {}
    chunk_counter = 1

    for idx, page in enumerate(pages):
        target_page = page
        next_page = pages[idx + 1] if idx + 1 < len(pages) else None

        per_page_chunks = _chunk_single_page_with_llm(
            target_page=target_page,
            next_page=next_page,
            doc_name=doc_name,
            page_images_dir=page_images_dir,
        )

        # Reindex chunks globally: chunk_1, chunk_2, ...
        for _, chunk_data in per_page_chunks.items():
            new_key = f"chunk_{chunk_counter}"

            # Make sure 'source' always exists and is a list
            src = chunk_data.get("source")
            if not src:
                src = [target_page["page_number"]]
            elif isinstance(src, int):
                src = [src]
            elif isinstance(src, str):
                # try to parse stringified JSON; fallback to target page
                try:
                    parsed = json.loads(src)
                    src = parsed
                except Exception:
                    src = [target_page["page_number"]]

            chunk_data["source"] = src
            global_chunks[new_key] = chunk_data
            chunk_counter += 1
        # 3) Save to JSON for embeddings/RAG
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(global_chunks, f, ensure_ascii=False, indent=2)

    return json_path

