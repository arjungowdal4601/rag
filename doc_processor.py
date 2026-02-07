import os
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from dotenv import load_dotenv
from docling_core.types.doc import (
    ImageRefMode,
    PictureItem,
    TableItem,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
load_dotenv()

# ==========================================================
# 1. OpenAI VLM setup (must be image-capable, e.g. gpt-4o)
# ==========================================================

VISION_MODEL_NAME = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
vision_llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=VISION_MODEL_NAME,
)

def _encode_image_to_b64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def describe_picture(image_path: Path) -> str:
    """
    Use OpenAI to describe a figure/image for retrieval.
    Returns markdown bullet points.
    """
    img_b64 = _encode_image_to_b64(image_path)

    messages = [                                                                                                                                                                                                                                                                                                                                                              
        SystemMessage(                                                                                                                                                                                                                                                                                                                                                              
            content=(                                                                                                                                                                                                                                                                                                                                                              
                f'''You are a helpful assistant that writes detailed, retrieval-friendly descriptions of figures from technical PDFs. 
                For each figure, produce a rich, information-dense description that carefully captures what the figure shows, 
                all visible labels (titles, axes, units, legends, annotations), 
                key trends and comparisons, important numeric values, and any notable patterns, thresholds, or anomalies. 
                Expand abbreviations or symbols when the meaning is clear from context, 
                and explicitly describe relationships between variables (e.g., “as X increases, Y decreases linearly”). 
                Write in clear, neutral prose with multiple sentences that prioritize factual detail over brevity, 
                making the text highly useful for search and embedding. Do not use headings, bullet points, or numbered lists, and do not mention the user, the prompt, or yourself '''                                                                                                                                                                                                                                                                                                                                                              
            )                                                                                                                                                                                                                                                                                                                                                              
        ),                                                                                                                                                                                                                                                                                                                                                              
        HumanMessage(                                                                                                                                                                                                                                                                                                                                                              
            content=[                                                                                                                                                                                                                                                                                                                                                              
                {                                                                                                                                                                                                                                                                                                                                                              
                    "type": "text",                                                                                                                                                                                                                                                                                                                                                              
                    "text": (                                                                                                                                                                                                                                                                                                                                                              
                        "Look at this image from a technical document and describe "                                                                                                                                                                                                                                                                                                                                                              
                        "what it shows in plain English so it is easy to search for later."                                                                                                                                                                                                                                                                                                                                                              
                    ),                                                                                                                                                                                                                                                                                                                                                              
                },                                                                                                                                                                                                                                                                                                                                                              
                {                                                                                                                                                                                                                                                                                                                                                              
                    "type": "image_url",                                                                                                                                                                                                                                                                                                                                                              
                    "image_url": {                                                                                                                                                                                                                                                                                                                                                              
                        "url": f"data:image/png;base64,{img_b64}"                                                                                                                                                                                                                                                                                                                                                              
                    },                                                                                                                                                                                                                                                                                                                                                              
                },                                                                                                                                                                                                                                                                                                                                                              
            ]                                                                                                                                                                                                                                                                                                                                                              
        ),                                                                                                                                                                                                                                                                                                                                                              
    ]                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                              
    resp = vision_llm.invoke(messages)                                                                                                                                                                                                                                                                                                                                                              
    return resp.content.strip()                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                              
def describe_table(image_path: Path) -> str:                                                                                                                                                                                                                                                                                                                                                              
    """
    Use OpenAI to turn a table image into bullet-point summary.
    Returns markdown bullet points (no raw table).
    """
    img_b64 = _encode_image_to_b64(image_path)

    messages = [
        SystemMessage(
            content=(
                    f''' You convert tables from technical PDFs into detailed, retrieval-friendly summaries. For each table, write a rich,
                    information-dense description that explains what the table is about, the meaning of each key column (including units, categories, and abbreviations),
                    the ranges and typical values, and any clear trends, patterns, or comparisons across rows or columns. 
                    Highlight important groupings, rankings, thresholds, outliers, 
                    and notable relationships between variables (for example, which items have the highest or lowest values, or how one column changes as another changes). 
                    Do not recreate the table or use tabular formatting. Write in clear, neutral prose with multiple sentences that prioritize factual detail over brevity,
                    making the text highly useful for search and embedding. Do not use headings, bullet points, or numbered lists, and do not mention the user, the prompt, or yourself.'''
            )
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Look at this table image. Describe its contents as bullet points "
                        "in plain English for a retrieval system. Start each line with '- '. "
                        "Do NOT add headings or extra commentary."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}"
                    },
                },
            ]
        ),
    ]

    resp = vision_llm.invoke(messages)
    return resp.content.strip()

# ------------------------------------------------------------------------------------------------------------

def inject_table_placeholders(
    md: str,
    placeholder: str = "[[DOC_TABLE]]",
) -> str:
    """
    Replace markdown table blocks with a single placeholder line.

    We detect tables in markdown like:

        | col1 | col2 |
        | ---  | ---  |
        | ...  | ...  |

    Everything from the header row through all data rows is removed
    and replaced by `placeholder`.
    """
    lines = md.splitlines()
    out_lines: List[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Heuristic: row with '|' followed by a separator row
        if "|" in stripped and not stripped.startswith("!["):
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # second row = header separator (only -, :, |, spaces)
                if (
                    "|" in next_line
                    and all(ch in "-:| " for ch in next_line if ch)
                ):
                    # Found a table — insert placeholder instead of table block
                    out_lines.append(placeholder)

                    # Skip header row + separator row
                    i += 2
                    # Skip data rows (lines that still look like table rows)
                    while i < len(lines) and "|" in lines[i]:
                        i += 1
                    # Optionally keep a blank line after table
                    if i < len(lines) and lines[i].strip() == "":
                        out_lines.append("")
                        i += 1
                    continue

        out_lines.append(line)
        i += 1

    return "\n".join(out_lines)


# ==========================================================
# 3. Main processor
# ==========================================================

def doc_processor_with_descriptions(
    pdf_path: str | Path,
    output_root: str | Path,
    page_separator_template: str = "\n\n--- PAGE {page_no} ---\n\n",
) -> Path:
    """
    Processes a PDF with Docling + Azure GPT-4o and produces:

      - {output_root}/page_images/        : full-page PNG images
      - {output_root}/image_png_images/   : PNG images of figures/pictures
      - {output_root}/table_images/       : PNG images of tables
      - {output_root}/{name}_with_descriptions.md

    In the markdown:
      * Each page is separated with `--- PAGE n ---`
      * Every figure AND table is represented as:

            ![Figure](relative/path/to/image.png)

            - bullet
            - bullet
            ...

        (no raw markdown table for tables).
    """

    pdf_path = Path(pdf_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # --- Output dirs ---
    page_dir = output_root / "page_images"
    picture_dir = output_root / "image_png_images"
    table_dir = output_root / "table_images"

    page_dir.mkdir(exist_ok=True)
    picture_dir.mkdir(exist_ok=True)
    table_dir.mkdir(exist_ok=True)

    # --- Docling converter ---
    IMAGE_RESOLUTION_SCALE = 2.0

    pipeline_options = PdfPipelineOptions()
    pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    conv_res = converter.convert(str(pdf_path))
    doc = conv_res.document
    doc_name = pdf_path.stem

    # ------------------------------------------------------
    # 3.1 Save page images
    # ------------------------------------------------------
    for page_no, page in sorted(doc.pages.items()):
        page_img_path = page_dir / f"{doc_name}-page-{page_no}.png"
        with page_img_path.open("wb") as fp:
            page.image.pil_image.save(fp, format="PNG")

    # ------------------------------------------------------
    # 3.2 Export pictures & tables as images + descriptions
    # ------------------------------------------------------
    page_pictures: Dict[int, List[Tuple[Path, str]]] = defaultdict(list)
    page_tables: Dict[int, List[Tuple[Path, str]]] = defaultdict(list)

    picture_counter = 0
    table_counter = 0

    for element, _level in doc.iterate_items():
        prov = getattr(element, "prov", [])
        page_no = prov[0].page_no if prov else None
        if page_no is None:
            continue

        # --- PictureItem ---
        if isinstance(element, PictureItem):
            picture_counter += 1
            img_path = picture_dir / f"{doc_name}-picture-{picture_counter}.png"
            with img_path.open("wb") as fp:
                element.get_image(doc).save(fp, "PNG")

            desc = describe_picture(img_path)
            page_pictures[page_no].append((img_path, desc))

        # --- TableItem ---
        elif isinstance(element, TableItem):
            table_counter += 1
            img_path = table_dir / f"{doc_name}-table-{table_counter}.png"
            with img_path.open("wb") as fp:
                element.get_image(doc).save(fp, "PNG")

            desc = describe_table(img_path)
            page_tables[page_no].append((img_path, desc))

    # ------------------------------------------------------
    # 3.3 Build markdown page by page
    #      - remove raw tables
    #      - replace picture + table placeholders
    # ------------------------------------------------------
    IMAGE_PLACEHOLDER = "[[DOC_IMAGE]]"
    TABLE_PLACEHOLDER = "[[DOC_TABLE]]"

    full_md_parts: List[str] = []

    for page_no in sorted(doc.pages.keys()):
        # Export this page as markdown with IMAGE placeholders
        page_md_raw = doc.export_to_markdown(
            image_mode=ImageRefMode.PLACEHOLDER,
            image_placeholder=IMAGE_PLACEHOLDER,
            page_no=page_no,
        )

        # Replace markdown tables with TABLE placeholders
        page_md = inject_table_placeholders(page_md_raw, TABLE_PLACEHOLDER)

        pictures = page_pictures.get(page_no, [])
        tables = page_tables.get(page_no, [])

        pic_idx = 0
        tbl_idx = 0
        cursor = 0
        out_chunks: List[str] = []

        while True:
            next_img = page_md.find(IMAGE_PLACEHOLDER, cursor)
            next_tbl = page_md.find(TABLE_PLACEHOLDER, cursor)

            if next_img == -1 and next_tbl == -1:
                out_chunks.append(page_md[cursor:])
                break

            # decide which placeholder comes next
            if next_tbl != -1 and (next_img == -1 or next_tbl < next_img):
                # handle TABLE_PLACEHOLDER
                pos = next_tbl
                out_chunks.append(page_md[cursor:pos])

                if tbl_idx < len(tables):
                    img_path, desc = tables[tbl_idx]
                    tbl_idx += 1
                    rel_path = os.path.relpath(img_path, output_root).replace("\\", "/")
                    replacement = f"![Figure]({rel_path})\n\n{desc}\n\n"
                else:
                    replacement = ""

                out_chunks.append(replacement)
                cursor = pos + len(TABLE_PLACEHOLDER)

            else:
                # handle IMAGE_PLACEHOLDER
                pos = next_img
                out_chunks.append(page_md[cursor:pos])

                if pic_idx < len(pictures):
                    img_path, desc = pictures[pic_idx]
                    pic_idx += 1
                    rel_path = os.path.relpath(img_path, output_root).replace("\\", "/")
                    replacement = f"![Figure]({rel_path})\n\n{desc}\n\n"
                else:
                    replacement = ""

                out_chunks.append(replacement)
                cursor = pos + len(IMAGE_PLACEHOLDER)

        page_md_with_desc = "".join(out_chunks).strip()

        # Add page separator and page content
        full_md_parts.append(page_separator_template.format(page_no=page_no))
        if page_md_with_desc:
            full_md_parts.append(page_md_with_desc)

    final_md = "\n".join(full_md_parts).strip() + "\n"

    # ------------------------------------------------------
    # 3.4 Save final markdown
    # ------------------------------------------------------
    md_path = output_root / f"processed_doc.md"
    md_path.write_text(final_md, encoding="utf-8")
    return md_path
