import os
import re
import gc
import base64
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Tuple
from pypdf import PdfReader

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from docling_core.types.doc import (
    DocItemLabel,
    FormulaItem,
    ImageRefMode,
    PictureItem,
    TableItem,
    TextItem,
)
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


load_dotenv()

# =============================================================================
# OpenAI vision model setup
# =============================================================================
# Kept intentionally.
# Your current pipeline depends on OpenAI vision to generate retrieval-friendly
# descriptions for figures, tables, and formulas.
VISION_MODEL_NAME = os.getenv("OPENAI_VISION_MODEL", "gpt-5.2")
vision_llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model=VISION_MODEL_NAME,
)

# Placeholder tokens used during markdown replacement.
IMAGE_PLACEHOLDER = "[[DOC_IMAGE]]"
TABLE_PLACEHOLDER = "[[DOC_TABLE]]"
FORMULA_PLACEHOLDER = "[[DOC_FORMULA]]"


# =============================================================================
# Small data containers
# =============================================================================
@dataclass
class OutputPaths:
    """All folders and files written by the processor."""

    root: Path
    page_images: Path
    picture_images: Path
    table_images: Path
    formula_images: Path
    pages_md: Path
    final_markdown: Path


@dataclass
class Asset:
    """One extracted visual asset and its generated description."""

    path: Path
    description: str
    extracted_text: Optional[str] = None


# =============================================================================
# Utility helpers
# =============================================================================
def encode_image_to_base64(image_path: Path) -> str:
    """Read an image file and return a base64 string for OpenAI vision."""
    with image_path.open("rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")


def cleanup_memory() -> None:
    """Run basic cleanup between page chunks."""
    gc.collect()


def get_pdf_page_count(pdf_path: Path) -> int:
    """Return the total number of pages in the PDF."""
    with pdf_path.open("rb") as file:
        reader = PdfReader(file)
        return len(reader.pages)


def create_output_paths(output_root: Path, write_page_md_files: bool) -> OutputPaths:
    """Create all output folders used by the pipeline."""
    output_root.mkdir(parents=True, exist_ok=True)

    paths = OutputPaths(
        root=output_root,
        page_images=output_root / "page_images",
        picture_images=output_root / "image_png_images",
        table_images=output_root / "table_images",
        formula_images=output_root / "formula_images",
        pages_md=output_root / "pages_md",
        final_markdown=output_root / "processed_doc.md",
    )

    paths.page_images.mkdir(exist_ok=True)
    paths.picture_images.mkdir(exist_ok=True)
    paths.table_images.mkdir(exist_ok=True)
    paths.formula_images.mkdir(exist_ok=True)

    if write_page_md_files:
        paths.pages_md.mkdir(exist_ok=True)

    return paths


# =============================================================================
# OpenAI description helpers
# =============================================================================
def describe_picture(image_path: Path) -> str:
    """Describe a figure or diagram in dense retrieval-friendly prose."""
    image_b64 = encode_image_to_base64(image_path)

    messages = [
        SystemMessage(
            content=(
                """You are a helpful assistant that writes detailed, retrieval-friendly descriptions of figures from technical PDFs.
For each figure, produce a rich, information-dense description that carefully captures what the figure shows,
all visible labels (titles, axes, units, legends, annotations), key trends and comparisons, important numeric values,
and any notable patterns, thresholds, or anomalies.
Expand abbreviations or symbols when the meaning is clear from context, and explicitly describe relationships between variables.
Write in clear, neutral prose with multiple sentences that prioritize factual detail over brevity.
Do not use headings, bullet points, or numbered lists, and do not mention the user, the prompt, or yourself."""
            )
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Describe what this figure shows in plain English so it is easy to search for later.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ]
        ),
    ]

    response = vision_llm.invoke(messages)
    return response.content.strip()


def describe_table(image_path: Path) -> str:
    """Describe a table image in prose, without reconstructing the table."""
    image_b64 = encode_image_to_base64(image_path)

    messages = [
        SystemMessage(
            content=(
                """You convert tables from technical PDFs into detailed, retrieval-friendly summaries.
For each table, write a rich, information-dense description that explains what the table is about,
the meaning of each key column (including units, categories, and abbreviations), the ranges and typical values,
and any clear trends, patterns, or comparisons across rows or columns.
Highlight important groupings, rankings, thresholds, outliers, and notable relationships between variables.
Do not recreate the table or use tabular formatting.
Write in clear, neutral prose with multiple sentences that prioritize factual detail over brevity.
Do not use headings, bullet points, or numbered lists, and do not mention the user, the prompt, or yourself."""
            )
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "Describe this table for retrieval in plain English. Do not use headings or markdown table formatting.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ]
        ),
    ]

    response = vision_llm.invoke(messages)
    return response.content.strip()


def describe_formula(image_path: Path, extracted_latex: Optional[str] = None) -> str:
    """Describe a formula crop.

    Expected output:
    - first line starts with 'LaTeX: ...'
    - remaining text explains the equation briefly and factually
    """
    image_b64 = encode_image_to_base64(image_path)
    latex_hint = (extracted_latex or "").strip()
    fallback_latex_line = f"LaTeX: {latex_hint}" if latex_hint else "LaTeX: (not provided)"

    messages = [
        SystemMessage(
            content=(
                """You describe mathematical formulas cropped from technical PDFs for retrieval.
You will be given a cropped image of an equation and sometimes an extracted LaTeX string.
Your output must:
1) Start with a single line: 'LaTeX: ...' (use the provided LaTeX if given; otherwise transcribe the equation to LaTeX as best as you can).
2) Then write a short, factual explanation of what the equation expresses.
3) If variable meanings are obvious from the equation text itself, mention them; otherwise do not guess.
Do not use headings, bullet points, or numbered lists."""
            )
        ),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Here is a cropped equation image. "
                        + (
                            f"The parser extracted this LaTeX: {latex_hint}"
                            if latex_hint
                            else "No LaTeX was provided."
                        )
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ]
        ),
    ]

    response = vision_llm.invoke(messages)
    text = response.content.strip()

    if not text.lower().startswith("latex:"):
        text = f"{fallback_latex_line}\n{text}".strip()

    return text


# =============================================================================
# Markdown placeholder helpers
# =============================================================================
def inject_table_placeholders(markdown: str, placeholder: str = TABLE_PLACEHOLDER) -> str:
    """Replace markdown table blocks with one placeholder line.

    This preserves the current pipeline behavior:
    later we replace that placeholder with the saved table image and its
    retrieval-friendly description.
    """
    lines = markdown.splitlines()
    output_lines: List[str] = []
    index = 0

    while index < len(lines):
        current_line = lines[index]
        stripped = current_line.strip()

        # Simple heuristic for markdown tables:
        # header row containing '|' followed by a separator row.
        if "|" in stripped and not stripped.startswith("!["):
            if index + 1 < len(lines):
                next_line = lines[index + 1].strip()
                looks_like_separator = "|" in next_line and all(
                    char in "-:| " for char in next_line if char
                )
                if looks_like_separator:
                    output_lines.append(placeholder)
                    index += 2

                    while index < len(lines) and "|" in lines[index]:
                        index += 1

                    if index < len(lines) and lines[index].strip() == "":
                        output_lines.append("")
                        index += 1
                    continue

        output_lines.append(current_line)
        index += 1

    return "\n".join(output_lines)


def inject_formula_placeholders(markdown: str, placeholder: str = FORMULA_PLACEHOLDER) -> str:
    r"""Replace common display-math blocks with one placeholder line."""
    patterns = [
        r"(?s)\$\$.*?\$\$",
        r"(?s)\\\[.*?\\\]",
        r"(?s)\\begin\{equation\*?\}.*?\\end\{equation\*?\}",
        r"(?s)\\begin\{align\*?\}.*?\\end\{align\*?\}",
        r"(?s)\\begin\{gather\*?\}.*?\\end\{gather\*?\}",
        r"(?s)\\begin\{multline\*?\}.*?\\end\{multline\*?\}",
    ]

    updated_markdown = markdown
    for pattern in patterns:
        updated_markdown = re.sub(pattern, f"\n{placeholder}\n", updated_markdown)

    return updated_markdown


# =============================================================================
# Docling setup and conversion helpers
# =============================================================================
def build_docling_converter(images_scale: float) -> DocumentConverter:
    """Create a simple CPU-only Docling converter.

    Simplifications compared with the older file:
    - CPU only
    - no GPU / CUDA / MPS branching
    - no threaded pipeline options
    - no manual batch-size tuning
    - no queue tuning

    Kept because the current process depends on them:
    - OCR
    - table structure extraction
    - page images
    - picture images
    - table images
    - formula enrichment
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.do_formula_enrichment = True
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True
    pipeline_options.images_scale = float(images_scale)
    pipeline_options.accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CPU)

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def convert_pdf_chunk(
    converter: DocumentConverter,
    pdf_path: Path,
    page_range: Optional[Tuple[int, int]],
):
    """Convert either the full PDF or a requested page range."""
    if page_range is None:
        return converter.convert(str(pdf_path))
    return converter.convert(str(pdf_path), page_range=page_range)


# =============================================================================
# Asset extraction helpers
# =============================================================================
def save_page_images(doc, doc_name: str, paths: OutputPaths) -> None:
    """Save one rendered PNG per page."""
    for page_no, page in sorted(doc.pages.items()):
        page_image_path = paths.page_images / f"{doc_name}-page-{page_no}.png"
        with page_image_path.open("wb") as file:
            page.image.pil_image.save(file, format="PNG")


def collect_assets_from_doc(
    doc,
    doc_name: str,
    paths: OutputPaths,
    counters: Dict[str, int],
) -> Tuple[DefaultDict[int, List[Asset]], DefaultDict[int, List[Asset]], DefaultDict[int, List[Asset]]]:
    """Extract pictures, tables, and formulas from one converted chunk.

    Important:
    The order is intentionally preserved from doc.iterate_items() so the final
    markdown behaves the same as your current code.
    """
    page_pictures: DefaultDict[int, List[Asset]] = defaultdict(list)
    page_tables: DefaultDict[int, List[Asset]] = defaultdict(list)
    page_formulas: DefaultDict[int, List[Asset]] = defaultdict(list)

    for element, _level in doc.iterate_items():
        provenance = getattr(element, "prov", [])
        page_no = provenance[0].page_no if provenance else None
        if page_no is None:
            continue

        if isinstance(element, PictureItem):
            counters["picture"] += 1
            image_path = paths.picture_images / f"{doc_name}-picture-{counters['picture']}.png"
            image = element.get_image(doc)
            if image is None:
                continue

            with image_path.open("wb") as file:
                image.save(file, "PNG")

            page_pictures[page_no].append(
                Asset(
                    path=image_path,
                    description=describe_picture(image_path),
                )
            )
            continue

        if isinstance(element, TableItem):
            counters["table"] += 1
            image_path = paths.table_images / f"{doc_name}-table-{counters['table']}.png"
            image = element.get_image(doc)
            if image is None:
                continue

            with image_path.open("wb") as file:
                image.save(file, "PNG")

            page_tables[page_no].append(
                Asset(
                    path=image_path,
                    description=describe_table(image_path),
                )
            )
            continue

        is_formula = isinstance(element, FormulaItem) or (
            isinstance(element, TextItem) and getattr(element, "label", None) == DocItemLabel.FORMULA
        )
        if not is_formula:
            continue

        counters["formula"] += 1
        image_path = paths.formula_images / f"{doc_name}-formula-{counters['formula']}.png"
        image = element.get_image(doc)
        if image is None:
            continue

        with image_path.open("wb") as file:
            image.save(file, "PNG")

        extracted_latex = (getattr(element, "text", None) or "").strip() or None
        page_formulas[page_no].append(
            Asset(
                path=image_path,
                description=describe_formula(image_path, extracted_latex=extracted_latex),
                extracted_text=extracted_latex,
            )
        )

    return page_pictures, page_tables, page_formulas


# =============================================================================
# Page markdown rendering helpers
# =============================================================================
def build_replacement_block(kind: str, asset: Asset, output_root: Path) -> str:
    """Build the markdown block inserted in place of a placeholder."""
    relative_path = os.path.relpath(asset.path, output_root).replace("\\", "/")

    if kind == "image":
        label = "Figure"
    elif kind == "table":
        label = "Table"
    else:
        label = "Formula"

    return f"![{label}]({relative_path})\n\n{asset.description}\n\n"


def render_page_markdown(
    doc,
    page_no: int,
    output_root: Path,
    pictures: List[Asset],
    tables: List[Asset],
    formulas: List[Asset],
) -> str:
    """Build the final markdown for one page.

    Flow:
    1. Ask Docling for page markdown with image placeholders.
    2. Replace markdown tables and display formulas with custom placeholders.
    3. Walk through the page text from left to right.
    4. Replace each placeholder with the matching image and description.
    """
    raw_page_markdown = doc.export_to_markdown(
        image_mode=ImageRefMode.PLACEHOLDER,
        image_placeholder=IMAGE_PLACEHOLDER,
        page_no=page_no,
    )

    page_markdown = inject_table_placeholders(raw_page_markdown, TABLE_PLACEHOLDER)
    page_markdown = inject_formula_placeholders(page_markdown, FORMULA_PLACEHOLDER)

    picture_index = 0
    table_index = 0
    formula_index = 0
    cursor = 0
    rendered_parts: List[str] = []

    while True:
        next_image = page_markdown.find(IMAGE_PLACEHOLDER, cursor)
        next_table = page_markdown.find(TABLE_PLACEHOLDER, cursor)
        next_formula = page_markdown.find(FORMULA_PLACEHOLDER, cursor)

        if next_image == -1 and next_table == -1 and next_formula == -1:
            rendered_parts.append(page_markdown[cursor:])
            break

        candidates = [(next_image, "image"), (next_table, "table"), (next_formula, "formula")]
        candidates = [(position, kind) for position, kind in candidates if position != -1]
        position, kind = min(candidates, key=lambda item: item[0])

        rendered_parts.append(page_markdown[cursor:position])

        if kind == "image":
            replacement = ""
            if picture_index < len(pictures):
                replacement = build_replacement_block("image", pictures[picture_index], output_root)
                picture_index += 1
            rendered_parts.append(replacement)
            cursor = position + len(IMAGE_PLACEHOLDER)
            continue

        if kind == "table":
            replacement = ""
            if table_index < len(tables):
                replacement = build_replacement_block("table", tables[table_index], output_root)
                table_index += 1
            rendered_parts.append(replacement)
            cursor = position + len(TABLE_PLACEHOLDER)
            continue

        replacement = ""
        if formula_index < len(formulas):
            replacement = build_replacement_block("formula", formulas[formula_index], output_root)
            formula_index += 1
        rendered_parts.append(replacement)
        cursor = position + len(FORMULA_PLACEHOLDER)

    return "".join(rendered_parts).strip()


def process_converted_chunk(
    doc,
    doc_name: str,
    output_root: Path,
    paths: OutputPaths,
    page_separator_template: str,
    write_page_md_files: bool,
    counters: Dict[str, int],
) -> List[str]:
    """Process one converted chunk and return rendered page blocks."""
    save_page_images(doc, doc_name, paths)

    page_pictures, page_tables, page_formulas = collect_assets_from_doc(
        doc=doc,
        doc_name=doc_name,
        paths=paths,
        counters=counters,
    )

    page_blocks: List[str] = []

    for page_no in sorted(doc.pages.keys()):
        page_markdown_with_descriptions = render_page_markdown(
            doc=doc,
            page_no=page_no,
            output_root=output_root,
            pictures=page_pictures.get(page_no, []),
            tables=page_tables.get(page_no, []),
            formulas=page_formulas.get(page_no, []),
        )

        page_block = page_separator_template.format(page_no=page_no)
        if page_markdown_with_descriptions:
            page_block += page_markdown_with_descriptions + "\n"
        else:
            page_block += "\n"

        page_blocks.append(page_block)

        if write_page_md_files:
            page_file = paths.pages_md / f"page_{page_no:04d}.md"
            page_file.write_text(page_block, encoding="utf-8")

    return page_blocks



# =============================================================================
# Public API
# =============================================================================
def doc_processor_with_descriptions(
    pdf_path: str | Path,
    output_root: str | Path,
    page_range: Optional[Tuple[int, int]] = None,
    images_scale: float = 1.6,
) -> Path:
    """Convert a PDF into a stitched markdown file with image/table/formula descriptions.

    Simplified fixed design:
    - page-wise conversion only (1 page at a time)
    - per-page markdown files are always written
    - page separator format is fixed
    - if page_range is given, only those pages are processed
    - if page_range is not given, total pages are detected first and then processed
    """

    pdf_path = Path(pdf_path)
    output_root = Path(output_root)

    page_separator_template = "\n\n--- PAGE {page_no} ---\n\n"

    paths = create_output_paths(output_root, write_page_md_files=True)
    converter = build_docling_converter(images_scale=images_scale)
    doc_name = pdf_path.stem

    counters = {"picture": 0, "table": 0, "formula": 0}
    stitched_parts: List[str] = []

    if page_range is not None:
        start_page, end_page = page_range
    else:
        start_page = 1
        end_page = get_pdf_page_count(pdf_path)

    start_page = int(start_page)
    end_page = int(end_page)

    processed_pages = 0

    for page_no in range(start_page, end_page + 1):
        conversion_result = convert_pdf_chunk(converter, pdf_path, (page_no, page_no))

        rendered_page_parts = process_converted_chunk(
        doc=conversion_result.document,
        doc_name=doc_name,
        output_root=output_root,
        paths=paths,
        page_separator_template=page_separator_template,
        write_page_md_files=True,
        counters=counters,
        )

        stitched_parts.extend(rendered_page_parts)

        processed_pages += 1
        cleanup_memory()

    final_markdown = "\n".join(stitched_parts).strip() + "\n"
    paths.final_markdown.write_text(final_markdown, encoding="utf-8")

    print(f"Processed {processed_pages} page(s) from: {pdf_path.name}")
    print(f"Final markdown written to: {paths.final_markdown}")

    return paths.final_markdown