import os
import re
import gc
import base64
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ==========================================================
# OpenAI VLM setup (image-capable)
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
    """Describe a figure/image for retrieval (dense, factual prose)."""
    img_b64 = _encode_image_to_b64(image_path)
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
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                },
            ]
        ),
    ]
    resp = vision_llm.invoke(messages)
    return resp.content.strip()


def describe_table(image_path: Path) -> str:
    """Describe a table image for retrieval (dense, factual prose; no table reconstruction)."""
    img_b64 = _encode_image_to_b64(image_path)
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
                    "text": "Describe this table for retrieval (plain English, no headings, no markdown table).",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                },
            ]
        ),
    ]
    resp = vision_llm.invoke(messages)
    return resp.content.strip()


def describe_formula(image_path: Path, extracted_latex: Optional[str] = None) -> str:
    """Describe a formula crop for retrieval.

    Output format:
      - First line starts with: "LaTeX: ..."
      - Then a short factual explanation (no guessing about variable meaning)
    """
    img_b64 = _encode_image_to_b64(image_path)
    latex_hint = (extracted_latex or "").strip()
    latex_line = f"LaTeX: {latex_hint}" if latex_hint else "LaTeX: (not provided)"

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
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                },
            ]
        ),
    ]

    resp = vision_llm.invoke(messages)
    text = resp.content.strip()
    if not text.lower().startswith("latex:"):
        text = f"{latex_line}\n{text}".strip()
    return text


# ------------------------------------------------------------------------------------------------------------


def inject_table_placeholders(md: str, placeholder: str = "[[DOC_TABLE]]") -> str:
    """Replace markdown table blocks with a single placeholder line."""
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
                if "|" in next_line and all(ch in "-:| " for ch in next_line if ch):
                    out_lines.append(placeholder)
                    i += 2
                    while i < len(lines) and "|" in lines[i]:
                        i += 1
                    if i < len(lines) and lines[i].strip() == "":
                        out_lines.append("")
                        i += 1
                    continue

        out_lines.append(line)
        i += 1
    return "\n".join(out_lines)


def inject_formula_placeholders(md: str, placeholder: str = "[[DOC_FORMULA]]") -> str:
    r"""Replace common Markdown/LaTeX display-math blocks with a placeholder.

    Covered patterns:
      - $$ ... $$
      - \\[ ... \\]
      - \\begin{equation/align/gather/multline...} ... \\end{...}
    """
    patterns = [
        r"(?s)\$\$.*?\$\$",
        r"(?s)\\\[.*?\\\]",
        r"(?s)\\begin\{equation\*?\}.*?\\end\{equation\*?\}",
        r"(?s)\\begin\{align\*?\}.*?\\end\{align\*?\}",
        r"(?s)\\begin\{gather\*?\}.*?\\end\{gather\*?\}",
        r"(?s)\\begin\{multline\*?\}.*?\\end\{multline\*?\}",
    ]
    out = md
    for pat in patterns:
        out = re.sub(pat, f"\n{placeholder}\n", out)
    return out


# ==========================================================
# Main processor
# ==========================================================


def _get_pdf_page_count_best_effort(pdf_path: Path) -> Optional[int]:
    """Return number of pages in a PDF using lightweight readers (best-effort)."""
    for mod_name in ("pypdf", "PyPDF2"):
        try:
            mod = __import__(mod_name)
            PdfReader = getattr(mod, "PdfReader")
            with pdf_path.open("rb") as f:
                return len(PdfReader(f).pages)
        except Exception:
            continue
    return None


def doc_processor_with_descriptions(
    pdf_path: str | Path,
    output_root: str | Path,
    page_separator_template: str = "\n\n--- PAGE {page_no} ---\n\n",
    *,
    # ===== Memory / device knobs =====
    device: str = "cpu",
    num_threads: int = 4,
    images_scale: float = 1.6,
    # Batching knobs for StandardPdfPipeline (threaded mode). Lower => less memory.
    ocr_batch_size: int = 1,
    layout_batch_size: int = 1,
    table_batch_size: int = 1,
    queue_max_size: int = 2,
    # If you try CUDA and it OOMs, automatically retry on CPU.
    retry_cpu_on_cuda_oom: bool = True,
    # ===== Page-wise conversion knobs =====
    pages_per_chunk: int = 1,
    page_range: Optional[Tuple[int, int]] = None,
    write_page_md_files: bool = True,
) -> Path:
    """Convert a PDF to Markdown, exporting page images, pictures, tables, and formulas.

    Why this version exists:
      - On small GPUs (e.g., 2GB VRAM), Docling's layout stage often OOMs.
      - Processing page-by-page (or small chunks) + batch sizes of 1 keeps peak memory low.
      - You can also force CPU via AcceleratorOptions (official Docling way).

    Outputs:
      - {output_root}/page_images/        : full-page PNG images
      - {output_root}/image_png_images/   : PNG images of figures/pictures
      - {output_root}/table_images/       : PNG images of tables
      - {output_root}/formula_images/     : PNG images of formulas (equation crops)
      - {output_root}/pages_md/           : per-page Markdown files (optional)
      - {output_root}/processed_doc.md    : final stitched Markdown

    Notes:
      - `page_range` is 1-based, as used in Docling.
      - `pages_per_chunk=1` processes one page at a time.
    """

    # Lazy imports so we can control accelerator/device per-call.
    from docling_core.types.doc import (
        ImageRefMode,
        PictureItem,
        TableItem,
        FormulaItem,
        TextItem,
        DocItemLabel,
    )
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pdf_path = Path(pdf_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    # --- Output dirs ---
    page_dir = output_root / "page_images"
    picture_dir = output_root / "image_png_images"
    table_dir = output_root / "table_images"
    formula_dir = output_root / "formula_images"
    pages_md_dir = output_root / "pages_md"

    for d in (page_dir, picture_dir, table_dir, formula_dir):
        d.mkdir(exist_ok=True)
    if write_page_md_files:
        pages_md_dir.mkdir(exist_ok=True)

    # --- Accelerator selection (official Docling way) ---
    device_norm = (device or "cpu").strip().lower()
    if device_norm in {"cpu", "host"}:
        accel_device = AcceleratorDevice.CPU
    elif device_norm in {"auto"}:
        accel_device = AcceleratorDevice.AUTO
    elif device_norm in {"cuda", "gpu"}:
        accel_device = AcceleratorDevice.CUDA
    elif device_norm in {"mps"}:
        accel_device = AcceleratorDevice.MPS
    elif device_norm in {"xpu"}:
        accel_device = AcceleratorDevice.XPU
    else:
        # Docling supports passing raw device strings like "cuda:1".
        accel_device = device

    accelerator_options = AcceleratorOptions(num_threads=num_threads, device=accel_device)

    # Use ThreadedPdfPipelineOptions so we can explicitly control batching/backpressure.
    pipeline_options = ThreadedPdfPipelineOptions(
        ocr_batch_size=int(ocr_batch_size),
        layout_batch_size=int(layout_batch_size),
        table_batch_size=int(table_batch_size),
        queue_max_size=int(queue_max_size),
    )
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.images_scale = float(images_scale)

    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True

    # Formula extraction (Docling enrichment)
    pipeline_options.do_formula_enrichment = True

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    def _convert_with_optional_retry(pr: Optional[Tuple[int, int]] = None):
        """Run Docling conversion, optionally limited to a page range; retry on CPU if CUDA OOM."""
        try:
            if pr is None:
                return converter.convert(str(pdf_path))
            return converter.convert(str(pdf_path), page_range=pr)
        except Exception as e:
            msg = str(e)
            if (
                retry_cpu_on_cuda_oom
                and ("CUDA out of memory" in msg or "torch.OutOfMemoryError" in msg)
                and device_norm not in {"cpu", "host"}
            ):
                # Best-effort cache cleanup
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                cpu_accel = AcceleratorOptions(num_threads=num_threads, device=AcceleratorDevice.CPU)
                pipeline_options.accelerator_options = cpu_accel
                cpu_converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )
                if pr is None:
                    return cpu_converter.convert(str(pdf_path))
                return cpu_converter.convert(str(pdf_path), page_range=pr)
            raise

    # ------------------------------------------------------
    # Globals across chunks (for stable filenames & stitching)
    # ------------------------------------------------------
    doc_name = pdf_path.stem
    picture_counter = 0
    table_counter = 0
    formula_counter = 0

    IMAGE_PLACEHOLDER = "[[DOC_IMAGE]]"
    TABLE_PLACEHOLDER = "[[DOC_TABLE]]"
    FORMULA_PLACEHOLDER = "[[DOC_FORMULA]]"

    stitched_parts: List[str] = []

    def _process_doc_chunk(doc) -> None:
        """Save assets + build per-page Markdown for the pages present in this DoclingDocument."""
        nonlocal picture_counter, table_counter, formula_counter

        # Save page images
        for page_no, page in sorted(doc.pages.items()):
            page_img_path = page_dir / f"{doc_name}-page-{page_no}.png"
            with page_img_path.open("wb") as fp:
                page.image.pil_image.save(fp, format="PNG")

        # Collect pictures/tables/formulas for this chunk
        page_pictures: Dict[int, List[Tuple[Path, str]]] = defaultdict(list)
        page_tables: Dict[int, List[Tuple[Path, str]]] = defaultdict(list)
        page_formulas: Dict[int, List[Tuple[Path, str]]] = defaultdict(list)

        for element, _level in doc.iterate_items():
            prov = getattr(element, "prov", [])
            page_no = prov[0].page_no if prov else None
            if page_no is None:
                continue

            if isinstance(element, PictureItem):
                picture_counter += 1
                img_path = picture_dir / f"{doc_name}-picture-{picture_counter}.png"
                img = element.get_image(doc)
                if img is not None:
                    with img_path.open("wb") as fp:
                        img.save(fp, "PNG")
                    desc = describe_picture(img_path)
                    page_pictures[page_no].append((img_path, desc))

            elif isinstance(element, TableItem):
                table_counter += 1
                img_path = table_dir / f"{doc_name}-table-{table_counter}.png"
                img = element.get_image(doc)
                if img is not None:
                    with img_path.open("wb") as fp:
                        img.save(fp, "PNG")
                    desc = describe_table(img_path)
                    page_tables[page_no].append((img_path, desc))

            elif isinstance(element, FormulaItem) or (
                isinstance(element, TextItem) and getattr(element, "label", None) == DocItemLabel.FORMULA
            ):
                formula_counter += 1
                img_path = formula_dir / f"{doc_name}-formula-{formula_counter}.png"
                img = element.get_image(doc)
                if img is not None:
                    with img_path.open("wb") as fp:
                        img.save(fp, "PNG")
                    extracted_latex = (getattr(element, "text", None) or "").strip() or None
                    desc = describe_formula(img_path, extracted_latex=extracted_latex)
                    page_formulas[page_no].append((img_path, desc))

        # Build Markdown for each page in this chunk
        for page_no in sorted(doc.pages.keys()):
            page_md_raw = doc.export_to_markdown(
                image_mode=ImageRefMode.PLACEHOLDER,
                image_placeholder=IMAGE_PLACEHOLDER,
                page_no=page_no,
            )
            page_md = inject_table_placeholders(page_md_raw, TABLE_PLACEHOLDER)
            page_md = inject_formula_placeholders(page_md, FORMULA_PLACEHOLDER)

            pictures = page_pictures.get(page_no, [])
            tables = page_tables.get(page_no, [])
            formulas = page_formulas.get(page_no, [])

            pic_idx = tbl_idx = frm_idx = 0
            cursor = 0
            out_chunks: List[str] = []

            while True:
                next_img = page_md.find(IMAGE_PLACEHOLDER, cursor)
                next_tbl = page_md.find(TABLE_PLACEHOLDER, cursor)
                next_frm = page_md.find(FORMULA_PLACEHOLDER, cursor)

                if next_img == -1 and next_tbl == -1 and next_frm == -1:
                    out_chunks.append(page_md[cursor:])
                    break

                candidates = [(next_img, "image"), (next_tbl, "table"), (next_frm, "formula")]
                candidates = [(pos, kind) for (pos, kind) in candidates if pos != -1]
                pos, kind = min(candidates, key=lambda x: x[0])

                out_chunks.append(page_md[cursor:pos])

                if kind == "table":
                    if tbl_idx < len(tables):
                        img_path, desc = tables[tbl_idx]
                        tbl_idx += 1
                        rel_path = os.path.relpath(img_path, output_root).replace("\\", "/")
                        replacement = f"![Table]({rel_path})\n\n{desc}\n\n"
                    else:
                        replacement = ""
                    out_chunks.append(replacement)
                    cursor = pos + len(TABLE_PLACEHOLDER)

                elif kind == "formula":
                    if frm_idx < len(formulas):
                        img_path, desc = formulas[frm_idx]
                        frm_idx += 1
                        rel_path = os.path.relpath(img_path, output_root).replace("\\", "/")
                        replacement = f"![Formula]({rel_path})\n\n{desc}\n\n"
                    else:
                        replacement = ""
                    out_chunks.append(replacement)
                    cursor = pos + len(FORMULA_PLACEHOLDER)

                else:  # image
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
            page_block = page_separator_template.format(page_no=page_no) + (page_md_with_desc + "\n" if page_md_with_desc else "\n")
            stitched_parts.append(page_block)

            if write_page_md_files:
                (pages_md_dir / f"page_{page_no:04d}.md").write_text(page_block, encoding="utf-8")

    # ------------------------------------------------------
    # Page-wise conversion loop
    # ------------------------------------------------------
    if pages_per_chunk is None or int(pages_per_chunk) <= 0:
        # Whole document in one shot
        conv_res = _convert_with_optional_retry(None)
        _process_doc_chunk(conv_res.document)
    else:
        start_page, end_page = (page_range if page_range is not None else (1, -1))
        start_page = int(start_page)

        # If end_page is unknown, try to detect it; otherwise fall back to probe-loop.
        detected_pages = None if int(end_page) == -1 else int(end_page)
        if detected_pages is None:
            detected_pages = _get_pdf_page_count_best_effort(pdf_path)

        chunk_size = int(pages_per_chunk)

        if detected_pages is not None:
            if page_range is None:
                end_page = int(detected_pages)
            else:
                end_page = int(end_page)

            for chunk_start in range(start_page, end_page + 1, chunk_size):
                chunk_end = min(end_page, chunk_start + chunk_size - 1)
                conv_res = _convert_with_optional_retry((chunk_start, chunk_end))
                _process_doc_chunk(conv_res.document)

                # Aggressive cleanup between chunks
                del conv_res
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        else:
            # Probe page-by-page until Docling yields no pages / errors.
            page_no = start_page
            while True:
                if page_range is not None and page_no > int(end_page):
                    break
                try:
                    conv_res = _convert_with_optional_retry((page_no, page_no))
                except Exception:
                    break

                doc = conv_res.document
                if not getattr(doc, "pages", None):
                    break

                _process_doc_chunk(doc)

                del conv_res
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass

                page_no += 1

    final_md = "\n".join(stitched_parts).strip() + "\n"
    md_path = output_root / "processed_doc.md"
    md_path.write_text(final_md, encoding="utf-8")
    return md_path

