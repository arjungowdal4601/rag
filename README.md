# RAG Pipeline (OpenAI + Chroma + Streamlit)

This repo contains a simple end-to-end RAG workflow:

1) Convert a PDF into page-level markdown with figure/table descriptions.
2) Chunk the markdown with an OpenAI vision model.
3) Embed chunks and store them in ChromaDB.
4) Ask questions via a Streamlit app that retrieves chunks and answers.

## Files

- `doc_processor.py`: PDF -> page images + figure/table images + `processed_doc.md`.
- `chunking.py`: Page-aware semantic chunking with OpenAI vision + writes `chunks.json`.
- `vectorization.py`: Embeds chunks and upserts them into Chroma.
- `app.py`: Streamlit RAG UI.
- `main.ipynb`: Example end-to-end pipeline runner.

## Setup

Create a `.env` file with:

```
OPENAI_API_KEY=...     # OpenAI API key
OPENAI_VISION_MODEL=... # Optional, defaults to gpt-4o-mini
OPENAI_CHAT_MODEL=...   # Optional, defaults to gpt-4o-mini
OPENAI_EMBED_MODEL=...  # Optional, defaults to text-embedding-3-large
DOC_NAME=sample_2      # Used by app.py for page image lookup
PAGE_IMAGES_DIR=doc_assets/page_images
```

Install dependencies (examples, adjust to your environment):

```
pip install -r requirements.txt
```

## Run the pipeline

From a notebook:

```
from chunking import chunk_markdown_with_llm
from doc_processor import doc_processor_with_descriptions
from vectorization import ingest_chunks_to_chroma

pdf_path = "sample_2.pdf"
output_dir = "doc_assets"
md_path = doc_processor_with_descriptions(pdf_path, output_root=output_dir)
json_path = chunk_markdown_with_llm(md_path)
ingest_chunks_to_chroma(json_path, chroma_path="doc_assets/embedding_chroma_db/", collection_name="embedding")
```

## Run the app

```
streamlit run app.py
```

## Notes

- `doc_processor_with_descriptions` writes `doc_assets/processed_doc.md`.
- The chunker derives the document name from the markdown filename, while page images
  are named from the original PDF stem. If you want page images in chunking, keep
  the markdown filename aligned with the PDF stem, or adjust the code to pass the
  correct document name.
