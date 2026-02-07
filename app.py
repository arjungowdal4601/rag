# helper_v5.py

import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import streamlit as st
from dotenv import load_dotenv
import chromadb
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage


# ==========================================================
# 1. Environment + global configuration
# ==========================================================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY must be set in .env")

# Embedding config (same as used during indexing)
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

# Chat model config (for rephrasing + answering)
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

# Chroma config
CHROMA_PATH = "doc_assets/embedding_chroma_db/"
CHROMA_COLLECTION_NAME = "embedding"

# Document / page images config
DOC_NAME = os.getenv("DOC_NAME", "sample_2")
PAGE_IMAGES_DIR = Path(os.getenv("PAGE_IMAGES_DIR", "doc_assets/page_images"))

SIMILARITY_THRESHOLD = 0.65  # 80% similarity

# ==========================================================
# 2. Initialize models + vector store
# ==========================================================
@st.cache_resource
def get_embeddings_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        model=EMBED_MODEL,
    )


@st.cache_resource
def get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=CHAT_MODEL,
        temperature=0.1,
    )


@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


embeddings = get_embeddings_model()
qa_llm = get_chat_model()
collection = get_chroma_collection()

# ==========================================================
# 3. Utility helpers
# ==========================================================
def encode_image_to_b64(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_pages_str(pages_str: str) -> List[int]:
    if not pages_str:
        return []
    out: List[int] = []
    for p in pages_str.split(","):
        p = p.strip()
        if p.isdigit():
            out.append(int(p))
    return out


def load_page_images(pages: List[int]) -> List[Dict[str, Any]]:
    """Load page screenshots and return list of {page, b64}."""
    unique_pages = sorted(set(pages))
    images: List[Dict[str, Any]] = []

    for p in unique_pages:
        img_path = PAGE_IMAGES_DIR / f"{DOC_NAME}-page-{p}.png"
        b64 = encode_image_to_b64(img_path)
        if b64:
            images.append({"page": p, "b64": b64})

    return images


# ==========================================================
# 4. Core RAG pipeline functions
# ==========================================================
def rephrase_query(user_query: str) -> str:
    """Use LLM to rewrite the user query into a clean retrieval query."""
    messages = [
        SystemMessage(
            content=(
                "You rewrite user questions into short, precise search queries "
                "that are ideal for semantic vector retrieval. "
                "Keep the core meaning, but remove filler words. "
                "Return ONLY the rewritten query text, nothing else."
            )
        ),
        HumanMessage(content=user_query),
    ]
    resp = qa_llm.invoke(messages)
    return resp.content.strip()


def retrieve_chunks(
    query: str,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    max_results: int = 10,
) -> List[Dict[str, Any]]:
    """Retrieve chunks from Chroma using precomputed embeddings & cosine similarity."""
    query_vec = embeddings.embed_query(query)

    # ❗ FIX: do NOT include "ids" here – Chroma returns ids by default.
    res = collection.query(
        query_embeddings=[query_vec],
        n_results=max_results,
        include=["documents", "metadatas", "distances"],
    )

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    ids = res.get("ids", [[]])[0]  # always present; no need to ask via `include`

    results: List[Dict[str, Any]] = []
    for doc, meta, dist, cid in zip(docs, metas, dists, ids):
        if dist is None:
            sim = None
        else:
            # For cosine, distance = 1 - similarity
            sim = 1.0 - dist

        pages_str = meta.get("pages", "") if isinstance(meta, dict) else ""
        results.append(
            {
                "id": cid,
                "text": doc,
                "pages_str": pages_str,
                "pages": parse_pages_str(pages_str),
                "similarity": sim,
            }
        )

    # Filter by similarity threshold (>= 0.8)
    filtered = [
        r
        for r in results
        if r["similarity"] is not None and r["similarity"] >= similarity_threshold
    ]

    # Fallback: if nothing passes threshold, at least return top 3
    if not filtered:
        filtered = results[:3]

    return filtered


def generate_answer(
    original_question: str,
    retrieved_chunks: List[Dict[str, Any]],
    page_images: List[Dict[str, Any]],
) -> str:
    """Call OpenAI chat model with text chunks and page screenshots to generate answer."""
    # Build context text
    ctx_parts: List[str] = []
    for r in retrieved_chunks:
        pages_display = ",".join(str(p) for p in r["pages"]) if r["pages"] else "?"
        sim = r["similarity"]
        sim_str = f"(similarity ≈ {sim:.2f})" if isinstance(sim, float) else ""
        ctx_parts.append(
            f"[Pages {pages_display}] {sim_str}\n{r['text']}"
        )

    context_text = "\n\n-----\n\n".join(ctx_parts)

    text_block = (
        "You are an assistant answering questions about the research paper "
        "'Attention Is All You Need'. You have:\n"
        "- Text chunks retrieved from a vector store.\n"
        "- Page screenshots for visual context.\n\n"
        "Use ONLY this information to answer the question. "
        "If the answer is not clearly contained in the context, say that you don't know.\n\n"
        f"CONTEXT CHUNKS:\n{context_text}\n\n"
        f"QUESTION: {original_question}\n\n"
        "Answer in clear, structured points or table if required be creative with output formate. Mention relevant page numbers in your answer."
    )

    # Build multimodal HumanMessage: text + images
    human_content: List[Dict[str, Any]] = [{"type": "text", "text": text_block}]
    for img in page_images:
        human_content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img['b64']}",
                },
            }
        )

    messages = [
        SystemMessage(
            content=(
                "You are a precise, grounded question-answering system for a specific PDF. "
                "You never hallucinate; if the context is insufficient, say you don't know."
            )
        ),
        HumanMessage(content=human_content),
    ]

    resp = qa_llm.invoke(messages)
    return resp.content.strip()


def answer_question(user_query: str) -> Tuple[str, str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Full pipeline: rephrase → retrieve → load images → answer."""
    rephrased = rephrase_query(user_query)
    retrieved = retrieve_chunks(rephrased)

    # Collect pages used
    pages: List[int] = []
    for r in retrieved:
        pages.extend(r["pages"])
    page_images = load_page_images(pages)

    answer = generate_answer(user_query, retrieved, page_images)
    return answer, rephrased, retrieved, page_images


# ==========================================================
# 5. Streamlit UI
# ==========================================================
def main():
    st.set_page_config(
        page_title="Regulation co-pilot",
        page_icon="📘",
        layout="wide",
    )

    st.title("📘 Regulation co-pilot — RAG QA")
    st.markdown(
        "Ask any question about the paper. The system will:\n"
        "1. Rephrase your query for better retrieval\n"
        "2. Search the vector store (only chunks ≥ 80% similarity)\n"
        "3. Use text + page screenshots to answer\n"
    )

    user_query = st.text_area(
        "🔹 Your question",
        height=100,
        placeholder="e.g., What is multi-head attention and why is it useful?",
    )
    run_button = st.button("🔍 Ask")

    if run_button and user_query.strip():
        with st.spinner("Thinking with RAG..."):
            answer, rephrased, retrieved, page_images = answer_question(user_query.strip())

        st.subheader("✅ Answer")
        st.write(answer)

        with st.expander("🔎 Debug: Retrieval details", expanded=False):
            st.markdown("**Rephrased query for retrieval:**")
            st.code(rephrased)

            st.markdown("**Retrieved chunks (after 0.8 similarity filter):**")
            for r in retrieved:
                pages_str = ",".join(str(p) for p in r["pages"]) if r["pages"] else "?"
                sim_str = f"{r['similarity']:.3f}" if isinstance(r["similarity"], float) else "N/A"
                st.markdown(
                    f"**ID:** `{r['id']}`  |  **Pages:** {pages_str}  |  **Similarity:** {sim_str}"
                )
                st.text_area(
                    label="Chunk text",
                    value=r["text"],
                    height=180,
                    key=f"chunk_{r['id']}",
                )

            if page_images:
                st.markdown("**Page screenshots used:**")
                cols = st.columns(min(len(page_images), 3))
                for i, img in enumerate(page_images):
                    with cols[i % len(cols)]:
                        st.image(
                            base64.b64decode(img["b64"]),
                            caption=f"Page {img['page']}",
                            use_column_width=True,
                        )
    else:
        st.info("Enter a question above and click **Ask** to run the RAG pipeline.")


if __name__ == "__main__":
    main()
