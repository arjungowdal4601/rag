import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import chromadb  # modern Chroma client


# ---------------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------------
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")

# ---------------------------------------------------------
# 2. Initialize OpenAI Embeddings
# ---------------------------------------------------------
embeddings = OpenAIEmbeddings(
    api_key=openai_key,
    model=EMBEDDING_MODEL,
)


def ingest_chunks_to_chroma(
    json_path: str,
    chroma_path: str,
    collection_name: str,
) -> int:
    """
    Load chunk JSON, embed with OpenAI, and upsert into a Chroma collection.

    Parameters
    ----------
    json_path : str
        Path to the JSON file containing chunks.
        Expected format: {chunk_id: {"refreshed_content": "...", "source": [page_nums...]}, ...}
    chroma_path : str, optional
        Folder path where Chroma will store its persistent data.
    collection_name : str, optional
        Name of the Chroma collection.

    Returns
    -------
    int
        Number of chunks successfully inserted.
    """


    # ---------------------------------------------------------
    # 3. Initialize local ChromaDB (persistent, new API)
    # ---------------------------------------------------------
    chroma_client = chromadb.PersistentClient(path=chroma_path)

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # optional, but fine to keep
    )

    # ---------------------------------------------------------
    # 4. Load your chunk JSON file
    # ---------------------------------------------------------
    with open(json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # ---------------------------------------------------------
    # 5. Prepare documents, metadatas, and ids
    # ---------------------------------------------------------
    documents = []
    metadatas = []
    ids = []

    for chunk_id, data in chunks.items():
        # Take the refreshed text for embedding
        text = data.get("refreshed_content", "").strip()
        source_pages = data.get("source", [])

        if not text:
            continue  # skip empty

        documents.append(text)
        ids.append(str(chunk_id))  # ensure string ids

        # Chroma metadata values MUST be scalar types, not lists.
        # So we convert the pages list into a string like "1" or "1,2".
        if isinstance(source_pages, list):
            pages_str = ",".join(str(p) for p in source_pages)
        else:
            pages_str = str(source_pages)

        metadatas.append({"pages": pages_str})

    if not documents:
        print("⚠️ No non-empty chunks found to embed. Nothing was inserted.")
        return 0

    # ---------------------------------------------------------
    # 6. Compute embeddings with OpenAI
    # ---------------------------------------------------------
    print("🔄 Generating embeddings from OpenAI...")
    vectors = embeddings.embed_documents(documents)
    print(f"✅ Generated {len(vectors)} embeddings for {len(documents)} documents.")

    # ---------------------------------------------------------
    # 7. Insert into ChromaDB
    # ---------------------------------------------------------
    collection.upsert(
        documents=documents,
        embeddings=vectors,
        metadatas=metadatas,
        ids=ids,
    )

    print("\n🎉 DONE — Chunks stored in ChromaDB successfully!")
    print(f"📦 Location on disk: {chroma_path}")
    print(f"📚 Collection name : {collection_name}")
    print(f"🔢 Total chunks    : {len(documents)}")

    return len(documents)

