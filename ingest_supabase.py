"""
ingest_supabase.py

This script is a ONE-TIME ingestion utility.
Its purpose is to take a document (PDF), extract readable text from it,
split the text into smaller chunks, generate embeddings for each chunk,
and then insert those chunks + embeddings into a Supabase PostgreSQL table.

Why this exists:
- Your RAG chatbot needs a vector database to retrieve relevant text chunks.
- This script prepares that vector database by populating the "chunks" table.

High-level flow:
1) Read PDF file from disk
2) Extract text page-by-page
3) Chunk the text into overlapping segments
4) Generate embeddings for each chunk using Ollama embedding endpoint
5) Insert the chunk records into Supabase table "chunks"
"""

# ============================================================
# 1) IMPORTS
# ============================================================

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any
from dotenv import load_dotenv
from supabase import create_client
import fitz  # PyMuPDF
import requests
from tqdm import tqdm


# ============================================================
# 2) LOAD ENVIRONMENT VARIABLES
# ============================================================

# This loads variables from a .env file into the runtime environment.
# Example: SUPABASE_URL, SUPABASE_SERVICE_KEY, DOCUMENT_PATH, etc.
load_dotenv()


# ============================================================
# 3) CONFIGURATION (CENTRALIZED SETTINGS)
# ============================================================

@dataclass
class Settings:
    """
    This class is used to store all configuration values in one place.
    Instead of reading os.getenv() everywhere in the code, we read once,
    validate once, and then pass settings around.

    This makes the script:
    - easier to debug
    - easier to maintain
    - easier to migrate into a bigger codebase later
    """

    # Supabase connection details (service key is required for insert operations)
    SUPABASE_URL: str
    SUPABASE_SERVICE_KEY: str

    # Document ingestion path
    DOC_PATH: str

    # Ollama embedding API configuration
    OLLAMA_EMBED_URL: str
    OLLAMA_EMBED_MODEL: str

    # Batch size for DB inserts (and embedding calls per loop)
    BATCH_SIZE: int

    # Chunking strategy configuration
    CHUNK_SIZE: int
    CHUNK_OVERLAP: int


def load_settings() -> Settings:
    """
    This function reads environment variables and builds a Settings object.

    Why this is important:
    - If any required variable is missing, we fail early with a clear error.
    - This avoids wasting time by failing halfway through ingestion.
    """

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

    if not supabase_url or not supabase_key:
        raise SystemExit(
            "Missing Supabase credentials.\n\n"
            "Please set these in your .env file:\n"
            "  SUPABASE_URL=...\n"
            "  SUPABASE_SERVICE_KEY=...\n\n"
            "Note: Use the SERVICE ROLE KEY only on backend scripts (never frontend)."
        )

    return Settings(
        SUPABASE_URL=supabase_url,
        SUPABASE_SERVICE_KEY=supabase_key,
        DOC_PATH=os.getenv("DOCUMENT_PATH", "docs/DeepankarResume.pdf"),
        OLLAMA_EMBED_URL=os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings"),
        OLLAMA_EMBED_MODEL=os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large"),
        BATCH_SIZE=int(os.getenv("BATCH_SIZE", "6")),
        CHUNK_SIZE=int(os.getenv("CHUNK_SIZE", "1200")),
        CHUNK_OVERLAP=int(os.getenv("CHUNK_OVERLAP", "200")),
    )


# ============================================================
# 4) SUPABASE CLIENT SETUP
# ============================================================

def create_supabase_client(settings: Settings):
    """
    This function creates the Supabase client.

    Why this is needed:
    - Supabase provides a Python client which can insert/select/update rows.
    - We use the service role key because ingestion is a privileged operation.
    - This script is NOT meant to run in the browser.
    """
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_KEY)


# ============================================================
# 5) PDF TEXT EXTRACTION HELPERS
# ============================================================

def extract_pages_from_pdf(path: str) -> List[Dict[str, Any]]:
    """
    This function extracts raw text from a PDF using PyMuPDF.

    Output format:
    [
      {"page": 1, "text": "...."},
      {"page": 2, "text": "...."},
      ...
    ]

    Important behavior:
    - We skip pages that are empty or contain only whitespace.
    - This helps avoid creating useless chunks and embeddings.
    """
    doc = fitz.open(path)

    pages = []
    for i in range(len(doc)):
        page_text = doc[i].get_text("text")

        # We only keep pages that contain meaningful text.
        # This prevents ingesting blank pages.
        if page_text and page_text.strip():
            pages.append({"page": i + 1, "text": page_text})

    return pages


# ============================================================
# 6) CHUNKING HELPERS
# ============================================================

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """
    This function splits a long text into smaller overlapping chunks.

    Why chunking is required:
    - LLMs and embedding models cannot handle extremely long text in one call.
    - Smaller chunks improve retrieval quality because results are more specific.

    What overlap means:
    - Overlap ensures important sentences that sit near chunk boundaries
      are not lost.
    - Example: If chunk 1 ends mid-topic, chunk 2 starts slightly earlier
      so the topic is still captured.

    Example:
    text length = 5000
    chunk_size = 1200
    overlap = 200

    Then chunks are created like:
    - chunk1: 0..1200
    - chunk2: 1000..2200
    - chunk3: 2000..3200
    ...
    """
    chunks = []
    start = 0
    L = len(text)

    while start < L:
        end = start + chunk_size

        # We trim whitespace so stored chunks look clean in DB.
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start forward, but keep overlap region.
        start = max(0, end - overlap)

    return chunks


# ============================================================
# 7) EMBEDDING HELPERS (OLLAMA)
# ============================================================

def get_embedding_ollama(text: str, embed_url: str, model: str) -> List[float]:
    """
    This function calls Ollama embedding endpoint and returns an embedding vector.

    What "embedding vector" means:
    - A list of floating point numbers representing semantic meaning of the text.
    - Similar texts have embeddings closer to each other in vector space.

    Request format expected by Ollama:
    {
      "model": "mxbai-embed-large",
      "prompt": "some text"
    }

    Why timeout matters:
    - Embedding generation can sometimes be slow.
    - Timeout ensures the script doesn't hang forever.
    """
    payload = {"model": model, "prompt": text}

    try:
        r = requests.post(embed_url, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()

        # Most common response format:
        # { "embedding": [ ... ] }
        if "embedding" in data:
            return data["embedding"]

        # Some versions may return:
        # { "data": [ { "embedding": [...] } ] }
        if isinstance(data.get("data"), list) and len(data["data"]) > 0:
            return data["data"][0].get("embedding", [])

        # If we cannot find embedding, it means response format changed
        # or an unexpected error occurred.
        raise RuntimeError("Unexpected embedding response: " + json.dumps(data))

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama embedding request failed: {str(e)}") from e


# ============================================================
# 8) SUPABASE INSERT HELPERS
# ============================================================

def insert_chunks_batch(supabase, rows: List[Dict[str, Any]]):
    """
    This function inserts a batch of chunk records into Supabase.

    Why batching is important:
    - Inserting 1 row at a time is slow.
    - Batch inserts reduce network calls and speed up ingestion.

    Table expected in Supabase:
    public.chunks

    Columns expected:
    - document_title (text)
    - chunk_text (text)
    - chunk_tokens (int)
    - embedding (vector type / float array depending on your schema)
    - meta (jsonb)
    """
    try:
        res = supabase.table("chunks").insert(rows).execute()

        # Supabase client versions differ slightly.
        # Some return inserted rows in res.data.
        if hasattr(res, "data") and res.data:
            print(f"Successfully inserted {len(rows)} chunks")
        else:
            print("Insert request executed, but no inserted rows returned in response.")

        return res

    except Exception as e:
        print(f"Insert failed: {str(e)}")
        raise


# ============================================================
# 9) MAIN INGESTION PIPELINE
# ============================================================

def build_chunk_records(pages: List[Dict[str, Any]], document_title: str, settings: Settings) -> List[Dict[str, Any]]:
    """
    This function converts extracted pages into chunk records.

    Output format is a list of items like:
    {
      "document_title": "...",
      "chunk_text": "...",
      "chunk_tokens": 123,
      "meta": {"page": 1, "chunk_index": 0}
    }

    We create chunk_tokens mainly for:
    - debugging
    - analytics
    - future filtering (optional)
    """
    all_items = []

    for p in pages:
        chunks = chunk_text(
            p["text"],
            chunk_size=settings.CHUNK_SIZE,
            overlap=settings.CHUNK_OVERLAP
        )

        for idx, chunk in enumerate(chunks):
            meta = {
                "page": p["page"],
                "chunk_index": idx
            }

            all_items.append({
                "document_title": document_title,
                "chunk_text": chunk,
                # "chunk_tokens": len(chunk.split()),
                "meta": meta
            })

    return all_items


def ingest_document(path: str, settings: Settings, supabase):
    """
    This is the main function that runs ingestion end-to-end.

    Steps performed:
    1) Extract pages from PDF
    2) Chunk text
    3) For each chunk -> call Ollama -> generate embedding
    4) Insert into Supabase in batches

    This is called once per document.
    """
    if not os.path.exists(path):
        raise SystemExit(f"Document not found at path: {path}")

    document_title = os.path.splitext(os.path.basename(path))[0]

    # Step 1: Extract pages
    pages = extract_pages_from_pdf(path)
    print(f"Extracted pages with text: {len(pages)}")

    # Step 2: Build chunk records
    all_items = build_chunk_records(pages, document_title, settings)
    print(f"Total chunks created: {len(all_items)}")

    if not all_items:
        print("No chunks to ingest. Exiting.")
        return

    # Step 3 + 4: Embed + Insert in batches
    for i in tqdm(range(0, len(all_items), settings.BATCH_SIZE), desc="Embedding + inserting batches"):
        batch = all_items[i:i + settings.BATCH_SIZE]

        # Generate embeddings for each chunk in this batch
        embeddings = []
        for item in batch:
            emb = get_embedding_ollama(
                text=item["chunk_text"],
                embed_url=settings.OLLAMA_EMBED_URL,
                model=settings.OLLAMA_EMBED_MODEL
            )
            embeddings.append(emb)

        # Prepare DB rows for insertion
        rows = []
        for item, emb in zip(batch, embeddings):
            rows.append({
                "document_title": item["document_title"],
                "chunk_text": item["chunk_text"],
                # "chunk_tokens": item["chunk_tokens"],
                "embedding": emb,
                "meta": item["meta"]
            })

        # Insert into Supabase
        insert_chunks_batch(supabase, rows)


# ============================================================
# 10) SCRIPT ENTRYPOINT
# ============================================================

def main():
    """
    This is the entrypoint of the script.

    It loads settings, creates Supabase client, and runs ingestion.
    Keeping this separate makes it easier to test individual functions.
    """
    settings = load_settings()
    supabase = create_supabase_client(settings)

    ingest_document(settings.DOC_PATH, settings, supabase)
    print("Ingestion complete ✅")


if __name__ == "__main__":
    main()
