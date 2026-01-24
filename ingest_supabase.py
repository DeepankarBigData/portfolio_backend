# ingest_supabase.py one time
import os
import json
from dotenv import load_dotenv
from supabase import create_client
import fitz  # PyMuPDF
import requests
from tqdm import tqdm

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
DOC_PATH = os.getenv("DOCUMENT_PATH", "docs/SABARI.pdf")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def extract_pages(path):
    doc = fitz.open(path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text")
        if text and text.strip():
            pages.append({"page": i+1, "text": text})
    return pages

def chunk_text(text, chunk_size=1200, overlap=200):
    chunks=[]
    start=0
    L=len(text)
    while start < L:
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = max(0, end - overlap)
    return chunks

def get_embedding_ollama(text):
    payload = {"model": OLLAMA_EMBED_MODEL, "prompt": text}  # Ollama uses "prompt" not "input"
    r = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    if "embedding" in data:
        return data["embedding"]
    if isinstance(data.get("data"), list):
        return data["data"][0].get("embedding")
    raise RuntimeError("Unexpected embedding response: " + json.dumps(data))

def upsert_batch(rows):
    try:
        res = supabase.table("chunks").insert(rows).execute()
        # Supabase 2.x returns data directly, check if it's successful
        if hasattr(res, 'data') and res.data:
            print(f"Successfully inserted {len(rows)} chunks")
        else:
            print("Warning: Insert may have failed - no data returned")
        return res
    except Exception as e:
        print(f"Upsert failed: {str(e)}")
        raise

def ingest(path):
    pages = extract_pages(path)
    print("pages:", len(pages))
    all_items=[]
    title = os.path.splitext(os.path.basename(path))[0]
    for p in pages:
        chs = chunk_text(p["text"], chunk_size=1200, overlap=200)
        for idx, c in enumerate(chs):
            meta = {"page": p["page"], "chunk_index": idx}
            all_items.append({"document_title": title, "chunk_text": c, "chunk_tokens": len(c.split()), "meta": meta})
    print("total chunks:", len(all_items))

    # embeddings in batches
    for i in tqdm(range(0, len(all_items), BATCH_SIZE)):
        batch = all_items[i:i+BATCH_SIZE]
        embeddings = []
        for item in batch:
            emb = get_embedding_ollama(item["chunk_text"])
            embeddings.append(emb)
        rows = []
        for item, emb in zip(batch, embeddings):
            row = {
                "document_title": item["document_title"],
                "chunk_text": item["chunk_text"],
                "chunk_tokens": item["chunk_tokens"],
                "embedding": emb,
                "meta": item["meta"]
            }
            rows.append(row)
        upsert_batch(rows)

if __name__ == "__main__":
    ingest(DOC_PATH)
    print("done")
