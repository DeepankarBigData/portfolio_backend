"""
app.py
FastAPI RAG API using:
- Supabase PostgreSQL + pgvector for retrieval
- Ollama embeddings + generation for answering

Flow:
User Query -> Embedding -> Vector Search -> Prompt Build -> LLM Answer -> Cache -> Response
"""

# ============================================================
# 1) IMPORTS
# ============================================================

import os
import atexit
import requests
import hashlib
from contextlib import contextmanager
from urllib.parse import urlparse, urlunparse, quote_plus

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
from dotenv import load_dotenv

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool


# ============================================================
# 2) LOAD ENVIRONMENT VARIABLES
# ============================================================

# Loads values from .env into os.environ
load_dotenv()


# ============================================================
# 3) CONFIGURATION (ENV VARS)
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings")
OLLAMA_GEN_URL = os.getenv("OLLAMA_GEN_URL", "http://localhost:11434/api/chat")

OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
OLLAMA_GEN_MODEL = os.getenv("OLLAMA_GEN_MODEL", "llama3")

VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))

# Allowed origins for CORS (frontend requests)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",") if os.getenv("ALLOWED_ORIGINS") else ["*"]


# ============================================================
# 4) DATABASE URL VALIDATION + FIXING
# ============================================================

def validate_and_fix_database_url(url: str) -> str | None:
    """
    Validates DATABASE_URL format and reconstructs it safely.
    
    Why needed?
    - Supabase DB passwords often contain special characters.
    - If password isn't URL-encoded, connection will fail.
    - Some malformed URLs break parsing.
    
    Expected format:
    postgresql://user:password@host:port/database
    """
    if not url:
        return None

    # Must start with correct PostgreSQL protocol
    if not url.startswith(("postgresql://", "postgres://")):
        return None

    try:
        parsed = urlparse(url)

        # Must have hostname
        if not parsed.hostname:
            return None

        # If password is missing but URL has "@", it is likely malformed
        if not parsed.password and "@" in url:
            return None

        # Build netloc properly
        if parsed.password:
            # URL encode password (handles special characters)
            password_encoded = quote_plus(parsed.password)
            netloc = f"{parsed.username}:{password_encoded}@{parsed.hostname}"
        else:
            netloc = f"{parsed.username}@{parsed.hostname}" if parsed.username else parsed.hostname

        if parsed.port:
            netloc += f":{parsed.port}"

        # Ensure database path exists
        fixed_url = urlunparse((
            parsed.scheme,
            netloc,
            parsed.path or "/postgres",
            parsed.params,
            parsed.query,
            parsed.fragment
        ))

        return fixed_url

    except Exception as e:
        print(f"Warning: Could not parse DATABASE_URL: {e}")
        return None


# ============================================================
# 5) CONSTRUCT DATABASE_URL (Fallback for Supabase Components)
# ============================================================

def build_database_url_if_missing() -> str:
    """
    If DATABASE_URL is missing/invalid, try constructing from Supabase env variables.
    """
    global DATABASE_URL

    # 1) If user already provided DATABASE_URL -> validate/fix it
    if DATABASE_URL:
        fixed_url = validate_and_fix_database_url(DATABASE_URL)
        if fixed_url:
            DATABASE_URL = fixed_url
            return DATABASE_URL
        else:
            print(f"Warning: DATABASE_URL format may be incorrect: {DATABASE_URL[:50]}...")
            print("Expected format: postgresql://user:password@host:port/database")

    # 2) If DATABASE_URL invalid -> try building from Supabase parts
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_db_password = os.getenv("SUPABASE_DB_PASSWORD")
    supabase_db_host = os.getenv("SUPABASE_DB_HOST")
    supabase_db_name = os.getenv("SUPABASE_DB_NAME", "postgres")
    supabase_db_user = os.getenv("SUPABASE_DB_USER", "postgres")

    if supabase_db_host and supabase_db_password:
        db_password_encoded = quote_plus(supabase_db_password)
        DATABASE_URL = f"postgresql://{supabase_db_user}:{db_password_encoded}@{supabase_db_host}:5432/{supabase_db_name}"
        print("Constructed DATABASE_URL from Supabase components")
        return DATABASE_URL

    # 3) If only SUPABASE_URL exists but no DATABASE_URL -> explain clearly
    if supabase_url:
        raise SystemExit(
            "DATABASE_URL not found or invalid in .env file.\n\n"
            "SUPABASE_URL is set, but you also need DATABASE_URL.\n\n"
            "Get it from:\n"
            "Supabase Dashboard > Settings > Database > Connection string (URI)\n\n"
            "Example:\n"
            "postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres"
        )

    # 4) Nothing exists -> fail fast
    raise SystemExit(
        "DATABASE_URL required in .env file.\n\n"
        "Get it from: Supabase Dashboard > Settings > Database > Connection string (URI)\n"
        "Format: postgresql://postgres:[PASSWORD]@db.[PROJECT].supabase.co:5432/postgres"
    )


# Ensure DATABASE_URL exists now
DATABASE_URL = build_database_url_if_missing()


# ============================================================
# 6) DATABASE CONNECTION POOL SETUP
# ============================================================

def ensure_db_url_has_required_params(url: str) -> str:
    """
    Ensures:
    - sslmode=require (Supabase requires SSL)
    - connect_timeout=10 (avoid long hangs)
    """
    if "sslmode" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"

    if "connect_timeout" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}connect_timeout=10"

    return url


def mask_database_url(url: str) -> str:
    """
    Prints safe URL without exposing password.
    """
    safe_url = url
    if "@" in safe_url:
        parts = safe_url.split("@")
        if ":" in parts[0]:
            user_pass = parts[0].split(":", 1)
            safe_url = f"{user_pass[0]}:***@{parts[1]}"
    return safe_url


DATABASE_URL = ensure_db_url_has_required_params(DATABASE_URL)

# Create connection pool (shared across requests)
try:
    parsed = urlparse(DATABASE_URL)
    if not parsed.hostname:
        raise ValueError("DATABASE_URL missing hostname")
    if not parsed.username:
        raise ValueError("DATABASE_URL missing username")

    print(f"Connecting to database: {mask_database_url(DATABASE_URL)}")
    print(f"Connecting to hostname: {parsed.hostname}")

    # Direct test connection (gives better startup error)
    try:
        print("Testing direct database connection...")
        test_conn = psycopg2.connect(DATABASE_URL, connect_timeout=15)
        with test_conn.cursor() as cur:
            cur.execute("SELECT 1")
        test_conn.close()
        print("Direct connection test successful")
    except psycopg2.OperationalError as e:
        raise SystemExit(f"Database connection failed: {str(e)}") from e

    # Pool for production: prevents reconnect per request
    connection_pool = SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        dsn=DATABASE_URL
    )

    print("Database connection pool created successfully")

except Exception as e:
    raise SystemExit(f"Failed to create database connection pool: {str(e)}") from e


@contextmanager
def get_db_connection():
    """
    Context manager to safely:
    - borrow connection from pool
    - return it after use

    This prevents leaking DB connections.
    """
    conn = None
    try:
        conn = connection_pool.getconn()
        yield conn
    finally:
        if conn:
            connection_pool.putconn(conn)


# ============================================================
# 7) FASTAPI APP INITIALIZATION
# ============================================================

app = FastAPI(title="RAG API (Supabase + Ollama)")

# CORS: allows browser frontend to call backend APIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],             # for development, allow all
    allow_credentials=False,         # must be False when allow_origins="*"
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ============================================================
# 8) REQUEST/RESPONSE MODELS
# ============================================================

class QueryRequest(BaseModel):
    """
    Incoming request payload for RAG query endpoint.
    
    query: user question
    k: how many chunks to retrieve from vector DB
    session_id: optional (future use: chat history / memory)
    """
    query: str
    k: int = 4
    session_id: str | None = None


# ============================================================
# 9) 
# These helper functions do two main jobs:

# (1) Convert the USER'S QUERY TEXT into an EMBEDDING VECTOR.
#     - The "text" here means the exact question typed by the user in the UI.
#     - Example user query: "What projects has Sabari done?"
#     - This text is sent to Ollama's /api/embeddings endpoint.
#     - Ollama returns a list of floats (vector), which we use for pgvector similarity search.

# (2) Convert the FINAL PROMPT into a GENERATED ANSWER.
#     - The "prompt" here is NOT only the user question.
#     - It is a combined text that includes:
#         a) System instructions (how the assistant should behave)
#         b) Retrieved context chunks from PostgreSQL (top-k results)
#         c) The original user question
#     - This full prompt is sent to Ollama's /api/chat endpoint.
#     - Ollama returns the final answer text that we show to the user.
# ============================================================

def get_embedding_ollama(text: str) -> list[float]:
    """
    Calls Ollama embedding endpoint.
    
    Ollama embedding API expects:
    { "model": "...", "prompt": "..." }

    Returns:
    embedding vector list[float]
    """
    payload = {"model": OLLAMA_EMBED_MODEL, "prompt": text}

    try:
        r = requests.post(OLLAMA_EMBED_URL, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()

        # Most common response format:
        if "embedding" in data:
            return data["embedding"]

        # Some versions return:
        # { "data": [ { "embedding": [...] } ] }
        if isinstance(data.get("data"), list) and len(data["data"]) > 0:
            return data["data"][0].get("embedding", [])

        raise RuntimeError(f"Unexpected embedding response: {data}")

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama embedding request failed: {str(e)}") from e


def generate_from_ollama(prompt: str, max_tokens: int = 128) -> str:
    """
    Calls Ollama chat endpoint for text generation.

    This uses /api/chat format:
    {
      "model": "...",
      "messages": [{"role": "user", "content": "..."}]
    }

    Returns:
    final answer string
    """
    payload = {
        "model": OLLAMA_GEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "options": {
            "temperature": 0.0,
            "num_predict": max_tokens,
            "num_ctx": 2048,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "num_thread": 4
        },
        "stream": False
    }

    try:
        r = requests.post(OLLAMA_GEN_URL, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()

        # Most common response format:
        if "message" in data and isinstance(data["message"], dict):
            content = data["message"].get("content", "")
            if content and content.strip():
                return content.strip()

        # Fallback for older formats
        if "response" in data:
            return str(data["response"]).strip()
        if "text" in data:
            return str(data["text"]).strip()

        raise RuntimeError(f"Unexpected generation response structure: {data}")

    except requests.exceptions.Timeout:
        raise RuntimeError("Ollama generation request timed out after 120 seconds") from None
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama generation request failed: {str(e)}") from e


# ============================================================
# 10) VECTOR SEARCH HELPERS (pgvector)
# ============================================================

def vector_to_literal(vec: list[float]) -> str:
    """
    Converts python list -> pgvector literal string.
    Example: [0.1,0.2,...]
    """
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


def pg_vector_search(query_vec: list[float], k: int = 4) -> list[dict]:
    """
    Runs pgvector similarity search.
    
    Uses <#> operator (distance).
    We convert distance to score using: 1 - distance
    """
    vec_lit = vector_to_literal(query_vec)

    sql = """
    SELECT chunk_id, document_title, chunk_text, meta,
      1 - (embedding <#> %s::vector) AS score
    FROM public.chunks
    ORDER BY embedding <#> %s::vector
    LIMIT %s;
    """

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, (vec_lit, vec_lit, k))
                rows = cur.fetchall()
        return rows

    except psycopg2.Error as e:
        raise RuntimeError(f"Database query failed: {str(e)}") from e


# ============================================================
# 11) SIMPLE IN-MEMORY CACHE (FOR SPEED)
# ============================================================

# cache_key -> {"answer": ..., "sources": ..., "timings": ...}
_answer_cache = {}
_cache_max_size = 100


def _get_cache_key(query: str, k: int) -> str:
    """
    Creates stable cache key from query+k.
    Using md5 because it is small and fast.
    """
    cache_string = f"{query.lower().strip()}:{k}"
    return hashlib.md5(cache_string.encode()).hexdigest()


def _get_cached_answer(cache_key: str):
    return _answer_cache.get(cache_key)


def _cache_answer(cache_key: str, answer: str, sources: list, timings: dict):
    """
    Adds result to cache.
    If full, removes oldest entry (FIFO style).
    """
    if len(_answer_cache) >= _cache_max_size:
        oldest_key = next(iter(_answer_cache))
        del _answer_cache[oldest_key]

    _answer_cache[cache_key] = {
        "answer": answer,
        "sources": sources,
        "timings": timings,
        "cached": True
    }


# ============================================================
# 12) DEBUG / HEALTH ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    """
    Very fast health check.
    Used by load balancers.
    """
    return {"status": "ok", "message": "Server is running"}


@app.get("/health/detailed")
def health_detailed():
    """
    Slower health check:
    - DB ping
    - Ollama quick ping
    """
    import time
    start = time.time()

    # DB test
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {str(e)}"

    # Ollama test
    try:
        r = requests.get(OLLAMA_EMBED_URL.replace("/api/embeddings", "/api/tags"), timeout=0.5)
        ollama_status = "ok" if r.status_code == 200 else "error"
    except Exception:
        ollama_status = "error"

    elapsed = time.time() - start

    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "database": db_status,
        "ollama": ollama_status,
        "response_time_ms": round(elapsed * 1000, 2)
    }


@app.get("/test-embedding")
def test_embedding():
    """
    Quick embedding test endpoint.
    """
    import time
    test_text = "Hello world"
    start = time.time()

    try:
        emb = get_embedding_ollama(test_text)
        return {
            "success": True,
            "embedding_length": len(emb) if emb else 0,
            "time_elapsed": f"{(time.time() - start):.2f}s"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/test-ollama")
def test_ollama():
    """
    Quick generation test endpoint.
    """
    import time
    test_prompt = "Say hello in one sentence."
    start = time.time()

    try:
        result = generate_from_ollama(test_prompt, max_tokens=50)
        return {
            "success": True,
            "response": result,
            "response_length": len(result) if result else 0,
            "time_elapsed": f"{(time.time() - start):.2f}s"
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/cache/stats")
def cache_stats():
    """
    Debug endpoint to inspect cache.
    """
    return {
        "cache_size": len(_answer_cache),
        "cache_max_size": _cache_max_size,
        "cache_keys_preview": list(_answer_cache.keys())[:10]
    }


@app.get("/cache/clear")
def clear_cache():
    """
    Clears cache manually.
    Useful during development/testing.
    """
    global _answer_cache
    size = len(_answer_cache)
    _answer_cache = {}
    return {"message": f"Cache cleared. Removed {size} entries."}


# ============================================================
# 13) MAIN RAG ENDPOINT
# ============================================================

@app.post("/api/rag/query")
def rag_query(req: QueryRequest):
    """
    Main endpoint:
    1) Validate query
    2) Check cache
    3) Create embedding
    4) Retrieve top-k chunks from pgvector
    5) Build prompt with context
    6) Generate answer using Ollama
    7) Cache and return response
    """
    import logging
    import time

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # ----------------------------
    # Step 0: Validate input
    # ----------------------------
    q = (req.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query required")

    # ----------------------------
    # Step 1: Cache check
    # ----------------------------
    cache_key = _get_cache_key(q, req.k)
    cached = _get_cached_answer(cache_key)

    if cached:
        logger.info(f"Cache HIT for query: {q}")
        return {
            "answer": cached["answer"],
            "sources": cached["sources"],
            "timings": {**cached["timings"], "cached": True}
        }

    logger.info(f"Cache MISS - Processing query: {q}")

    # ----------------------------
    # Step 2: Embedding
    # ----------------------------
    t0 = time.time()
    try:
        q_emb = get_embedding_ollama(q)
        t_embed = time.time() - t0
        logger.info(f"Embedding completed: dim={len(q_emb)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"embedding failed: {e}") from e

    # ----------------------------
    # Step 3: Vector Search
    # ----------------------------
    t0 = time.time()
    try:
        results = pg_vector_search(q_emb, k=req.k)
        t_search = time.time() - t0
        logger.info(f"Vector search completed: found={len(results)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vector search failed: {e}") from e

    if not results:
        return {"answer": "I don't have relevant information for that query.", "sources": []}

    # ----------------------------
    # Step 4: Build Context
    # ----------------------------
    context_parts = []
    sources = []

    for r in results:
        txt = r.get("chunk_text", "")
        truncated = txt if len(txt) < 1200 else txt[:1200] + "..."

        title = (
            r.get("document_title")
            or (r.get("meta") or {}).get("title")
            or "Source"
        )

        context_parts.append(f"[{title}] {truncated}")

        sources.append({
            "chunk_id": r.get("chunk_id"),
            "chunk_text": truncated,
            "score": float(r.get("score", 0)),
            "meta": r.get("meta") or {}
        })

    # Reduce context length so Ollama stays fast
    max_context_length = 1500
    truncated_context = []
    current_length = 0

    for ctx in context_parts:
        if current_length + len(ctx) > max_context_length:
            break
        truncated_context.append(ctx)
        current_length += len(ctx)

    prompt = (
        "SYSTEM: You are Sabari's portfolio assistant. Use only provided chunks. "
        "If unknown, say 'I don't know'.\n\n"
        "CONTEXT:\n" + "\n\n".join(truncated_context) +
        f"\n\nUSER QUESTION: {q}\n\n"
        "INSTRUCTIONS: Provide a concise factual answer (3-6 sentences). List source titles."
    )

    # ----------------------------
    # Step 5: LLM Generation
    # ----------------------------
    t0 = time.time()
    try:
        answer = generate_from_ollama(prompt)
        t_llm = time.time() - t0

        if not answer or not answer.strip():
            answer = "I couldn't generate a response. Please try rephrasing your question."

        logger.info(f"LLM completed: length={len(answer)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM failed: {e}") from e

    # ----------------------------
    # Step 6: Timings + Cache
    # ----------------------------
    total_time = t_embed + t_search + t_llm
    timings = {
        "embed_seconds": round(t_embed, 2),
        "search_seconds": round(t_search, 2),
        "llm_seconds": round(t_llm, 2),
        "total_seconds": round(total_time, 2),
        "cached": False
    }

    _cache_answer(cache_key, answer, sources, timings)

    return {"answer": answer, "sources": sources, "timings": timings}


# ============================================================
# 14) CLEANUP ON SHUTDOWN
# ============================================================

# When server stops, close all DB connections
atexit.register(lambda: connection_pool.closeall() if connection_pool else None)
