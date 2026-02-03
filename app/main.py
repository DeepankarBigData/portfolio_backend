import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .config import ALLOWED_ORIGINS
from .schemas import QueryRequest
from .services.llm import generate_from_groq
# from .services.cache import get_cache_key, get_cached_answer, cache_answer, get_cache_stats, clear_cache
import os
# ============================================================
# FASTAPI APP INITIALIZATION
# ============================================================

app = FastAPI(title="RAG API")

# Load markdown context on startup
# It runs when the ASGI application is created and the server is ready to start serving requests.
markdown_context = ""
@app.on_event("startup")
def startup_event():
    global markdown_context
    try:
        with open("output_document.md", "r", encoding="utf-8") as f:
            markdown_context = f.read()

        print(f"Loaded markdown context: {len(markdown_context)} characters")

    except Exception as e:
        print(f"Failed to load markdown context: {e}")
        markdown_context = ""


# CORS: allows browser frontend to call backend APIs
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    return {"status": "ok", "message": "API is running"}

@app.get("/health")
def health():
    """
    Very fast health check.
    """
    return {"status": "ok", "message": "Server is running"}

# @app.get("/cache/stats")
# def api_cache_stats():
#     return get_cache_stats()

# @app.get("/cache/clear")
# def api_clear_cache():
#     size = clear_cache()
#     return {"message": f"Cache cleared. Removed {size} entries."}


@app.post("/api/rag/query")
def rag_query(req: QueryRequest):
    """
    Main endpoint:
    1) Validate query
    2) Check cache
    3) Create embedding (Ollama)
    4) Retrieve top-k chunks (pgvector)
    5) Build prompt with context
    6) Generate answer (Groq)
    7) Cache and return response
    """
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
    # # ----------------------------
    # cache_key = get_cache_key(q, req.k)
    # cached = get_cached_answer(cache_key)

    # if cached:
    #     logger.info(f"Cache HIT for query: {q}")
    #     return {
    #         "answer": cached["answer"],
    #         "sources": cached["sources"],
    #         "timings": {**cached["timings"], "cached": True}
    #     }

    # logger.info(f"Cache MISS - Processing query: {q}")

    prompt = f"{q}"
    print(prompt)
    # ----------------------------
    # Step 5: LLM Generation (Groq) with Markdown Context
    # ----------------------------


    print(f"Markdown Context from env: {markdown_context}")
    t0 = time.time()
    try:
        answer = generate_from_groq(prompt, markdown_context=markdown_context)
        print(f"Request Context: {markdown_context}")
        print("Generated Answer:", answer)
        t_llm = time.time() - t0

        if not answer or not answer.strip():
            answer = "I couldn't generate a response. Please try rephrasing your question."

        logger.info(f"LLM completed: length={len(answer)}")

    except Exception as e:
        # Log specific error
        logger.error(f"Groq LLM failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM failed: {e}") from e

    # ----------------------------
    # Step 6: Timings + Cache
    # # ----------------------------
    # total_time = t_embed + t_search + t_llm
    # timings = {
    #     "embed_seconds": round(t_embed, 2),
    #     "search_seconds": round(t_search, 2),
    #     "llm_seconds": round(t_llm, 2),
    #     "total_seconds": round(total_time, 2),
    #     "cached": False
    # }

    # cache_answer(cache_key, answer, sources, timings)

    # Build a minimal timings and sources to return (embedding/search were disabled)
    timings = {
        "llm_seconds": round(t_llm, 2),
        "total_seconds": round(t_llm, 2),
        "cached": False
    }
    sources = []

    # Return the response expected by the frontend
    return {"answer": answer, "sources": sources, "timings": timings}
