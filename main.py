import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import answer, semantic_cache

app = FastAPI(title="Optimized RAG Service with Semantic Caching")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
INDEX_PATH = os.path.join(STATIC_DIR, "index.html")

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class Question(BaseModel):
    question: str


@app.get("/")
@app.get("/index.html")
@app.get("/index")
def index():
    """Serve interactive frontend web UI."""
    if os.path.exists(INDEX_PATH):
        return FileResponse(INDEX_PATH)
    raise HTTPException(status_code=404, detail="Frontend index.html not found.")


@app.get("/health")
def health():
    """Quick check that the server is alive."""
    return {"status": "ok"}


@app.post("/ask")
def ask(body: Question):
    """
    Query the RAG pipeline.
    Uses semantic caching: if a semantically similar question was asked before,
    the answer is returned from the cache with 0 LLM cost and sub-10ms latency.
    """
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        return answer(question)
    except Exception as e:
        # Most common cause: Ollama is not running or the model isn't downloaded
        raise HTTPException(status_code=500, detail=f"LLM call failed: {e}")


@app.get("/cache/stats")
def get_cache_stats():
    """Retrieve semantic cache hit rate, query count, and stored entries."""
    return semantic_cache.stats()


@app.post("/cache/clear")
def clear_cache():
    """Clear all semantic cache entries."""
    semantic_cache.clear()
    return {"status": "cache cleared", "stats": semantic_cache.stats()}


@app.get("/cache/entries")
def get_cache_entries():
    """Retrieve list of currently cached questions and metadata."""
    return [
        {
            "id": i,
            "question": e.get("question", ""),
            "answer": e.get("answer", "")[:200] + ("..." if len(e.get("answer", "")) > 200 else ""),
            "sources": e.get("sources", []),
            "model": e.get("model", ""),
            "timestamp": e.get("timestamp", 0),
        }
        for i, e in enumerate(semantic_cache.entries)
    ]