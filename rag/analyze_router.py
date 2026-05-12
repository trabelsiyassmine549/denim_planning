"""
rag/analyze_router.py  —  POST /api/planning/analyze
=====================================================
FastAPI router that exposes the RAG analysis endpoint.
Add to api.py with:  app.include_router(analyze_router)

v5.0: Removed SKIP_FORMATTER_LLM from health endpoint (flag deleted in rag_engine v5.0).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Any

from rag.rag_engine import analyze, index_planning_rows

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    planningId: int
    question: str
    # Pre-fetched SQL query results from the .NET backend (keys A–F)
    # Each value is a list of row dicts from the respective diagnostic query.
    # The .NET PlanningController fetches these and forwards them here.
    sqlData: Optional[Dict[str, List[Dict[str, Any]]]] = None


class AnalyzeResponse(BaseModel):
    planningId: int
    question: str
    answer: str


class IndexRequest(BaseModel):
    planningId: int
    planningText: str   # free-text summary of the planning rows to index


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/api/planning/analyze", response_model=AnalyzeResponse)
async def analyze_planning(req: AnalyzeRequest):
    """
    RAG-powered planning analysis (v5.0 — Mistral-Primary).

    The Angular frontend calls this (via the .NET proxy) after a planning
    is generated. It receives structured SQL data (queries A–F) and the
    user's natural-language question, and returns a structured expert answer.

    v5.0: Improvement questions now go through Mistral as the primary reasoning
    engine instead of returning deterministic bullet strings directly. Python
    acts as fact extractor, prompt builder, and post-LLM validator.

    NOTE: sqlData must be populated by the .NET backend for improvement/analysis
    questions to work. An empty sqlData will cause the RAG engine to return
    a data-missing refusal rather than calling Mistral.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question is required")

    db_rows = req.sqlData or {}

    # Pre-call validation — warn on missing keys so routing bugs surface in
    # logs before they cause silent analyzer failures.
    REQUIRED_FOR_IMPROVEMENT = {"A", "B", "C", "D", "E", "F"}
    present = {k for k, v in db_rows.items() if isinstance(v, list) and len(v) > 0}
    missing = REQUIRED_FOR_IMPROVEMENT - present
    if missing:
        print(
            f"[ANALYZE] WARNING: sqlData missing or empty keys: {sorted(missing)} "
            f"(planning_id={req.planningId}, question={req.question!r:.60})"
        )

    # Log a one-line summary of what arrived so column-name bugs are visible.
    for key in sorted(db_rows):
        rows = db_rows[key]
        cols = list(rows[0].keys()) if rows else []
        print(
            f"[ANALYZE] Query {key}: {len(rows)} rows | "
            f"columns={cols}"
        )

    try:
        answer = await analyze(
            planning_id=req.planningId,
            question=req.question,
            db_rows=db_rows,
        )
        return AnalyzeResponse(
            planningId=req.planningId,
            question=req.question,
            answer=answer,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG analysis error: {str(e)}")


@router.post("/api/planning/index")
async def index_planning(req: IndexRequest):
    """
    Index a newly generated planning's text summary into FAISS.
    Called after /api/planning/run saves the result.
    """
    try:
        await index_planning_rows(req.planningId, req.planningText)
        return {"status": "indexed", "planningId": req.planningId}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")


@router.get("/api/planning/analyze/health")
async def analyze_health():
    """Check RAG engine health (v5.0 — Mistral-Primary mode)."""
    from rag.rag_engine import _faiss_index, EMBED_MODEL, LLM_MODEL, OLLAMA_URL
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
        ollama_ok = True
    except Exception as e:
        models = []
        ollama_ok = False

    return {
        "status":        "ok" if ollama_ok else "degraded",
        "ollama":        ollama_ok,
        "models":        models,
        "embedModel":    EMBED_MODEL,
        "llmModel":      LLM_MODEL,
        "mode":          "mistral-primary-v5.0",  # replaces skipFormatter flag
        "faissVectors":  _faiss_index.index.ntotal if _faiss_index.index else 0,
        "faissChunks":   len(_faiss_index.texts),
    }