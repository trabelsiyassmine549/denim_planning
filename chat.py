"""
chat.py  —  Unified Chatbot for the Denim Washing Production Planner
=====================================================================
ONE chatbot endpoint that automatically uses RAG when a planningId is provided.

Behaviour:
  • planningId = null / absent  →  General assistant (Mistral direct, fast)
  • planningId = <int>          →  Expert RAG mode (FAISS + SQL A-F + Mistral)

The Angular chatbot passes planningId whenever the user has a planning open.
No UI mode switch needed — the backend decides automatically.

OPTIMISATIONS vs previous version:
  • General mode num_predict : 400  → 300
  • General mode num_ctx     : 2048 → 1536  (faster for short Q&A)
  • General mode timeout     : 300s → 180s  (fail fast if Ollama hangs)
"""

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from rag.rag_engine import analyze as rag_analyze, _is_gibberish, _detect_language   # RAG pipeline

router = APIRouter()

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "mistral"

# FIX: Reduced num_predict and num_ctx for general mode — short answers need less.
OLLAMA_OPTIONS = {
    "num_predict": 300,    # ↓ was 400
    "num_ctx":     1536,   # ↓ was 2048
    "temperature": 0.3,
    "top_p":       0.8,
}

GENERAL_SYSTEM_PROMPT = """You are an industrial planning assistant for the Micwic denim washing workshop.

LANGUAGE RULE — ABSOLUTE PRIORITY:
Detect the language of the user's last message and respond in that EXACT language.
- French question → French answer.
- English question → English answer.
- Never mix languages.

Rules:
- Short, direct answers (2–4 sentences unless more detail is explicitly requested).
- No lengthy introductions. No "Sure!", no "Of course".
- Practical and concrete.
- If you don't know, say so clearly.
- If the question is gibberish or incomprehensible, reply: "Je n'ai pas compris votre question. Pouvez-vous la reformuler ?" (French) or "I did not understand your question. Could you rephrase it?" (English)."""


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages:   List[ChatMessage]
    planningId: Optional[int] = None          # ← KEY: triggers RAG when set
    sqlData:    Optional[dict] = None         # pre-fetched SQL rows A-F from .NET
    stream:     Optional[bool] = False


class ChatResponse(BaseModel):
    reply:    str
    mode:     str   # "general" or "rag"


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/api/chat/health")
async def chat_health():
    from rag.rag_engine import _faiss_index
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
        return {
            "status":       "ok",
            "ollama":       True,
            "models":       models,
            "active_model": OLLAMA_MODEL,
            "rag_vectors":  _faiss_index.index.ntotal if _faiss_index.index else 0,
        }
    except Exception as e:
        return {"status": "degraded", "ollama": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Unified chat endpoint
# ---------------------------------------------------------------------------

@router.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Single chat endpoint.
    - If planningId is provided → RAG mode (expert planning analysis)
    - If planningId is None     → General assistant mode
    """

    # ── RAG MODE — planning context available ──────────────────────────────
    if req.planningId is not None:
        user_question = next(
            (m.content for m in reversed(req.messages) if m.role == "user"),
            ""
        )
        if not user_question.strip():
            raise HTTPException(status_code=400, detail="Empty question")

        # Gibberish guard — short-circuit before any DB/LLM calls
        if _is_gibberish(user_question):
            lang = _detect_language(user_question)
            clarify = (
                "I did not understand your question. Could you please rephrase it?"
                if lang == 'en' else
                "Je n'ai pas compris votre question. Pouvez-vous la reformuler ?"
            )
            return ChatResponse(reply=clarify, mode="rag")

        try:
            reply = await rag_analyze(
                planning_id=req.planningId,
                question=user_question,
                db_rows=req.sqlData or {},
            )
            return ChatResponse(reply=reply, mode="rag")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"RAG error: {str(e)}")

    # ── GENERAL MODE — no planning context ────────────────────────────────
    messages = [{"role": "system", "content": GENERAL_SYSTEM_PROMPT}]
    messages += [{"role": m.role, "content": m.content} for m in req.messages]

    payload = {
        "model":      OLLAMA_MODEL,
        "messages":   messages,
        "stream":     False,
        "keep_alive": "10m",
        "options":    OLLAMA_OPTIONS,
    }

    try:
        # FIX: timeout 180s for general mode (was 300s) — fail faster if Ollama hangs
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Ollama timed out. Vérifiez qu'Ollama tourne correctement.",
        )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start it with: ollama serve",
        )

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Ollama error: {r.text[:300]}")

    reply = r.json().get("message", {}).get("content", "")
    return ChatResponse(reply=reply, mode="general")