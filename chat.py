"""
chat.py  —  Unified Chatbot for the Denim Washing Production Planner
=====================================================================
ONE chatbot endpoint that automatically uses RAG when a planningId is provided.

Behaviour:
  • planningId = null / absent  →  General assistant (Mistral direct, fast)
  • planningId = <int>          →  Expert RAG mode (FAISS + SQL A-F + Mistral)

The Angular chatbot passes planningId whenever the user has a planning open.
No UI mode switch needed — the backend decides automatically.

TIMEOUT FIXES (v2.1):
  • General mode num_predict: 500 → 300  (shorter answers, faster)
  • General mode num_ctx:    2048 → 1536 (less context needed for general Q&A)
  • General mode timeout:     180 → 120s (fail faster; RAG mode handles planning Qs)
  • Added explicit planningId-missing warning in the general-mode response so
    the user knows to open a planning if they want planning-specific answers.

IMPORTANT — Angular frontend requirement:
  When the user has a planning open, the frontend MUST send planningId in the
  request body, otherwise the question hits the general mode which:
    1. Has no SQL data → cannot answer planning questions
    2. Calls Mistral with no planning context → slow and useless
    3. Times out frequently on CPU-only machines
  Ensure your Angular ChatService always includes planningId when one is active.
"""

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from rag.rag_engine import analyze as rag_analyze, _is_gibberish, _detect_language

router = APIRouter()

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "mistral"

# Reduced num_predict and num_ctx for faster general-mode responses.
# General mode is for non-planning questions only; concise answers are appropriate.
OLLAMA_OPTIONS = {
    "num_predict": 300,
    "num_ctx":     1536,
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
- If the question is gibberish or incomprehensible, reply: "Je n'ai pas compris votre question. Pouvez-vous la reformuler ?" (French) or "I did not understand your question. Could you rephrase it?" (English).
- If the question is clearly about a specific planning (schedules, machine load, delays, makespan, fragmentation), reply: "Pour analyser un planning spécifique, veuillez ouvrir un planning dans l'application." (French) or "To analyze a specific planning, please open a planning in the application." (English). Do NOT attempt to answer from general knowledge."""


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

    IMPORTANT: The Angular frontend must always include planningId in the
    request when a planning is open. If planningId is missing, planning-specific
    questions will hit general mode and either timeout or give wrong answers.
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

        # Warn if sqlData is missing — improvement path will return a refusal
        # rather than calling Mistral, but log it for debugging.
        if not req.sqlData:
            print(
                f"[CHAT] WARNING: planningId={req.planningId} but sqlData is empty — "
                "RAG analyzers will return a data-missing refusal."
            )
        else:
            # Log a brief summary of what arrived (mirrors analyze_router logging)
            for key in sorted(req.sqlData):
                rows = req.sqlData[key]
                cols = list(rows[0].keys()) if rows else []
                print(
                    f"[CHAT] sqlData Query {key}: {len(rows)} rows | "
                    f"columns={cols}"
                )

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
    # Log a warning when improvement-type questions reach general mode.
    # This is a strong signal that the Angular frontend is not sending planningId.
    user_question_general = next(
        (m.content for m in reversed(req.messages) if m.role == "user"),
        ""
    )
    _PLANNING_SIGNALS = [
        "améliorer", "amélioration", "optimiser", "planning", "makespan",
        "machine", "retard", "commande", "improve", "planning", "schedule",
        "bottleneck", "delay", "order",
    ]
    if any(sig in user_question_general.lower() for sig in _PLANNING_SIGNALS):
        print(
            f"[CHAT] WARNING: planning-related question reached GENERAL mode "
            f"(planningId=None). Frontend may not be sending planningId. "
            f"Question: {user_question_general!r}"
        )

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
        # timeout reduced to 120s for general mode.
        # General-mode answers are short (2-4 sentences, num_predict=300),
        # so 120s is more than enough on CPU. If Ollama is truly stuck,
        # fail fast rather than blocking the UI for 3 minutes.
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail=(
                "Ollama a mis trop de temps à répondre. "
                "Vérifiez qu'Ollama tourne correctement (ollama serve). "
                "Si vous posez une question sur un planning spécifique, "
                "assurez-vous d'avoir un planning ouvert dans l'application."
            ),
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