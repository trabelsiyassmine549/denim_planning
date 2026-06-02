import re
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from chatbot.rag.rag_engine import analyze
from chatbot.redis_cache import get_memory, add_to_memory, clear_memory

router = APIRouter(prefix="/api/chat", tags=["chat"])


# ── Schemas 
class ChatRequest(BaseModel):
    question:   str
    planningId: Optional[int] = None
    sessionId:  Optional[str] = None  


class ChatResponse(BaseModel):
    reply:     str
    sessionId: str
    mode:      str   # "rag" | "general"


# ── Input validation 

_VALID_KEYWORDS = {
    "résumé", "resume", "summary", "alertes", "alerts", "retards",
    "machines", "planning", "commandes", "orders", "bilan", "rapport",
    "help", "aide", "bonjour", "hello", "hi", "status", "statut",
}

_VOWELS = set("aeiouàâäéèêëîïôùûüœæ")


def _is_intelligible(text: str) -> bool:
    """
    Return True if the question looks like a real human message.
    Rejects keyboard mash, random symbols, and very short non-word inputs.
    """
    stripped = text.strip()

    if len(stripped) < 3:
        return False

    if stripped.lower() in _VALID_KEYWORDS:
        return True

    if not re.search(r'[a-zA-ZÀ-ÿ]{2,}', stripped):
        return False

    non_space = re.sub(r'\s', '', stripped)
    if not non_space:
        return False

    letter_count = len(re.findall(r'[a-zA-ZÀ-ÿ]', non_space))
    if letter_count / len(non_space) < 0.4:
        return False

    symbol_count = len(re.findall(r'[^a-zA-ZÀ-ÿ0-9\s]', non_space))
    if symbol_count / len(non_space) > 0.7:
        return False

    letters_only = [c for c in stripped.lower() if c.isalpha()]
    if len(letters_only) >= 4:
        vowel_ratio = sum(1 for c in letters_only if c in _VOWELS) / len(letters_only)
        if vowel_ratio < 0.10:
            return False

    return True


_GIBBERISH_REPLY = (
    "Je ne comprends pas votre message. "
    "Posez-moi une question sur le planning, les commandes, les machines ou les alertes."
)


def _is_complete(reply: str) -> bool:
    # Return True if the reply looks like a complete sentence.
    if not reply:
        return False
    return reply.strip()[-1] in ".!?»)"


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question vide.")

    session_id = req.sessionId if req.sessionId else str(uuid.uuid4())

    # Reject gibberish immediately — do not call Mistral
    if not _is_intelligible(req.question):
        print(f"[VALIDATION] Rejected gibberish input: {req.question!r}")
        return ChatResponse(
            reply=_GIBBERISH_REPLY,
            sessionId=session_id,
            mode="general",
        )

    memory = get_memory(session_id) or []

    # Run RAG pipeline
    try:
        reply = await analyze(
            question=req.question.strip(),
            planning_id=req.planningId,
            memory=memory,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    add_to_memory(session_id, "user", req.question.strip())
    if _is_complete(reply):
        add_to_memory(session_id, "assistant", reply)
    else:
        print(f"[MEMORY] Skipped saving incomplete reply for session {session_id}: ...{reply[-60:]!r}")

    mode = "rag" if req.planningId is not None else "general"
    return ChatResponse(reply=reply, sessionId=session_id, mode=mode)


@router.delete("/{session_id}")
async def clear_session(session_id: str):
    clear_memory(session_id)
    return {"status": "cleared", "sessionId": session_id}


@router.get("/health")
async def health():
    import httpx
    from chatbot.redis_cache import REDIS_OK
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get("http://localhost:11434/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
        ollama_ok = True
    except Exception:
        models = []
        ollama_ok = False

    return {
        "status": "ok" if ollama_ok else "degraded",
        "ollama": ollama_ok,
        "models": models,
        "redis":  REDIS_OK,
    }