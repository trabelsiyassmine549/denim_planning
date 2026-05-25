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
    sessionId:  Optional[str] = None   # client generates once per conversation


class ChatResponse(BaseModel):
    reply:     str
    sessionId: str
    mode:      str   # "rag" | "general"


# ── Input validation 

# Common single-word commands that are valid even if short.
_VALID_KEYWORDS = {
    "résumé", "resume", "summary", "alertes", "alerts", "retards",
    "machines", "planning", "commandes", "orders", "bilan", "rapport",
    "help", "aide", "bonjour", "hello", "hi", "status", "statut",
}

# Vowels covering ASCII + French accented characters.
_VOWELS = set("aeiouàâäéèêëîïôùûüœæ")


def _is_intelligible(text: str) -> bool:
    """
    Return True if the question looks like a real human message.
    Rejects keyboard mash, random symbols, and very short non-word inputs.

    Rules (all must pass):
      1. At least 3 characters after stripping whitespace.
      2. At least one recognisable word (2+ consecutive letters).
      3. The ratio of letter characters to total non-space characters is ≥ 0.4
         (catches "vhbjnk,l;m" — mostly consonants with punctuation, but the
         ratio check is language-agnostic and won't reject French/Arabic).
      4. Not more than 70 % of characters being non-alphanumeric symbols
         (rejects pure symbol spam like "!@#$%^&").
      5. NEW — Vowel ratio ≥ 10 % of all letters.
         All-consonant strings like "ffffklsld", "jhfbvf", "vhbjnk" have 0 %
         vowels and are almost certainly keyboard mash.  Real words in French,
         English, or Arabic transliteration always contain some vowels.
         10 % is intentionally low to avoid false positives on short words,
         acronyms (CMD, SQL, RAG), or transliterated Arabic which is sometimes
         vowel-light in Latin script.
    """
    stripped = text.strip()

    # Rule 1 — minimum length
    if len(stripped) < 3:
        return False

    # Allow known single-keyword commands regardless of other rules
    if stripped.lower() in _VALID_KEYWORDS:
        return True

    # Rule 2 — must contain at least one word (2+ letters in a row)
    if not re.search(r'[a-zA-ZÀ-ÿ]{2,}', stripped):
        return False

    non_space = re.sub(r'\s', '', stripped)
    if not non_space:
        return False

    # Rule 3 — letter ratio ≥ 0.4
    letter_count = len(re.findall(r'[a-zA-ZÀ-ÿ]', non_space))
    if letter_count / len(non_space) < 0.4:
        return False

    # Rule 4 — symbol ratio ≤ 0.7
    symbol_count = len(re.findall(r'[^a-zA-ZÀ-ÿ0-9\s]', non_space))
    if symbol_count / len(non_space) > 0.7:
        return False

    # Rule 5 — vowel ratio ≥ 10 % of all letters.
    # Only applied when there are enough letters to make the ratio meaningful
    # (≥ 4 letters).  Shorter strings that passed Rules 1-4 are trusted.
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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_complete(reply: str) -> bool:
    """
    Return True if the reply looks like a complete sentence.
    A truncated Mistral response typically ends mid-word or mid-sentence without
    any terminal punctuation. Storing such a reply in Redis would corrupt the
    conversation memory for all subsequent turns.
    """
    if not reply:
        return False
    return reply.strip()[-1] in ".!?»)"


# ── Endpoint ──────────────────────────────────────────────────────────────────

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
        # Log so we can monitor how often truncation still occurs after the
        # num_predict / num_ctx fixes in rag_engine.py.
        print(f"[MEMORY] Skipped saving incomplete reply for session {session_id}: ...{reply[-60:]!r}")

    mode = "rag" if req.planningId is not None else "general"
    return ChatResponse(reply=reply, sessionId=session_id, mode=mode)


@router.delete("/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation memory for a session (e.g. on planning change)."""
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