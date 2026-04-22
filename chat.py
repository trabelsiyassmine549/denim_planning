"""
chat.py  —  Ollama/Mistral chatbot endpoint for the Denim Planner
=================================================================
"""

import httpx
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "mistral"

# Ollama generation options — tuned for speed
OLLAMA_OPTIONS = {
    "num_predict": 200,    # max tokens to generate (short answers)
    "num_ctx":     2048,   # smaller context window = faster
    "temperature": 0.3,    # lower = more focused, less wandering
    "top_p":       0.8,
}

SYSTEM_PROMPT = """You are a concise production planning assistant for a denim washing factory.
Rules:
- Keep answers short (2-4 sentences max unless detail is explicitly requested).
- Be direct and practical — no long introductions.
- Answer in the same language the user writes in (French or English).
- If you don't know the answer, say you don't know instead of making something up."""


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: Optional[bool] = True


@router.get("/api/chat/health")
async def chat_health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            models = [m["name"] for m in r.json().get("models", [])]
        return {
            "status": "ok",
            "ollama": True,
            "models": models,
            "active_model": OLLAMA_MODEL,
        }
    except Exception as e:
        return {"status": "degraded", "ollama": False, "error": str(e)}


@router.post("/api/chat")
async def chat(req: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m.role, "content": m.content} for m in req.messages]

    payload = {
        "model":      OLLAMA_MODEL,
        "messages":   messages,
        "stream":     False,
        "keep_alive": "10m",   # keep model loaded in memory between requests
        "options":    OLLAMA_OPTIONS,
    }

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start it with: ollama serve",
        )

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Ollama error: {r.text[:300]}")

    data = r.json()
    reply = data.get("message", {}).get("content", "")
    return {"reply": reply}


@router.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m.role, "content": m.content} for m in req.messages]

    payload = {
        "model":      OLLAMA_MODEL,
        "messages":   messages,
        "stream":     True,
        "keep_alive": "10m",
        "options":    OLLAMA_OPTIONS,
    }

    async def generate():
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            token = chunk.get("message", {}).get("content", "")
                            if token:
                                yield f"data: {json.dumps({'token': token})}\n\n"
                        except json.JSONDecodeError:
                            pass
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")