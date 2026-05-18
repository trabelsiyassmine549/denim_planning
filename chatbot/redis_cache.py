"""
redis_cache.py — Redis-backed cache + conversation memory for the chatbot.

Two responsibilities:
  1. SQL query cache     : avoid re-querying the DB for identical questions
                           within the same planning session (TTL = 5 min).
  2. Conversation memory : store the last N turns per session so Mistral
                           has conversational context without a client-side
                           message history.

Redis key schema:
  chatbot:sql:{planning_id}:{query_hash}   → JSON-encoded query result
  chatbot:mem:{session_id}                 → JSON-encoded list of {role, content}
  chatbot:ctx:{planning_id}                → JSON-encoded planning summary facts

If Redis is unavailable, every method degrades gracefully (returns None / []).
"""

import hashlib
import json
import os
from typing import Any, List, Dict, Optional

try:
    import redis

    _redis = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", 0)),
        decode_responses=True,
        socket_connect_timeout=2,
    )
    _redis.ping()
    REDIS_OK = True
    print("[REDIS] Connected.")
except Exception as e:
    _redis = None  # type: ignore
    REDIS_OK = False
    print(f"[REDIS] Unavailable — running without cache: {e}")

# ── TTLs ──────────────────────────────────────────────────────────────────────
SQL_TTL = 300        # 5 min — SQL cache per planning/query
MEM_TTL = 3600       # 1 hour — conversation memory per session
CTX_TTL = 600        # 10 min — planning context summary


def _safe(fn):
    """Decorator: swallow Redis errors so the app never crashes on cache miss."""
    def wrapper(*args, **kwargs):
        if not REDIS_OK or _redis is None:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"[REDIS] Error in {fn.__name__}: {e}")
            return None
    return wrapper


# ── SQL cache ─────────────────────────────────────────────────────────────────

def _sql_key(planning_id: Optional[int], sql: str) -> str:
    h = hashlib.md5(sql.encode()).hexdigest()[:12]
    pid = planning_id or "general"
    return f"chatbot:sql:{pid}:{h}"


@_safe
def get_sql_cache(planning_id: Optional[int], sql: str) -> Optional[Any]:
    key = _sql_key(planning_id, sql)
    val = _redis.get(key)  # type: ignore
    return json.loads(val) if val else None


@_safe
def set_sql_cache(planning_id: Optional[int], sql: str, data: Any) -> None:
    key = _sql_key(planning_id, sql)
    _redis.setex(key, SQL_TTL, json.dumps(data, default=str))  # type: ignore


# ── Conversation memory ────────────────────────────────────────────────────────

MAX_TURNS = 6   # keep last 6 turns (3 user + 3 assistant) in memory

def _mem_key(session_id: str) -> str:
    return f"chatbot:mem:{session_id}"


@_safe
def get_memory(session_id: str) -> List[Dict[str, str]]:
    key = _mem_key(session_id)
    val = _redis.get(key)  # type: ignore
    return json.loads(val) if val else []


@_safe
def add_to_memory(session_id: str, role: str, content: str) -> None:
    key = _mem_key(session_id)
    val = _redis.get(key)  # type: ignore
    history: List[Dict] = json.loads(val) if val else []
    history.append({"role": role, "content": content})
    # Keep last MAX_TURNS * 2 messages (each turn = user + assistant)
    history = history[-(MAX_TURNS * 2):]
    _redis.setex(key, MEM_TTL, json.dumps(history))  # type: ignore


@_safe
def clear_memory(session_id: str) -> None:
    _redis.delete(_mem_key(session_id))  # type: ignore


# ── Planning context cache ─────────────────────────────────────────────────────

def _ctx_key(planning_id: int) -> str:
    return f"chatbot:ctx:{planning_id}"


@_safe
def get_planning_context(planning_id: int) -> Optional[Dict]:
    key = _ctx_key(planning_id)
    val = _redis.get(key)  # type: ignore
    return json.loads(val) if val else None


@_safe
def set_planning_context(planning_id: int, context: Dict) -> None:
    key = _ctx_key(planning_id)
    _redis.setex(key, CTX_TTL, json.dumps(context, default=str))  # type: ignore
