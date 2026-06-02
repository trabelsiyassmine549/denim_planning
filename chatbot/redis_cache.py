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
    _redis = None  
    REDIS_OK = False
    print(f"[REDIS] Unavailable — running without cache: {e}")

SQL_TTL = 300        # 5 min 
MEM_TTL = 3600       # 1 hour 
CTX_TTL = 600        # 10 min 


def _safe(fn):
    def wrapper(*args, **kwargs):
        if not REDIS_OK or _redis is None:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"[REDIS] Error in {fn.__name__}: {e}")
            return None
    return wrapper


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


# ── Conversation memory 

MAX_TURNS = 6  

def _mem_key(session_id: str) -> str:
    return f"chatbot:mem:{session_id}"


@_safe
def get_memory(session_id: str) -> List[Dict[str, str]]:
    key = _mem_key(session_id)
    val = _redis.get(key)  
    return json.loads(val) if val else []


@_safe
def add_to_memory(session_id: str, role: str, content: str) -> None:
    key = _mem_key(session_id)
    val = _redis.get(key)  
    history: List[Dict] = json.loads(val) if val else []
    history.append({"role": role, "content": content})
    history = history[-(MAX_TURNS * 2):]
    _redis.setex(key, MEM_TTL, json.dumps(history))  

@_safe
def clear_memory(session_id: str) -> None:
    _redis.delete(_mem_key(session_id)) 


# ── Planning context cache 

def _ctx_key(planning_id: int) -> str:
    return f"chatbot:ctx:{planning_id}"


@_safe
def get_planning_context(planning_id: int) -> Optional[Dict]:
    key = _ctx_key(planning_id)
    val = _redis.get(key) 
    return json.loads(val) if val else None


@_safe
def set_planning_context(planning_id: int, context: Dict) -> None:
    key = _ctx_key(planning_id)
    _redis.setex(key, CTX_TTL, json.dumps(context, default=str)) 
