"""
faiss_index.py — Semantic retrieval via FAISS + nomic-embed-text (Ollama).

What gets indexed:
  - PlanningRows  : one text chunk per (planning, machine, operation) group
  - Commandes     : one chunk per commande (status, recette, urgence, dates)
  - Alerts        : one chunk per active alert

Index is built at FastAPI startup and refreshed whenever a new planning
is saved (call rebuild_for_planning(planning_id) from your planning endpoint).

Retrieval:
  retrieve(question, planning_id, top_k) → list of relevant text chunks
  These chunks are injected into the Mistral prompt alongside the SQL context.

If FAISS or nomic-embed-text is unavailable, every function degrades
gracefully (returns []) so the chatbot still works via SQL-only context.

FIX: every internal helper was doing 'from db import query' (bare module name).
     Python can't find a top-level 'db' module when the package is 'chatbot'.
     All imports are now 'from chatbot.db import query'.

FIX v2 (retrieve): Added top_k == 0 early-return guard.
     When SQL_ONLY_INTENTS fire (amélioration, makespan, résumé, etc.),
     rag_engine calls retrieve() with top_k=0. Without the guard, the function
     still called _embed([question]) (wasting ~1s) and then passed k=0 to
     faiss.IndexFlatIP.search(), which raises an error on some faiss-cpu builds.
     The empty-string exception message caused the misleading log line:
       [FAISS] Retrieve error:
     Adding 'if top_k == 0: return []' before any FAISS/embed work eliminates
     the error and the wasted embedding call.

FIX v2 (retrieve error logging): Changed bare 'except Exception as e: print(e)'
     to 'print(repr(e))' + 'traceback.format_exc()' so future errors are
     visible in the logs instead of printing an empty string.
"""

import asyncio
import hashlib
import pickle
import traceback
from pathlib import Path
from typing import List, Optional

import httpx
import numpy as np

# ── FAISS import (optional) ───────────────────────────────────────────────────
try:
    import faiss
    FAISS_OK = True
except ImportError:
    faiss = None  # type: ignore
    FAISS_OK = False
    print("[FAISS] faiss-cpu not installed — semantic retrieval disabled.")

OLLAMA_URL  = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM   = 768   # nomic-embed-text output dimension

# Persist index to disk so it survives restarts
_INDEX_PATH = Path(__file__).parent / "faiss_store" / "index.bin"
_META_PATH  = Path(__file__).parent / "faiss_store" / "meta.pkl"
_INDEX_PATH.parent.mkdir(exist_ok=True)


# ── In-memory store ───────────────────────────────────────────────────────────

class FaissStore:
    def __init__(self):
        self.index: Optional[object] = None   # faiss.IndexFlatIP
        self.texts: List[str] = []
        self.meta:  List[dict] = []           # {planning_id, table, key}

    def is_ready(self) -> bool:
        return self.index is not None and len(self.texts) > 0

    def build(self, texts: List[str], meta: List[dict], embeddings: np.ndarray):
        if not FAISS_OK:
            return
        norms  = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms  = np.where(norms == 0, 1, norms)
        normed = (embeddings / norms).astype("float32")

        self.index = faiss.IndexFlatIP(EMBED_DIM)
        self.index.add(normed)
        self.texts = texts
        self.meta  = meta

    def add(self, texts: List[str], meta: List[dict], embeddings: np.ndarray):
        """Add new vectors without rebuilding the full index."""
        if not FAISS_OK or not self.is_ready():
            self.build(texts, meta, embeddings)
            return
        norms  = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms  = np.where(norms == 0, 1, norms)
        normed = (embeddings / norms).astype("float32")
        self.index.add(normed)
        self.texts.extend(texts)
        self.meta.extend(meta)

    def remove_planning(self, planning_id: int):
        """Remove all vectors for a given planning_id and rebuild index."""
        if not FAISS_OK or not self.is_ready():
            return
        keep = [i for i, m in enumerate(self.meta) if m.get("planning_id") != planning_id]
        if len(keep) == len(self.texts):
            return  # nothing to remove
        kept_texts = [self.texts[i] for i in keep]
        kept_meta  = [self.meta[i]  for i in keep]
        all_vecs   = np.zeros((self.index.ntotal, EMBED_DIM), dtype="float32")
        for i in range(self.index.ntotal):
            all_vecs[i] = self.index.reconstruct(i)
        kept_vecs = all_vecs[keep]
        new_index = faiss.IndexFlatIP(EMBED_DIM)
        if len(kept_vecs) > 0:
            new_index.add(kept_vecs)
        self.index = new_index
        self.texts = kept_texts
        self.meta  = kept_meta

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[str]:
        if not FAISS_OK or not self.is_ready():
            return []
        q    = query_embedding.astype("float32").reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score > 0.3:
                results.append(self.texts[idx])
        return results

    def save(self):
        if not FAISS_OK or not self.is_ready():
            return
        faiss.write_index(self.index, str(_INDEX_PATH))
        with open(_META_PATH, "wb") as f:
            pickle.dump({"texts": self.texts, "meta": self.meta}, f)

    def load(self) -> bool:
        if not FAISS_OK or not _INDEX_PATH.exists() or not _META_PATH.exists():
            return False
        try:
            self.index = faiss.read_index(str(_INDEX_PATH))
            with open(_META_PATH, "rb") as f:
                data = pickle.load(f)
            self.texts = data["texts"]
            self.meta  = data["meta"]
            print(f"[FAISS] Loaded {self.index.ntotal} vectors from disk.")
            return True
        except Exception as e:
            print(f"[FAISS] Load failed: {e}")
            return False


_store = FaissStore()


# ── Embedding via Ollama ──────────────────────────────────────────────────────

async def _embed(texts: List[str]) -> np.ndarray:
    """Embed a list of texts using nomic-embed-text via Ollama."""
    vectors    = []
    batch_size = 16
    async with httpx.AsyncClient(timeout=120) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                try:
                    r   = await client.post(
                        f"{OLLAMA_URL}/api/embeddings",
                        json={"model": EMBED_MODEL, "prompt": text},
                    )
                    vec = r.json().get("embedding", [])
                    vectors.append(vec if vec else [0.0] * EMBED_DIM)
                except Exception as e:
                    print(f"[FAISS] Embed error: {e}")
                    vectors.append([0.0] * EMBED_DIM)
    return np.array(vectors, dtype="float32")


# ── Text chunk builders (DB rows → natural language) ─────────────────────────

def _planning_chunks(planning_id: int) -> tuple[List[str], List[dict]]:
    from chatbot.db import query

    # STRING_AGG requires SQL Server 2017+.
    # Try it first; fall back to the FOR XML PATH trick (SQL Server 2008+)
    # if the server raises a syntax error.
    try:
        rows = query("""
            SELECT MachineName, NomOperation,
                   COUNT(*)         AS NbLots,
                   SUM(LotSize)     AS TotalPieces,
                   SUM(DureeTotale) AS TotalMinutes,
                   MIN(DateStart)   AS DebutReel,
                   MAX(DateEnd)     AS FinReelle,
                   STRING_AGG(NumeroCommande, ', ') AS Commandes
            FROM PlanningRows
            WHERE PlanningId = ?
            GROUP BY MachineName, NomOperation
        """, (planning_id,))
    except Exception:
        # Fallback compatible with SQL Server 2008+.
        # STUFF removes the leading ', ' that FOR XML PATH prepends.
        rows = query("""
            SELECT
                pr.MachineName,
                pr.NomOperation,
                COUNT(*)         AS NbLots,
                SUM(pr.LotSize)     AS TotalPieces,
                SUM(pr.DureeTotale) AS TotalMinutes,
                MIN(pr.DateStart)   AS DebutReel,
                MAX(pr.DateEnd)     AS FinReelle,
                STUFF((
                    SELECT DISTINCT ', ' + sub.NumeroCommande
                    FROM PlanningRows sub
                    WHERE sub.PlanningId   = pr.PlanningId
                      AND sub.MachineName  = pr.MachineName
                      AND sub.NomOperation = pr.NomOperation
                    FOR XML PATH(''), TYPE
                ).value('.', 'NVARCHAR(MAX)'), 1, 2, '') AS Commandes
            FROM PlanningRows pr
            WHERE pr.PlanningId = ?
            GROUP BY pr.PlanningId, pr.MachineName, pr.NomOperation
        """, (planning_id,))

    texts, meta = [], []
    for r in rows:
        text = (
            f"Planning {planning_id} — machine {r['MachineName']} — "
            f"opération {r['NomOperation']}: "
            f"{r['NbLots']} lot(s), {r['TotalPieces']} pièces, "
            f"{r['TotalMinutes']} minutes planifiées, "
            f"du {str(r['DebutReel'])[:10]} au {str(r['FinReelle'])[:10]}. "
            f"Commandes concernées: {r['Commandes']}."
        )
        texts.append(text)
        meta.append({
            "planning_id": planning_id,
            "table": "PlanningRows",
            "key":   f"{r['MachineName']}_{r['NomOperation']}",
        })
    return texts, meta


def _commandes_chunks() -> tuple[List[str], List[dict]]:
    from chatbot.db import query
    rows = query("""
        SELECT c.NumeroCommande, c.Statut, c.DateExport, c.Urgence,
               c.Quantite, r.NomRecette, c.DateCreation
        FROM Commandes c
        LEFT JOIN Recettes r ON r.Id = c.RecetteId
        WHERE c.Statut NOT IN ('Annulé')
        ORDER BY c.DateExport
    """)
    texts, meta = [], []
    for r in rows:
        text = (
            f"Commande {r['NumeroCommande']}: statut {r['Statut']}, "
            f"recette {r.get('NomRecette', 'inconnue')}, "
            f"quantité {r['Quantite']} pièces, "
            f"urgence niveau {r['Urgence']}, "
            f"date export {str(r['DateExport'])[:10]}."
        )
        texts.append(text)
        meta.append({"planning_id": None, "table": "Commandes", "key": r['NumeroCommande']})
    return texts, meta


def _alerts_chunks() -> tuple[List[str], List[dict]]:
    from chatbot.db import query
    rows = query("""
        SELECT Type, Severity, Message, NumeroCommande,
               MachineName, GeneratedAt
        FROM Alerts
        WHERE IsDismissed = 0
        ORDER BY Severity DESC, GeneratedAt DESC
    """)
    texts, meta = [], []
    for r in rows:
        text = (
            f"Alerte {r['Severity']} ({r['Type']}): {r['Message']} "
            f"— commande {r.get('NumeroCommande', 'N/A')}, "
            f"machine {r.get('MachineName', 'N/A')}, "
            f"générée le {str(r.get('GeneratedAt', ''))[:10]}."
        )
        texts.append(text)
        meta.append({
            "planning_id": None,
            "table": "Alerts",
            "key":   str(r.get('GeneratedAt', '')),
        })
    return texts, meta


# ── Public API ────────────────────────────────────────────────────────────────

async def build_index():
    """
    Build the full FAISS index from scratch.
    Called at FastAPI startup. Loads from disk if available, rebuilds otherwise.
    """
    if not FAISS_OK:
        return

    if _store.load():
        print(f"[FAISS] Using cached index ({_store.index.ntotal} vectors).")
        return

    print("[FAISS] Building index from database...")
    all_texts, all_meta = [], []

    from chatbot.db import query as _q

    cmd_texts, cmd_meta = _commandes_chunks()
    all_texts += cmd_texts
    all_meta  += cmd_meta
    print(f"[FAISS] Commandes: {len(cmd_texts)} chunks")

    alert_texts, alert_meta = _alerts_chunks()
    all_texts += alert_texts
    all_meta  += alert_meta
    print(f"[FAISS] Alerts: {len(alert_texts)} chunks")

    recent_plannings = _q("SELECT TOP 5 Id FROM Plannings ORDER BY DateGeneration DESC")
    for p in recent_plannings:
        pid = p["Id"]
        pt, pm = _planning_chunks(pid)
        all_texts += pt
        all_meta  += pm
        print(f"[FAISS] Planning {pid}: {len(pt)} chunks")

    if not all_texts:
        print("[FAISS] No data to index.")
        return

    print(f"[FAISS] Embedding {len(all_texts)} chunks...")
    embeddings = await _embed(all_texts)
    _store.build(all_texts, all_meta, embeddings)
    _store.save()
    print(f"[FAISS] Index ready: {_store.index.ntotal} vectors.")


async def rebuild_for_planning(planning_id: int):
    """
    Called after a new planning is generated and saved.
    Removes old vectors for this planning_id and adds fresh ones.
    """
    if not FAISS_OK:
        return
    print(f"[FAISS] Refreshing index for planning {planning_id}...")
    _store.remove_planning(planning_id)

    texts, meta = _planning_chunks(planning_id)
    if not texts:
        return
    embeddings = await _embed(texts)
    _store.add(texts, meta, embeddings)
    _store.save()
    print(f"[FAISS] Planning {planning_id}: {len(texts)} chunks added.")


async def retrieve(question: str, planning_id: Optional[int] = None, top_k: int = 5) -> List[str]:
    """
    Semantic search: embed the question and return the most relevant chunks.
    Returns [] gracefully if FAISS is not ready (chatbot falls back to SQL-only).

    FIX v2: Added top_k == 0 early-return guard.
    When SQL_ONLY_INTENTS fire, rag_engine passes top_k=0 to skip FAISS.
    Without this guard the function still called _embed() (wasting ~1s) and
    then called faiss.IndexFlatIP.search(q, 0), which raises an exception on
    some faiss-cpu builds — printing the misleading log line:
      [FAISS] Retrieve error:    ← empty because the exception has no message
    The guard short-circuits before any work is done.
    """
    # FIX v2: skip immediately — no embed call, no FAISS call, no error log
    if top_k == 0:
        return []

    if not FAISS_OK or not _store.is_ready():
        return []

    try:
        embeddings = await _embed([question])
        results    = _store.search(embeddings[0], top_k=top_k * 2)

        if planning_id:
            subset_idx = [
                i for i, m in enumerate(_store.meta)
                if m.get("planning_id") == planning_id
            ]
            if FAISS_OK and len(subset_idx) >= top_k:
                all_vecs = np.zeros((_store.index.ntotal, EMBED_DIM), dtype="float32")
                for i in range(_store.index.ntotal):
                    all_vecs[i] = _store.index.reconstruct(i)
                sub_vecs = all_vecs[subset_idx]
                mini     = faiss.IndexFlatIP(EMBED_DIM)
                mini.add(sub_vecs)
                q    = embeddings[0].astype("float32").reshape(1, -1)
                norm = np.linalg.norm(q)
                if norm > 0:
                    q = q / norm
                k = min(top_k, mini.ntotal)
                scores, idxs = mini.search(q, k)
                planning_results = [
                    _store.texts[subset_idx[i]]
                    for score, i in zip(scores[0], idxs[0])
                    if i >= 0 and score > 0.3
                ]
                seen   = set(planning_results)
                merged = planning_results + [r for r in results if r not in seen]
                return merged[:top_k]

        return results[:top_k]

    except Exception as e:
        # FIX v2: log repr(e) + full traceback so the error is never silent
        print(f"[FAISS] Retrieve error: {repr(e)}")
        print(traceback.format_exc())
        return []