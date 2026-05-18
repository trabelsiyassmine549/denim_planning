"""
api.py — FastAPI entry point for the Denim Washing Production Planner
========================================================================
FIXES applied:
  1. Removed call to ensure_domain_knowledge_indexed() — function never existed,
     caused NameError on every startup.
  2. Fixed /api/planning/health endpoint — was importing from the wrong module
     path 'chatBotSystem.rag.rag_engine' (typo) and referencing a private
     symbol that doesn't exist there.  Now uses the FaissStore instance
     directly from chatbot.rag.faiss_index.
  3. All 'from db import …' calls inside faiss_index.py were using a bare
     'db' module name instead of 'chatbot.db'.  The fix is in faiss_index.py
     (see that file).  api.py itself is clean — listed here for completeness.
"""

from datetime import date, datetime
import traceback
import httpx

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from data.fetcher import load_live_data, validate
from models.schemas import RunRequest, RunResponse, GanttRow
from solver.cp_sat_solver import (
    HAS_ORTOOLS,
    run_lns,
    run_cpsat,
    lns_fallback,
    collect_split_warnings,
)
from utils.time_utils import PPD, date_to_day_offset, working_day_date, pm_to_clock

from chatbot.rag.faiss_index import build_index, rebuild_for_planning, _store as _faiss_store
from chatbot.chat_router import router as chatbot_router

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Denim Planner Optimiser", version="2.0.0")

app.include_router(chatbot_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "https://localhost:4200"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

OLLAMA_URL   = "http://localhost:11434"
OLLAMA_MODEL = "mistral"

# ---------------------------------------------------------------------------
# Helper — shift solver rows
# ---------------------------------------------------------------------------

def _shift_rows(rows: list, now_pm: int) -> list:
    shifted = []
    for r in rows:
        s_pm = r.startPM + now_pm
        e_pm = r.endPM + now_pm

        s_day = pm_to_clock(s_pm)[0]
        e_day = pm_to_clock(e_pm)[0]

        shifted.append(r.model_copy(update={
            "startPM":   s_pm,
            "endPM":     e_pm,
            "dateStart": working_day_date(s_day).isoformat(),
            "dateEnd":   working_day_date(e_day).isoformat(),
        }))
    return shifted

# ---------------------------------------------------------------------------
# Late warning helper
# ---------------------------------------------------------------------------

def _collect_late_warnings(rows: list, start_date) -> list[str]:
    from datetime import date as _date

    today = _date.today()
    worst = {}
    for r in rows:
        cmd = r.numeroCommande
        if cmd not in worst or r.dateEnd > worst[cmd][0]:
            worst[cmd] = (r.dateEnd, r.dateExport)

    late_warnings = []
    for cmd, (date_end_str, date_export_str) in sorted(worst.items()):
        try:
            date_end    = _date.fromisoformat(date_end_str)
            date_export = _date.fromisoformat(date_export_str)
        except ValueError:
            continue

        if date_end > date_export:
            days_late = (date_end - date_export).days
            late_warnings.append(
                f"LATE_WARNING|{cmd}|{date_end_str}|{date_export_str}|{days_late}"
            )
        elif date_export < today:
            days_late = (today - date_export).days
            late_warnings.append(
                f"LATE_WARNING|{cmd}|{date_end_str}|{date_export_str}|{days_late}"
            )

    return late_warnings

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():

    # 1. Warmup LLM (Ollama chat)
    print("[STARTUP] Pre-loading Mistral...")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            await client.post(f"{OLLAMA_URL}/api/chat", json={
                "model":    OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                "stream":   False,
                "keep_alive": "10m",
                "options":  {"num_predict": 1},
            })
        print("[STARTUP] Mistral ready.")
    except Exception as e:
        print(f"[STARTUP] LLM warmup skipped: {e}")

    # 2. Warmup embeddings model
    print("[STARTUP] Pre-loading embeddings model...")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            await client.post(f"{OLLAMA_URL}/api/embeddings", json={
                "model":  "nomic-embed-text",
                "prompt": "warmup",
            })
        print("[STARTUP] Embeddings ready.")
    except Exception as e:
        print(f"[STARTUP] Embedding warmup skipped: {e}")

    # 3. Build FAISS semantic index (RAG)
    # FIX: wrapped in try/except so a DB error (e.g. empty tables on first run)
    # never blocks the chatbot from starting.
    print("[STARTUP] Building FAISS index...")
    try:
        await build_index()
        print("[STARTUP] FAISS index ready.")
    except Exception as e:
        print(f"[STARTUP] FAISS index failed (chatbot still works via SQL-only): {e}")

    # NOTE: ensure_domain_knowledge_indexed() was removed — it was never defined
    # and caused a NameError on every startup.

# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/api/planning/health")
async def health():
    # FIX: was importing from 'chatBotSystem.rag.rag_engine' (wrong path + typo)
    # and referencing '_faiss_index' which doesn't exist there.
    # Now reads directly from the _store singleton imported at the top of this file.
    return {
        "status":      "ok",
        "faissVectors": _faiss_store.index.ntotal if _faiss_store.index is not None else 0,
        "faissReady":  _faiss_store.is_ready(),
    }

# ---------------------------------------------------------------------------
# After planning saved → FAISS refresh hook
# ---------------------------------------------------------------------------

async def _after_planning_saved(planning_id: int):
    try:
        await rebuild_for_planning(planning_id)
    except Exception as e:
        print(f"[FAISS] Refresh failed for planning {planning_id}: {e}")

# ---------------------------------------------------------------------------
# Run planning endpoint
# ---------------------------------------------------------------------------

@app.post("/api/planning/run", response_model=RunResponse)
async def run_planning(req: RunRequest):

    # Anchor datetime
    if req.startDatetime:
        try:
            _start = datetime.fromisoformat(req.startDatetime)
        except ValueError:
            raise HTTPException(422, "Invalid ISO datetime format")
    else:
        _start = datetime.now()

    now_pm    = _start.hour * 60 + _start.minute
    today_iso = _start.date().isoformat()

    try:
        commandes, machines, ops_by_recette = await load_live_data(
            req.token, req.commandeIds
        )
    except Exception as e:
        raise HTTPException(500, f"Data loading error: {str(e)}")

    machines_ok = [m for m in machines if m.is_available()]

    if not machines_ok:
        raise HTTPException(400, "No machines available")

    warnings = validate(commandes, machines, ops_by_recette)

    lns_state = run_lns(
        commandes, ops_by_recette, machines_ok,
        max_machines_per_op=max(1, min(3, req.maxMachinesPerOp))
    )

    # CP-SAT or fallback
    if not HAS_ORTOOLS:
        rows        = lns_fallback(commandes, ops_by_recette, lns_state)
        makespan_pm = max((r.endPM for r in rows), default=0)
        rows        = _shift_rows(rows, now_pm)
        late_warnings = _collect_late_warnings(rows, _start.date())

        return RunResponse(
            status="feasible_lns_only",
            makespanDays=makespan_pm // PPD,
            makespanPM=makespan_pm,
            startDate=today_iso,
            rows=rows,
            warnings=warnings + late_warnings + ["OR-Tools missing"],
        )

    result = run_cpsat(
        commandes, ops_by_recette, machines_ok, lns_state,
        horizon=(max(date_to_day_offset(c.DateExport) for c in commandes) + 15) * PPD
    )

    rows          = result.extract_rows(commandes, ops_by_recette, lns_state)
    rows          = _shift_rows(rows, now_pm)
    late_warnings = _collect_late_warnings(rows, _start.date())

    return RunResponse(
        status="optimal",
        makespanDays=result.makespan_pm // PPD,
        makespanPM=result.makespan_pm,
        startDate=today_iso,
        rows=rows,
        warnings=warnings + late_warnings,
    )