"""
api.py  —  FastAPI entry point for the Denim Washing Production Planner
========================================================================
UPDATED: Added RAG analysis endpoint + FAISS index startup
         Schedule is anchored to a user-chosen datetime (or now if not provided).
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
from chat import router as chat_router, OLLAMA_URL, OLLAMA_MODEL
from rag.analyze_router import router as analyze_router
from rag.rag_engine import ensure_domain_knowledge_indexed


# ---------------------------------------------------------------------------
# Helper — shift solver rows so PM=0 maps to the chosen start minute
# ---------------------------------------------------------------------------

def _shift_rows(rows: list, now_pm: int) -> list:
    """
    The solver produces PM values where PM=0 = 00h00 today.
    Shift every startPM / endPM by now_pm so the schedule starts
    at the actual chosen time, not midnight.
    Also recompute dateStart / dateEnd to match the shifted times.
    """
    shifted = []
    for r in rows:
        s_pm = r.startPM + now_pm
        e_pm = r.endPM   + now_pm
        s_day = pm_to_clock(s_pm)[0]
        e_day = pm_to_clock(e_pm)[0]
        shifted.append(r.model_copy(update={
            "startPM":   s_pm,
            "endPM":     e_pm,
            "dateStart": working_day_date(s_day).isoformat(),
            "dateEnd":   working_day_date(e_day).isoformat(),
        }))
    return shifted


app = FastAPI(title="Denim Planner Optimiser", version="2.0.0")
app.include_router(chat_router)
app.include_router(analyze_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "https://localhost:4200"],
    allow_credentials=True,
    allow_headers=["*"],
    allow_methods=["*"],
)


# ---------------------------------------------------------------------------
# Startup — warm up Ollama + build FAISS index
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    # 1. Pre-load Mistral into Ollama memory
    print("[STARTUP] Pre-loading Mistral into Ollama...")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            await client.post(f"{OLLAMA_URL}/api/chat", json={
                "model":      OLLAMA_MODEL,
                "messages":   [{"role": "user", "content": "hi"}],
                "stream":     False,
                "keep_alive": "10m",
                "options":    {"num_predict": 1},
            })
        print("[STARTUP] Mistral ready.")
    except Exception as e:
        print(f"[STARTUP] Ollama warmup skipped: {e}")

    # 2. Pre-load nomic-embed-text into Ollama memory
    print("[STARTUP] Pre-loading nomic-embed-text...")
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            await client.post(f"{OLLAMA_URL}/api/embeddings", json={
                "model":  "nomic-embed-text",
                "prompt": "warmup",
            })
        print("[STARTUP] nomic-embed-text ready.")
    except Exception as e:
        print(f"[STARTUP] Embed model warmup skipped: {e}")

    # 3. Build / reload FAISS domain knowledge index
    print("[STARTUP] Initialising FAISS RAG index...")
    try:
        await ensure_domain_knowledge_indexed()
        print("[STARTUP] RAG index ready.")
    except Exception as e:
        print(f"[STARTUP] FAISS index failed (RAG degraded): {e}")


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Late-delivery warning helper
# ---------------------------------------------------------------------------

def _collect_late_warnings(rows: list, start_date) -> list[str]:
    """
    Compare each row's dateEnd against its dateExport.
    Returns one LATE_WARNING token per *commande* that will finish after
    its export deadline, so the frontend and the warning banner both know.

    Token format:
      LATE_WARNING|<numeroCommande>|<dateEnd>|<dateExport>|<daysLate>
    """
    from datetime import date as _date, timedelta

    # today's date is used to check if dateExport is already in the past
    today = _date.today()

    # Collect worst (latest) dateEnd per commande
    worst: dict[str, tuple[str, str]] = {}  # cmd -> (dateEnd, dateExport)
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
            # Deadline already passed before planning even started
            days_late = (today - date_export).days
            late_warnings.append(
                f"LATE_WARNING|{cmd}|{date_end_str}|{date_export_str}|{days_late}"
            )

    return late_warnings


@app.get("/api/planning/health")
async def health():
    from rag.rag_engine import _faiss_index
    return {
        "status":      "ok",
        "ortools":     HAS_ORTOOLS,
        "startDate":   date.today().isoformat(),
        "faissVectors": _faiss_index.index.ntotal if _faiss_index.index else 0,
    }


# ---------------------------------------------------------------------------
# Run planning
# ---------------------------------------------------------------------------

@app.post("/api/planning/run", response_model=RunResponse)
async def run_planning(req: RunRequest):

    # ── Determine the schedule anchor datetime ────────────────────────────
    # Priority: explicit startDatetime from the user → datetime.now() fallback.
    if req.startDatetime:
        try:
            _start = datetime.fromisoformat(req.startDatetime)
        except ValueError:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid startDatetime format: '{req.startDatetime}'. "
                       f"Expected ISO-8601, e.g. '2026-05-10T08:30:00'.",
            )
    else:
        _start = datetime.now()

    # now_pm  : intra-day minute offset (0–1439), used to shift the solver output
    # today_iso: calendar date that becomes the Gantt's day-0 label
    now_pm      = _start.hour * 60 + _start.minute
    today_iso   = _start.date().isoformat()

    print(f"\n{'='*60}")
    print(f"[API] POST /api/planning/run")
    print(f"[API] commandeIds = {req.commandeIds}")
    print(f"[API] maxMachinesPerOp = {req.maxMachinesPerOp}")
    print(f"[API] startDatetime = {req.startDatetime!r}  →  {_start.strftime('%Y-%m-%d %H:%M')}")
    print(f"[API] now_pm = {now_pm}")
    print(f"{'='*60}")

    try:
        commandes, machines, ops_by_recette = await load_live_data(
            req.token, req.commandeIds
        )
    except Exception as e:
        print(f"[API] ERROR loading data: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Data loading error: {str(e)}")

    print(f"[API] Loaded {len(commandes)} commandes, {len(machines)} machines")

    if not commandes:
        raise HTTPException(status_code=400, detail="No 'En attente' commandes found")

    machines_ok = [m for m in machines if m.is_available()]
    if not machines_ok:
        raise HTTPException(status_code=400, detail="No functional machines available")

    warnings = validate(commandes, machines, ops_by_recette)
    if warnings:
        print(f"[API] Validation warnings: {warnings}")

    max_machines_per_op = max(1, min(3, req.maxMachinesPerOp))

    try:
        lns_state = run_lns(
            commandes, ops_by_recette, machines_ok,
            max_machines_per_op=max_machines_per_op,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LNS solver error: {str(e)}")

    if max_machines_per_op > 1:
        try:
            split_warnings = collect_split_warnings(
                commandes, ops_by_recette, lns_state, max_machines_per_op
            )
            warnings = warnings + split_warnings
        except Exception as e:
            print(f"[API] WARNING: collect_split_warnings failed: {e}")

    if not HAS_ORTOOLS:
        try:
            rows = lns_fallback(commandes, ops_by_recette, lns_state)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LNS fallback error: {str(e)}")
        makespan_pm = max((r.endPM for r in rows), default=0)  # elapsed minutes (solver-relative)
        rows = _shift_rows(rows, now_pm)
        # NOTE: do NOT add now_pm to makespan_pm.
        # makespan_pm is an *elapsed duration* from the schedule start, not an
        # absolute PM value.  Adding now_pm would double-count the start offset
        # and inflate the reported makespan (e.g. 15h displayed as 1d 8h).
        late_warnings = _collect_late_warnings(rows, _start.date())
        return RunResponse(
            status="feasible_lns_only",
            makespanDays=makespan_pm // PPD,
            makespanPM=makespan_pm,
            startDate=today_iso,
            rows=rows,
            warnings=warnings + late_warnings + ["OR-Tools not installed; LNS fallback used"],
        )

    try:
        max_export_day = max(date_to_day_offset(c.DateExport) for c in commandes)
        horizon        = (max_export_day + 15) * PPD
        result = run_cpsat(
            commandes, ops_by_recette, machines_ok, lns_state, horizon
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CP-SAT error: {str(e)}")

    makespan_pm = result.makespan_pm   # elapsed minutes (solver-relative)
    rows = result.extract_rows(commandes, ops_by_recette, lns_state)
    rows = _shift_rows(rows, now_pm)
    # NOTE: do NOT add now_pm to makespan_pm.
    # makespan_pm is an *elapsed duration* from the schedule start, not an
    # absolute PM value.  Adding now_pm would double-count the start offset
    # and inflate the reported makespan (e.g. 15h displayed as 1d 8h).

    print(f"[API] Done. makespan={makespan_pm} PM ({makespan_pm//PPD}d {makespan_pm%PPD//60}h), {len(rows)} rows generated")

    late_warnings = _collect_late_warnings(rows, _start.date())
    return RunResponse(
        status="optimal" if makespan_pm < horizon else "feasible",
        makespanDays=makespan_pm // PPD,
        makespanPM=makespan_pm,
        startDate=today_iso,
        rows=rows,
        warnings=warnings + late_warnings,
    )